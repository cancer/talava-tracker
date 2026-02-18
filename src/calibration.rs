use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;

use opencv::{
    calib3d,
    core::{Mat, Size, TermCriteria, TermCriteria_Type, Vector},
    objdetect::{
        self, CharucoBoard, CharucoDetector, Dictionary,
        PredefinedDictionaryType,
    },
    prelude::*,
};

use crate::config::CalibrationConfig;

// --- データ構造 ---

/// ボードパラメータ（再現用に保存）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoardParams {
    pub dictionary: String,
    pub squares_x: i32,
    pub squares_y: i32,
    pub square_length: f32,
    pub marker_length: f32,
}

/// 単一カメラのキャリブレーション結果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraCalibration {
    pub camera_index: i32,
    pub width: u32,
    pub height: u32,
    /// 内部パラメータ行列 K (row-major 3x3)
    pub intrinsic_matrix: [f64; 9],
    /// 歪み係数
    pub dist_coeffs: Vec<f64>,
    /// 回転ベクトル (Rodrigues)
    pub rvec: [f64; 3],
    /// 並進ベクトル
    pub tvec: [f64; 3],
    /// 再投影誤差
    pub reprojection_error: f64,
}

/// マルチカメラキャリブレーション結果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiCameraCalibration {
    pub board: BoardParams,
    pub cameras: Vec<CameraCalibration>,
}

// --- Save / Load ---

pub fn save_calibration(path: &str, cal: &MultiCameraCalibration) -> Result<()> {
    let json = serde_json::to_string_pretty(cal)?;
    fs::write(path, json).context("Failed to write calibration file")?;
    Ok(())
}

pub fn load_calibration(path: &str) -> Result<MultiCameraCalibration> {
    let content = fs::read_to_string(path).context("Failed to read calibration file")?;
    let cal: MultiCameraCalibration = serde_json::from_str(&content)?;
    Ok(cal)
}

// --- 辞書ヘルパー ---

pub fn parse_dictionary(name: &str) -> Result<Dictionary> {
    let dict_type = match name {
        "DICT_4X4_50" => PredefinedDictionaryType::DICT_4X4_50,
        "DICT_4X4_100" => PredefinedDictionaryType::DICT_4X4_100,
        "DICT_4X4_250" => PredefinedDictionaryType::DICT_4X4_250,
        "DICT_4X4_1000" => PredefinedDictionaryType::DICT_4X4_1000,
        "DICT_5X5_50" => PredefinedDictionaryType::DICT_5X5_50,
        "DICT_5X5_100" => PredefinedDictionaryType::DICT_5X5_100,
        "DICT_5X5_250" => PredefinedDictionaryType::DICT_5X5_250,
        "DICT_5X5_1000" => PredefinedDictionaryType::DICT_5X5_1000,
        "DICT_6X6_50" => PredefinedDictionaryType::DICT_6X6_50,
        "DICT_6X6_100" => PredefinedDictionaryType::DICT_6X6_100,
        "DICT_6X6_250" => PredefinedDictionaryType::DICT_6X6_250,
        "DICT_6X6_1000" => PredefinedDictionaryType::DICT_6X6_1000,
        _ => bail!("Unknown dictionary: {}", name),
    };
    objdetect::get_predefined_dictionary(dict_type).context("Failed to get predefined dictionary")
}

/// ChArUcoボードを作成
pub fn create_board(config: &CalibrationConfig) -> Result<CharucoBoard> {
    let dict = parse_dictionary(&config.dictionary)?;
    let size = Size::new(config.squares_x, config.squares_y);
    CharucoBoard::new_def(size, config.square_length, config.marker_length, &dict)
        .context("Failed to create CharucoBoard")
}

/// ChArUcoDetectorを作成（広角・斜め検出対応）
pub fn create_detector(board: &CharucoBoard) -> Result<CharucoDetector> {
    use objdetect::{CharucoParameters, DetectorParameters, DetectorParametersTrait, RefineParameters};

    let mut det_params = DetectorParameters::default()?;
    // 歪んだ四角形をより寛容に受け入れる（デフォルト0.03）
    det_params.set_polygonal_approx_accuracy_rate(0.08);
    // パースペクティブ補正の解像度を上げる（デフォルト4）
    det_params.set_perspective_remove_pixel_per_cell(8);
    // 適応的閾値の探索範囲を広げる（デフォルト max=23）
    det_params.set_adaptive_thresh_win_size_max(53);

    let charuco_params = CharucoParameters::default()?;
    let refine_params = RefineParameters::new(10.0, 3.0, true)?;

    CharucoDetector::new(board, &charuco_params, &det_params, refine_params)
        .context("Failed to create CharucoDetector")
}

// --- 内部パラメータキャリブレーション ---

/// 1フレームからChArUcoコーナーを検出
///
/// 戻り値: (corners, ids, corner_count)
/// corner_countが0の場合もcornersとidsは空のMatが返る。
pub fn detect_charuco_corners(
    detector: &CharucoDetector,
    image: &Mat,
) -> Result<(Mat, Mat, i32)> {
    let mut corners = Mat::default();
    let mut ids = Mat::default();

    detector
        .detect_board_def(image, &mut corners, &mut ids)
        .context("detect_board failed")?;

    let n = ids.rows();
    Ok((corners, ids, n))
}

/// 蓄積されたコーナーから内部パラメータをキャリブレーション
///
/// - all_corners: 各フレームのChArUcoコーナー座標
/// - all_ids: 各フレームのChArUcoコーナーID
/// - board: ChArUcoボード
/// - image_size: 画像サイズ
///
/// 戻り値: (camera_matrix, dist_coeffs, reprojection_error)
pub fn calibrate_intrinsics(
    all_corners: &[Mat],
    all_ids: &[Mat],
    board: &CharucoBoard,
    image_size: Size,
) -> Result<(Mat, Mat, f64)> {
    if all_corners.is_empty() {
        bail!("No calibration frames provided");
    }

    // match_image_points で 3D-2D 対応を構築
    let mut all_obj_points = Vector::<Mat>::new();
    let mut all_img_points = Vector::<Mat>::new();

    for i in 0..all_corners.len() {
        let mut obj_pts = Mat::default();
        let mut img_pts = Mat::default();
        board
            .match_image_points(&all_corners[i], &all_ids[i], &mut obj_pts, &mut img_pts)
            .context("match_image_points failed")?;

        if obj_pts.rows() >= 6 {
            all_obj_points.push(obj_pts);
            all_img_points.push(img_pts);
        }
    }

    if all_obj_points.len() < 3 {
        bail!(
            "Not enough valid frames for calibration (got {}, need >= 3)",
            all_obj_points.len()
        );
    }

    let mut camera_matrix = Mat::default();
    let mut dist_coeffs = Mat::default();
    let mut rvecs = Mat::default();
    let mut tvecs = Mat::default();

    let criteria = TermCriteria::new(
        TermCriteria_Type::COUNT as i32 + TermCriteria_Type::EPS as i32,
        100,
        1e-6,
    )?;

    let error = calib3d::calibrate_camera(
        &all_obj_points,
        &all_img_points,
        image_size,
        &mut camera_matrix,
        &mut dist_coeffs,
        &mut rvecs,
        &mut tvecs,
        0,
        criteria,
    )
    .context("calibrate_camera failed")?;

    Ok((camera_matrix, dist_coeffs, error))
}

// --- 外部パラメータキャリブレーション ---

/// 1フレームからsolvePnPで外部パラメータを推定
///
/// 戻り値: (rvec, tvec)
pub fn estimate_pose(
    corners: &Mat,
    ids: &Mat,
    board: &CharucoBoard,
    camera_matrix: &Mat,
    dist_coeffs: &Mat,
) -> Result<Option<(Mat, Mat)>> {
    let mut obj_pts = Mat::default();
    let mut img_pts = Mat::default();
    board
        .match_image_points(corners, ids, &mut obj_pts, &mut img_pts)
        .context("match_image_points failed")?;

    if obj_pts.rows() < 6 {
        return Ok(None);
    }

    let mut rvec = Mat::default();
    let mut tvec = Mat::default();

    let ok = calib3d::solve_pnp(
        &obj_pts,
        &img_pts,
        camera_matrix,
        dist_coeffs,
        &mut rvec,
        &mut tvec,
        false,
        calib3d::SOLVEPNP_ITERATIVE,
    )
    .context("solvePnP failed")?;

    if !ok {
        return Ok(None);
    }

    Ok(Some((rvec, tvec)))
}

/// カメラ0を原点として相対変換を計算
///
/// 入力: 各カメラの (rvec, tvec) — ボード座標系→カメラ座標系
/// 出力: 各カメラの (rvec, tvec) — カメラ0座標系基準
pub fn compute_relative_poses(
    poses: &[(Mat, Mat)],
) -> Result<Vec<([f64; 3], [f64; 3])>> {
    if poses.is_empty() {
        bail!("No poses provided");
    }

    /// rvec(Mat 3x1) → 回転行列 [[f64;3];3]
    fn rvec_to_rmat(rvec: &Mat) -> Result<[[f64; 3]; 3]> {
        let mut rmat = Mat::default();
        let mut jacobian = Mat::default();
        calib3d::rodrigues(rvec, &mut rmat, &mut jacobian)
            .context("Rodrigues failed")?;
        let mut r = [[0.0f64; 3]; 3];
        for row in 0..3 {
            for col in 0..3 {
                r[row][col] = *rmat.at_2d::<f64>(row as i32, col as i32)?;
            }
        }
        Ok(r)
    }

    /// tvec(Mat 3x1 or 1x3) → [f64;3]
    fn mat_to_vec3(m: &Mat) -> Result<[f64; 3]> {
        if m.rows() == 1 {
            Ok([*m.at_2d::<f64>(0, 0)?, *m.at_2d::<f64>(0, 1)?, *m.at_2d::<f64>(0, 2)?])
        } else {
            Ok([*m.at_2d::<f64>(0, 0)?, *m.at_2d::<f64>(1, 0)?, *m.at_2d::<f64>(2, 0)?])
        }
    }

    /// 回転行列 → rvec [f64;3]
    fn rmat_to_rvec(r: &[[f64; 3]; 3]) -> Result<[f64; 3]> {
        let mut rmat = Mat::zeros(3, 3, opencv::core::CV_64F)?.to_mat()?;
        for row in 0..3 {
            for col in 0..3 {
                *rmat.at_2d_mut::<f64>(row as i32, col as i32)? = r[row][col];
            }
        }
        let mut rvec = Mat::default();
        let mut jacobian = Mat::default();
        calib3d::rodrigues(&rmat, &mut rvec, &mut jacobian)
            .context("Rodrigues failed")?;
        if rvec.rows() == 1 {
            Ok([*rvec.at_2d::<f64>(0, 0)?, *rvec.at_2d::<f64>(0, 1)?, *rvec.at_2d::<f64>(0, 2)?])
        } else {
            Ok([*rvec.at_2d::<f64>(0, 0)?, *rvec.at_2d::<f64>(1, 0)?, *rvec.at_2d::<f64>(2, 0)?])
        }
    }

    fn mul_3x3(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
        let mut c = [[0.0f64; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    c[i][j] += a[i][k] * b[k][j];
                }
            }
        }
        c
    }

    fn mul_3x3_vec(m: &[[f64; 3]; 3], v: &[f64; 3]) -> [f64; 3] {
        [
            m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
            m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
            m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
        ]
    }

    fn transpose(m: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
        [
            [m[0][0], m[1][0], m[2][0]],
            [m[0][1], m[1][1], m[2][1]],
            [m[0][2], m[1][2], m[2][2]],
        ]
    }

    let r0 = rvec_to_rmat(&poses[0].0)?;
    let t0 = mat_to_vec3(&poses[0].1)?;
    let r0_t = transpose(&r0);

    let mut results = Vec::new();

    for (i, (rvec, tvec)) in poses.iter().enumerate() {
        if i == 0 {
            results.push(([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]));
            continue;
        }

        let ri = rvec_to_rmat(rvec)?;
        let ti = mat_to_vec3(tvec)?;

        // R_rel = R_i * R_0^T
        let r_rel = mul_3x3(&ri, &r0_t);

        // t_rel = t_i - R_rel * t_0
        let rt0 = mul_3x3_vec(&r_rel, &t0);
        let t_rel = [ti[0] - rt0[0], ti[1] - rt0[1], ti[2] - rt0[2]];

        // R_rel → rvec
        let rv = rmat_to_rvec(&r_rel)?;

        results.push((rv, t_rel));
    }

    Ok(results)
}

// --- Mat ↔ 配列変換ヘルパー ---

/// 3x3 Mat (f64) → [f64; 9] row-major
pub fn mat3x3_to_array(mat: &Mat) -> Result<[f64; 9]> {
    if mat.rows() != 3 || mat.cols() != 3 {
        bail!("Expected 3x3 matrix, got {}x{}", mat.rows(), mat.cols());
    }
    let mut arr = [0.0f64; 9];
    for r in 0..3 {
        for c in 0..3 {
            arr[r * 3 + c] = *mat.at_2d::<f64>(r as i32, c as i32)?;
        }
    }
    Ok(arr)
}

/// Vec<f64> from Mat (Nx1 or 1xN)
pub fn mat_to_vec(mat: &Mat) -> Result<Vec<f64>> {
    let n = mat.rows().max(mat.cols()) as usize;
    let mut v = Vec::with_capacity(n);
    let is_row = mat.rows() == 1;
    for i in 0..n {
        let val = if is_row {
            *mat.at_2d::<f64>(0, i as i32)?
        } else {
            *mat.at_2d::<f64>(i as i32, 0)?
        };
        v.push(val);
    }
    Ok(v)
}
