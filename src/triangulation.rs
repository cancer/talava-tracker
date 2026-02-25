use nalgebra::{Matrix3, Matrix3x4, Matrix4, Vector3, Vector4};

/// カメラパラメータ（射影行列 P = K[R|t]）
#[derive(Debug, Clone)]
pub struct CameraParams {
    pub projection: Matrix3x4<f32>,
    pub image_width: f32,
    pub image_height: f32,
    /// 内部パラメータ行列（歪み補正用）
    pub intrinsic: Matrix3<f32>,
    /// 歪み係数 [k1, k2, p1, p2, k3]
    pub dist_coeffs: [f32; 5],
}

impl CameraParams {
    /// FOV + 位置 + 回転から射影行列を構築
    ///
    /// - fov_v_deg: 垂直画角（度）
    /// - width, height: 画像解像度
    /// - position: カメラ位置 [x, y, z] メートル
    /// - rotation_deg: カメラ回転 [rx, ry, rz] 度 (Euler XYZ)
    pub fn from_config(
        fov_v_deg: f32,
        width: u32,
        height: u32,
        position: [f32; 3],
        rotation_deg: [f32; 3],
    ) -> Self {
        let w = width as f32;
        let h = height as f32;
        let image_width = w;
        let image_height = h;

        // 内部パラメータ行列 K
        let fy = h / (2.0 * (fov_v_deg.to_radians() / 2.0).tan());
        let fx = fy; // 正方ピクセルを仮定
        let cx = w / 2.0;
        let cy = h / 2.0;

        let k = Matrix3::new(fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0);

        // 回転行列 R (Euler XYZ)
        let rx = rotation_deg[0].to_radians();
        let ry = rotation_deg[1].to_radians();
        let rz = rotation_deg[2].to_radians();

        let rot_x = Matrix3::new(
            1.0,
            0.0,
            0.0,
            0.0,
            rx.cos(),
            -rx.sin(),
            0.0,
            rx.sin(),
            rx.cos(),
        );
        let rot_y = Matrix3::new(
            ry.cos(),
            0.0,
            ry.sin(),
            0.0,
            1.0,
            0.0,
            -ry.sin(),
            0.0,
            ry.cos(),
        );
        let rot_z = Matrix3::new(
            rz.cos(),
            -rz.sin(),
            0.0,
            rz.sin(),
            rz.cos(),
            0.0,
            0.0,
            0.0,
            1.0,
        );
        let r = rot_z * rot_y * rot_x;

        // 並進ベクトル t = -R * position
        let pos = Vector3::new(position[0], position[1], position[2]);
        let t = -(r * pos);

        // P = K * [R | t]
        let mut rt = Matrix3x4::zeros();
        for i in 0..3 {
            for j in 0..3 {
                rt[(i, j)] = r[(i, j)];
            }
            rt[(i, 3)] = t[i];
        }

        let projection = k * rt;
        Self { projection, image_width, image_height, intrinsic: k, dist_coeffs: [0.0; 5] }
    }

    /// キャリブレーション結果から射影行列を構築
    ///
    /// - intrinsic: 内部パラメータ行列 K (row-major 3x3, f64)
    /// - rvec: 回転ベクトル (Rodrigues, f64)
    /// - tvec: 並進ベクトル (f64)
    pub fn from_calibration(intrinsic: &[f64; 9], dist_coeffs: &[f64; 5], rvec: &[f64; 3], tvec: &[f64; 3], width: u32, height: u32) -> Self {
        // K行列 (row-major → nalgebra column-major)
        let k = Matrix3::new(
            intrinsic[0] as f32, intrinsic[1] as f32, intrinsic[2] as f32,
            intrinsic[3] as f32, intrinsic[4] as f32, intrinsic[5] as f32,
            intrinsic[6] as f32, intrinsic[7] as f32, intrinsic[8] as f32,
        );

        // Rodrigues → 回転行列
        let theta = (rvec[0] * rvec[0] + rvec[1] * rvec[1] + rvec[2] * rvec[2]).sqrt();
        let r = if theta < 1e-10 {
            Matrix3::<f32>::identity()
        } else {
            let kx = rvec[0] / theta;
            let ky = rvec[1] / theta;
            let kz = rvec[2] / theta;
            let ct = theta.cos() as f32;
            let st = theta.sin() as f32;
            let vt = 1.0 - ct;
            let (kx, ky, kz) = (kx as f32, ky as f32, kz as f32);

            Matrix3::new(
                ct + kx * kx * vt,      kx * ky * vt - kz * st, kx * kz * vt + ky * st,
                ky * kx * vt + kz * st, ct + ky * ky * vt,      ky * kz * vt - kx * st,
                kz * kx * vt - ky * st, kz * ky * vt + kx * st, ct + kz * kz * vt,
            )
        };

        let t = Vector3::new(tvec[0] as f32, tvec[1] as f32, tvec[2] as f32);

        // P = K * [R | t]
        let mut rt = Matrix3x4::zeros();
        for i in 0..3 {
            for j in 0..3 {
                rt[(i, j)] = r[(i, j)];
            }
            rt[(i, 3)] = t[i];
        }

        let projection = k * rt;
        let dc: [f32; 5] = [
            dist_coeffs[0] as f32, dist_coeffs[1] as f32,
            dist_coeffs[2] as f32, dist_coeffs[3] as f32,
            dist_coeffs[4] as f32,
        ];

        // fx/fy比の検証: 正方ピクセルなら~1.0、異常値はキャリブレーション品質の問題を示す
        let fx = intrinsic[0];
        let fy = intrinsic[4];
        if fy.abs() > 1e-10 {
            let ratio = fx / fy;
            if ratio > 1.2 || ratio < 0.8 {
                eprintln!("WARNING: from_calibration() fx/fy ratio={:.3} (fx={:.1}, fy={:.1}). Non-square pixels suggest calibration issue.",
                    ratio, fx, fy);
            }
        }

        Self {
            projection,
            image_width: width as f32,
            image_height: height as f32,
            intrinsic: k,
            dist_coeffs: dc,
        }
    }

    /// 実際の解像度に合わせて内部パラメータと射影行列をリスケール
    ///
    /// キャリブレーション時と異なる解像度でカメラを使用する場合、
    /// K行列のfx,fy,cx,cyを比例調整してP=K[R|t]を再計算する。
    /// アスペクト比が異なる場合（例: 1920x1080 → 720x1280）は
    /// カメラの物理的な向き変更を意味し、この関数では対応できない。
    /// その場合はfalseを返す。
    pub fn rescale_to(&mut self, actual_width: u32, actual_height: u32) -> bool {
        let cal_w = self.image_width;
        let cal_h = self.image_height;
        let act_w = actual_width as f32;
        let act_h = actual_height as f32;

        // 同一なら何もしない
        if (cal_w - act_w).abs() < 1.0 && (cal_h - act_h).abs() < 1.0 {
            return true;
        }

        // アスペクト比チェック
        let cal_ratio = cal_w / cal_h;
        let act_ratio = act_w / act_h;
        if (cal_ratio - act_ratio).abs() / cal_ratio > 0.05 {
            // アスペクト比が5%以上異なる → 物理的向き変更の可能性
            return false;
        }

        // K行列をスケーリング: fx,cx → *sx, fy,cy → *sy
        let sx = act_w / cal_w;
        let sy = act_h / cal_h;

        // K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        // K_new = [[fx*sx, 0, cx*sx], [0, fy*sy, cy*sy], [0, 0, 1]]
        // P = K * [R|t] なので、P_new = S * P where S = diag(sx, sy, 1)
        // (Sは行に掛かるスケーリング)
        for j in 0..4 {
            self.projection[(0, j)] *= sx;
            self.projection[(1, j)] *= sy;
        }

        self.intrinsic[(0, 0)] *= sx; // fx
        self.intrinsic[(0, 2)] *= sx; // cx
        self.intrinsic[(1, 1)] *= sy; // fy
        self.intrinsic[(1, 2)] *= sy; // cy

        self.image_width = act_w;
        self.image_height = act_h;

        true
    }

    /// 歪んだピクセル座標を歪み補正して理想ピクセル座標に変換
    /// Newton-Raphson法による歪み補正（大きな歪み係数でも収束）
    pub fn undistort_point(&self, u_dist: f32, v_dist: f32) -> (f32, f32) {
        let fx = self.intrinsic[(0, 0)];
        let fy = self.intrinsic[(1, 1)];
        let cx = self.intrinsic[(0, 2)];
        let cy = self.intrinsic[(1, 2)];
        let [k1, k2, p1, p2, k3] = self.dist_coeffs;

        // 歪み係数がゼロなら補正不要
        if k1 == 0.0 && k2 == 0.0 && p1 == 0.0 && p2 == 0.0 && k3 == 0.0 {
            return (u_dist, v_dist);
        }

        // ピクセル→正規化カメラ座標（歪みあり = ターゲット）
        let xd = (u_dist - cx) / fx;
        let yd = (v_dist - cy) / fy;

        // Newton-Raphson: 順方向歪みモデル f(x,y) = target を解く
        // f_x(x,y) = x*R + 2*p1*x*y + p2*(r2 + 2*x^2)
        // f_y(x,y) = y*R + p1*(r2 + 2*y^2) + 2*p2*x*y
        // R = 1 + k1*r2 + k2*r4 + k3*r6
        let mut x = xd;
        let mut y = yd;
        let mut best_x = x;
        let mut best_y = y;
        let mut best_residual = f32::MAX;

        for _ in 0..30 {
            let r2 = x * x + y * y;
            let r4 = r2 * r2;
            let r6 = r4 * r2;
            let radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;
            let dr_dr2 = k1 + 2.0 * k2 * r2 + 3.0 * k3 * r4;

            // 順方向歪み適用
            let fx_val = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x) - xd;
            let fy_val = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y - yd;

            let residual = fx_val * fx_val + fy_val * fy_val;
            if residual < best_residual {
                best_residual = residual;
                best_x = x;
                best_y = y;
            }

            // 十分収束したら終了
            if residual < 1e-12 {
                break;
            }

            // ヤコビアン
            let j00 = radial + 2.0 * x * x * dr_dr2 + 2.0 * p1 * y + 6.0 * p2 * x;
            let j01 = 2.0 * x * y * dr_dr2 + 2.0 * p1 * x + 2.0 * p2 * y;
            let j10 = j01; // 対称
            let j11 = radial + 2.0 * y * y * dr_dr2 + 6.0 * p1 * y + 2.0 * p2 * x;

            let det = j00 * j11 - j01 * j10;
            if det.abs() < 1e-10 {
                break; // 特異ヤコビアン → best値を使用
            }

            let dx = -(j11 * fx_val - j01 * fy_val) / det;
            let dy = -(-j10 * fx_val + j00 * fy_val) / det;

            x += dx;
            y += dy;
        }

        (best_x * fx + cx, best_y * fy + cy)
    }
}

/// 単一3D点のDLT三角測量
///
/// N台のカメラの2D観測から3D座標を推定。
/// 各カメラについて x × (P · X) = 0 の形で2行追加し、SVDで解く。
pub fn triangulate_point(cameras: &[&CameraParams], points_2d: &[(f32, f32)]) -> (f32, f32, f32) {
    let n = cameras.len();
    assert!(n >= 2);

    // A行列: 2N × 4
    let mut a = Matrix4::zeros(); // A^T * A (4x4) を直接構築

    for i in 0..n {
        let p = &cameras[i].projection;
        let (u, v) = points_2d[i];

        // row1: u * P[2] - P[0]
        // row2: v * P[2] - P[1]
        let row1 = Vector4::new(
            u * p[(2, 0)] - p[(0, 0)],
            u * p[(2, 1)] - p[(0, 1)],
            u * p[(2, 2)] - p[(0, 2)],
            u * p[(2, 3)] - p[(0, 3)],
        );
        let row2 = Vector4::new(
            v * p[(2, 0)] - p[(1, 0)],
            v * p[(2, 1)] - p[(1, 1)],
            v * p[(2, 2)] - p[(1, 2)],
            v * p[(2, 3)] - p[(1, 3)],
        );

        a += row1 * row1.transpose();
        a += row2 * row2.transpose();
    }

    // SVD of A^T*A — 最小固有値の固有ベクトルが解
    let eigen = a.symmetric_eigen();
    let mut min_idx = 0;
    let mut min_val = eigen.eigenvalues[0].abs();
    for i in 1..4 {
        let v = eigen.eigenvalues[i].abs();
        if v < min_val {
            min_val = v;
            min_idx = i;
        }
    }

    let x = eigen.eigenvectors.column(min_idx);
    let w = x[3];
    if w.abs() < 1e-10 {
        return (0.0, 0.0, 0.0);
    }

    (x[0] / w, x[1] / w, x[2] / w)
}

/// 3D点のリプロジェクションエラーを計算
pub fn reprojection_error(
    cameras: &[&CameraParams],
    points_2d: &[(f32, f32)],
    point_3d: &Vector4<f32>,
) -> f32 {
    let mut max_err = 0.0f32;
    for i in 0..cameras.len() {
        let projected = cameras[i].projection * point_3d;
        if projected[2].abs() < 1e-6 {
            return f32::MAX;
        }
        let u_proj = projected[0] / projected[2];
        let v_proj = projected[1] / projected[2];
        let (u_obs, v_obs) = points_2d[i];
        let err = ((u_proj - u_obs).powi(2) + (v_proj - v_obs).powi(2)).sqrt();
        max_err = max_err.max(err);
    }
    max_err
}

/// N台のカメラの2Dポーズから3Dポーズを三角測量
///
/// - 各キーポイントについて、confidence_threshold以上のカメラの観測を集める
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_camera_params_from_config() {
        let params = CameraParams::from_config(55.0, 640, 480, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]);
        // 原点カメラ、回転なし: P ≈ K * [I | 0]
        // P[0,2] ≈ cx = 320, P[1,2] ≈ cy = 240
        assert!((params.projection[(0, 2)] - 320.0).abs() < 1.0);
        assert!((params.projection[(1, 2)] - 240.0).abs() < 1.0);
    }

    #[test]
    fn test_triangulate_two_cameras() {
        // カメラ1: 原点、前方(+Z)を向く
        let cam1 = CameraParams::from_config(55.0, 640, 480, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]);
        // カメラ2: 右に1m、前方を向く
        let cam2 = CameraParams::from_config(55.0, 640, 480, [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]);

        // 3D点 (0.5, 0.0, 3.0) を各カメラに投影
        let target = Vector4::new(0.5, 0.0, 3.0, 1.0);

        let proj1 = cam1.projection * target;
        let u1 = proj1[0] / proj1[2];
        let v1 = proj1[1] / proj1[2];

        let proj2 = cam2.projection * target;
        let u2 = proj2[0] / proj2[2];
        let v2 = proj2[1] / proj2[2];

        let cameras = [&cam1, &cam2];
        let points = [(u1, v1), (u2, v2)];
        let (x, y, z) = triangulate_point(&cameras, &points);

        assert!(
            (x - 0.5).abs() < 0.01,
            "x: expected 0.5, got {}",
            x
        );
        assert!(y.abs() < 0.01, "y: expected 0.0, got {}", y);
        assert!(
            (z - 3.0).abs() < 0.01,
            "z: expected 3.0, got {}",
            z
        );
    }

    /// calibration.jsonの実データを使った三角測量シミュレーション
    /// 既知の3D点を各カメラに投影→三角測量→復元精度を検証
    #[test]
    fn test_calibration_data_self_consistency() {
        let dc = [0.0f64; 5]; // 歪み補正無効（tracker_bevy.rsと同じ）

        // Camera 0: reference
        let cam0 = CameraParams::from_calibration(
            &[1272.7304511139373, 0.0, 962.2823957723114,
              0.0, 1215.9025385457269, 504.1560959749078,
              0.0, 0.0, 1.0],
            &dc,
            &[0.0, 0.0, 0.0],
            &[0.0, 0.0, 0.0],
            1920, 1080,
        );
        // Camera 3
        let cam3 = CameraParams::from_calibration(
            &[232.02611418613375, 0.0, 950.3215350268297,
              0.0, 422.2621858675812, 545.5253015655518,
              0.0, 0.0, 1.0],
            &dc,
            &[-0.23531352491872176, 0.1441572745289576, 0.17254723995808235],
            &[-0.16138131662249985, -0.7185672849291409, -0.871966102994757],
            1920, 1080,
        );
        // Camera 4
        let cam4 = CameraParams::from_calibration(
            &[456.399288994886, 0.0, 941.8959669069045,
              0.0, 1544.018695645093, 524.4074246034186,
              0.0, 0.0, 1.0],
            &dc,
            &[-0.1925106008619544, -0.5582459992900398, -0.06365120750007275],
            &[-1.2687434272642264, -0.5338435605797942, 1.6824551205347025],
            1920, 1080,
        );

        // 既知の3D点: 参照カメラ(cam0)の前方2mに立つ人
        let target = Vector4::new(0.0, 0.0, 2.0, 1.0);

        let all_cams: [(&CameraParams, &str); 3] = [
            (&cam0, "cam0"), (&cam3, "cam3"), (&cam4, "cam4"),
        ];

        // 各カメラに投影
        let mut projections = Vec::new();
        for (cam, name) in &all_cams {
            let proj = cam.projection * target;
            let w = proj[2];
            if w.abs() < 1e-6 {
                panic!("{}: point behind camera (w={})", name, w);
            }
            let u = proj[0] / w;
            let v = proj[1] / w;
            let in_frame = u >= 0.0 && u <= cam.image_width && v >= 0.0 && v <= cam.image_height;
            eprintln!("{}: projected ({:.1}, {:.1}) in_frame={} w={:.3}", name, u, v, in_frame, w);
            projections.push((u, v));
        }

        // 全ペアの三角測量を試行
        let pairs = [(0, 1, "cam0+cam3"), (0, 2, "cam0+cam4"), (1, 2, "cam3+cam4")];
        for (i, j, name) in &pairs {
            let pair_cams = [all_cams[*i].0, all_cams[*j].0];
            let pair_pts = [projections[*i], projections[*j]];
            let (x, y, z) = triangulate_point(&pair_cams, &pair_pts);
            let p3d = Vector4::new(x, y, z, 1.0);
            let err = reprojection_error(&pair_cams, &pair_pts, &p3d);

            eprintln!("{}: triangulated ({:.4}, {:.4}, {:.4}) reproj_err={:.2}px",
                name, x, y, z, err);

            // 自己整合性テスト: 同じ行列で投影→三角測量なので誤差はほぼゼロ
            assert!(err < 1.0, "{}: self-consistency reproj err too high: {:.2}px", name, err);
            assert!((x - 0.0).abs() < 0.05, "{}: x={:.4} expected ~0.0", name, x);
            assert!((y - 0.0).abs() < 0.05, "{}: y={:.4} expected ~0.0", name, y);
            assert!((z - 2.0).abs() < 0.05, "{}: z={:.4} expected ~2.0", name, z);
        }
    }

    /// ノイズ耐性テスト: 各カメラペアが2Dノイズにどの程度敏感か検証
    /// broken intrinsicsのカメラペアはノイズに対して不安定になるはず
    #[test]
    fn test_calibration_noise_sensitivity() {
        let dc = [0.0f64; 5];

        let cam0 = CameraParams::from_calibration(
            &[1272.7304511139373, 0.0, 962.2823957723114,
              0.0, 1215.9025385457269, 504.1560959749078,
              0.0, 0.0, 1.0],
            &dc, &[0.0, 0.0, 0.0], &[0.0, 0.0, 0.0], 1920, 1080,
        );
        let cam3 = CameraParams::from_calibration(
            &[232.02611418613375, 0.0, 950.3215350268297,
              0.0, 422.2621858675812, 545.5253015655518,
              0.0, 0.0, 1.0],
            &dc,
            &[-0.23531352491872176, 0.1441572745289576, 0.17254723995808235],
            &[-0.16138131662249985, -0.7185672849291409, -0.871966102994757],
            1920, 1080,
        );
        let cam4 = CameraParams::from_calibration(
            &[456.399288994886, 0.0, 941.8959669069045,
              0.0, 1544.018695645093, 524.4074246034186,
              0.0, 0.0, 1.0],
            &dc,
            &[-0.1925106008619544, -0.5582459992900398, -0.06365120750007275],
            &[-1.2687434272642264, -0.5338435605797942, 1.6824551205347025],
            1920, 1080,
        );

        let target = Vector4::new(0.0, 0.0, 2.0, 1.0);

        let all_cams: [(&CameraParams, &str); 3] = [
            (&cam0, "cam0"), (&cam3, "cam3"), (&cam4, "cam4"),
        ];

        // 正確な2D投影
        let mut projections = Vec::new();
        for (cam, _) in &all_cams {
            let proj = cam.projection * target;
            let u = proj[0] / proj[2];
            let v = proj[1] / proj[2];
            projections.push((u, v));
        }

        // ノイズ追加 (各カメラの2D点に±10pxのオフセット)
        let noise = 10.0f32;
        let pairs = [(0usize, 1usize, "cam0+cam3"), (0, 2, "cam0+cam4"), (1, 2, "cam3+cam4")];

        eprintln!("\n--- Noise sensitivity (±{:.0}px) ---", noise);
        for (i, j, name) in &pairs {
            let pair_cams = [all_cams[*i].0, all_cams[*j].0];
            // 4つの方向でノイズを試す
            let noisy_offsets = [
                (noise, 0.0, -noise, 0.0),
                (0.0, noise, 0.0, -noise),
                (noise, noise, -noise, -noise),
                (-noise, noise, noise, -noise),
            ];
            let mut max_3d_error = 0.0f32;
            for (du1, dv1, du2, dv2) in &noisy_offsets {
                let p1 = (projections[*i].0 + du1, projections[*i].1 + dv1);
                let p2 = (projections[*j].0 + du2, projections[*j].1 + dv2);
                let (x, y, z) = triangulate_point(&pair_cams, &[p1, p2]);
                let err_3d = ((x - 0.0).powi(2) + (y - 0.0).powi(2) + (z - 2.0).powi(2)).sqrt();
                max_3d_error = max_3d_error.max(err_3d);
            }
            eprintln!("{}: max 3D error with ±{:.0}px noise = {:.3}m", name, noise, max_3d_error);
            // 10px noise → 合理的なカメラペアなら3Dエラーは1m未満
            // （壊れたキャリブレーションでは数十m以上になる可能性あり）
        }
    }

    /// fx/fy比の検証: 正常なカメラは1.0に近い、非正方ピクセルは異常
    #[test]
    fn test_calibration_intrinsic_quality() {
        // Camera 0
        let fx0 = 1272.7304511139373f64;
        let fy0 = 1215.9025385457269;
        let ratio0 = fx0 / fy0;
        eprintln!("Camera 0: fx={:.1}, fy={:.1}, fx/fy={:.3}", fx0, fy0, ratio0);
        assert!((ratio0 - 1.0).abs() < 0.1, "Camera 0 fx/fy ratio abnormal: {:.3}", ratio0);

        // Camera 3
        let fx3 = 232.02611418613375;
        let fy3 = 422.2621858675812;
        let ratio3 = fx3 / fy3;
        eprintln!("Camera 3: fx={:.1}, fy={:.1}, fx/fy={:.3}", fx3, fy3, ratio3);
        // fx/fy=0.55 — 非常に異常
        // これはキャリブレーションの問題を示す
        // テストとしては「異常であること」を記録するだけ
        if (ratio3 - 1.0f64).abs() > 0.15 {
            eprintln!("  WARNING: Camera 3 has non-square pixels (fx/fy={:.3}). Calibration may be invalid.", ratio3);
        }

        // Camera 4
        let fx4 = 456.399288994886;
        let fy4 = 1544.018695645093;
        let ratio4 = fx4 / fy4;
        eprintln!("Camera 4: fx={:.1}, fy={:.1}, fx/fy={:.3}", fx4, fy4, ratio4);
        if (ratio4 - 1.0f64).abs() > 0.15 {
            eprintln!("  WARNING: Camera 4 has non-square pixels (fx/fy={:.3}). Calibration may be invalid.", ratio4);
        }
    }

    /// 解像度ミスマッチのシミュレーション:
    /// cam4がキャリブレーション時1920x1080、実際720x1280(portrait)の場合、
    /// 正規化座標→ピクセル変換でどの程度の誤差が出るか
    #[test]
    fn test_resolution_mismatch_impact() {
        let dc = [0.0f64; 5];

        let cam0 = CameraParams::from_calibration(
            &[1272.7304511139373, 0.0, 962.2823957723114,
              0.0, 1215.9025385457269, 504.1560959749078,
              0.0, 0.0, 1.0],
            &dc, &[0.0, 0.0, 0.0], &[0.0, 0.0, 0.0], 1920, 1080,
        );
        let cam4 = CameraParams::from_calibration(
            &[456.399288994886, 0.0, 941.8959669069045,
              0.0, 1544.018695645093, 524.4074246034186,
              0.0, 0.0, 1.0],
            &dc,
            &[-0.1925106008619544, -0.5582459992900398, -0.06365120750007275],
            &[-1.2687434272642264, -0.5338435605797942, 1.6824551205347025],
            1920, 1080,
        );

        // 既知の3D点
        let target = Vector4::new(0.0, 0.0, 2.0, 1.0);

        // cam0: 正しい投影（1920x1080で検出、1920x1080でマッピング）
        let p0 = cam0.projection * target;
        let u0_true = p0[0] / p0[2];
        let v0_true = p0[1] / p0[2];

        // cam4: 正しい投影
        let p4 = cam4.projection * target;
        let u4_true = p4[0] / p4[2];
        let v4_true = p4[1] / p4[2];

        // 正常ケース: 三角測量
        let (x, y, z) = triangulate_point(
            &[&cam0, &cam4], &[(u0_true, v0_true), (u4_true, v4_true)]);
        eprintln!("Normal: triangulated ({:.3}, {:.3}, {:.3})", x, y, z);
        assert!((z - 2.0).abs() < 0.01);

        // 異常ケース: cam4が実際は720x1280で動作していた場合
        // ポーズ検出器の出力: 正規化座標(0-1) at 720x1280
        // cam4の正しいピクセル座標をまず720x1280の正規化座標に変換
        // (これはcam4が1920x1080で正しく検出した場合の正規化座標)
        let _norm_x = u4_true / 1920.0;
        let _norm_y = v4_true / 1080.0;

        // もしカメラが720x1280(portrait)で、映像の中身が90度回転していたら
        // ポーズ検出器は回転後の画像で検出する。
        // 仮に同じ被写体を720x1280のフレームで検出した場合、
        // 正規化座標は全く異なる値になる。
        // ここでは最悪ケースとして、正規化座標はそのまま使われるが
        // マッピング先がcam4の1920x1080になる場合をシミュレート。
        // つまり u = norm_x * 1920 = u4_true (同じ) → 問題なし
        //
        // 本当の問題: ポーズ検出器が720x1280のフレームで検出し、
        // 出力する正規化座標が720x1280基準。これが1920x1080にマッピングされると:
        // - 中心点(0.5, 0.5)は同じ → OK
        // - 端の点でスケール差が出る
        //
        // 例: cam4真のピクセル (u4_true, v4_true) が 1920x1080 画像にある。
        // 同じ被写体がcam4実際フレーム720x1280にもいると仮定。
        // 720x1280の正規化座標は (u4_true / 1920 * (1920/720), v4_true / 1080 * (1080/1280))
        // = (u4_true / 720, v4_true / 1280) ← これが実際のポーズ検出器出力
        // triangle_poses はこれを * 1920 * 1080 する:
        // u_mapped = (u4_true / 720) * 1920 = u4_true * 2.667
        // v_mapped = (v4_true / 1280) * 1080 = v4_true * 0.844
        let u4_mismatched = (u4_true / 720.0) * 1920.0;
        let v4_mismatched = (v4_true / 1280.0) * 1080.0;

        let (x2, y2, z2) = triangulate_point(
            &[&cam0, &cam4], &[(u0_true, v0_true), (u4_mismatched, v4_mismatched)]);
        let err_3d = ((x2 - 0.0).powi(2) + (y2 - 0.0).powi(2) + (z2 - 2.0).powi(2)).sqrt();
        let p3d = Vector4::new(x2, y2, z2, 1.0);
        let reproj = reprojection_error(&[&cam0, &cam4], &[(u0_true, v0_true), (u4_mismatched, v4_mismatched)], &p3d);
        eprintln!("Resolution mismatch (1920x1080 cal, 720x1280 actual):");
        eprintln!("  cam4 pixel: true=({:.0},{:.0}) mismatched=({:.0},{:.0})",
            u4_true, v4_true, u4_mismatched, v4_mismatched);
        eprintln!("  triangulated ({:.3}, {:.3}, {:.3}) 3D_err={:.3}m reproj={:.0}px",
            x2, y2, z2, err_3d, reproj);
    }

    /// 実際のcalibration.jsonデータで立位人物のhip/foot z差を検証
    /// 歪み補正をゼロにした場合のz軸バイアスを計測
    #[test]
    fn test_standing_z_offset_with_real_calibration() {
        // === 実際のcalibration.jsonの値 ===
        let cam0_intrinsic = [
            1266.4166719940747, 0.0, 997.4319802056252,
            0.0, 1270.3438769581796, 528.8261022128323,
            0.0, 0.0, 1.0,
        ];
        let cam0_dc = [-0.2813001615738964, 14.981438197321392, 0.0032840674774454403, 0.01000190260805305, -391.99175222958104];
        let cam3_intrinsic = [
            1403.0019049820999, 0.0, 1038.6174061926272,
            0.0, 1414.5271546136971, 375.9119459435122,
            0.0, 0.0, 1.0,
        ];
        let cam3_dc = [-0.07804099579898437, 0.6223845753432052, -0.008701764962594302, 0.05526474771493587, 0.729633439752385];
        let cam3_rvec = [0.17189497194602363, 0.005084597149756517, 0.04079656389019682];
        let cam3_tvec = [-0.44565852935789385, 0.25581575547683455, 0.34145473784479474];

        let dc_zero = [0.0f64; 5];

        // P = K[R|t] （歪みなし版、本番と同じ）
        let cam0_nodist = CameraParams::from_calibration(&cam0_intrinsic, &dc_zero, &[0.0;3], &[0.0;3], 1920, 1080);
        let cam3_nodist = CameraParams::from_calibration(&cam3_intrinsic, &dc_zero, &cam3_rvec, &cam3_tvec, 1920, 1080);

        // 3D座標: cam0座標系で直立人物（hip z=2.0, foot z=2.0）
        let hip_3d = Vector4::new(0.0, 0.0, 2.0, 1.0);   // 腰（カメラ正面）
        let foot_3d = Vector4::new(0.0, 0.8, 2.0, 1.0);  // 足（0.8m下、同じz）

        // --- ケース1: 歪みなし投影 → 三角測量（自己整合性テスト） ---
        let hip_proj0 = cam0_nodist.projection * hip_3d;
        let hip_uv0 = (hip_proj0[0] / hip_proj0[2], hip_proj0[1] / hip_proj0[2]);
        let hip_proj3 = cam3_nodist.projection * hip_3d;
        let hip_uv3 = (hip_proj3[0] / hip_proj3[2], hip_proj3[1] / hip_proj3[2]);

        let foot_proj0 = cam0_nodist.projection * foot_3d;
        let foot_uv0 = (foot_proj0[0] / foot_proj0[2], foot_proj0[1] / foot_proj0[2]);
        let foot_proj3 = cam3_nodist.projection * foot_3d;
        let foot_uv3 = (foot_proj3[0] / foot_proj3[2], foot_proj3[1] / foot_proj3[2]);

        let (hx, hy, hz) = triangulate_point(&[&cam0_nodist, &cam3_nodist], &[hip_uv0, hip_uv3]);
        let (fx, fy, fz) = triangulate_point(&[&cam0_nodist, &cam3_nodist], &[foot_uv0, foot_uv3]);

        let no_dist_z_diff = (fz - hz).abs();
        eprintln!("=== Case 1: No distortion (self-consistency) ===");
        eprintln!("hip: ({:.4}, {:.4}, {:.4})", hx, hy, hz);
        eprintln!("foot: ({:.4}, {:.4}, {:.4})", fx, fy, fz);
        eprintln!("z diff: {:.4}m", no_dist_z_diff);
        assert!(no_dist_z_diff < 0.001, "self-consistency: hip-foot z diff should be ~0, got {:.4}", no_dist_z_diff);

        // --- ケース2: 歪みあり投影 → 歪みなし三角測量（実際の使用パターン） ---
        // 実際のシステム: カメラは歪んだ画像を出力 → ポーズ検出器がそれを検出
        // → 歪んだ2D座標を、歪みなしP行列で三角測量する → zバイアスが発生
        //
        // 歪みモデル: x_dist = x * (1 + k1*r² + k2*r⁴ + k3*r⁶) + tangential
        // cam0の歪みは大きい (k2=14.98, k3=-391.99) → 画像端で大きなバイアス

        // 3D→正規化カメラ座標→歪み適用→ピクセル座標
        fn project_with_distortion(
            intrinsic: &[f64; 9], dc: &[f64; 5],
            rvec: &[f64; 3], tvec: &[f64; 3],
            point: &Vector4<f32>,
        ) -> (f32, f32) {
            let fx = intrinsic[0] as f32;
            let fy = intrinsic[4] as f32;
            let cx = intrinsic[2] as f32;
            let cy = intrinsic[5] as f32;
            let [k1, k2, p1, p2, k3] = [dc[0] as f32, dc[1] as f32, dc[2] as f32, dc[3] as f32, dc[4] as f32];

            // 回転行列
            let theta = (rvec[0]*rvec[0] + rvec[1]*rvec[1] + rvec[2]*rvec[2]).sqrt();
            let r = if theta < 1e-10 {
                nalgebra::Matrix3::<f32>::identity()
            } else {
                let kx = (rvec[0] / theta) as f32;
                let ky = (rvec[1] / theta) as f32;
                let kz = (rvec[2] / theta) as f32;
                let ct = (theta as f32).cos();
                let st = (theta as f32).sin();
                let vt = 1.0 - ct;
                nalgebra::Matrix3::new(
                    ct + kx*kx*vt, kx*ky*vt - kz*st, kx*kz*vt + ky*st,
                    ky*kx*vt + kz*st, ct + ky*ky*vt, ky*kz*vt - kx*st,
                    kz*kx*vt - ky*st, kz*ky*vt + kx*st, ct + kz*kz*vt,
                )
            };
            let t = nalgebra::Vector3::new(tvec[0] as f32, tvec[1] as f32, tvec[2] as f32);

            // ワールド → カメラ座標
            let pw = nalgebra::Vector3::new(point[0], point[1], point[2]);
            let pc = r * pw + t;
            if pc[2].abs() < 1e-6 { return (0.0, 0.0); }

            // 正規化カメラ座標
            let xn = pc[0] / pc[2];
            let yn = pc[1] / pc[2];
            let r2 = xn*xn + yn*yn;
            let r4 = r2*r2;
            let r6 = r4*r2;

            // 歪みモデル適用
            let radial = 1.0 + k1*r2 + k2*r4 + k3*r6;
            let xd = xn * radial + 2.0*p1*xn*yn + p2*(r2 + 2.0*xn*xn);
            let yd = yn * radial + p1*(r2 + 2.0*yn*yn) + 2.0*p2*xn*yn;

            (xd * fx + cx, yd * fy + cy)
        }

        let hip_dist0 = project_with_distortion(&cam0_intrinsic, &cam0_dc, &[0.0;3], &[0.0;3], &hip_3d);
        let hip_dist3 = project_with_distortion(&cam3_intrinsic, &cam3_dc, &cam3_rvec, &cam3_tvec, &hip_3d);
        let foot_dist0 = project_with_distortion(&cam0_intrinsic, &cam0_dc, &[0.0;3], &[0.0;3], &foot_3d);
        let foot_dist3 = project_with_distortion(&cam3_intrinsic, &cam3_dc, &cam3_rvec, &cam3_tvec, &foot_3d);

        eprintln!("\n=== Case 2: Distorted projection -> undistorted triangulation ===");
        eprintln!("cam0 hip  pixel: ({:.1}, {:.1}) undist: ({:.1}, {:.1})", hip_dist0.0, hip_dist0.1, hip_uv0.0, hip_uv0.1);
        eprintln!("cam0 foot pixel: ({:.1}, {:.1}) undist: ({:.1}, {:.1})", foot_dist0.0, foot_dist0.1, foot_uv0.0, foot_uv0.1);
        eprintln!("cam0 hip  distortion shift: ({:.1}, {:.1})px", hip_dist0.0 - hip_uv0.0, hip_dist0.1 - hip_uv0.1);
        eprintln!("cam0 foot distortion shift: ({:.1}, {:.1})px", foot_dist0.0 - foot_uv0.0, foot_dist0.1 - foot_uv0.1);

        // 歪み投影の2D座標で三角測量
        let (hx2, hy2, hz2) = triangulate_point(&[&cam0_nodist, &cam3_nodist], &[hip_dist0, hip_dist3]);
        let (fx2, fy2, fz2) = triangulate_point(&[&cam0_nodist, &cam3_nodist], &[foot_dist0, foot_dist3]);

        let dist_z_diff = (fz2 - hz2).abs();
        eprintln!("hip:  ({:.4}, {:.4}, {:.4})", hx2, hy2, hz2);
        eprintln!("foot: ({:.4}, {:.4}, {:.4})", fx2, fy2, fz2);
        eprintln!("z diff: {:.4}m (this is the systematic z bias from ignoring distortion)", dist_z_diff);

        // 歪みを無視した場合のzバイアスがどの程度か記録
        // これが0.3-0.5mなら、ログで観測したhip-foot zオフセットの原因
        eprintln!("\n=== Summary ===");
        eprintln!("Without distortion error: z diff = {:.4}m", no_dist_z_diff);
        eprintln!("With distortion error:    z diff = {:.4}m", dist_z_diff);
    }

    #[test]
    fn test_rescale_intrinsics() {
        let dc = [0.0f64; 5];
        let mut cam = CameraParams::from_calibration(
            &[1272.73, 0.0, 962.28,
              0.0, 1215.90, 504.16,
              0.0, 0.0, 1.0],
            &dc, &[0.0, 0.0, 0.0], &[0.0, 0.0, 0.0], 1920, 1080,
        );

        // 同一解像度 → 変更なし
        assert!(cam.rescale_to(1920, 1080));

        // 半分の解像度（同アスペクト比）
        let mut cam_half = cam.clone();
        assert!(cam_half.rescale_to(960, 540));
        assert!((cam_half.image_width - 960.0).abs() < 0.1);
        // 半分の解像度ではfxも半分
        assert!((cam_half.intrinsic[(0, 0)] - 1272.73 * 0.5).abs() < 1.0);

        // 異なるアスペクト比 → false
        let mut cam_portrait = cam.clone();
        assert!(!cam_portrait.rescale_to(720, 1280));
    }
}
