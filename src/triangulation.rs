use nalgebra::{Matrix3, Matrix3x4, Matrix4, Vector3, Vector4};

use crate::pose::{Keypoint, KeypointIndex, Pose};

/// リプロジェクションエラー閾値（ピクセル）
/// 三角測量した3D点をカメラに再投影した際の最大許容誤差
const MAX_REPROJ_ERROR: f32 = 75.0;

/// カメラパラメータ（射影行列 P = K[R|t]）
#[derive(Debug, Clone)]
pub struct CameraParams {
    pub projection: Matrix3x4<f32>,
    pub image_width: f32,
    pub image_height: f32,
    /// 内部パラメータ行列（歪み補正用）
    intrinsic: Matrix3<f32>,
    /// 歪み係数 [k1, k2, p1, p2, k3]
    dist_coeffs: [f32; 5],
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
        Self {
            projection,
            image_width: width as f32,
            image_height: height as f32,
            intrinsic: k,
            dist_coeffs: dc,
        }
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
fn triangulate_point(cameras: &[&CameraParams], points_2d: &[(f32, f32)]) -> (f32, f32, f32) {
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

/// N台のカメラの2Dポーズから3Dポーズを三角測量
///
/// - 各キーポイントについて、confidence_threshold以上のカメラの観測を集める
/// - 2台以上の有効観測があるキーポイントのみ三角測量
/// - 1台以下の観測のキーポイントはconfidence=0
pub fn triangulate_poses(
    cameras: &[&CameraParams],
    poses_2d: &[&Pose],
    confidence_threshold: f32,
) -> Pose {
    assert_eq!(cameras.len(), poses_2d.len());
    let n = cameras.len();

    let mut keypoints = [Keypoint::default(); KeypointIndex::COUNT];

    for kp_idx in 0..KeypointIndex::COUNT {
        // 有効な観測を集める
        let mut valid_cameras = Vec::new();
        let mut valid_points = Vec::new();
        let mut conf_sum = 0.0f32;

        for cam_idx in 0..n {
            let kp = &poses_2d[cam_idx].keypoints[kp_idx];
            if kp.confidence >= confidence_threshold {
                valid_cameras.push(cameras[cam_idx]);
                // 正規化座標(0-1)→歪みピクセル座標→歪み補正ピクセル座標
                let cam = cameras[cam_idx];
                let u_dist = kp.x * cam.image_width;
                let v_dist = kp.y * cam.image_height;
                let (u, v) = cam.undistort_point(u_dist, v_dist);
                valid_points.push((u, v));
                conf_sum += kp.confidence;
            }
        }

        if valid_cameras.len() >= 2 {
            let (x, y, z) = triangulate_point(&valid_cameras, &valid_points);

            // リプロジェクションエラーチェック: 三角測量結果をカメラに再投影し、
            // 元の2D検出との誤差が大きい場合は棄却
            let point_3d = Vector4::new(x, y, z, 1.0);
            let mut max_reproj_err = 0.0f32;
            for i in 0..valid_cameras.len() {
                let projected = valid_cameras[i].projection * point_3d;
                if projected[2].abs() < 1e-6 {
                    max_reproj_err = f32::MAX;
                    break;
                }
                let u_proj = projected[0] / projected[2];
                let v_proj = projected[1] / projected[2];
                let (u_obs, v_obs) = valid_points[i];
                let err = ((u_proj - u_obs).powi(2) + (v_proj - v_obs).powi(2)).sqrt();
                max_reproj_err = max_reproj_err.max(err);
            }

            if max_reproj_err < MAX_REPROJ_ERROR {
                let avg_conf = conf_sum / valid_cameras.len() as f32;
                keypoints[kp_idx] = Keypoint::new_3d(x, y, z, avg_conf);
            }
        }
    }

    Pose::new(keypoints)
}

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

    #[test]
    fn test_triangulate_poses_min_cameras() {
        let cam1 = CameraParams::from_config(55.0, 640, 480, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]);
        let cam2 = CameraParams::from_config(55.0, 640, 480, [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]);

        // 1台のカメラでしか見えないキーポイント → confidence=0
        let mut kp1 = [Keypoint::default(); KeypointIndex::COUNT];
        kp1[0] = Keypoint::new(0.5, 0.5, 0.9); // Nose visible in cam1

        let kp2 = [Keypoint::default(); KeypointIndex::COUNT];
        // Nose not visible in cam2 (confidence=0)

        let pose1 = Pose::new(kp1);
        let pose2 = Pose::new(kp2);

        let result = triangulate_poses(&[&cam1, &cam2], &[&pose1, &pose2], 0.3);

        // Nose: only 1 camera → should not be triangulated
        assert_eq!(result.get(KeypointIndex::Nose).confidence, 0.0);
    }
}
