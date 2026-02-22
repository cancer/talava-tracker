//! Inference server: receives JPEG frames over TCP, runs ONNX pose estimation,
//! triangulates 3D poses, and sends tracker data via VMT/OSC.
//!
//! Self-contained: only imports from `talava_tracker::protocol`.
//! All other functionality (pose detection, triangulation, tracking, VMT) is inline.

use std::io::Write;
use std::net::UdpSocket;
use std::sync::{mpsc, Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use nalgebra::{Matrix3, Matrix3x4, Matrix4, Vector3, Vector4};
use ndarray::Array4;
use opencv::core::{AlgorithmHint, Mat, Rect, Scalar, Size, Vector, CV_32FC3};
use opencv::prelude::*;
use opencv::{imgcodecs, imgproc};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;
use rosc::{encoder, OscMessage, OscPacket, OscType};
use serde::Deserialize;

use talava_tracker::protocol::{
    self, ClientMessage, ServerMessage,
};

// ===========================================================================
// Config (reads inference_server.toml)
// ===========================================================================

#[derive(Debug, Deserialize)]
struct Config {
    #[serde(default = "default_listen_addr")]
    listen_addr: String,
    #[serde(default = "default_vmt_addr")]
    vmt_addr: String,
    #[serde(default = "default_model")]
    model: String,
    #[serde(default = "default_detector")]
    detector: String,
    #[serde(default = "default_interp_mode")]
    interpolation_mode: String,
    #[serde(default)]
    mirror_x: bool,
    #[serde(default = "default_offset_y")]
    offset_y: f32,
    #[serde(default)]
    foot_y_offset: f32,
    #[serde(default)]
    filter: FilterConfig,
}

fn default_listen_addr() -> String { "0.0.0.0:9000".to_string() }
fn default_vmt_addr() -> String { "127.0.0.1:39570".to_string() }
fn default_model() -> String { "spinepose_medium".to_string() }
fn default_detector() -> String { "keypoint".to_string() }
fn default_interp_mode() -> String { "extrapolate".to_string() }
fn default_offset_y() -> f32 { 1.0 }

#[derive(Debug, Deserialize, Clone)]
struct FilterConfig {
    #[serde(default = "default_pos_min_cutoff")]
    position_min_cutoff: f32,
    #[serde(default = "default_pos_beta")]
    position_beta: f32,
    #[serde(default = "default_rot_min_cutoff")]
    rotation_min_cutoff: f32,
    #[serde(default = "default_rot_beta")]
    rotation_beta: f32,
    #[serde(default)]
    lower_body_position_beta: Option<f32>,
}

fn default_pos_min_cutoff() -> f32 { 1.5 }
fn default_pos_beta() -> f32 { 0.3 }
fn default_rot_min_cutoff() -> f32 { 1.0 }
fn default_rot_beta() -> f32 { 0.01 }

impl Default for FilterConfig {
    fn default() -> Self {
        Self {
            position_min_cutoff: default_pos_min_cutoff(),
            position_beta: default_pos_beta(),
            rotation_min_cutoff: default_rot_min_cutoff(),
            rotation_beta: default_rot_beta(),
            lower_body_position_beta: None,
        }
    }
}

// ===========================================================================
// Keypoint / Pose types (37 keypoints)
// ===========================================================================

const KP_COUNT: usize = 37;

// Keypoint indices
const KP_NOSE: usize = 0;
const KP_LEFT_SHOULDER: usize = 5;
const KP_RIGHT_SHOULDER: usize = 6;
const KP_LEFT_HIP: usize = 11;
const KP_RIGHT_HIP: usize = 12;
const KP_LEFT_KNEE: usize = 13;
const KP_RIGHT_KNEE: usize = 14;
const KP_LEFT_ANKLE: usize = 15;
const KP_RIGHT_ANKLE: usize = 16;
const KP_HEAD: usize = 17;
const KP_NECK: usize = 18;
const KP_HIP: usize = 19;
const KP_LEFT_BIG_TOE: usize = 20;
const KP_RIGHT_BIG_TOE: usize = 21;
const KP_LEFT_SMALL_TOE: usize = 22;
const KP_RIGHT_SMALL_TOE: usize = 23;
const KP_LEFT_HEEL: usize = 24;
const KP_RIGHT_HEEL: usize = 25;

#[derive(Debug, Clone, Copy, Default)]
struct Keypoint {
    x: f32,
    y: f32,
    z: f32,
    confidence: f32,
}

impl Keypoint {
    fn new(x: f32, y: f32, confidence: f32) -> Self {
        Self { x, y, z: 0.0, confidence }
    }
    fn new_3d(x: f32, y: f32, z: f32, confidence: f32) -> Self {
        Self { x, y, z, confidence }
    }
    fn is_valid(&self, threshold: f32) -> bool {
        self.confidence >= threshold
    }
}

#[derive(Debug, Clone)]
struct Pose {
    keypoints: [Keypoint; KP_COUNT],
}

impl Pose {
    fn new(keypoints: [Keypoint; KP_COUNT]) -> Self {
        Self { keypoints }
    }
    fn get(&self, idx: usize) -> &Keypoint {
        &self.keypoints[idx]
    }
}

impl Default for Pose {
    fn default() -> Self {
        Self { keypoints: [Keypoint::default(); KP_COUNT] }
    }
}

// ===========================================================================
// TrackerPose (position + rotation)
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq)]
struct TrackerPose {
    position: [f32; 3],
    rotation: [f32; 4],
}

impl TrackerPose {
    fn new(position: [f32; 3], rotation: [f32; 4]) -> Self {
        Self { position, rotation }
    }
}

// ===========================================================================
// Camera parameters + Triangulation
// ===========================================================================

const MAX_REPROJ_ERROR: f32 = 120.0;
const MAX_HIP_Z_JUMP: f32 = 0.3;

#[derive(Debug, Clone)]
struct CameraParams {
    projection: Matrix3x4<f32>,
    image_width: f32,
    image_height: f32,
    intrinsic: Matrix3<f32>,
    dist_coeffs: [f32; 5],
}

impl CameraParams {
    fn from_calibration(
        intrinsic: &[f64; 9],
        dist_coeffs: &[f64; 5],
        rvec: &[f64; 3],
        tvec: &[f64; 3],
        width: u32,
        height: u32,
    ) -> Self {
        let k = Matrix3::new(
            intrinsic[0] as f32, intrinsic[1] as f32, intrinsic[2] as f32,
            intrinsic[3] as f32, intrinsic[4] as f32, intrinsic[5] as f32,
            intrinsic[6] as f32, intrinsic[7] as f32, intrinsic[8] as f32,
        );

        // Rodrigues -> rotation matrix
        let theta = (rvec[0] * rvec[0] + rvec[1] * rvec[1] + rvec[2] * rvec[2]).sqrt();
        let r = if theta < 1e-10 {
            Matrix3::<f32>::identity()
        } else {
            let kx = (rvec[0] / theta) as f32;
            let ky = (rvec[1] / theta) as f32;
            let kz = (rvec[2] / theta) as f32;
            let ct = (theta as f32).cos();
            let st = (theta as f32).sin();
            let vt = 1.0 - ct;
            Matrix3::new(
                ct + kx * kx * vt,      kx * ky * vt - kz * st, kx * kz * vt + ky * st,
                ky * kx * vt + kz * st, ct + ky * ky * vt,      ky * kz * vt - kx * st,
                kz * kx * vt - ky * st, kz * ky * vt + kx * st, ct + kz * kz * vt,
            )
        };

        let t = Vector3::new(tvec[0] as f32, tvec[1] as f32, tvec[2] as f32);

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

    fn rescale_to(&mut self, actual_width: u32, actual_height: u32) -> bool {
        let cal_w = self.image_width;
        let cal_h = self.image_height;
        let act_w = actual_width as f32;
        let act_h = actual_height as f32;

        if (cal_w - act_w).abs() < 1.0 && (cal_h - act_h).abs() < 1.0 {
            return true;
        }

        let cal_ratio = cal_w / cal_h;
        let act_ratio = act_w / act_h;
        if (cal_ratio - act_ratio).abs() / cal_ratio > 0.05 {
            return false;
        }

        let sx = act_w / cal_w;
        let sy = act_h / cal_h;

        for j in 0..4 {
            self.projection[(0, j)] *= sx;
            self.projection[(1, j)] *= sy;
        }

        self.intrinsic[(0, 0)] *= sx;
        self.intrinsic[(0, 2)] *= sx;
        self.intrinsic[(1, 1)] *= sy;
        self.intrinsic[(1, 2)] *= sy;

        self.image_width = act_w;
        self.image_height = act_h;
        true
    }

    /// Newton-Raphson undistortion
    fn undistort_point(&self, u_dist: f32, v_dist: f32) -> (f32, f32) {
        let fx = self.intrinsic[(0, 0)];
        let fy = self.intrinsic[(1, 1)];
        let cx = self.intrinsic[(0, 2)];
        let cy = self.intrinsic[(1, 2)];
        let [k1, k2, p1, p2, k3] = self.dist_coeffs;

        if k1 == 0.0 && k2 == 0.0 && p1 == 0.0 && p2 == 0.0 && k3 == 0.0 {
            return (u_dist, v_dist);
        }

        let xd = (u_dist - cx) / fx;
        let yd = (v_dist - cy) / fy;

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

            let fx_val = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x) - xd;
            let fy_val = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y - yd;

            let residual = fx_val * fx_val + fy_val * fy_val;
            if residual < best_residual {
                best_residual = residual;
                best_x = x;
                best_y = y;
            }

            if residual < 1e-12 {
                break;
            }

            let j00 = radial + 2.0 * x * x * dr_dr2 + 2.0 * p1 * y + 6.0 * p2 * x;
            let j01 = 2.0 * x * y * dr_dr2 + 2.0 * p1 * x + 2.0 * p2 * y;
            let j10 = j01;
            let j11 = radial + 2.0 * y * y * dr_dr2 + 6.0 * p1 * y + 2.0 * p2 * x;

            let det = j00 * j11 - j01 * j10;
            if det.abs() < 1e-10 {
                break;
            }

            let dx = -(j11 * fx_val - j01 * fy_val) / det;
            let dy = -(-j10 * fx_val + j00 * fy_val) / det;

            x += dx;
            y += dy;
        }

        (best_x * fx + cx, best_y * fy + cy)
    }
}

/// DLT triangulation of a single 3D point from N camera observations
fn triangulate_point(cameras: &[&CameraParams], points_2d: &[(f32, f32)]) -> (f32, f32, f32) {
    let n = cameras.len();
    assert!(n >= 2);

    let mut a = Matrix4::zeros();

    for i in 0..n {
        let p = &cameras[i].projection;
        let (u, v) = points_2d[i];

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

fn reprojection_error(
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

fn find_hip_reference_pair(
    cameras: &[&CameraParams],
    observations: &[Vec<Option<(f32, f32, f32)>>],
    prev_pair: Option<(usize, usize)>,
) -> Option<(usize, usize)> {
    let kp_idx = KP_LEFT_HIP;
    let n = cameras.len();

    let mut valid_cameras = Vec::new();
    let mut valid_points = Vec::new();
    let mut valid_cam_indices = Vec::new();

    for cam_idx in 0..n {
        if let Some((u, v, _conf)) = observations[kp_idx][cam_idx] {
            valid_cameras.push(cameras[cam_idx]);
            valid_points.push((u, v));
            valid_cam_indices.push(cam_idx);
        }
    }

    if valid_cameras.len() < 2 {
        return None;
    }

    // Hysteresis: prefer previous pair if still valid
    if let Some((prev_ci, prev_cj)) = prev_pair {
        let pi = valid_cam_indices.iter().position(|&c| c == prev_ci);
        let pj = valid_cam_indices.iter().position(|&c| c == prev_cj);
        if let (Some(vi), Some(vj)) = (pi, pj) {
            let pair_cams = [valid_cameras[vi], valid_cameras[vj]];
            let pair_pts = [valid_points[vi], valid_points[vj]];
            let (px, py, pz) = triangulate_point(&pair_cams, &pair_pts);
            let p3d = Vector4::new(px, py, pz, 1.0);
            let prev_err = reprojection_error(&pair_cams, &pair_pts, &p3d);
            if prev_err < MAX_REPROJ_ERROR {
                return Some((prev_ci, prev_cj));
            }
        }
    }

    let mut best_err = f32::MAX;
    let mut best_pair = None;

    let vc = valid_cameras.len();
    for i in 0..vc {
        for j in (i + 1)..vc {
            let pair_cams = [valid_cameras[i], valid_cameras[j]];
            let pair_pts = [valid_points[i], valid_points[j]];
            let (px, py, pz) = triangulate_point(&pair_cams, &pair_pts);
            let p3d = Vector4::new(px, py, pz, 1.0);
            let pair_err = reprojection_error(&pair_cams, &pair_pts, &p3d);

            if pair_err < best_err {
                best_err = pair_err;
                best_pair = Some((valid_cam_indices[i], valid_cam_indices[j]));
            }
        }
    }

    if best_err < MAX_REPROJ_ERROR { best_pair } else { None }
}

/// Triangulate 3D pose from N camera 2D observations
fn triangulate_poses_multi(
    cameras: &[&CameraParams],
    poses_2d: &[&Pose],
    confidence_threshold: f32,
    prev_hip_ref_pair: &mut Option<(usize, usize)>,
    prev_hip_z: &mut Option<f32>,
) -> Pose {
    assert_eq!(cameras.len(), poses_2d.len());
    let n = cameras.len();

    // Pre-compute undistorted 2D observations: [kp_idx][cam_idx]
    let mut observations: Vec<Vec<Option<(f32, f32, f32)>>> = Vec::with_capacity(KP_COUNT);
    for kp_idx in 0..KP_COUNT {
        let mut cam_obs = Vec::with_capacity(n);
        for cam_idx in 0..n {
            let kp = &poses_2d[cam_idx].keypoints[kp_idx];
            if kp.confidence >= confidence_threshold {
                let cam = cameras[cam_idx];
                let u_dist = kp.x * cam.image_width;
                let v_dist = kp.y * cam.image_height;
                let (u, v) = cam.undistort_point(u_dist, v_dist);
                cam_obs.push(Some((u, v, kp.confidence)));
            } else {
                cam_obs.push(None);
            }
        }
        observations.push(cam_obs);
    }

    let hip_ref_pair = find_hip_reference_pair(cameras, &observations, *prev_hip_ref_pair);

    let mut keypoints = [Keypoint::default(); KP_COUNT];

    for kp_idx in 0..KP_COUNT {
        // Try reference pair first
        if let Some((ci, cj)) = hip_ref_pair {
            if let (Some((ui, vi, conf_i)), Some((uj, vj, conf_j))) =
                (observations[kp_idx][ci], observations[kp_idx][cj])
            {
                let pair_cams = [cameras[ci], cameras[cj]];
                let pair_pts = [(ui, vi), (uj, vj)];
                let (x, y, z) = triangulate_point(&pair_cams, &pair_pts);
                let p3d = Vector4::new(x, y, z, 1.0);
                let err = reprojection_error(&pair_cams, &pair_pts, &p3d);

                if err < MAX_REPROJ_ERROR {
                    let avg_conf = (conf_i + conf_j) / 2.0;
                    keypoints[kp_idx] = Keypoint::new_3d(x, y, z, avg_conf);
                    continue;
                }
            }
        }

        // Fallback: per-keypoint pair selection
        let mut valid_cameras = Vec::new();
        let mut valid_points = Vec::new();
        let mut valid_confs = Vec::new();
        let mut valid_cam_indices = Vec::new();

        for cam_idx in 0..n {
            if let Some((u, v, conf)) = observations[kp_idx][cam_idx] {
                valid_cameras.push(cameras[cam_idx]);
                valid_points.push((u, v));
                valid_confs.push(conf);
                valid_cam_indices.push(cam_idx);
            }
        }

        if valid_cameras.len() < 2 {
            continue;
        }

        let (x, y, z) = triangulate_point(&valid_cameras, &valid_points);
        let point_3d = Vector4::new(x, y, z, 1.0);
        let err = reprojection_error(&valid_cameras, &valid_points, &point_3d);

        if err < MAX_REPROJ_ERROR {
            let avg_conf: f32 = valid_confs.iter().sum::<f32>() / valid_confs.len() as f32;
            keypoints[kp_idx] = Keypoint::new_3d(x, y, z, avg_conf);
            continue;
        }

        // Try all pairs
        let mut best_err = f32::MAX;
        let mut best_point = (0.0f32, 0.0f32, 0.0f32);
        let mut best_conf = 0.0f32;

        let vc = valid_cameras.len();
        for i in 0..vc {
            for j in (i + 1)..vc {
                let pair_cams = [valid_cameras[i], valid_cameras[j]];
                let pair_pts = [valid_points[i], valid_points[j]];
                let (px, py, pz) = triangulate_point(&pair_cams, &pair_pts);
                let p3d = Vector4::new(px, py, pz, 1.0);
                let pair_err = reprojection_error(&pair_cams, &pair_pts, &p3d);

                if pair_err < best_err {
                    best_err = pair_err;
                    best_point = (px, py, pz);
                    best_conf = (valid_confs[i] + valid_confs[j]) / 2.0;
                }
            }
        }

        if best_err < MAX_REPROJ_ERROR {
            keypoints[kp_idx] = Keypoint::new_3d(best_point.0, best_point.1, best_point.2, best_conf);
        }
    }

    *prev_hip_ref_pair = hip_ref_pair;

    // Hip z-jump rejection
    let lh = &keypoints[KP_LEFT_HIP];
    let rh = &keypoints[KP_RIGHT_HIP];
    let current_hip_z = if lh.confidence > 0.0 && rh.confidence > 0.0 {
        Some((lh.z + rh.z) / 2.0)
    } else if lh.confidence > 0.0 {
        Some(lh.z)
    } else if rh.confidence > 0.0 {
        Some(rh.z)
    } else {
        None
    };

    if let Some(cur_z) = current_hip_z {
        if let Some(pz) = *prev_hip_z {
            let z_jump = (cur_z - pz).abs();
            if z_jump > MAX_HIP_Z_JUMP {
                return Pose::new([Keypoint::default(); KP_COUNT]);
            }
        }
        *prev_hip_z = Some(cur_z);
    }

    Pose::new(keypoints)
}

// ===========================================================================
// Model types + ONNX inference
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq)]
enum ModelType { MoveNet, SpinePose, RTMW3D }

struct PoseDetector {
    session: Session,
    model_type: ModelType,
}

impl PoseDetector {
    fn new(model_path: &str, model_type: ModelType) -> Result<Self> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .commit_from_file(model_path)
            .context("Failed to load ONNX model")?;
        Ok(Self { session, model_type })
    }

    fn detect(&mut self, input: Array4<f32>) -> Result<Pose> {
        match self.model_type {
            ModelType::MoveNet => self.detect_movenet(input),
            ModelType::SpinePose => self.detect_spinepose(input),
            ModelType::RTMW3D => self.detect_rtmw3d(input),
        }
    }

    fn detect_movenet(&mut self, input: Array4<f32>) -> Result<Pose> {
        let input_tensor = Tensor::from_array(input)?;
        let outputs = self.session
            .run(ort::inputs!["serving_default_input_0" => input_tensor])
            .context("Inference failed")?;
        let output: ndarray::ArrayViewD<f32> = outputs["StatefulPartitionedCall_0"]
            .try_extract_array().context("Failed to extract output")?;

        let mut keypoints = [Keypoint::default(); KP_COUNT];
        for i in 0..17 {
            let y = output[[0, 0, i, 0]];
            let x = output[[0, 0, i, 1]];
            let confidence = output[[0, 0, i, 2]];
            keypoints[i] = Keypoint::new(x, y, confidence);
        }
        Ok(Pose::new(keypoints))
    }

    fn detect_spinepose(&mut self, input: Array4<f32>) -> Result<Pose> {
        let input_tensor = Tensor::from_array(input)?;
        let outputs = self.session
            .run(ort::inputs!["input" => input_tensor])
            .context("Inference failed")?;
        let simcc_x: ndarray::ArrayViewD<f32> = outputs["simcc_x"]
            .try_extract_array().context("Failed to extract simcc_x")?;
        let simcc_y: ndarray::ArrayViewD<f32> = outputs["simcc_y"]
            .try_extract_array().context("Failed to extract simcc_y")?;

        let mut keypoints = [Keypoint::default(); KP_COUNT];
        for i in 0..KP_COUNT {
            let mut max_x_val = f32::NEG_INFINITY;
            let mut max_x_idx = 0usize;
            for j in 0..384 {
                let v = simcc_x[[0, i, j]];
                if v > max_x_val { max_x_val = v; max_x_idx = j; }
            }
            let mut max_y_val = f32::NEG_INFINITY;
            let mut max_y_idx = 0usize;
            for j in 0..512 {
                let v = simcc_y[[0, i, j]];
                if v > max_y_val { max_y_val = v; max_y_idx = j; }
            }
            let x = max_x_idx as f32 / (2.0 * 192.0);
            let y = max_y_idx as f32 / (2.0 * 256.0);
            let avg_logit = (max_x_val + max_y_val) / 2.0;
            let confidence = 1.0 / (1.0 + (-avg_logit).exp());
            keypoints[i] = Keypoint::new(x, y, confidence);
        }
        Ok(Pose::new(keypoints))
    }

    fn detect_rtmw3d(&mut self, input: Array4<f32>) -> Result<Pose> {
        let input_tensor = Tensor::from_array(input)?;
        let outputs = self.session
            .run(ort::inputs!["input" => input_tensor])
            .context("Inference failed")?;
        let simcc_x: ndarray::ArrayViewD<f32> = outputs["output"]
            .try_extract_array().context("Failed to extract simcc_x")?;
        let simcc_y: ndarray::ArrayViewD<f32> = outputs["1554"]
            .try_extract_array().context("Failed to extract simcc_y")?;
        let simcc_z: ndarray::ArrayViewD<f32> = outputs["1556"]
            .try_extract_array().context("Failed to extract simcc_z")?;

        const SIMCC_SPLIT_RATIO: f32 = 2.0;
        const MODEL_W: f32 = 288.0;
        const MODEL_H: f32 = 384.0;
        const Z_RANGE: f32 = 2.1744869;

        let mut raw = [(0.0f32, 0.0f32, 0.0f32, 0.0f32); 133];
        for i in 0..133 {
            let mut mx_val = f32::NEG_INFINITY; let mut mx_idx = 0usize;
            for j in 0..576 { let v = simcc_x[[0, i, j]]; if v > mx_val { mx_val = v; mx_idx = j; } }
            let mut my_val = f32::NEG_INFINITY; let mut my_idx = 0usize;
            for j in 0..768 { let v = simcc_y[[0, i, j]]; if v > my_val { my_val = v; my_idx = j; } }
            let mut mz_val = f32::NEG_INFINITY; let mut mz_idx = 0usize;
            for j in 0..576 { let v = simcc_z[[0, i, j]]; if v > mz_val { mz_val = v; mz_idx = j; } }

            let x = mx_idx as f32 / SIMCC_SPLIT_RATIO / MODEL_W;
            let y = my_idx as f32 / SIMCC_SPLIT_RATIO / MODEL_H;
            let z_raw = mz_idx as f32 / SIMCC_SPLIT_RATIO;
            let z = (z_raw / (MODEL_H / 2.0) - 1.0) * Z_RANGE;
            let confidence = mx_val.min(my_val);
            let confidence = if confidence <= 0.0 { 0.0 } else { confidence };
            raw[i] = (x, y, z, confidence);
        }

        let mut keypoints = [Keypoint::default(); KP_COUNT];
        for i in 0..17 {
            let (x, y, z, c) = raw[i];
            keypoints[i] = Keypoint::new_3d(x, y, z, c);
        }
        // Foot mapping: RTMW3D -> our indices
        const FOOT_MAP: [(usize, usize); 6] = [
            (17, KP_LEFT_BIG_TOE), (18, KP_LEFT_SMALL_TOE), (19, KP_LEFT_HEEL),
            (20, KP_RIGHT_BIG_TOE), (21, KP_RIGHT_SMALL_TOE), (22, KP_RIGHT_HEEL),
        ];
        for &(src, dst) in &FOOT_MAP {
            let (x, y, z, c) = raw[src];
            keypoints[dst] = Keypoint::new_3d(x, y, z, c);
        }
        // Synthetic: Head = Nose
        let (nx, ny, nz, nc) = raw[0];
        keypoints[KP_HEAD] = Keypoint::new_3d(nx, ny, nz, nc);
        // Neck = shoulder midpoint
        let (lsx, lsy, lsz, lsc) = raw[5];
        let (rsx, rsy, rsz, rsc) = raw[6];
        if lsc > 0.0 && rsc > 0.0 {
            keypoints[KP_NECK] = Keypoint::new_3d(
                (lsx + rsx) / 2.0, (lsy + rsy) / 2.0, (lsz + rsz) / 2.0, lsc.min(rsc),
            );
        }
        // Hip = hip midpoint
        let (lhx, lhy, lhz, lhc) = raw[11];
        let (rhx, rhy, rhz, rhc) = raw[12];
        if lhc > 0.0 && rhc > 0.0 {
            keypoints[KP_HIP] = Keypoint::new_3d(
                (lhx + rhx) / 2.0, (lhy + rhy) / 2.0, (lhz + rhz) / 2.0, lhc.min(rhc),
            );
        }
        Ok(Pose::new(keypoints))
    }
}

// ===========================================================================
// Preprocessing
// ===========================================================================

#[derive(Debug, Clone, Copy)]
struct LetterboxInfo {
    pad_left: f32,
    pad_top: f32,
    content_width: f32,
    content_height: f32,
}

impl LetterboxInfo {
    fn identity() -> Self {
        Self { pad_left: 0.0, pad_top: 0.0, content_width: 1.0, content_height: 1.0 }
    }
}

fn unletterbox_pose(pose: &Pose, info: &LetterboxInfo) -> Pose {
    let mut keypoints = [Keypoint::default(); KP_COUNT];
    for i in 0..KP_COUNT {
        let kp = &pose.keypoints[i];
        keypoints[i] = Keypoint {
            x: (kp.x - info.pad_left) / info.content_width,
            y: (kp.y - info.pad_top) / info.content_height,
            z: kp.z,
            confidence: kp.confidence,
        };
    }
    Pose::new(keypoints)
}

fn preprocess_for_movenet(frame: &Mat) -> Result<Array4<f32>> {
    let mut rgb = Mat::default();
    imgproc::cvt_color(frame, &mut rgb, imgproc::COLOR_BGR2RGB, 0, AlgorithmHint::ALGO_HINT_DEFAULT)?;
    let mut resized = Mat::default();
    imgproc::resize(&rgb, &mut resized, Size::new(192, 192), 0.0, 0.0, imgproc::INTER_LINEAR)?;
    let mut float_mat = Mat::default();
    resized.convert_to(&mut float_mat, CV_32FC3, 1.0, 0.0)?;

    let (h, w) = (192usize, 192usize);
    let mut tensor = Array4::<f32>::zeros((1, h, w, 3));
    let dst = tensor.as_slice_mut().unwrap();
    let row_floats = w * 3;
    let data = float_mat.data_bytes()?;
    let step = float_mat.mat_step().get(0);
    for y in 0..h {
        let src = unsafe { std::slice::from_raw_parts(data.as_ptr().add(y * step) as *const f32, row_floats) };
        dst[y * row_floats..(y + 1) * row_floats].copy_from_slice(src);
    }
    Ok(tensor)
}

fn preprocess_imagenet_nchw(frame: &Mat, width: i32, height: i32) -> Result<(Array4<f32>, LetterboxInfo)> {
    let frame_w = frame.cols() as f32;
    let frame_h = frame.rows() as f32;
    let target_w = width as f32;
    let target_h = height as f32;

    let scale = (target_w / frame_w).min(target_h / frame_h);
    let new_w = (frame_w * scale).round() as i32;
    let new_h = (frame_h * scale).round() as i32;
    let pad_left = (width - new_w) / 2;
    let pad_top = (height - new_h) / 2;

    let mut rgb = Mat::default();
    imgproc::cvt_color(frame, &mut rgb, imgproc::COLOR_BGR2RGB, 0, AlgorithmHint::ALGO_HINT_DEFAULT)?;
    let mut resized = Mat::default();
    imgproc::resize(&rgb, &mut resized, Size::new(new_w, new_h), 0.0, 0.0, imgproc::INTER_LINEAR)?;

    let mean_rgb = Scalar::new(123.675, 116.28, 103.53, 0.0);
    let mut padded = Mat::default();
    opencv::core::copy_make_border(
        &resized, &mut padded,
        pad_top, height - new_h - pad_top,
        pad_left, width - new_w - pad_left,
        opencv::core::BORDER_CONSTANT, mean_rgb,
    )?;

    let mut float_mat = Mat::default();
    padded.convert_to(&mut float_mat, CV_32FC3, 1.0, 0.0)?;

    let (h, w) = (height as usize, width as usize);
    let mean = [123.675f32, 116.28, 103.53];
    let inv_std = [1.0 / 58.395f32, 1.0 / 57.12, 1.0 / 57.375];
    let mut tensor = Array4::<f32>::zeros((1, 3, h, w));
    let dst = tensor.as_slice_mut().unwrap();
    let ch_stride = h * w;
    let data = float_mat.data_bytes()?;
    let step = float_mat.mat_step().get(0);
    for y_idx in 0..h {
        let row_ptr = unsafe {
            std::slice::from_raw_parts(data.as_ptr().add(y_idx * step) as *const f32, w * 3)
        };
        let row_offset = y_idx * w;
        for x_idx in 0..w {
            let src_base = x_idx * 3;
            let dst_base = row_offset + x_idx;
            dst[dst_base] = (row_ptr[src_base] - mean[0]) * inv_std[0];
            dst[ch_stride + dst_base] = (row_ptr[src_base + 1] - mean[1]) * inv_std[1];
            dst[2 * ch_stride + dst_base] = (row_ptr[src_base + 2] - mean[2]) * inv_std[2];
        }
    }

    let letterbox = LetterboxInfo {
        pad_left: pad_left as f32 / target_w,
        pad_top: pad_top as f32 / target_h,
        content_width: new_w as f32 / target_w,
        content_height: new_h as f32 / target_h,
    };

    Ok((tensor, letterbox))
}

fn preprocess_for_spinepose(frame: &Mat) -> Result<(Array4<f32>, LetterboxInfo)> {
    preprocess_imagenet_nchw(frame, 192, 256)
}

fn preprocess_for_rtmw3d(frame: &Mat) -> Result<(Array4<f32>, LetterboxInfo)> {
    preprocess_imagenet_nchw(frame, 288, 384)
}

// ===========================================================================
// Person detection + Crop
// ===========================================================================

#[derive(Debug, Clone, Copy)]
struct BBox { x: f32, y: f32, width: f32, height: f32 }

#[derive(Debug, Clone, Copy)]
struct CropRegion { x: f32, y: f32, width: f32, height: f32 }

impl CropRegion {
    fn full() -> Self { Self { x: 0.0, y: 0.0, width: 1.0, height: 1.0 } }
    fn is_full(&self) -> bool { self.width >= 1.0 && self.height >= 1.0 }
}

fn bbox_from_keypoints(pose: &Pose, frame_w: u32, frame_h: u32, conf_thresh: f32) -> Option<BBox> {
    let mut min_x = f32::MAX; let mut min_y = f32::MAX;
    let mut max_x = f32::MIN; let mut max_y = f32::MIN;
    let mut count = 0u32;
    for kp in &pose.keypoints {
        if kp.confidence >= conf_thresh {
            let px = kp.x * frame_w as f32;
            let py = kp.y * frame_h as f32;
            min_x = min_x.min(px); min_y = min_y.min(py);
            max_x = max_x.max(px); max_y = max_y.max(py);
            count += 1;
        }
    }
    if count < 2 { return None; }
    Some(BBox { x: min_x, y: min_y, width: max_x - min_x, height: max_y - min_y })
}

fn crop_for_pose(frame: &Mat, bbox: &BBox, frame_w: u32, frame_h: u32) -> Result<(Mat, CropRegion)> {
    let expand = 1.25;
    let cx = bbox.x + bbox.width / 2.0;
    let cy = bbox.y + bbox.height / 2.0;
    let mut w = bbox.width * expand;
    let mut h = bbox.height * expand;
    let target_ratio = 4.0 / 3.0;
    let current_ratio = h / w;
    if current_ratio < target_ratio { h = w * target_ratio; }
    else { w = h / target_ratio; }

    let mut x = cx - w / 2.0;
    let mut y = cy - h / 2.0;
    let fw = frame_w as f32;
    let fh = frame_h as f32;
    x = x.max(0.0); y = y.max(0.0);
    w = w.min(fw - x); h = h.min(fh - y);

    let roi = Rect::new(x as i32, y as i32, (w as i32).max(1), (h as i32).max(1));
    let cropped = Mat::roi(frame, roi)?;
    let crop_region = CropRegion { x: x / fw, y: y / fh, width: w / fw, height: h / fh };
    Ok((cropped.try_clone()?, crop_region))
}

fn remap_pose(pose: &Pose, crop: &CropRegion) -> Pose {
    let mut keypoints = [Keypoint::default(); KP_COUNT];
    for i in 0..KP_COUNT {
        let kp = &pose.keypoints[i];
        keypoints[i] = Keypoint {
            x: crop.x + kp.x * crop.width,
            y: crop.y + kp.y * crop.height,
            z: kp.z,
            confidence: kp.confidence,
        };
    }
    Pose::new(keypoints)
}

struct PersonDetector {
    session: Session,
    input_size: i32,
}

impl PersonDetector {
    fn new(model_path: &str, input_size: i32) -> Result<Self> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .commit_from_file(model_path)
            .context("Failed to load person detection model")?;
        Ok(Self { session, input_size })
    }

    fn detect(&mut self, frame: &Mat) -> Result<Option<BBox>> {
        let frame_w = frame.cols();
        let frame_h = frame.rows();
        let input = self.preprocess(frame)?;
        let input_tensor = Tensor::from_array(input)?;
        let outputs = self.session
            .run(ort::inputs!["images" => input_tensor])
            .context("Person detection failed")?;
        let output: ndarray::ArrayViewD<f32> = outputs["output0"]
            .try_extract_array().context("Failed to extract output")?;

        let n_det = output.shape()[2];
        let mut best_score: f32 = 0.0;
        let mut best_idx: Option<usize> = None;
        for i in 0..n_det {
            let score = output[[0, 4, i]];
            if score > best_score && score >= 0.25 {
                best_score = score;
                best_idx = Some(i);
            }
        }
        let Some(idx) = best_idx else { return Ok(None); };

        let cx = output[[0, 0, idx]];
        let cy = output[[0, 1, idx]];
        let w = output[[0, 2, idx]];
        let h = output[[0, 3, idx]];
        let sx = frame_w as f32 / self.input_size as f32;
        let sy = frame_h as f32 / self.input_size as f32;
        Ok(Some(BBox {
            x: (cx - w / 2.0) * sx, y: (cy - h / 2.0) * sy,
            width: w * sx, height: h * sy,
        }))
    }

    fn preprocess(&self, frame: &Mat) -> Result<Array4<f32>> {
        let size = self.input_size;
        let mut rgb = Mat::default();
        imgproc::cvt_color(frame, &mut rgb, imgproc::COLOR_BGR2RGB, 0, AlgorithmHint::ALGO_HINT_DEFAULT)?;
        let mut resized = Mat::default();
        imgproc::resize(&rgb, &mut resized, Size::new(size, size), 0.0, 0.0, imgproc::INTER_LINEAR)?;
        let mut float_mat = Mat::default();
        resized.convert_to(&mut float_mat, CV_32FC3, 1.0, 0.0)?;

        let s = size as usize;
        let mut tensor = Array4::<f32>::zeros((1, 3, s, s));
        let data = float_mat.data_bytes()?;
        let step = float_mat.mat_step().get(0);
        for y in 0..s {
            let row_ptr = unsafe { std::slice::from_raw_parts(data.as_ptr().add(y * step) as *const f32, s * 3) };
            for x in 0..s {
                for c in 0..3 {
                    tensor[[0, c, y, x]] = row_ptr[x * 3 + c] / 255.0;
                }
            }
        }
        Ok(tensor)
    }
}

// ===========================================================================
// One Euro Filter
// ===========================================================================

struct LowPassFilter { prev: Option<f32> }

impl LowPassFilter {
    fn new() -> Self { Self { prev: None } }
    fn filter(&mut self, value: f32, alpha: f32) -> f32 {
        match self.prev {
            Some(prev) => { let r = alpha * value + (1.0 - alpha) * prev; self.prev = Some(r); r }
            None => { self.prev = Some(value); value }
        }
    }
    fn reset(&mut self) { self.prev = None; }
}

fn smoothing_factor(te: f32, cutoff: f32) -> f32 {
    let r = 2.0 * std::f32::consts::PI * cutoff * te;
    r / (r + 1.0)
}

struct ScalarFilter {
    min_cutoff: f32, beta: f32, d_cutoff: f32,
    x_filter: LowPassFilter, dx_filter: LowPassFilter,
    prev_value: Option<f32>,
}

impl ScalarFilter {
    fn new(min_cutoff: f32, beta: f32, d_cutoff: f32) -> Self {
        Self {
            min_cutoff, beta, d_cutoff,
            x_filter: LowPassFilter::new(), dx_filter: LowPassFilter::new(),
            prev_value: None,
        }
    }
    fn filter(&mut self, value: f32, dt: f32) -> f32 {
        let dx = match self.prev_value {
            Some(prev) => if dt > 0.0 { (value - prev) / dt } else { 0.0 },
            None => 0.0,
        };
        self.prev_value = Some(value);
        let edx = self.dx_filter.filter(dx, smoothing_factor(dt, self.d_cutoff));
        let cutoff = self.min_cutoff + self.beta * edx.abs();
        self.x_filter.filter(value, smoothing_factor(dt, cutoff))
    }
    fn reset(&mut self) {
        self.x_filter.reset(); self.dx_filter.reset(); self.prev_value = None;
    }
}

struct PoseFilter {
    position: [ScalarFilter; 3],
    rotation: [ScalarFilter; 4],
    prev_rotation: Option<[f32; 4]>,
    last_time: Option<Instant>,
}

impl PoseFilter {
    fn new(pos_min_cutoff: f32, pos_beta: f32, rot_min_cutoff: f32, rot_beta: f32) -> Self {
        let d_cutoff = 1.0;
        Self {
            position: std::array::from_fn(|_| ScalarFilter::new(pos_min_cutoff, pos_beta, d_cutoff)),
            rotation: std::array::from_fn(|_| ScalarFilter::new(rot_min_cutoff, rot_beta, d_cutoff)),
            prev_rotation: None, last_time: None,
        }
    }

    fn from_config(config: &FilterConfig) -> Self {
        Self::new(config.position_min_cutoff, config.position_beta, config.rotation_min_cutoff, config.rotation_beta)
    }

    fn from_config_lower_body(config: &FilterConfig) -> Self {
        let beta = config.lower_body_position_beta.unwrap_or(config.position_beta);
        Self::new(config.position_min_cutoff, beta, config.rotation_min_cutoff, config.rotation_beta)
    }

    fn apply(&mut self, pose: TrackerPose) -> TrackerPose {
        let now = Instant::now();
        let dt = match self.last_time {
            Some(t) => { let d = now.duration_since(t).as_secs_f32(); if d > 0.0 { d } else { 1.0 / 90.0 } }
            None => { self.last_time = Some(now); self.prev_rotation = Some(pose.rotation); return pose; }
        };
        self.last_time = Some(now);

        let position = [
            self.position[0].filter(pose.position[0], dt),
            self.position[1].filter(pose.position[1], dt),
            self.position[2].filter(pose.position[2], dt),
        ];

        let mut rot = pose.rotation;
        if let Some(ref prev) = self.prev_rotation {
            let dot = prev[0] * rot[0] + prev[1] * rot[1] + prev[2] * rot[2] + prev[3] * rot[3];
            if dot < 0.0 { rot = [-rot[0], -rot[1], -rot[2], -rot[3]]; }
        }
        self.prev_rotation = Some(rot);

        let mut rotation = [
            self.rotation[0].filter(rot[0], dt),
            self.rotation[1].filter(rot[1], dt),
            self.rotation[2].filter(rot[2], dt),
            self.rotation[3].filter(rot[3], dt),
        ];
        let len = (rotation[0].powi(2) + rotation[1].powi(2) + rotation[2].powi(2) + rotation[3].powi(2)).sqrt();
        if len > 0.0 { for v in &mut rotation { *v /= len; } }

        TrackerPose::new(position, rotation)
    }

    fn reset(&mut self) {
        for f in &mut self.position { f.reset(); }
        for f in &mut self.rotation { f.reset(); }
        self.prev_rotation = None; self.last_time = None;
    }
}

// ===========================================================================
// Extrapolator / Lerper
// ===========================================================================

struct Extrapolator {
    current: Option<TrackerPose>,
    velocity: [f32; 3],
    last_update: Option<Instant>,
}

impl Extrapolator {
    fn new() -> Self { Self { current: None, velocity: [0.0; 3], last_update: None } }

    fn update(&mut self, pose: TrackerPose) {
        let now = Instant::now();
        if let (Some(prev_pose), Some(last_t)) = (self.current, self.last_update) {
            let dt = now.duration_since(last_t).as_secs_f32();
            if dt > 0.0 {
                self.velocity = [
                    (pose.position[0] - prev_pose.position[0]) / dt,
                    (pose.position[1] - prev_pose.position[1]) / dt,
                    (pose.position[2] - prev_pose.position[2]) / dt,
                ];
            }
        }
        self.current = Some(pose);
        self.last_update = Some(now);
    }

    fn predict(&self, dt_secs: f32) -> Option<TrackerPose> {
        self.current.map(|cur| TrackerPose::new(
            [
                cur.position[0] + self.velocity[0] * dt_secs,
                cur.position[1] + self.velocity[1] * dt_secs,
                cur.position[2] + self.velocity[2] * dt_secs,
            ],
            cur.rotation,
        ))
    }
}

struct Lerper { start: Option<TrackerPose>, end: Option<TrackerPose> }

impl Lerper {
    fn new() -> Self { Self { start: None, end: None } }

    fn update(&mut self, pose: TrackerPose, current_t: f32) {
        match self.start {
            None => { self.start = Some(pose); self.end = Some(pose); }
            Some(_) => { self.start = self.interpolate(current_t); self.end = Some(pose); }
        }
    }

    fn interpolate(&self, t: f32) -> Option<TrackerPose> {
        let start = self.start?;
        let end = self.end?;
        let t = t.clamp(0.0, 1.0);
        let position = [
            (1.0 - t) * start.position[0] + t * end.position[0],
            (1.0 - t) * start.position[1] + t * end.position[1],
            (1.0 - t) * start.position[2] + t * end.position[2],
        ];
        // nlerp
        let dot = start.rotation[0] * end.rotation[0] + start.rotation[1] * end.rotation[1]
            + start.rotation[2] * end.rotation[2] + start.rotation[3] * end.rotation[3];
        let sign = if dot < 0.0 { -1.0 } else { 1.0 };
        let mut rotation = [
            (1.0 - t) * start.rotation[0] + t * sign * end.rotation[0],
            (1.0 - t) * start.rotation[1] + t * sign * end.rotation[1],
            (1.0 - t) * start.rotation[2] + t * sign * end.rotation[2],
            (1.0 - t) * start.rotation[3] + t * sign * end.rotation[3],
        ];
        let len = (rotation[0].powi(2) + rotation[1].powi(2) + rotation[2].powi(2) + rotation[3].powi(2)).sqrt();
        if len > 0.0 { for v in &mut rotation { *v /= len; } }
        Some(TrackerPose::new(position, rotation))
    }
}

// ===========================================================================
// Body tracker
// ===========================================================================

const CONFIDENCE_THRESHOLD: f32 = 0.3;

struct BodyCalibration {
    hip_x: f32, hip_y: f32,
    shoulder_y: f32,
    torso_height: f32,
    hip_z: f32,
    yaw_shoulder: f32,
    yaw_left_foot: f32, yaw_right_foot: f32,
    left_knee_offset: Option<(f32, f32)>,
    right_knee_offset: Option<(f32, f32)>,
    yaw_left_knee: f32, yaw_right_knee: f32,
    tilt_angle: f32,
}

struct BodyTracker {
    mirror_x: bool,
    offset_y: f32,
    foot_y_offset: f32,
    calibration: Option<BodyCalibration>,
}

struct BodyPoses {
    hip: Option<TrackerPose>,
    left_foot: Option<TrackerPose>,
    right_foot: Option<TrackerPose>,
    chest: Option<TrackerPose>,
    left_knee: Option<TrackerPose>,
    right_knee: Option<TrackerPose>,
}

impl BodyTracker {
    fn new(mirror_x: bool, offset_y: f32, foot_y_offset: f32) -> Self {
        Self { mirror_x, offset_y, foot_y_offset, calibration: None }
    }

    fn is_calibrated(&self) -> bool { self.calibration.is_some() }

    fn compute_tilt_angle(pose: &Pose) -> f32 {
        let lh = pose.get(KP_LEFT_HIP);
        let rh = pose.get(KP_RIGHT_HIP);
        if !lh.is_valid(0.2) || !rh.is_valid(0.2) { return 0.0; }
        let hip_y = (lh.y + rh.y) / 2.0;
        let hip_z = (lh.z + rh.z) / 2.0;

        let la = pose.get(KP_LEFT_ANKLE);
        let ra = pose.get(KP_RIGHT_ANKLE);
        let lk = pose.get(KP_LEFT_KNEE);
        let rk = pose.get(KP_RIGHT_KNEE);

        let (lower_y, lower_z) = if la.is_valid(0.2) && ra.is_valid(0.2) {
            ((la.y + ra.y) / 2.0, (la.z + ra.z) / 2.0)
        } else if lk.is_valid(0.2) && rk.is_valid(0.2) {
            ((lk.y + rk.y) / 2.0, (lk.z + rk.z) / 2.0)
        } else { return 0.0; };

        f32::atan2(lower_z - hip_z, lower_y - hip_y)
    }

    fn rotate_pose_yz(pose: &Pose, tilt: f32) -> Pose {
        let (sin_t, cos_t) = tilt.sin_cos();
        let lh = pose.get(KP_LEFT_HIP);
        let rh = pose.get(KP_RIGHT_HIP);
        let ref_z = if lh.is_valid(0.2) && rh.is_valid(0.2) {
            (lh.z + rh.z) / 2.0
        } else { 0.0 };

        let mut rotated = pose.clone();
        for kp in rotated.keypoints.iter_mut() {
            let y = kp.y;
            let z = kp.z;
            kp.y = y * cos_t + ref_z * sin_t;
            kp.z = -y * sin_t + z * cos_t;
        }
        rotated
    }

    fn calibrate(&mut self, pose: &Pose) -> bool {
        let tilt_angle = Self::compute_tilt_angle(pose);
        let rotated = Self::rotate_pose_yz(pose, tilt_angle);
        let pose = &rotated;

        let lh = pose.get(KP_LEFT_HIP);
        let rh = pose.get(KP_RIGHT_HIP);
        if !lh.is_valid(0.2) || !rh.is_valid(0.2) { return false; }

        let hip_x = (lh.x + rh.x) / 2.0;
        let hip_y = (lh.y + rh.y) / 2.0;
        let hip_z = (lh.z + rh.z) / 2.0;

        let ls = pose.get(KP_LEFT_SHOULDER);
        let rs = pose.get(KP_RIGHT_SHOULDER);
        let shoulder_y = if ls.is_valid(0.2) && rs.is_valid(0.2) {
            (ls.y + rs.y) / 2.0
        } else { 0.5 };

        let torso_height = self.compute_torso_height(pose).unwrap_or(0.0);
        let yaw_shoulder = self.compute_shoulder_yaw(pose);
        let yaw_left_foot = self.compute_foot_yaw(pose, KP_LEFT_KNEE, KP_LEFT_ANKLE);
        let yaw_right_foot = self.compute_foot_yaw(pose, KP_RIGHT_KNEE, KP_RIGHT_ANKLE);

        let lk = pose.get(KP_LEFT_KNEE);
        let left_knee_offset = if lk.is_valid(0.2) { Some((lk.x - hip_x, lk.y - hip_y)) } else { None };
        let rk = pose.get(KP_RIGHT_KNEE);
        let right_knee_offset = if rk.is_valid(0.2) { Some((rk.x - hip_x, rk.y - hip_y)) } else { None };

        let yaw_left_knee = self.compute_knee_yaw(pose, KP_LEFT_HIP, KP_LEFT_KNEE);
        let yaw_right_knee = self.compute_knee_yaw(pose, KP_RIGHT_HIP, KP_RIGHT_KNEE);

        self.calibration = Some(BodyCalibration {
            hip_x, hip_y, shoulder_y, torso_height, hip_z,
            yaw_shoulder, yaw_left_foot, yaw_right_foot,
            left_knee_offset, right_knee_offset,
            yaw_left_knee, yaw_right_knee, tilt_angle,
        });
        true
    }

    fn compute(&self, pose: &Pose) -> BodyPoses {
        let tilt = self.calibration.as_ref().map_or(0.0, |c| c.tilt_angle);
        let rotated;
        let pose = if tilt.abs() > 0.001 {
            rotated = Self::rotate_pose_yz(pose, tilt);
            &rotated
        } else { pose };

        let lh = pose.get(KP_LEFT_HIP);
        let rh = pose.get(KP_RIGHT_HIP);
        let hip_center = if lh.is_valid(0.2) && rh.is_valid(0.2) {
            Some(((lh.x + rh.x) / 2.0, (lh.y + rh.y) / 2.0))
        } else { None };

        let hip_z = self.estimate_depth(pose);
        let (lf_z, rf_z, ch_z, lk_z, rk_z) = (
            self.keypoint_depth(pose, KP_LEFT_ANKLE, hip_z),
            self.keypoint_depth(pose, KP_RIGHT_ANKLE, hip_z),
            hip_z,
            self.keypoint_depth(pose, KP_LEFT_KNEE, hip_z),
            self.keypoint_depth(pose, KP_RIGHT_KNEE, hip_z),
        );

        let mut left_foot = self.compute_foot(pose, hip_center, lf_z, KP_LEFT_KNEE, KP_LEFT_ANKLE, true);
        let mut right_foot = self.compute_foot(pose, hip_center, rf_z, KP_RIGHT_KNEE, KP_RIGHT_ANKLE, false);
        let mut left_knee = self.compute_knee(pose, hip_center, lk_z, KP_LEFT_HIP, KP_LEFT_KNEE, true);
        let mut right_knee = self.compute_knee(pose, hip_center, rk_z, KP_RIGHT_HIP, KP_RIGHT_KNEE, false);

        Self::reject_duplicate_pair(&mut left_foot, &mut right_foot);
        Self::reject_duplicate_pair(&mut left_knee, &mut right_knee);

        BodyPoses {
            hip: self.compute_hip(pose, hip_center, hip_z),
            left_foot, right_foot,
            chest: self.compute_chest(pose, hip_center, ch_z),
            left_knee, right_knee,
        }
    }

    fn convert_position(&self, x: f32, y: f32, hip_x: f32, hip_y: f32, pos_z: f32) -> [f32; 3] {
        let (ref_x, ref_y) = match &self.calibration {
            Some(cal) => (cal.hip_x, cal.hip_y),
            None => (0.5, 0.5),
        };
        let global_x = ref_x - hip_x;
        let global_y = ref_y - hip_y;
        let mut pos_x = global_x + (hip_x - x);
        if self.mirror_x { pos_x = -pos_x; }
        let pos_y = self.offset_y + (global_y + (hip_y - y));
        [pos_x, pos_y, pos_z]
    }

    fn compute_torso_height(&self, pose: &Pose) -> Option<f32> {
        let ls = pose.get(KP_LEFT_SHOULDER);
        let rs = pose.get(KP_RIGHT_SHOULDER);
        let lh = pose.get(KP_LEFT_HIP);
        let rh = pose.get(KP_RIGHT_HIP);
        if !ls.is_valid(0.2) || !rs.is_valid(0.2) || !lh.is_valid(0.2) || !rh.is_valid(0.2) {
            return None;
        }
        let shoulder_y = (ls.y + rs.y) / 2.0;
        let hip_y = (lh.y + rh.y) / 2.0;
        Some((hip_y - shoulder_y).abs())
    }

    fn estimate_depth(&self, pose: &Pose) -> f32 {
        let lh = pose.get(KP_LEFT_HIP);
        let rh = pose.get(KP_RIGHT_HIP);
        if lh.is_valid(0.2) && rh.is_valid(0.2) && (lh.z.abs() > 0.001 || rh.z.abs() > 0.001) {
            let hip_z = (lh.z + rh.z) / 2.0;
            let ref_z = self.calibration.as_ref().map_or(0.0, |c| c.hip_z);
            return ref_z - hip_z;
        }
        0.0
    }

    fn keypoint_depth(&self, pose: &Pose, kp_idx: usize, fallback: f32) -> f32 {
        let kp = pose.get(kp_idx);
        if kp.is_valid(0.2) && kp.z.abs() > 0.001 {
            let ref_z = self.calibration.as_ref().map_or(0.0, |c| c.hip_z);
            ref_z - kp.z
        } else { fallback }
    }

    fn compute_shoulder_yaw(&self, pose: &Pose) -> f32 {
        let ls = pose.get(KP_LEFT_SHOULDER);
        let rs = pose.get(KP_RIGHT_SHOULDER);
        if ls.is_valid(0.2) && rs.is_valid(0.2) {
            let dx = if self.mirror_x { rs.x - ls.x } else { ls.x - rs.x };
            let dy = rs.y - ls.y;
            f32::atan2(dy, dx)
        } else { 0.0 }
    }

    fn compute_foot_yaw(&self, pose: &Pose, knee_idx: usize, ankle_idx: usize) -> f32 {
        let knee = pose.get(knee_idx);
        let ankle = pose.get(ankle_idx);
        if knee.is_valid(0.2) && ankle.is_valid(0.2) {
            let raw_dx = ankle.x - knee.x;
            let dx = if self.mirror_x { raw_dx } else { -raw_dx };
            let dy = ankle.y - knee.y;
            f32::atan2(dx, dy)
        } else { 0.0 }
    }

    fn compute_knee_yaw(&self, pose: &Pose, hip_idx: usize, knee_idx: usize) -> f32 {
        let hip = pose.get(hip_idx);
        let knee = pose.get(knee_idx);
        if hip.is_valid(0.2) && knee.is_valid(0.2) {
            let raw_dx = knee.x - hip.x;
            let dx = if self.mirror_x { raw_dx } else { -raw_dx };
            let dy = knee.y - hip.y;
            f32::atan2(dx, dy)
        } else { 0.0 }
    }

    fn yaw_to_quaternion(yaw: f32) -> [f32; 4] {
        let half = yaw / 2.0;
        [0.0, half.sin(), 0.0, half.cos()]
    }

    fn yaw_pitch_to_quaternion(yaw: f32, pitch: f32) -> [f32; 4] {
        let (sy, cy) = (yaw / 2.0).sin_cos();
        let (sp, cp) = (pitch / 2.0).sin_cos();
        [cy * sp, sy * cp, -sy * sp, cy * cp]
    }

    fn reject_duplicate_pair(left: &mut Option<TrackerPose>, right: &mut Option<TrackerPose>) {
        if let (Some(l), Some(r)) = (left.as_ref(), right.as_ref()) {
            let dx = l.position[0] - r.position[0];
            let dy = l.position[1] - r.position[1];
            if (dx * dx + dy * dy).sqrt() < 0.05 { *left = None; *right = None; }
        }
    }

    fn compute_hip(&self, pose: &Pose, hip_center: Option<(f32, f32)>, pos_z: f32) -> Option<TrackerPose> {
        let (hip_x, hip_y) = hip_center?;
        let position = self.convert_position(hip_x, hip_y, hip_x, hip_y, pos_z);
        let ref_yaw = self.calibration.as_ref().map_or(0.0, |c| c.yaw_shoulder);
        let yaw = self.compute_shoulder_yaw(pose) - ref_yaw;
        Some(TrackerPose::new(position, Self::yaw_to_quaternion(yaw)))
    }

    fn compute_foot(&self, pose: &Pose, hip_center: Option<(f32, f32)>, pos_z: f32,
        knee_idx: usize, ankle_idx: usize, is_left: bool,
    ) -> Option<TrackerPose> {
        let (hip_x, hip_y) = hip_center?;
        let knee = pose.get(knee_idx);
        let ankle = pose.get(ankle_idx);
        let ankle_valid = ankle.is_valid(0.2);
        let knee_valid = knee.is_valid(0.2);
        let (ax, ay) = if ankle_valid { (ankle.x, ankle.y) }
            else if knee_valid { (knee.x, knee.y) }
            else { return None; };

        let mut position = self.convert_position(ax, ay, hip_x, hip_y, pos_z);
        position[1] += self.foot_y_offset;

        let yaw = if knee_valid && ankle_valid {
            let ref_yaw = self.calibration.as_ref().map_or(0.0, |c| {
                if is_left { c.yaw_left_foot } else { c.yaw_right_foot }
            });
            self.compute_foot_yaw(pose, knee_idx, ankle_idx) - ref_yaw
        } else { 0.0 };
        Some(TrackerPose::new(position, Self::yaw_to_quaternion(yaw)))
    }

    fn compute_chest(&self, pose: &Pose, hip_center: Option<(f32, f32)>, pos_z: f32) -> Option<TrackerPose> {
        let (hip_x, hip_y) = hip_center?;
        let ls = pose.get(KP_LEFT_SHOULDER);
        let rs = pose.get(KP_RIGHT_SHOULDER);
        let ls_valid = ls.is_valid(0.2);
        let rs_valid = rs.is_valid(0.2);
        if !ls_valid && !rs_valid { return None; }

        let shoulder_y = match (ls_valid, rs_valid) {
            (true, true) => (ls.y + rs.y) / 2.0,
            (true, false) => ls.y,
            (false, true) => rs.y,
            _ => unreachable!(),
        };
        let x = if ls_valid && rs_valid { (ls.x + rs.x) / 2.0 } else { hip_x };
        let y = shoulder_y + (hip_y - shoulder_y) * 0.35;

        let position = self.convert_position(x, y, hip_x, hip_y, pos_z);
        let ref_yaw = self.calibration.as_ref().map_or(0.0, |c| c.yaw_shoulder);
        let yaw = self.compute_shoulder_yaw(pose) - ref_yaw;
        Some(TrackerPose::new(position, Self::yaw_to_quaternion(yaw)))
    }

    fn compute_knee(&self, pose: &Pose, hip_center: Option<(f32, f32)>, pos_z: f32,
        hip_idx: usize, knee_idx: usize, is_left: bool,
    ) -> Option<TrackerPose> {
        let (hip_x, hip_y) = hip_center?;
        let knee = pose.get(knee_idx);

        let (kx, ky, has_keypoint) = if knee.is_valid(0.2) {
            (knee.x, knee.y, true)
        } else if let Some(cal) = &self.calibration {
            let offset = if is_left { cal.left_knee_offset } else { cal.right_knee_offset };
            if let Some((ox, oy)) = offset { (hip_x + ox, hip_y + oy, false) }
            else { return None; }
        } else { return None; };

        let position = self.convert_position(kx, ky, hip_x, hip_y, pos_z);
        let yaw = if has_keypoint {
            let ref_yaw = self.calibration.as_ref().map_or(0.0, |c| {
                if is_left { c.yaw_left_knee } else { c.yaw_right_knee }
            });
            self.compute_knee_yaw(pose, hip_idx, knee_idx) - ref_yaw
        } else { 0.0 };
        Some(TrackerPose::new(position, Self::yaw_pitch_to_quaternion(yaw, 0.0)))
    }
}

// ===========================================================================
// VMT (OSC sending)
// ===========================================================================

struct VmtClient {
    socket: UdpSocket,
    target_addr: String,
}

impl VmtClient {
    fn new(target_addr: &str) -> Result<Self> {
        let socket = UdpSocket::bind("0.0.0.0:0")?;
        Ok(Self { socket, target_addr: target_addr.to_string() })
    }

    fn send(&self, index: i32, enable: i32, pose: &TrackerPose) -> Result<()> {
        let msg = OscMessage {
            addr: "/VMT/Room/Unity".to_string(),
            args: vec![
                OscType::Int(index), OscType::Int(enable), OscType::Float(0.0),
                OscType::Float(pose.position[0]), OscType::Float(pose.position[1]),
                OscType::Float(pose.position[2]), OscType::Float(pose.rotation[0]),
                OscType::Float(pose.rotation[1]), OscType::Float(pose.rotation[2]),
                OscType::Float(pose.rotation[3]),
            ],
        };
        let data = encoder::encode(&OscPacket::Message(msg))?;
        self.socket.send_to(&data, &self.target_addr)?;
        Ok(())
    }
}

// ===========================================================================
// Logging
// ===========================================================================

type LogFile = Arc<Mutex<std::io::BufWriter<std::fs::File>>>;

fn open_log_file() -> Result<LogFile> {
    std::fs::create_dir_all("logs")?;
    let ts = chrono::Local::now().format("%Y%m%d_%H%M%S");
    let path = format!("logs/inference_{}.log", ts);
    let file = std::fs::File::create(&path)?;
    eprintln!("Log: {}", path);
    Ok(Arc::new(Mutex::new(std::io::BufWriter::new(file))))
}

macro_rules! log {
    ($logfile:expr, $($arg:tt)*) => {{
        let msg = format!($($arg)*);
        println!("{}", msg);
        if let Ok(mut f) = $logfile.lock() {
            let _ = writeln!(f, "{}", msg);
        }
    }};
}

// ===========================================================================
// TCP types + receive loop
// ===========================================================================

const TRACKER_COUNT: usize = 6;
const TRACKER_INDICES: [i32; TRACKER_COUNT] = [0, 1, 2, 3, 4, 5];
const TRIANGULATION_TIMEOUT_MS: f32 = 100.0;

struct DecodedFrame { camera_id: u8, width: u32, height: u32, mat: Mat }
struct DecodedFrameSet { #[allow(dead_code)] timestamp_us: u64, frames: Vec<DecodedFrame> }

enum TcpEvent {
    CalibrationReceived(protocol::CalibrationData),
    FrameSet(DecodedFrameSet),
    TriggerPoseCalibration,
}

async fn tcp_receive_loop(
    stream: tokio::net::TcpStream,
    tx: mpsc::SyncSender<TcpEvent>,
) -> Result<()> {
    let mut stream = protocol::message_stream(stream);

    loop {
        let msg: ClientMessage = protocol::recv_message(&mut stream).await?;
        match msg {
            ClientMessage::CameraCalibration { data } => {
                let _ = tx.send(TcpEvent::CalibrationReceived(data));
                protocol::send_message(&mut stream, &ServerMessage::CameraCalibrationAck { ok: true, error: None }).await?;
                protocol::send_message(&mut stream, &ServerMessage::Ready).await?;
            }
            ClientMessage::FrameSet { timestamp_us, frames } => {
                let decoded: Vec<DecodedFrame> = frames.into_iter().filter_map(|f| {
                    let buf = Vector::<u8>::from_iter(f.jpeg_data.into_iter());
                    let mat = imgcodecs::imdecode(&buf, imgcodecs::IMREAD_COLOR).ok()?;
                    if mat.empty() { return None; }
                    Some(DecodedFrame { camera_id: f.camera_id, width: f.width as u32, height: f.height as u32, mat })
                }).collect();
                if !decoded.is_empty() {
                    let _ = tx.try_send(TcpEvent::FrameSet(DecodedFrameSet { timestamp_us, frames: decoded }));
                }
            }
            ClientMessage::TriggerPoseCalibration => {
                let _ = tx.send(TcpEvent::TriggerPoseCalibration);
            }
        }
    }
}

// ===========================================================================
// Filter setup
// ===========================================================================

fn make_filters(config: &FilterConfig) -> [PoseFilter; TRACKER_COUNT] {
    std::array::from_fn(|i| match i {
        1 | 2 | 4 | 5 => PoseFilter::from_config_lower_body(config),
        _ => PoseFilter::from_config(config),
    })
}

// ===========================================================================
// Inference loop (sync, blocking)
// ===========================================================================

#[allow(clippy::too_many_arguments)]
fn run_inference_loop(
    rx: &mpsc::Receiver<TcpEvent>,
    config: &Config,
    detector: &mut PoseDetector,
    person_detector: &mut Option<PersonDetector>,
    model_type: ModelType,
    vmt: &VmtClient,
    logfile: &LogFile,
) {
    let mut camera_params: Vec<CameraParams> = Vec::new();
    let mut calibrated = false;
    let mut prev_poses: Vec<Option<Pose>> = Vec::new();
    let mut poses_2d: Vec<Option<Pose>> = Vec::new();
    let mut pose_timestamps: Vec<Option<Instant>> = Vec::new();
    let mut cam_widths: Vec<u32> = Vec::new();
    let mut cam_heights: Vec<u32> = Vec::new();
    let mut num_cameras: usize = 0;

    let mut body_tracker = BodyTracker::new(config.mirror_x, config.offset_y, config.foot_y_offset);
    let mut filters: [PoseFilter; TRACKER_COUNT] = make_filters(&config.filter);
    let mut extrapolators: [Extrapolator; TRACKER_COUNT] = std::array::from_fn(|_| Extrapolator::new());
    let mut lerpers: [Lerper; TRACKER_COUNT] = std::array::from_fn(|_| Lerper::new());
    let mut last_poses: [Option<TrackerPose>; TRACKER_COUNT] = [None; TRACKER_COUNT];
    let mut last_update_times: [Instant; TRACKER_COUNT] = [Instant::now(); TRACKER_COUNT];
    let mut stale: [bool; TRACKER_COUNT] = [false; TRACKER_COUNT];
    let mut reject_count: [u32; TRACKER_COUNT] = [0; TRACKER_COUNT];
    let mut last_inference_time = Instant::now();

    // Triangulation state
    let mut prev_hip_ref_pair: Option<(usize, usize)> = None;
    let mut prev_hip_z: Option<f32> = None;

    // Calibration state
    let mut cal_deadline: Option<Instant> = None;
    let mut auto_cal_triggered = false;
    let mut first_hip_time: Option<Instant> = None;
    let mut last_hip_time: Option<Instant> = None;
    let mut recent_hip_z: Vec<f32> = Vec::new();

    // FPS
    let mut fps_frame_count: u32 = 0;
    let mut fps_inference_count: u32 = 0;
    let mut fps_timer = Instant::now();

    let mut pose_3d: Option<Pose> = None;
    let detector_mode = config.detector.clone();
    let interp_mode = config.interpolation_mode.clone();

    log!(logfile, "Waiting for calibration data from client...");

    loop {
        let event = match rx.recv_timeout(Duration::from_millis(16)) {
            Ok(ev) => Some(ev),
            Err(mpsc::RecvTimeoutError::Timeout) => None,
            Err(mpsc::RecvTimeoutError::Disconnected) => { log!(logfile, "TCP channel disconnected"); break; }
        };

        if let Some(ev) = event {
            match ev {
                TcpEvent::CalibrationReceived(cal) => {
                    log!(logfile, "Calibration received: {} cameras", cal.cameras.len());
                    camera_params.clear();
                    cam_widths.clear();
                    cam_heights.clear();

                    for cam_cal in &cal.cameras {
                        let dc: [f64; 5] = [
                            cam_cal.dist_coeffs.get(0).copied().unwrap_or(0.0),
                            cam_cal.dist_coeffs.get(1).copied().unwrap_or(0.0),
                            cam_cal.dist_coeffs.get(2).copied().unwrap_or(0.0),
                            cam_cal.dist_coeffs.get(3).copied().unwrap_or(0.0),
                            cam_cal.dist_coeffs.get(4).copied().unwrap_or(0.0),
                        ];
                        let cp = CameraParams::from_calibration(
                            &cam_cal.intrinsic_matrix, &dc,
                            &cam_cal.rvec, &cam_cal.tvec,
                            cam_cal.width, cam_cal.height,
                        );
                        camera_params.push(cp);
                        cam_widths.push(cam_cal.width);
                        cam_heights.push(cam_cal.height);
                        log!(logfile, "  Camera {}: {}x{}", cam_cal.camera_index, cam_cal.width, cam_cal.height);
                    }

                    num_cameras = cal.cameras.len();
                    prev_poses = vec![None; num_cameras];
                    poses_2d = vec![None; num_cameras];
                    pose_timestamps = vec![None; num_cameras];
                    calibrated = true;

                    // Reset tracker state
                    body_tracker = BodyTracker::new(config.mirror_x, config.offset_y, config.foot_y_offset);
                    filters = make_filters(&config.filter);
                    extrapolators = std::array::from_fn(|_| Extrapolator::new());
                    lerpers = std::array::from_fn(|_| Lerper::new());
                    last_poses = [None; TRACKER_COUNT];
                    last_update_times = [Instant::now(); TRACKER_COUNT];
                    stale = [false; TRACKER_COUNT];
                    reject_count = [0; TRACKER_COUNT];
                    last_inference_time = Instant::now();
                    auto_cal_triggered = false;
                    first_hip_time = None;
                    last_hip_time = None;
                    recent_hip_z.clear();
                    cal_deadline = None;
                    pose_3d = None;
                    prev_hip_ref_pair = None;
                    prev_hip_z = None;

                    log!(logfile, "Ready for inference ({} cameras)", num_cameras);
                }
                TcpEvent::FrameSet(frame_set) if calibrated => {
                    for decoded in &frame_set.frames {
                        let cam_idx = decoded.camera_id as usize;
                        if cam_idx >= num_cameras { continue; }

                        let width = decoded.width;
                        let height = decoded.height;

                        if width != cam_widths[cam_idx] || height != cam_heights[cam_idx] {
                            if camera_params[cam_idx].rescale_to(width, height) {
                                log!(logfile, "Rescaled cam{}: {}x{} -> {}x{}", cam_idx, cam_widths[cam_idx], cam_heights[cam_idx], width, height);
                                cam_widths[cam_idx] = width;
                                cam_heights[cam_idx] = height;
                            }
                        }

                        // Person detection -> crop
                        let (input_frame, crop_region) = match detector_mode.as_str() {
                            "keypoint" => {
                                match prev_poses[cam_idx].as_ref()
                                    .and_then(|p| bbox_from_keypoints(p, width, height, CONFIDENCE_THRESHOLD))
                                {
                                    Some(bbox) => match crop_for_pose(&decoded.mat, &bbox, width, height) {
                                        Ok(v) => v,
                                        Err(_) => (decoded.mat.clone(), CropRegion::full()),
                                    },
                                    None => (decoded.mat.clone(), CropRegion::full()),
                                }
                            }
                            "yolo" => {
                                if let Some(ref mut pd) = person_detector {
                                    match pd.detect(&decoded.mat) {
                                        Ok(Some(bbox)) => match crop_for_pose(&decoded.mat, &bbox, width, height) {
                                            Ok(v) => v,
                                            Err(_) => (decoded.mat.clone(), CropRegion::full()),
                                        },
                                        _ => (decoded.mat.clone(), CropRegion::full()),
                                    }
                                } else { (decoded.mat.clone(), CropRegion::full()) }
                            }
                            _ => (decoded.mat.clone(), CropRegion::full()),
                        };

                        // Preprocess
                        let (input, letterbox) = match model_type {
                            ModelType::MoveNet => match preprocess_for_movenet(&input_frame) {
                                Ok(v) => (v, LetterboxInfo::identity()),
                                Err(e) => { log!(logfile, "preprocess error: {}", e); continue; }
                            },
                            ModelType::SpinePose => match preprocess_for_spinepose(&input_frame) {
                                Ok(v) => v,
                                Err(e) => { log!(logfile, "preprocess error: {}", e); continue; }
                            },
                            ModelType::RTMW3D => match preprocess_for_rtmw3d(&input_frame) {
                                Ok(v) => v,
                                Err(e) => { log!(logfile, "preprocess error: {}", e); continue; }
                            },
                        };

                        // Inference
                        let mut pose = match detector.detect(input) {
                            Ok(v) => v,
                            Err(e) => { log!(logfile, "inference error: {}", e); continue; }
                        };

                        pose = unletterbox_pose(&pose, &letterbox);
                        if !crop_region.is_full() { pose = remap_pose(&pose, &crop_region); }

                        prev_poses[cam_idx] = Some(pose.clone());
                        poses_2d[cam_idx] = Some(pose);
                        pose_timestamps[cam_idx] = Some(Instant::now());
                        fps_inference_count += 1;
                    }

                    // --- Triangulation ---
                    if num_cameras >= 2 {
                        let all_ready = poses_2d.iter().all(|p| p.is_some());
                        let ready_count = poses_2d.iter().filter(|p| p.is_some()).count();

                        let should_triangulate = if all_ready { true }
                        else if ready_count >= 2 {
                            pose_timestamps.iter().filter_map(|t| *t).min()
                                .map(|ts| ts.elapsed().as_secs_f32() * 1000.0 > TRIANGULATION_TIMEOUT_MS)
                                .unwrap_or(false)
                        } else { false };

                        if should_triangulate {
                            let mut available_poses: Vec<&Pose> = Vec::new();
                            let mut available_params: Vec<&CameraParams> = Vec::new();
                            for i in 0..num_cameras {
                                if let Some(ref pose) = poses_2d[i] {
                                    available_poses.push(pose);
                                    available_params.push(&camera_params[i]);
                                }
                            }

                            if available_poses.len() >= 2 {
                                let mut tri_pose = triangulate_poses_multi(
                                    &available_params, &available_poses, CONFIDENCE_THRESHOLD,
                                    &mut prev_hip_ref_pair, &mut prev_hip_z,
                                );

                                for kp in tri_pose.keypoints.iter_mut() {
                                    if kp.confidence > 0.0
                                        && (kp.x.abs() > 10.0 || kp.y.abs() > 10.0
                                            || kp.z.abs() > 10.0 || kp.z < 0.0)
                                    { kp.confidence = 0.0; }
                                }

                                pose_3d = Some(tri_pose);
                            }

                            for p in poses_2d.iter_mut() { *p = None; }
                            for t in pose_timestamps.iter_mut() { *t = None; }
                        }
                    }
                }
                TcpEvent::TriggerPoseCalibration if calibrated => {
                    if cal_deadline.is_none() {
                        cal_deadline = Some(Instant::now() + Duration::from_secs(5));
                        log!(logfile, "Calibration in 5s...");
                    }
                }
                _ => {}
            }
        }

        if !calibrated { continue; }

        // --- Auto-calibration ---
        if !auto_cal_triggered && !body_tracker.is_calibrated() && cal_deadline.is_none() {
            let hip_z = pose_3d.as_ref().and_then(|p| {
                let lh = p.get(KP_LEFT_HIP);
                let rh = p.get(KP_RIGHT_HIP);
                if lh.is_valid(0.2) && rh.is_valid(0.2) { Some((lh.z + rh.z) / 2.0) } else { None }
            });
            if let Some(z) = hip_z {
                let now = Instant::now();
                if first_hip_time.is_none() { first_hip_time = Some(now); recent_hip_z.clear(); }
                last_hip_time = Some(now);
                recent_hip_z.push(z);
                if now.duration_since(first_hip_time.unwrap()).as_secs_f32() > 3.0 {
                    let n = recent_hip_z.len() as f32;
                    let mean = recent_hip_z.iter().sum::<f32>() / n;
                    let variance = recent_hip_z.iter().map(|z| (z - mean).powi(2)).sum::<f32>() / n;
                    let std_dev = variance.sqrt();
                    if std_dev < 0.3 {
                        auto_cal_triggered = true;
                        cal_deadline = Some(now);
                        log!(logfile, "Auto-calibration triggered (hip z stable: mean={:.2}, std={:.2}, n={})", mean, std_dev, recent_hip_z.len());
                    } else {
                        log!(logfile, "Auto-cal rejected: hip z unstable (mean={:.2}, std={:.2}, n={})", mean, std_dev, recent_hip_z.len());
                        first_hip_time = None; recent_hip_z.clear();
                    }
                }
            } else {
                let gap = last_hip_time.map_or(f32::MAX, |t| Instant::now().duration_since(t).as_secs_f32());
                if gap > 0.5 { first_hip_time = None; last_hip_time = None; recent_hip_z.clear(); }
            }
        }

        // --- Manual calibration deadline ---
        if let Some(deadline) = cal_deadline {
            if deadline.saturating_duration_since(Instant::now()).is_zero() {
                if let Some(ref pose) = pose_3d {
                    if body_tracker.calibrate(pose) {
                        filters = make_filters(&config.filter);
                        extrapolators = std::array::from_fn(|_| Extrapolator::new());
                        lerpers = std::array::from_fn(|_| Lerper::new());
                        last_poses = [None; TRACKER_COUNT];
                        log!(logfile, "Calibrated!");
                    } else {
                        log!(logfile, "Calibration failed: hip not detected");
                    }
                }
                cal_deadline = None;
            }
        }

        // --- Compute trackers ---
        if let Some(ref pose) = pose_3d {
            let lh = pose.get(KP_LEFT_HIP);
            let rh = pose.get(KP_RIGHT_HIP);
            let hip_valid = if lh.is_valid(0.2) && rh.is_valid(0.2) {
                let hx = (lh.x + rh.x) / 2.0;
                let hy = (lh.y + rh.y) / 2.0;
                let hz = (lh.z + rh.z) / 2.0;
                !(hx.abs() > 10.0 || hy.abs() > 10.0 || hz.abs() > 10.0 || hz < 0.0)
            } else { true };

            if hip_valid {
                let body_poses = body_tracker.compute(pose);
                let poses = [
                    body_poses.hip, body_poses.left_foot, body_poses.right_foot,
                    body_poses.chest, body_poses.left_knee, body_poses.right_knee,
                ];

                let dt = last_inference_time.elapsed().as_secs_f32();
                let lerp_t = (dt / (1.0 / 30.0)).min(1.0);
                let now = Instant::now();
                let max_displacement = 0.4;
                let max_limb_dist = 1.5;

                for i in 0..TRACKER_COUNT {
                    let mut accepted = false;
                    if let Some(p) = poses[i] {
                        let velocity_ok = match last_poses[i].as_ref() {
                            Some(last) => {
                                let dx = p.position[0] - last.position[0];
                                let dy = p.position[1] - last.position[1];
                                let dz = p.position[2] - last.position[2];
                                (dx * dx + dy * dy + dz * dz).sqrt() <= max_displacement
                            }
                            None => true,
                        };
                        let limb_ok = if i > 0 {
                            match last_poses[0].as_ref() {
                                Some(hip) => {
                                    let dx = p.position[0] - hip.position[0];
                                    let dy = p.position[1] - hip.position[1];
                                    let dz = p.position[2] - hip.position[2];
                                    let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                                    let anatomy_ok = match i {
                                        1 | 2 | 4 | 5 => p.position[1] < hip.position[1],
                                        _ => true,
                                    };
                                    dist <= max_limb_dist && anatomy_ok
                                }
                                None => false,
                            }
                        } else { true };

                        if velocity_ok && limb_ok {
                            let smoothed = filters[i].apply(p);
                            extrapolators[i].update(smoothed);
                            lerpers[i].update(smoothed, lerp_t);
                            last_poses[i] = Some(smoothed);
                            last_update_times[i] = now;
                            stale[i] = false;
                            reject_count[i] = 0;
                            accepted = true;
                        } else if !velocity_ok && limb_ok {
                            reject_count[i] += 1;
                            if reject_count[i] >= 5 {
                                let z_ok = match last_poses[i].as_ref() {
                                    Some(last) => (p.position[2] - last.position[2]).abs() <= 0.3,
                                    None => true,
                                };
                                if z_ok {
                                    filters[i].reset();
                                    extrapolators[i] = Extrapolator::new();
                                    lerpers[i] = Lerper::new();
                                    let smoothed = filters[i].apply(p);
                                    extrapolators[i].update(smoothed);
                                    lerpers[i].update(smoothed, lerp_t);
                                    last_poses[i] = Some(smoothed);
                                    last_update_times[i] = now;
                                    stale[i] = false;
                                    reject_count[i] = 0;
                                    accepted = true;
                                } else { reject_count[i] = 0; }
                            }
                        }
                    }
                    if !accepted && last_poses[i].is_some() {
                        let stale_time = now.duration_since(last_update_times[i]).as_secs_f32();
                        if stale_time > 3.0 { last_poses[i] = None; stale[i] = false; }
                        else if stale_time > 0.3 && !stale[i] {
                            stale[i] = true;
                            filters[i].reset();
                            extrapolators[i] = Extrapolator::new();
                            lerpers[i] = Lerper::new();
                        }
                    }
                }
                last_inference_time = now;
            }
        }

        // --- Send VMT ---
        if pose_3d.is_some() {
            for i in 0..TRACKER_COUNT {
                if !stale[i] {
                    if let Some(ref p) = last_poses[i] { let _ = vmt.send(TRACKER_INDICES[i], 1, p); }
                }
            }
        } else {
            let dt = last_inference_time.elapsed().as_secs_f32();
            match interp_mode.as_str() {
                "extrapolate" => {
                    for i in 0..TRACKER_COUNT {
                        if let Some(predicted) = extrapolators[i].predict(dt) {
                            let _ = vmt.send(TRACKER_INDICES[i], 1, &predicted);
                        }
                    }
                }
                "lerp" => {
                    let t = (dt / (1.0 / 30.0)).min(1.0);
                    for i in 0..TRACKER_COUNT {
                        if let Some(interpolated) = lerpers[i].interpolate(t) {
                            let _ = vmt.send(TRACKER_INDICES[i], 1, &interpolated);
                        }
                    }
                }
                _ => {
                    for i in 0..TRACKER_COUNT {
                        if let Some(ref p) = last_poses[i] { let _ = vmt.send(TRACKER_INDICES[i], 1, p); }
                    }
                }
            }
        }

        // --- FPS logging ---
        fps_frame_count += 1;
        let elapsed = fps_timer.elapsed().as_secs_f32();
        if elapsed >= 1.0 {
            let names = ["hip", "L_foot", "R_foot", "chest", "L_knee", "R_knee"];
            let mut parts = Vec::new();
            for i in 0..TRACKER_COUNT {
                if !stale[i] {
                    if let Some(ref p) = last_poses[i] {
                        parts.push(format!("{}({:.2},{:.2},{:.2})", names[i], p.position[0], p.position[1], p.position[2]));
                    }
                }
            }
            let diag_kps = [("LA", KP_LEFT_ANKLE), ("RA", KP_RIGHT_ANKLE), ("LK", KP_LEFT_KNEE), ("RK", KP_RIGHT_KNEE)];
            let mut cam_diag = String::new();
            for (cam_idx, prev) in prev_poses.iter().enumerate() {
                if let Some(ref p) = prev {
                    let confs: Vec<String> = diag_kps.iter()
                        .map(|(name, idx)| format!("{}={:.2}", name, p.get(*idx).confidence))
                        .collect();
                    cam_diag.push_str(&format!(" cam{}[{}]", cam_idx, confs.join(",")));
                }
            }
            log!(logfile, "FPS: {:.1} (infer: {}) | {}{}",
                fps_frame_count as f32 / elapsed, fps_inference_count,
                if parts.is_empty() { "no pose".to_string() } else { parts.join(" ") },
                cam_diag,
            );
            fps_frame_count = 0;
            fps_inference_count = 0;
            fps_timer = Instant::now();
        }
    }
}

// ===========================================================================
// Main
// ===========================================================================

#[tokio::main]
async fn main() -> Result<()> {
    let config_str = std::fs::read_to_string("inference_server.toml")
        .context("failed to read inference_server.toml")?;
    let config: Config = toml::from_str(&config_str)?;
    let logfile = open_log_file()?;

    log!(logfile, "Inference Server (standalone)");
    log!(logfile, "Listen: {}", config.listen_addr);
    log!(logfile, "VMT target: {}", config.vmt_addr);
    log!(logfile, "Interpolation: {}", config.interpolation_mode);

    let (model_path, model_type): (&str, ModelType) = match config.model.as_str() {
        "movenet" => ("models/movenet_lightning.onnx", ModelType::MoveNet),
        "spinepose_small" => ("models/spinepose_small.onnx", ModelType::SpinePose),
        "spinepose_medium" => ("models/spinepose_medium.onnx", ModelType::SpinePose),
        "rtmw3d" => ("models/rtmw3d-x.onnx", ModelType::RTMW3D),
        other => { bail!("Unknown model: {}", other); }
    };
    log!(logfile, "Model: {}", config.model);
    log!(logfile, "Detector: {}", config.detector);

    let mut detector = PoseDetector::new(model_path, model_type)?;
    log!(logfile, "Pose model loaded");

    let mut person_detector: Option<PersonDetector> = match config.detector.as_str() {
        "yolo" => {
            let pd = PersonDetector::new("models/yolov8n.onnx", 160)?;
            log!(logfile, "Person detector loaded");
            Some(pd)
        }
        _ => None,
    };

    let vmt = VmtClient::new(&config.vmt_addr)?;
    log!(logfile, "VMT client ready");

    let listener = tokio::net::TcpListener::bind(&config.listen_addr).await?;
    log!(logfile, "Listening on {}", config.listen_addr);
    log!(logfile, "");

    loop {
        let (tcp_stream, addr) = listener.accept().await?;
        log!(logfile, "Client connected: {}", addr);

        let (tx, rx) = mpsc::sync_channel::<TcpEvent>(4);

        let tcp_task = tokio::spawn(async move {
            if let Err(e) = tcp_receive_loop(tcp_stream, tx).await {
                eprintln!("TCP error: {}", e);
            }
        });

        tokio::task::block_in_place(|| {
            run_inference_loop(&rx, &config, &mut detector, &mut person_detector, model_type, &vmt, &logfile);
        });

        tcp_task.abort();
        log!(logfile, "Client disconnected, waiting for next connection...");
    }
}
