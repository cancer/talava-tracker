//! Inference server: receives JPEG frames over TCP, runs ONNX pose estimation,
//! triangulates 3D poses, and sends tracker data via VMT/OSC.

use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use nalgebra::Vector4;
use ndarray::Array4;
use opencv::core::{Mat, Size, Vector, CV_32FC3};
use opencv::prelude::*;
use opencv::{imgcodecs, imgproc};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;
use serde::Deserialize;

use talava_tracker::pose::{Keypoint, Pose};
use talava_tracker::pose::preprocess::{
    LetterboxInfo, preprocess_for_movenet, preprocess_for_spinepose, preprocess_for_rtmw3d,
    unletterbox_pose,
};
use talava_tracker::pose::crop::{
    BBox, CropRegion, bbox_from_keypoints, crop_for_pose, remap_pose,
};
use talava_tracker::protocol::{
    self, ClientMessage, ServerMessage,
};
use talava_tracker::triangulation::{
    CameraParams, triangulate_point, reprojection_error,
};
use talava_tracker::vmt::{TrackerPose, VmtClient};

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
    verbose: bool,
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
    lower_body_position_min_cutoff: Option<f32>,
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
            lower_body_position_min_cutoff: None,
            lower_body_position_beta: None,
        }
    }
}

// ===========================================================================
// Keypoint indices (usize aliases for KeypointIndex enum values)
// ===========================================================================

use talava_tracker::pose::keypoint::KeypointIndex;

#[allow(dead_code)]
const KP_COUNT: usize = KeypointIndex::COUNT;

#[allow(dead_code)]
const KP_NOSE: usize = KeypointIndex::Nose as usize;
const KP_LEFT_SHOULDER: usize = KeypointIndex::LeftShoulder as usize;
const KP_RIGHT_SHOULDER: usize = KeypointIndex::RightShoulder as usize;
const KP_LEFT_HIP: usize = KeypointIndex::LeftHip as usize;
const KP_RIGHT_HIP: usize = KeypointIndex::RightHip as usize;
const KP_LEFT_KNEE: usize = KeypointIndex::LeftKnee as usize;
const KP_RIGHT_KNEE: usize = KeypointIndex::RightKnee as usize;
const KP_LEFT_ANKLE: usize = KeypointIndex::LeftAnkle as usize;
const KP_RIGHT_ANKLE: usize = KeypointIndex::RightAnkle as usize;
const KP_HEAD: usize = KeypointIndex::Head as usize;
const KP_NECK: usize = KeypointIndex::Neck as usize;
const KP_HIP: usize = KeypointIndex::Hip as usize;
const KP_LEFT_BIG_TOE: usize = KeypointIndex::LeftBigToe as usize;
const KP_RIGHT_BIG_TOE: usize = KeypointIndex::RightBigToe as usize;
const KP_LEFT_SMALL_TOE: usize = KeypointIndex::LeftSmallToe as usize;
const KP_RIGHT_SMALL_TOE: usize = KeypointIndex::RightSmallToe as usize;
const KP_LEFT_HEEL: usize = KeypointIndex::LeftHeel as usize;
const KP_RIGHT_HEEL: usize = KeypointIndex::RightHeel as usize;

const KP_NAMES: &[&str] = &[
    "Nose","LEye","REye","LEar","REar","LShoulder","RShoulder","LElbow","RElbow",
    "LWrist","RWrist","LHip","RHip","LKnee","RKnee","LAnkle","RAnkle",
    "Head","Neck","Hip","LBigToe","RBigToe","LSmallToe","RSmallToe","LHeel","RHeel",
    "26","27","28","29","30","31","32","33","34","35","36",
];

// ===========================================================================
// Triangulation helpers (uses library CameraParams, triangulate_point, reprojection_error)
// ===========================================================================

const MAX_REPROJ_ERROR: f32 = 120.0;
const MAX_HIP_Z_JUMP: f32 = 0.3;
const MAX_HIP_Z_REJECT_STREAK: u32 = 5;

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
/// Check hip z-jump and update state. Returns `true` if the frame should be rejected.
fn check_hip_z_jump(
    current_hip_z: Option<f32>,
    prev_hip_z: &mut Option<f32>,
    reject_count: &mut u32,
) -> bool {
    if let Some(cur_z) = current_hip_z {
        if let Some(pz) = *prev_hip_z {
            let z_jump = (cur_z - pz).abs();
            if z_jump > MAX_HIP_Z_JUMP {
                *reject_count += 1;
                if *reject_count >= MAX_HIP_Z_REJECT_STREAK {
                    eprintln!("z-jump reject streak reached {}, resetting prev_hip_z to {:.3}", *reject_count, cur_z);
                    *prev_hip_z = Some(cur_z);
                    *reject_count = 0;
                } else {
                    return true; // reject
                }
            } else {
                *reject_count = 0;
            }
        }
        *prev_hip_z = Some(cur_z);
    }
    false
}

fn triangulate_poses_multi(
    cameras: &[&CameraParams],
    poses_2d: &[&Pose],
    confidence_threshold: f32,
    prev_hip_ref_pair: &mut Option<(usize, usize)>,
    prev_hip_z: &mut Option<f32>,
    prev_hip_z_reject_count: &mut u32,
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

    if check_hip_z_jump(current_hip_z, prev_hip_z, prev_hip_z_reject_count) {
        return Pose::new([Keypoint::default(); KP_COUNT]);
    }

    Pose::new(keypoints)
}

// ===========================================================================
// Model types + ONNX inference
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq)]
enum ModelType { MoveNet, SpinePose, RTMW3D }

fn build_session(model_path: &str) -> Result<Session> {
    let builder = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?;

    #[cfg(feature = "cuda")]
    let builder = {
        eprintln!("[ort] Attempting CUDA execution provider...");
        builder.with_execution_providers([ort::execution_providers::CUDAExecutionProvider::default().build()])?
    };

    builder.commit_from_file(model_path).context("Failed to load ONNX model")
}

struct PoseDetector {
    session: Session,
    model_type: ModelType,
}

impl PoseDetector {
    fn new(model_path: &str, model_type: ModelType) -> Result<Self> {
        let session = build_session(model_path)?;
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

/// Invalidate keypoints near image boundaries where model receptive field is clipped.
fn sanitize_pose(pose: &Pose) -> Pose {
    // Full boundary check (X and Y): limbs and shoulders
    const FULL_BOUNDARY: &[usize] = &[
        KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER,
        KP_LEFT_KNEE, KP_RIGHT_KNEE, KP_LEFT_ANKLE, KP_RIGHT_ANKLE,
        KP_LEFT_BIG_TOE, KP_RIGHT_BIG_TOE, KP_LEFT_SMALL_TOE, KP_RIGHT_SMALL_TOE,
        KP_LEFT_HEEL, KP_RIGHT_HEEL,
    ];
    // X-only boundary check: hips (Y direction allowed to be at edge for lower body)
    const X_ONLY_BOUNDARY: &[usize] = &[KP_LEFT_HIP, KP_RIGHT_HIP];

    let mut sanitized = pose.clone();
    for &idx in FULL_BOUNDARY {
        let kp = &mut sanitized.keypoints[idx];
        if kp.x <= 0.02 || kp.x >= 0.98 || kp.y <= 0.02 || kp.y >= 0.98 {
            kp.confidence = 0.0;
        }
    }
    for &idx in X_ONLY_BOUNDARY {
        let kp = &mut sanitized.keypoints[idx];
        if kp.x <= 0.02 || kp.x >= 0.98 {
            kp.confidence = 0.0;
        }
    }
    sanitized
}

struct PersonDetector {
    session: Session,
    input_size: i32,
}

impl PersonDetector {
    fn new(model_path: &str, input_size: i32) -> Result<Self> {
        let session = build_session(model_path)?;
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
        imgproc::cvt_color_def(frame, &mut rgb, imgproc::COLOR_BGR2RGB)?;
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
        let min_cutoff = config.lower_body_position_min_cutoff.unwrap_or(config.position_min_cutoff);
        let beta = config.lower_body_position_beta.unwrap_or(config.position_beta);
        Self::new(min_cutoff, beta, config.rotation_min_cutoff, config.rotation_beta)
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
const BODY_CONFIDENCE_THRESHOLD: f32 = 0.2;

struct BodyCalibration {
    hip_x: f32, hip_y: f32,
    hip_z: f32,
    yaw_shoulder: f32,
    yaw_left_foot: f32, yaw_right_foot: f32,
    left_knee_offset: Option<(f32, f32)>,
    right_knee_offset: Option<(f32, f32)>,
    yaw_left_knee: f32, yaw_right_knee: f32,
    pitch_left_knee: f32, pitch_right_knee: f32,
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
        let lh = pose.get_by_index(KP_LEFT_HIP);
        let rh = pose.get_by_index(KP_RIGHT_HIP);
        if !lh.is_valid(BODY_CONFIDENCE_THRESHOLD) || !rh.is_valid(BODY_CONFIDENCE_THRESHOLD) { return 0.0; }
        let hip_y = (lh.y + rh.y) / 2.0;
        let hip_z = (lh.z + rh.z) / 2.0;

        let la = pose.get_by_index(KP_LEFT_ANKLE);
        let ra = pose.get_by_index(KP_RIGHT_ANKLE);
        let lk = pose.get_by_index(KP_LEFT_KNEE);
        let rk = pose.get_by_index(KP_RIGHT_KNEE);

        let (lower_y, lower_z) = if la.is_valid(BODY_CONFIDENCE_THRESHOLD) && ra.is_valid(BODY_CONFIDENCE_THRESHOLD) {
            ((la.y + ra.y) / 2.0, (la.z + ra.z) / 2.0)
        } else if lk.is_valid(BODY_CONFIDENCE_THRESHOLD) && rk.is_valid(BODY_CONFIDENCE_THRESHOLD) {
            ((lk.y + rk.y) / 2.0, (lk.z + rk.z) / 2.0)
        } else { return 0.0; };

        f32::atan2(lower_z - hip_z, lower_y - hip_y)
    }

    fn rotate_pose_yz(pose: &Pose, tilt: f32) -> Pose {
        let (sin_t, cos_t) = tilt.sin_cos();
        let lh = pose.get_by_index(KP_LEFT_HIP);
        let rh = pose.get_by_index(KP_RIGHT_HIP);
        let ref_z = if lh.is_valid(BODY_CONFIDENCE_THRESHOLD) && rh.is_valid(BODY_CONFIDENCE_THRESHOLD) {
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

        let lh = pose.get_by_index(KP_LEFT_HIP);
        let rh = pose.get_by_index(KP_RIGHT_HIP);
        if !lh.is_valid(BODY_CONFIDENCE_THRESHOLD) || !rh.is_valid(BODY_CONFIDENCE_THRESHOLD) { return false; }

        let hip_x = (lh.x + rh.x) / 2.0;
        let hip_y = (lh.y + rh.y) / 2.0;
        let hip_z = (lh.z + rh.z) / 2.0;

        let yaw_shoulder = self.compute_shoulder_yaw(pose);
        let yaw_left_foot = self.compute_foot_yaw(pose, KP_LEFT_KNEE, KP_LEFT_ANKLE);
        let yaw_right_foot = self.compute_foot_yaw(pose, KP_RIGHT_KNEE, KP_RIGHT_ANKLE);

        let lk = pose.get_by_index(KP_LEFT_KNEE);
        let left_knee_offset = if lk.is_valid(BODY_CONFIDENCE_THRESHOLD) { Some((lk.x - hip_x, lk.y - hip_y)) } else { None };
        let rk = pose.get_by_index(KP_RIGHT_KNEE);
        let right_knee_offset = if rk.is_valid(BODY_CONFIDENCE_THRESHOLD) { Some((rk.x - hip_x, rk.y - hip_y)) } else { None };

        let yaw_left_knee = self.compute_knee_yaw(pose, KP_LEFT_HIP, KP_LEFT_KNEE);
        let yaw_right_knee = self.compute_knee_yaw(pose, KP_RIGHT_HIP, KP_RIGHT_KNEE);
        let pitch_left_knee = self.compute_knee_pitch(pose, KP_LEFT_HIP, KP_LEFT_KNEE);
        let pitch_right_knee = self.compute_knee_pitch(pose, KP_RIGHT_HIP, KP_RIGHT_KNEE);

        self.calibration = Some(BodyCalibration {
            hip_x, hip_y, hip_z,
            yaw_shoulder, yaw_left_foot, yaw_right_foot,
            left_knee_offset, right_knee_offset,
            yaw_left_knee, yaw_right_knee,
            pitch_left_knee, pitch_right_knee,
            tilt_angle,
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

        let lh = pose.get_by_index(KP_LEFT_HIP);
        let rh = pose.get_by_index(KP_RIGHT_HIP);
        let hip_center = if lh.is_valid(BODY_CONFIDENCE_THRESHOLD) && rh.is_valid(BODY_CONFIDENCE_THRESHOLD) {
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

    fn estimate_depth(&self, pose: &Pose) -> f32 {
        let lh = pose.get_by_index(KP_LEFT_HIP);
        let rh = pose.get_by_index(KP_RIGHT_HIP);
        if lh.is_valid(BODY_CONFIDENCE_THRESHOLD) && rh.is_valid(BODY_CONFIDENCE_THRESHOLD) && (lh.z.abs() > 0.001 || rh.z.abs() > 0.001) {
            let hip_z = (lh.z + rh.z) / 2.0;
            let ref_z = self.calibration.as_ref().map_or(0.0, |c| c.hip_z);
            return ref_z - hip_z;
        }
        0.0
    }

    fn keypoint_depth(&self, pose: &Pose, kp_idx: usize, fallback: f32) -> f32 {
        let kp = pose.get_by_index(kp_idx);
        if kp.is_valid(BODY_CONFIDENCE_THRESHOLD) && kp.z.abs() > 0.001 {
            let ref_z = self.calibration.as_ref().map_or(0.0, |c| c.hip_z);
            ref_z - kp.z
        } else { fallback }
    }

    fn compute_shoulder_yaw(&self, pose: &Pose) -> f32 {
        let ls = pose.get_by_index(KP_LEFT_SHOULDER);
        let rs = pose.get_by_index(KP_RIGHT_SHOULDER);
        if ls.is_valid(BODY_CONFIDENCE_THRESHOLD) && rs.is_valid(BODY_CONFIDENCE_THRESHOLD) {
            let dx = if self.mirror_x { rs.x - ls.x } else { ls.x - rs.x };
            let dy = rs.y - ls.y;
            f32::atan2(dy, dx)
        } else { 0.0 }
    }

    fn compute_foot_yaw(&self, pose: &Pose, knee_idx: usize, ankle_idx: usize) -> f32 {
        let knee = pose.get_by_index(knee_idx);
        let ankle = pose.get_by_index(ankle_idx);
        if knee.is_valid(BODY_CONFIDENCE_THRESHOLD) && ankle.is_valid(BODY_CONFIDENCE_THRESHOLD) {
            let raw_dx = ankle.x - knee.x;
            let dx = if self.mirror_x { raw_dx } else { -raw_dx };
            let dy = ankle.y - knee.y;
            f32::atan2(dx, dy)
        } else { 0.0 }
    }

    fn compute_knee_yaw(&self, pose: &Pose, hip_idx: usize, knee_idx: usize) -> f32 {
        let hip = pose.get_by_index(hip_idx);
        let knee = pose.get_by_index(knee_idx);
        if hip.is_valid(BODY_CONFIDENCE_THRESHOLD) && knee.is_valid(BODY_CONFIDENCE_THRESHOLD) {
            let raw_dx = knee.x - hip.x;
            let dx = if self.mirror_x { raw_dx } else { -raw_dx };
            let dy = knee.y - hip.y;
            f32::atan2(dx, dy)
        } else { 0.0 }
    }

    fn compute_knee_pitch(&self, pose: &Pose, hip_idx: usize, knee_idx: usize) -> f32 {
        let hip = pose.get_by_index(hip_idx);
        let knee = pose.get_by_index(knee_idx);
        if hip.is_valid(BODY_CONFIDENCE_THRESHOLD) && knee.is_valid(BODY_CONFIDENCE_THRESHOLD) {
            let dz = knee.z - hip.z;
            let dy = knee.y - hip.y;
            f32::atan2(dz, dy)
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
        let knee = pose.get_by_index(knee_idx);
        let ankle = pose.get_by_index(ankle_idx);
        let ankle_valid = ankle.is_valid(BODY_CONFIDENCE_THRESHOLD);
        let knee_valid = knee.is_valid(BODY_CONFIDENCE_THRESHOLD);
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
        let ls = pose.get_by_index(KP_LEFT_SHOULDER);
        let rs = pose.get_by_index(KP_RIGHT_SHOULDER);
        let ls_valid = ls.is_valid(BODY_CONFIDENCE_THRESHOLD);
        let rs_valid = rs.is_valid(BODY_CONFIDENCE_THRESHOLD);
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
        let knee = pose.get_by_index(knee_idx);

        let (kx, ky, has_keypoint) = if knee.is_valid(BODY_CONFIDENCE_THRESHOLD) {
            (knee.x, knee.y, true)
        } else if let Some(cal) = &self.calibration {
            let offset = if is_left { cal.left_knee_offset } else { cal.right_knee_offset };
            if let Some((ox, oy)) = offset { (hip_x + ox, hip_y + oy, false) }
            else { return None; }
        } else { return None; };

        let position = self.convert_position(kx, ky, hip_x, hip_y, pos_z);
        let (yaw, pitch) = if has_keypoint {
            let ref_yaw = self.calibration.as_ref().map_or(0.0, |c| {
                if is_left { c.yaw_left_knee } else { c.yaw_right_knee }
            });
            let ref_pitch = self.calibration.as_ref().map_or(0.0, |c| {
                if is_left { c.pitch_left_knee } else { c.pitch_right_knee }
            });
            (
                self.compute_knee_yaw(pose, hip_idx, knee_idx) - ref_yaw,
                self.compute_knee_pitch(pose, hip_idx, knee_idx) - ref_pitch,
            )
        } else { (0.0, 0.0) };
        Some(TrackerPose::new(position, Self::yaw_pitch_to_quaternion(yaw, pitch)))
    }
}

// ===========================================================================
// Logging
// ===========================================================================

type LogFile = Arc<Mutex<std::io::BufWriter<std::fs::File>>>;

fn open_log_file() -> Result<(LogFile, String)> {
    std::fs::create_dir_all("logs")?;
    let ts = chrono::Local::now().format("%Y%m%d_%H%M%S");
    let path = format!("logs/inference_{}.log", ts);
    let file = std::fs::File::create(&path)?;
    eprintln!("Log: {}", path);
    Ok((Arc::new(Mutex::new(std::io::BufWriter::new(file))), path))
}

macro_rules! log {
    ($logfile:expr, $($arg:tt)*) => {{
        let msg = format!($($arg)*);
        eprintln!("{}", msg);
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

struct RawFrame { camera_id: u8, width: u32, height: u32, jpeg_data: Vec<u8> }
struct RawFrameSet { #[allow(dead_code)] timestamp_us: u64, frames: Vec<RawFrame> }

enum TcpEvent {
    CalibrationReceived(protocol::CalibrationData),
    FrameSet(RawFrameSet),
    TriggerPoseCalibration,
}

async fn tcp_receive_loop(
    stream: tokio::net::TcpStream,
    tx: mpsc::SyncSender<TcpEvent>,
    mut out_rx: tokio::sync::mpsc::Receiver<ServerMessage>,
    frame_drop_count: Arc<AtomicU32>,
) -> Result<()> {
    use futures::StreamExt as _;

    let framed = protocol::message_stream(stream);
    let (mut sink, mut reader) = framed.split();

    loop {
        tokio::select! {
            result = reader.next() => {
                let bytes = match result {
                    Some(Ok(b)) => b,
                    Some(Err(e)) => return Err(e.into()),
                    None => return Err(anyhow::anyhow!("connection closed")),
                };
                let msg: ClientMessage = bincode::deserialize(&bytes)?;
                match msg {
                    ClientMessage::CameraCalibration { data } => {
                        let _ = tx.send(TcpEvent::CalibrationReceived(data));
                        protocol::send_to_sink(&mut sink, &ServerMessage::CameraCalibrationAck { ok: true, error: None }).await?;
                        protocol::send_to_sink(&mut sink, &ServerMessage::Ready).await?;
                    }
                    ClientMessage::FrameSet { timestamp_us, frames } => {
                        let raw_frames: Vec<RawFrame> = frames.into_iter().map(|f| {
                            RawFrame { camera_id: f.camera_id, width: f.width as u32, height: f.height as u32, jpeg_data: f.jpeg_data }
                        }).collect();
                        if !raw_frames.is_empty() {
                            if tx.try_send(TcpEvent::FrameSet(RawFrameSet { timestamp_us, frames: raw_frames })).is_err() {
                                frame_drop_count.fetch_add(1, Ordering::Relaxed);
                            }
                        }
                    }
                    ClientMessage::TriggerPoseCalibration => {
                        let _ = tx.send(TcpEvent::TriggerPoseCalibration);
                    }
                }
            }
            Some(out_msg) = out_rx.recv() => {
                protocol::send_to_sink(&mut sink, &out_msg).await?;
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
    verbose: bool,
    log_path: &str,
    out_tx: &tokio::sync::mpsc::Sender<ServerMessage>,
    trigger_log_send: &AtomicBool,
    trigger_calibration: &AtomicBool,
    frame_drop_count: &AtomicU32,
) {
    let mut camera_params: Vec<CameraParams> = Vec::new();
    let mut calibrated = false;
    let mut prev_poses: Vec<Option<Pose>> = Vec::new();
    let mut poses_2d: Vec<Option<Pose>> = Vec::new();
    let mut pose_timestamps: Vec<Option<Instant>> = Vec::new();
    let mut cam_widths: Vec<u32> = Vec::new();
    let mut cam_heights: Vec<u32> = Vec::new();
    let mut num_cameras: usize = 0;
    let mut cam_id_to_idx: std::collections::HashMap<u8, usize> = std::collections::HashMap::new();

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
    let mut prev_hip_z_reject_count: u32 = 0;

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
    let mut fps_reject_velocity: u32 = 0;
    let mut fps_reject_limb: u32 = 0;
    let mut fps_reject_zjump: u32 = 0;
    let mut fps_no_pose: u32 = 0;

    let mut pose_3d: Option<Pose> = None;
    let detector_mode = config.detector.clone();
    let interp_mode = config.interpolation_mode.clone();

    log!(logfile, "Waiting for calibration data from client...");

    loop {
        // Check console triggers
        if trigger_calibration.swap(false, Ordering::Relaxed) && calibrated && cal_deadline.is_none() {
            cal_deadline = Some(Instant::now() + Duration::from_secs(5));
            log!(logfile, "Calibration in 5s... (console)");
        }
        if trigger_log_send.swap(false, Ordering::Relaxed) {
            if let Ok(mut f) = logfile.lock() {
                let _ = f.flush();
            }
            match std::fs::read(log_path) {
                Ok(content) => {
                    let filename = std::path::Path::new(log_path)
                        .file_name()
                        .map(|n| n.to_string_lossy().to_string())
                        .unwrap_or_else(|| "inference.log".to_string());
                    let len = content.len();
                    let msg = ServerMessage::LogData { filename, data: content };
                    let _ = out_tx.blocking_send(msg);
                    log!(logfile, "[log] sent log to camera ({} bytes)", len);
                }
                Err(e) => {
                    log!(logfile, "[log] failed to read log file: {}", e);
                }
            }
        }

        let event = match rx.recv_timeout(Duration::from_millis(1)) {
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
                    cam_id_to_idx.clear();

                    for (idx, cam_cal) in cal.cameras.iter().enumerate() {
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
                        if verbose {
                            log!(logfile, "  Camera {} (idx {}): {}x{} dist_coeffs=[{:.4},{:.4},{:.4},{:.4},{:.4}]",
                                cam_cal.camera_index, idx, cam_cal.width, cam_cal.height,
                                dc[0], dc[1], dc[2], dc[3], dc[4]);
                            log!(logfile, "    intrinsic: fx={:.1} fy={:.1} cx={:.1} cy={:.1}",
                                cp.intrinsic[(0,0)], cp.intrinsic[(1,1)], cp.intrinsic[(0,2)], cp.intrinsic[(1,2)]);
                        } else {
                            log!(logfile, "  Camera {} (idx {}): {}x{}", cam_cal.camera_index, idx, cam_cal.width, cam_cal.height);
                        }
                        camera_params.push(cp);
                        cam_widths.push(cam_cal.width);
                        cam_heights.push(cam_cal.height);
                        cam_id_to_idx.insert(cam_cal.camera_index as u8, idx);
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
                    prev_hip_z_reject_count = 0;

                    log!(logfile, "Ready for inference ({} cameras)", num_cameras);
                }
                TcpEvent::FrameSet(frame_set) if calibrated => {
                    for raw_frame in &frame_set.frames {
                        let cam_idx = match cam_id_to_idx.get(&raw_frame.camera_id) {
                            Some(&idx) => idx,
                            None => {
                                if verbose { log!(logfile, "[verbose] unknown camera_id={}, skipping", raw_frame.camera_id); }
                                continue;
                            }
                        };

                        let buf = Vector::<u8>::from_iter(raw_frame.jpeg_data.iter().copied());
                        let mat = match imgcodecs::imdecode(&buf, imgcodecs::IMREAD_COLOR) {
                            Ok(m) if !m.empty() => m,
                            _ => continue,
                        };

                        let width = raw_frame.width;
                        let height = raw_frame.height;

                        if verbose {
                            let mat_rows = mat.rows();
                            let mat_cols = mat.cols();
                            log!(logfile, "[verbose] cam{}: decoded={}x{} (expected {}x{})",
                                cam_idx, mat_cols, mat_rows, width, height);
                        }

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
                                    Some(bbox) => {
                                        if verbose { log!(logfile, "[verbose] cam{}: keypoint crop x={} y={} w={} h={}", cam_idx, bbox.x, bbox.y, bbox.width, bbox.height); }
                                        match crop_for_pose(&mat, &bbox, width, height) {
                                            Ok(v) => v,
                                            Err(_) => (mat.clone(), CropRegion::full()),
                                        }
                                    }
                                    None => {
                                        if verbose { log!(logfile, "[verbose] cam{}: no keypoint bbox, full frame", cam_idx); }
                                        (mat.clone(), CropRegion::full())
                                    }
                                }
                            }
                            "yolo" => {
                                if let Some(ref mut pd) = person_detector {
                                    match pd.detect(&mat) {
                                        Ok(Some(bbox)) => match crop_for_pose(&mat, &bbox, width, height) {
                                            Ok(v) => v,
                                            Err(_) => (mat.clone(), CropRegion::full()),
                                        },
                                        _ => (mat.clone(), CropRegion::full()),
                                    }
                                } else { (mat.clone(), CropRegion::full()) }
                            }
                            _ => (mat.clone(), CropRegion::full()),
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

                        if verbose {
                            log!(logfile, "[verbose] cam{}: letterbox pad_left={:.1} pad_top={:.1} content=({:.3},{:.3})",
                                cam_idx, letterbox.pad_left, letterbox.pad_top, letterbox.content_width, letterbox.content_height);
                        }

                        // Inference
                        let infer_start = Instant::now();
                        let mut pose = match detector.detect(input) {
                            Ok(v) => v,
                            Err(e) => { log!(logfile, "inference error: {}", e); continue; }
                        };
                        let infer_ms = infer_start.elapsed().as_secs_f64() * 1000.0;

                        pose = unletterbox_pose(&pose, &letterbox);
                        if !crop_region.is_full() { pose = remap_pose(&pose, &crop_region); }

                        if verbose {
                            // Log pre-sanitize keypoints for key joints
                            let diag_kps: &[(usize, &str)] = &[
                                (KP_LEFT_HIP, "LHip"), (KP_RIGHT_HIP, "RHip"),
                                (KP_LEFT_KNEE, "LKnee"), (KP_RIGHT_KNEE, "RKnee"),
                                (KP_LEFT_ANKLE, "LAnkle"), (KP_RIGHT_ANKLE, "RAnkle"),
                                (KP_LEFT_SHOULDER, "LShoulder"), (KP_RIGHT_SHOULDER, "RShoulder"),
                            ];
                            let kp_strs: Vec<String> = diag_kps.iter()
                                .map(|(idx, name)| {
                                    let kp = &pose.keypoints[*idx];
                                    format!("{}({:.3},{:.3},{:.2})", name, kp.x, kp.y, kp.confidence)
                                }).collect();
                            log!(logfile, "[verbose] cam{}: infer={:.1}ms 2d=[{}]",
                                cam_idx, infer_ms, kp_strs.join(" "));
                        }

                        let pre_sanitize = if verbose { Some(pose.clone()) } else { None };
                        pose = sanitize_pose(&pose);

                        if verbose {
                            if let Some(ref pre) = pre_sanitize {
                                let sanitized_kps: Vec<String> = (0..KP_COUNT)
                                    .filter(|&i| pre.keypoints[i].confidence > 0.0 && pose.keypoints[i].confidence == 0.0)
                                    .map(|i| {
                                        let kp = &pre.keypoints[i];
                                        format!("{}({:.3},{:.3})", KP_NAMES[i], kp.x, kp.y)
                                    }).collect();
                                if !sanitized_kps.is_empty() {
                                    log!(logfile, "[verbose] cam{}: sanitized out: {}", cam_idx, sanitized_kps.join(" "));
                                }
                            }
                        }

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

                        if verbose && !should_triangulate && ready_count >= 1 {
                            let oldest_ms = pose_timestamps.iter().filter_map(|t| *t).min()
                                .map(|ts| ts.elapsed().as_secs_f32() * 1000.0).unwrap_or(0.0);
                            log!(logfile, "[verbose] tri: waiting ({}/{} cams ready, oldest={:.0}ms)",
                                ready_count, num_cameras, oldest_ms);
                        }

                        if should_triangulate {
                            let mut available_poses: Vec<&Pose> = Vec::new();
                            let mut available_params: Vec<&CameraParams> = Vec::new();
                            let mut available_cam_indices: Vec<usize> = Vec::new();
                            for i in 0..num_cameras {
                                if let Some(ref pose) = poses_2d[i] {
                                    available_poses.push(pose);
                                    available_params.push(&camera_params[i]);
                                    available_cam_indices.push(i);
                                }
                            }

                            if verbose {
                                log!(logfile, "[verbose] tri: using {} cameras (idx {:?}), prev_ref_pair={:?}, prev_hip_z={:?}",
                                    available_poses.len(), available_cam_indices, prev_hip_ref_pair, prev_hip_z);
                            }

                            if available_poses.len() >= 2 {
                                let zjump_count_before = prev_hip_z_reject_count;
                                let mut tri_pose = triangulate_poses_multi(
                                    &available_params, &available_poses, CONFIDENCE_THRESHOLD,
                                    &mut prev_hip_ref_pair, &mut prev_hip_z,
                                    &mut prev_hip_z_reject_count,
                                );
                                if prev_hip_z_reject_count > zjump_count_before {
                                    fps_reject_zjump += 1;
                                }
                                if zjump_count_before > 0 && prev_hip_z_reject_count == 0 {
                                    log!(logfile, "z-jump reject streak reset (was {}), hip_z={:?}", zjump_count_before, prev_hip_z);
                                }

                                if verbose {
                                    log!(logfile, "[verbose] tri: ref_pair={:?} hip_z={:?}", prev_hip_ref_pair, prev_hip_z);
                                    let diag_kps: &[(usize, &str)] = &[
                                        (KP_LEFT_HIP, "LHip"), (KP_RIGHT_HIP, "RHip"),
                                        (KP_LEFT_KNEE, "LKnee"), (KP_RIGHT_KNEE, "RKnee"),
                                        (KP_LEFT_ANKLE, "LAnkle"), (KP_RIGHT_ANKLE, "RAnkle"),
                                        (KP_LEFT_SHOULDER, "LShoulder"), (KP_RIGHT_SHOULDER, "RShoulder"),
                                    ];
                                    let kp_strs: Vec<String> = diag_kps.iter()
                                        .filter(|(idx, _)| tri_pose.keypoints[*idx].confidence > 0.0)
                                        .map(|(idx, name)| {
                                            let kp = &tri_pose.keypoints[*idx];
                                            format!("{}({:.3},{:.3},{:.3} c={:.2})", name, kp.x, kp.y, kp.z, kp.confidence)
                                        }).collect();
                                    log!(logfile, "[verbose] tri 3d: [{}]", kp_strs.join(" "));
                                }

                                let mut oob_count = 0;
                                for kp in tri_pose.keypoints.iter_mut() {
                                    if kp.confidence > 0.0
                                        && (kp.x.abs() > 10.0 || kp.y.abs() > 10.0
                                            || kp.z.abs() > 10.0 || kp.z < 0.0)
                                    {
                                        oob_count += 1;
                                        kp.confidence = 0.0;
                                    }
                                }
                                if verbose && oob_count > 0 {
                                    log!(logfile, "[verbose] tri: {} keypoints out-of-bounds filtered", oob_count);
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
                let lh = p.get_by_index(KP_LEFT_HIP);
                let rh = p.get_by_index(KP_RIGHT_HIP);
                if lh.is_valid(BODY_CONFIDENCE_THRESHOLD) && rh.is_valid(BODY_CONFIDENCE_THRESHOLD) { Some((lh.z + rh.z) / 2.0) } else { None }
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
        let tracker_names = ["hip", "L_foot", "R_foot", "chest", "L_knee", "R_knee"];
        if let Some(ref pose) = pose_3d {
            let lh = pose.get_by_index(KP_LEFT_HIP);
            let rh = pose.get_by_index(KP_RIGHT_HIP);
            let hip_valid = if lh.is_valid(BODY_CONFIDENCE_THRESHOLD) && rh.is_valid(BODY_CONFIDENCE_THRESHOLD) {
                let hx = (lh.x + rh.x) / 2.0;
                let hy = (lh.y + rh.y) / 2.0;
                let hz = (lh.z + rh.z) / 2.0;
                if verbose {
                    log!(logfile, "[verbose] hip_check: ({:.3},{:.3},{:.3})", hx, hy, hz);
                }
                !(hx.abs() > 10.0 || hy.abs() > 10.0 || hz.abs() > 10.0 || hz < 0.0)
            } else {
                if verbose { log!(logfile, "[verbose] hip not valid (LHip c={:.2}, RHip c={:.2})", lh.confidence, rh.confidence); }
                true
            };

            if !hip_valid {
                if verbose { log!(logfile, "[verbose] hip OOB, skipping tracker compute"); }
            } else {
                let body_poses = body_tracker.compute(pose);
                let poses = [
                    body_poses.hip, body_poses.left_foot, body_poses.right_foot,
                    body_poses.chest, body_poses.left_knee, body_poses.right_knee,
                ];

                if verbose {
                    let raw_strs: Vec<String> = poses.iter().enumerate()
                        .map(|(i, p)| match p {
                            Some(tp) => format!("{}({:.3},{:.3},{:.3})", tracker_names[i], tp.position[0], tp.position[1], tp.position[2]),
                            None => format!("{}(none)", tracker_names[i]),
                        }).collect();
                    log!(logfile, "[verbose] body_tracker raw: {}", raw_strs.join(" "));
                }

                let dt = last_inference_time.elapsed().as_secs_f32();
                let lerp_t = (dt / (1.0 / 30.0)).min(1.0);
                let now = Instant::now();
                let max_displacement = 0.4;
                let max_limb_dist = 1.5;

                for i in 0..TRACKER_COUNT {
                    let mut accepted = false;
                    if let Some(p) = poses[i] {
                        let (velocity_ok, displacement) = match last_poses[i].as_ref() {
                            Some(last) => {
                                let dx = p.position[0] - last.position[0];
                                let dy = p.position[1] - last.position[1];
                                let dz = p.position[2] - last.position[2];
                                let d = (dx * dx + dy * dy + dz * dz).sqrt();
                                (d <= max_displacement, d)
                            }
                            None => (true, 0.0),
                        };
                        let (limb_ok, limb_dist) = if i > 0 {
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
                                    (dist <= max_limb_dist && anatomy_ok, dist)
                                }
                                None => (false, 0.0),
                            }
                        } else { (true, 0.0) };

                        if !velocity_ok { fps_reject_velocity += 1; }
                        if !limb_ok { fps_reject_limb += 1; }

                        if verbose && (!velocity_ok || !limb_ok) {
                            log!(logfile, "[verbose] {}: reject vel_ok={} (d={:.3}) limb_ok={} (d={:.3}) reject_count={}",
                                tracker_names[i], velocity_ok, displacement, limb_ok, limb_dist, reject_count[i]);
                        }

                        if velocity_ok && limb_ok {
                            let smoothed = filters[i].apply(p);
                            if verbose {
                                log!(logfile, "[verbose] {}: raw({:.3},{:.3},{:.3}) -> filtered({:.3},{:.3},{:.3})",
                                    tracker_names[i],
                                    p.position[0], p.position[1], p.position[2],
                                    smoothed.position[0], smoothed.position[1], smoothed.position[2]);
                            }
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
                                    if verbose { log!(logfile, "[verbose] {}: reset after {} rejects (z_ok)", tracker_names[i], reject_count[i]); }
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
                                } else {
                                    if verbose { log!(logfile, "[verbose] {}: reset blocked (z_jump too large)", tracker_names[i]); }
                                    reject_count[i] = 0;
                                }
                            }
                        }
                    }
                    if !accepted && last_poses[i].is_some() {
                        let stale_time = now.duration_since(last_update_times[i]).as_secs_f32();
                        if stale_time > 3.0 {
                            if verbose { log!(logfile, "[verbose] {}: expired (stale {:.1}s)", tracker_names[i], stale_time); }
                            last_poses[i] = None; stale[i] = false;
                        }
                        else if stale_time > 0.3 && !stale[i] {
                            if verbose { log!(logfile, "[verbose] {}: went stale ({:.1}s)", tracker_names[i], stale_time); }
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
            if verbose {
                let vmt_strs: Vec<String> = (0..TRACKER_COUNT)
                    .filter(|&i| !stale[i] && last_poses[i].is_some())
                    .map(|i| {
                        let p = last_poses[i].as_ref().unwrap();
                        format!("{}({:.3},{:.3},{:.3} q=[{:.3},{:.3},{:.3},{:.3}])",
                            tracker_names[i],
                            p.position[0], p.position[1], p.position[2],
                            p.rotation[0], p.rotation[1], p.rotation[2], p.rotation[3])
                    }).collect();
                if !vmt_strs.is_empty() {
                    log!(logfile, "[verbose] vmt send: {}", vmt_strs.join(" "));
                }
            }
            for i in 0..TRACKER_COUNT {
                if !stale[i] {
                    if let Some(ref p) = last_poses[i] { let _ = vmt.send(TRACKER_INDICES[i], 1, p); }
                }
            }
        } else {
            fps_no_pose += 1;
            let dt = last_inference_time.elapsed().as_secs_f32();
            if verbose { log!(logfile, "[verbose] no pose_3d, interpolating (mode={}, dt={:.3}s)", interp_mode, dt); }
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
                        .map(|(name, idx)| format!("{}={:.2}", name, p.get_by_index(*idx).confidence))
                        .collect();
                    cam_diag.push_str(&format!(" cam{}[{}]", cam_idx, confs.join(",")));
                }
            }
            let drops = frame_drop_count.swap(0, Ordering::Relaxed);
            log!(logfile, "FPS: {:.1} (infer: {} drop: {}) reject: vel={} limb={} zjump={} no_pose={} | {}{}",
                fps_frame_count as f32 / elapsed, fps_inference_count, drops,
                fps_reject_velocity, fps_reject_limb, fps_reject_zjump, fps_no_pose,
                if parts.is_empty() { "no pose".to_string() } else { parts.join(" ") },
                cam_diag,
            );
            if let Ok(mut f) = logfile.lock() {
                let _ = f.flush();
            }
            fps_frame_count = 0;
            fps_inference_count = 0;
            fps_reject_velocity = 0;
            fps_reject_limb = 0;
            fps_reject_zjump = 0;
            fps_no_pose = 0;
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
    let (logfile, log_path) = open_log_file()?;

    log!(logfile, "Inference Server ({})", env!("GIT_VERSION"));
    log!(logfile, "Listen: {}", config.listen_addr);
    log!(logfile, "VMT target: {}", config.vmt_addr);
    log!(logfile, "Interpolation: {}", config.interpolation_mode);
    if config.verbose { log!(logfile, "Verbose mode: ON"); }
    log!(logfile, "Press 'l' + Enter to send log to camera");

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

    // Console input: 'l' → send log, 'p' → pose calibration
    let trigger_log_send = Arc::new(AtomicBool::new(false));
    let trigger_calibration = Arc::new(AtomicBool::new(false));
    {
        let log_flag = Arc::clone(&trigger_log_send);
        let cal_flag = Arc::clone(&trigger_calibration);
        std::thread::spawn(move || {
            let stdin = std::io::stdin();
            let mut line = String::new();
            loop {
                line.clear();
                if stdin.read_line(&mut line).is_ok() {
                    match line.trim() {
                        "l" => {
                            eprintln!("[input] log send triggered");
                            log_flag.store(true, Ordering::Relaxed);
                        }
                        "p" => {
                            eprintln!("[input] pose calibration triggered");
                            cal_flag.store(true, Ordering::Relaxed);
                        }
                        _ => {}
                    }
                }
            }
        });
    }

    let bind_addr: std::net::SocketAddr = config.listen_addr.parse()
        .context("invalid listen_addr")?;
    let listener = tokio::net::TcpListener::bind(bind_addr).await?;
    log!(logfile, "Listening on {}", bind_addr);
    log!(logfile, "");

    loop {
        let (tcp_stream, addr) = listener.accept().await?;
        tcp_stream.set_nodelay(true)?;
        log!(logfile, "Client connected: {}", addr);

        let (tx, rx) = mpsc::sync_channel::<TcpEvent>(16);
        let (out_tx, out_rx) = tokio::sync::mpsc::channel::<ServerMessage>(4);
        let frame_drop_count = Arc::new(AtomicU32::new(0));
        let frame_drop_count2 = Arc::clone(&frame_drop_count);

        let tcp_task = tokio::spawn(async move {
            if let Err(e) = tcp_receive_loop(tcp_stream, tx, out_rx, frame_drop_count2).await {
                eprintln!("TCP error: {}", e);
            }
        });

        tokio::task::block_in_place(|| {
            run_inference_loop(
                &rx, &config, &mut detector, &mut person_detector, model_type, &vmt,
                &logfile, config.verbose, &log_path, &out_tx, &trigger_log_send, &trigger_calibration,
                &frame_drop_count,
            );
        });

        tcp_task.abort();
        log!(logfile, "Client disconnected, waiting for next connection...");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_z_jump_first_frame_no_reject() {
        let mut prev_z: Option<f32> = None;
        let mut reject_count: u32 = 0;
        // First frame: no previous value, should not reject
        let rejected = check_hip_z_jump(Some(1.0), &mut prev_z, &mut reject_count);
        assert!(!rejected);
        assert_eq!(prev_z, Some(1.0));
        assert_eq!(reject_count, 0);
    }

    #[test]
    fn test_z_jump_small_movement_no_reject() {
        let mut prev_z: Option<f32> = Some(1.0);
        let mut reject_count: u32 = 0;
        // Small movement within threshold
        let rejected = check_hip_z_jump(Some(1.2), &mut prev_z, &mut reject_count);
        assert!(!rejected);
        assert_eq!(prev_z, Some(1.2));
        assert_eq!(reject_count, 0);
    }

    #[test]
    fn test_z_jump_large_movement_rejects() {
        let mut prev_z: Option<f32> = Some(1.0);
        let mut reject_count: u32 = 0;
        // Large movement exceeding MAX_HIP_Z_JUMP (0.3)
        let rejected = check_hip_z_jump(Some(1.5), &mut prev_z, &mut reject_count);
        assert!(rejected);
        assert_eq!(prev_z, Some(1.0)); // prev_z unchanged
        assert_eq!(reject_count, 1);
    }

    #[test]
    fn test_z_jump_streak_reset_after_max_rejects() {
        let mut prev_z: Option<f32> = Some(1.0);
        let mut reject_count: u32 = 0;
        let new_z = 2.0; // Large jump

        // Reject frames 1 through MAX_HIP_Z_REJECT_STREAK-1
        for i in 1..MAX_HIP_Z_REJECT_STREAK {
            let rejected = check_hip_z_jump(Some(new_z), &mut prev_z, &mut reject_count);
            assert!(rejected, "frame {} should be rejected", i);
            assert_eq!(prev_z, Some(1.0), "prev_z should stay at old value during streak");
            assert_eq!(reject_count, i);
        }

        // Frame at MAX_HIP_Z_REJECT_STREAK: should reset and accept
        let rejected = check_hip_z_jump(Some(new_z), &mut prev_z, &mut reject_count);
        assert!(!rejected, "should accept after streak reaches max");
        assert_eq!(prev_z, Some(new_z), "prev_z should be updated to new value");
        assert_eq!(reject_count, 0, "reject count should be reset");
    }

    #[test]
    fn test_z_jump_streak_resets_on_normal_frame() {
        let mut prev_z: Option<f32> = Some(1.0);
        let mut reject_count: u32 = 3; // mid-streak

        // Normal frame within threshold resets the count
        let rejected = check_hip_z_jump(Some(1.1), &mut prev_z, &mut reject_count);
        assert!(!rejected);
        assert_eq!(reject_count, 0);
        assert_eq!(prev_z, Some(1.1));
    }

    #[test]
    fn test_z_jump_none_hip_z_no_reject() {
        let mut prev_z: Option<f32> = Some(1.0);
        let mut reject_count: u32 = 0;
        // No hip detected: should not reject, prev_z unchanged
        let rejected = check_hip_z_jump(None, &mut prev_z, &mut reject_count);
        assert!(!rejected);
        assert_eq!(prev_z, Some(1.0));
        assert_eq!(reject_count, 0);
    }

    #[test]
    fn test_z_jump_recovery_after_reset() {
        let mut prev_z: Option<f32> = Some(1.0);
        let mut reject_count: u32 = 0;
        let new_z = 2.0;

        // Exhaust streak to trigger reset
        for _ in 0..MAX_HIP_Z_REJECT_STREAK {
            check_hip_z_jump(Some(new_z), &mut prev_z, &mut reject_count);
        }
        assert_eq!(prev_z, Some(new_z));

        // Now normal operation resumes from new position
        let rejected = check_hip_z_jump(Some(2.1), &mut prev_z, &mut reject_count);
        assert!(!rejected);
        assert_eq!(prev_z, Some(2.1));
        assert_eq!(reject_count, 0);
    }

    #[test]
    fn test_undistort_identity() {
        let cp = CameraParams::from_calibration(
            &[500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0],
            &[0.0; 5], &[0.0; 3], &[0.0; 3], 640, 480,
        );
        let (u, v) = cp.undistort_point(400.0, 300.0);
        assert!((u - 400.0).abs() < 0.01, "u: expected 400.0, got {}", u);
        assert!((v - 300.0).abs() < 0.01, "v: expected 300.0, got {}", v);
    }

    #[test]
    fn test_triangulate_known_point() {
        let cam1 = CameraParams::from_config(55.0, 640, 480, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]);
        let cam2 = CameraParams::from_config(55.0, 640, 480, [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]);

        let target = nalgebra::Vector4::new(0.3, -0.2, 2.5, 1.0);
        let proj1 = cam1.projection * target;
        let proj2 = cam2.projection * target;
        let uv1 = (proj1[0] / proj1[2], proj1[1] / proj1[2]);
        let uv2 = (proj2[0] / proj2[2], proj2[1] / proj2[2]);

        let (x, y, z) = triangulate_point(&[&cam1, &cam2], &[uv1, uv2]);
        assert!((x - 0.3).abs() < 0.01, "x: expected 0.3, got {}", x);
        assert!((y - (-0.2)).abs() < 0.01, "y: expected -0.2, got {}", y);
        assert!((z - 2.5).abs() < 0.01, "z: expected 2.5, got {}", z);
    }

    #[test]
    fn test_one_euro_filter_constant() {
        let mut filter = ScalarFilter::new(1.0, 0.0, 1.0);
        let dt = 1.0 / 30.0;
        let value = 5.0;
        let mut last = 0.0;
        for _ in 0..100 {
            last = filter.filter(value, dt);
        }
        assert!((last - value).abs() < 0.01, "filter should converge to {}, got {}", value, last);
    }

    #[test]
    fn test_z_jump_reject_and_recover() {
        let mut prev_z: Option<f32> = Some(1.0);
        let mut reject_count: u32 = 0;

        // Large jump should be rejected
        assert!(check_hip_z_jump(Some(2.0), &mut prev_z, &mut reject_count));
        assert_eq!(reject_count, 1);

        // Continue rejecting until streak
        for _ in 1..MAX_HIP_Z_REJECT_STREAK - 1 {
            assert!(check_hip_z_jump(Some(2.0), &mut prev_z, &mut reject_count));
        }
        assert_eq!(reject_count, MAX_HIP_Z_REJECT_STREAK - 1);

        // Streak max → reset and accept
        assert!(!check_hip_z_jump(Some(2.0), &mut prev_z, &mut reject_count));
        assert_eq!(prev_z, Some(2.0));
        assert_eq!(reject_count, 0);

        // Normal movement from new position accepted
        assert!(!check_hip_z_jump(Some(2.1), &mut prev_z, &mut reject_count));
        assert_eq!(prev_z, Some(2.1));
    }

    // ===================================================================
    // Helper: build a Pose with specific keypoints set
    // ===================================================================

    fn make_pose(entries: &[(usize, f32, f32, f32, f32)]) -> Pose {
        let mut pose = Pose::default();
        for &(idx, x, y, z, conf) in entries {
            pose.keypoints[idx] = Keypoint::new_3d(x, y, z, conf);
        }
        pose
    }

    fn standing_pose() -> Pose {
        make_pose(&[
            (KP_LEFT_HIP,       0.45, 0.50, 1.0, 0.9),
            (KP_RIGHT_HIP,      0.55, 0.50, 1.0, 0.9),
            (KP_LEFT_SHOULDER,   0.42, 0.30, 1.0, 0.9),
            (KP_RIGHT_SHOULDER,  0.58, 0.30, 1.0, 0.9),
            (KP_LEFT_KNEE,      0.44, 0.65, 1.0, 0.9),
            (KP_RIGHT_KNEE,     0.56, 0.65, 1.0, 0.9),
            (KP_LEFT_ANKLE,     0.43, 0.80, 1.0, 0.9),
            (KP_RIGHT_ANKLE,    0.57, 0.80, 1.0, 0.9),
        ])
    }

    // ===================================================================
    // BodyTracker::compute() tests
    // ===================================================================

    #[test]
    fn test_body_compute_uncalibrated_all_keypoints() {
        let bt = BodyTracker::new(false, 0.0, 0.0);
        let pose = standing_pose();
        let result = bt.compute(&pose);

        // hip should exist (both hips valid)
        assert!(result.hip.is_some(), "hip should be present");
        // left/right foot should exist (ankles valid)
        assert!(result.left_foot.is_some(), "left_foot should be present");
        assert!(result.right_foot.is_some(), "right_foot should be present");
        // chest should exist (shoulders valid)
        assert!(result.chest.is_some(), "chest should be present");
        // knees should exist
        assert!(result.left_knee.is_some(), "left_knee should be present");
        assert!(result.right_knee.is_some(), "right_knee should be present");
    }

    #[test]
    fn test_body_compute_uncalibrated_positions() {
        let bt = BodyTracker::new(false, 0.0, 0.0);
        let pose = standing_pose();
        let result = bt.compute(&pose);

        let hip = result.hip.unwrap();
        // Uncalibrated: ref=(0.5,0.5), hip_center=(0.5,0.5)
        // convert_position: global=(0.5-0.5, 0.5-0.5)=(0,0), pos_x=0+(0.5-0.5)=0, pos_y=0+0=0
        assert!((hip.position[0]).abs() < 0.01, "hip x={}", hip.position[0]);
        assert!((hip.position[1]).abs() < 0.01, "hip y={}", hip.position[1]);

        // Feet: ankles at y=0.8, hip at y=0.5
        // pos_y = 0 + (0 + (0.5 - 0.8)) = -0.3
        let lf = result.left_foot.unwrap();
        assert!((lf.position[1] - (-0.3)).abs() < 0.01, "left_foot y={}", lf.position[1]);
    }

    #[test]
    fn test_body_compute_no_hips_returns_all_none() {
        let bt = BodyTracker::new(false, 0.0, 0.0);
        // Pose with no hip keypoints
        let pose = make_pose(&[
            (KP_LEFT_SHOULDER,  0.42, 0.30, 0.0, 0.9),
            (KP_RIGHT_SHOULDER, 0.58, 0.30, 0.0, 0.9),
            (KP_LEFT_ANKLE,     0.43, 0.80, 0.0, 0.9),
            (KP_RIGHT_ANKLE,    0.57, 0.80, 0.0, 0.9),
        ]);
        let result = bt.compute(&pose);

        assert!(result.hip.is_none(), "hip should be None without hip keypoints");
        assert!(result.left_foot.is_none(), "left_foot requires hip_center");
        assert!(result.right_foot.is_none(), "right_foot requires hip_center");
        assert!(result.chest.is_none(), "chest requires hip_center");
        assert!(result.left_knee.is_none(), "left_knee requires hip_center");
        assert!(result.right_knee.is_none(), "right_knee requires hip_center");
    }

    #[test]
    fn test_body_compute_missing_ankle_falls_back_to_knee() {
        let bt = BodyTracker::new(false, 0.0, 0.0);
        // Left ankle missing, left knee present
        let pose = make_pose(&[
            (KP_LEFT_HIP,       0.45, 0.50, 1.0, 0.9),
            (KP_RIGHT_HIP,      0.55, 0.50, 1.0, 0.9),
            (KP_LEFT_KNEE,      0.44, 0.65, 1.0, 0.9),
            (KP_RIGHT_KNEE,     0.56, 0.65, 1.0, 0.9),
            (KP_RIGHT_ANKLE,    0.57, 0.80, 1.0, 0.9),
            // KP_LEFT_ANKLE intentionally omitted
        ]);
        let result = bt.compute(&pose);

        // Left foot should fall back to knee position
        assert!(result.left_foot.is_some(), "left_foot should use knee as fallback");
        let lf = result.left_foot.unwrap();
        // knee at x=0.44, hip_center at x=0.5
        // convert_position: ref=(0.5,0.5), global=(0.5-0.5)=0, pos_x=0+(0.5-0.44)=0.06
        assert!((lf.position[0] - 0.06).abs() < 0.01, "left_foot x={}", lf.position[0]);
    }

    #[test]
    fn test_body_compute_missing_ankle_and_knee_returns_none() {
        let bt = BodyTracker::new(false, 0.0, 0.0);
        let pose = make_pose(&[
            (KP_LEFT_HIP,       0.45, 0.50, 1.0, 0.9),
            (KP_RIGHT_HIP,      0.55, 0.50, 1.0, 0.9),
            (KP_LEFT_SHOULDER,   0.42, 0.30, 1.0, 0.9),
            (KP_RIGHT_SHOULDER,  0.58, 0.30, 1.0, 0.9),
            // Left knee and ankle both omitted
            (KP_RIGHT_KNEE,     0.56, 0.65, 1.0, 0.9),
            (KP_RIGHT_ANKLE,    0.57, 0.80, 1.0, 0.9),
        ]);
        let result = bt.compute(&pose);

        assert!(result.left_foot.is_none(), "left_foot should be None without ankle and knee");
        assert!(result.right_foot.is_some(), "right_foot should be present");
    }

    #[test]
    fn test_body_compute_mirror_x() {
        let bt_normal = BodyTracker::new(false, 0.0, 0.0);
        let bt_mirror = BodyTracker::new(true, 0.0, 0.0);
        let pose = standing_pose();

        let normal = bt_normal.compute(&pose);
        let mirror = bt_mirror.compute(&pose);

        let nh = normal.hip.unwrap();
        let mh = mirror.hip.unwrap();
        // mirror_x flips the sign of pos_x
        assert!(
            (nh.position[0] + mh.position[0]).abs() < 0.01,
            "mirror hip x should be negated: normal={}, mirror={}",
            nh.position[0], mh.position[0]
        );
        // y should be identical
        assert!(
            (nh.position[1] - mh.position[1]).abs() < 0.01,
            "mirror hip y should be same: normal={}, mirror={}",
            nh.position[1], mh.position[1]
        );
    }

    #[test]
    fn test_body_compute_offset_y() {
        let bt = BodyTracker::new(false, 0.5, 0.0);
        let pose = standing_pose();
        let result = bt.compute(&pose);

        let hip = result.hip.unwrap();
        // offset_y=0.5 is added to all y positions
        assert!((hip.position[1] - 0.5).abs() < 0.01, "hip y with offset={}", hip.position[1]);
    }

    #[test]
    fn test_body_compute_foot_y_offset() {
        let bt = BodyTracker::new(false, 0.0, 0.1);
        let pose = standing_pose();
        let result = bt.compute(&pose);

        let bt_no_offset = BodyTracker::new(false, 0.0, 0.0);
        let result_no_offset = bt_no_offset.compute(&pose);

        let lf = result.left_foot.unwrap();
        let lf_no = result_no_offset.left_foot.unwrap();
        assert!(
            (lf.position[1] - lf_no.position[1] - 0.1).abs() < 0.01,
            "foot_y_offset should add 0.1: with={}, without={}",
            lf.position[1], lf_no.position[1]
        );
    }

    #[test]
    fn test_body_calibrate_and_compute() {
        let mut bt = BodyTracker::new(false, 0.0, 0.0);
        let calibration_pose = standing_pose();
        assert!(bt.calibrate(&calibration_pose), "calibration should succeed");
        assert!(bt.is_calibrated());

        // Compute with the same pose as calibration: all positions should be ~0
        let result = bt.compute(&calibration_pose);
        let hip = result.hip.unwrap();
        assert!((hip.position[0]).abs() < 0.01, "calibrated hip x={}", hip.position[0]);
        assert!((hip.position[1]).abs() < 0.01, "calibrated hip y={}", hip.position[1]);

        // Yaw should be ~0 (same pose as calibration)
        assert!((hip.rotation[1]).abs() < 0.01, "calibrated hip yaw={}", hip.rotation[1]);
    }

    #[test]
    fn test_body_calibrate_shifts_position() {
        let mut bt = BodyTracker::new(false, 0.0, 0.0);
        let calibration_pose = standing_pose();
        bt.calibrate(&calibration_pose);

        // Shifted pose: person moved right by 0.1
        let shifted = make_pose(&[
            (KP_LEFT_HIP,       0.55, 0.50, 1.0, 0.9),
            (KP_RIGHT_HIP,      0.65, 0.50, 1.0, 0.9),
            (KP_LEFT_SHOULDER,   0.52, 0.30, 1.0, 0.9),
            (KP_RIGHT_SHOULDER,  0.68, 0.30, 1.0, 0.9),
            (KP_LEFT_KNEE,      0.54, 0.65, 1.0, 0.9),
            (KP_RIGHT_KNEE,     0.66, 0.65, 1.0, 0.9),
            (KP_LEFT_ANKLE,     0.53, 0.80, 1.0, 0.9),
            (KP_RIGHT_ANKLE,    0.67, 0.80, 1.0, 0.9),
        ]);
        let result = bt.compute(&shifted);
        let hip = result.hip.unwrap();
        // Person moved right by 0.1 in image → hip_center=(0.6, 0.5)
        // convert_position: ref=(0.5,0.5), global=(0.5-0.6)=(-0.1, 0.5-0.5=0)
        // pos_x = -0.1 + (0.6 - 0.6) = -0.1
        assert!((hip.position[0] - (-0.1)).abs() < 0.02, "shifted hip x={}", hip.position[0]);
    }

    #[test]
    fn test_body_calibrate_fails_without_hips() {
        let mut bt = BodyTracker::new(false, 0.0, 0.0);
        let pose = make_pose(&[
            (KP_LEFT_SHOULDER,  0.42, 0.30, 0.0, 0.9),
            (KP_RIGHT_SHOULDER, 0.58, 0.30, 0.0, 0.9),
        ]);
        assert!(!bt.calibrate(&pose), "calibrate should fail without hips");
        assert!(!bt.is_calibrated());
    }

    #[test]
    fn test_body_compute_depth_from_z() {
        let bt = BodyTracker::new(false, 0.0, 0.0);
        let pose = make_pose(&[
            (KP_LEFT_HIP,   0.45, 0.50, 2.0, 0.9),
            (KP_RIGHT_HIP,  0.55, 0.50, 2.0, 0.9),
            (KP_LEFT_SHOULDER,  0.42, 0.30, 2.0, 0.9),
            (KP_RIGHT_SHOULDER, 0.58, 0.30, 2.0, 0.9),
            (KP_LEFT_ANKLE, 0.43, 0.80, 2.5, 0.9),
            (KP_RIGHT_ANKLE, 0.57, 0.80, 1.5, 0.9),
        ]);
        let result = bt.compute(&pose);

        // Uncalibrated: estimate_depth = ref_z(0) - hip_z(2.0) = -2.0
        let hip = result.hip.unwrap();
        assert!((hip.position[2] - (-2.0)).abs() < 0.01, "hip z={}", hip.position[2]);

        // Left ankle z=2.5 → keypoint_depth = 0 - 2.5 = -2.5
        let lf = result.left_foot.unwrap();
        assert!((lf.position[2] - (-2.5)).abs() < 0.01, "left_foot z={}", lf.position[2]);
    }

    #[test]
    fn test_body_compute_depth_calibrated() {
        let mut bt = BodyTracker::new(false, 0.0, 0.0);
        let cal_pose = make_pose(&[
            (KP_LEFT_HIP,   0.45, 0.50, 2.0, 0.9),
            (KP_RIGHT_HIP,  0.55, 0.50, 2.0, 0.9),
            (KP_LEFT_SHOULDER,  0.42, 0.30, 2.0, 0.9),
            (KP_RIGHT_SHOULDER, 0.58, 0.30, 2.0, 0.9),
            (KP_LEFT_KNEE,  0.44, 0.65, 2.0, 0.9),
            (KP_RIGHT_KNEE, 0.56, 0.65, 2.0, 0.9),
            (KP_LEFT_ANKLE, 0.43, 0.80, 2.0, 0.9),
            (KP_RIGHT_ANKLE, 0.57, 0.80, 2.0, 0.9),
        ]);
        bt.calibrate(&cal_pose);

        // Same pose → z should be 0 (ref_z - current_z = 2.0 - 2.0 = 0)
        let result = bt.compute(&cal_pose);
        let hip = result.hip.unwrap();
        assert!((hip.position[2]).abs() < 0.01, "calibrated hip z={}", hip.position[2]);

        // Person moved closer: z=1.5 → ref_z(2.0) - 1.5 = 0.5
        let closer_pose = make_pose(&[
            (KP_LEFT_HIP,   0.45, 0.50, 1.5, 0.9),
            (KP_RIGHT_HIP,  0.55, 0.50, 1.5, 0.9),
            (KP_LEFT_SHOULDER,  0.42, 0.30, 1.5, 0.9),
            (KP_RIGHT_SHOULDER, 0.58, 0.30, 1.5, 0.9),
            (KP_LEFT_KNEE,  0.44, 0.65, 1.5, 0.9),
            (KP_RIGHT_KNEE, 0.56, 0.65, 1.5, 0.9),
            (KP_LEFT_ANKLE, 0.43, 0.80, 1.5, 0.9),
            (KP_RIGHT_ANKLE, 0.57, 0.80, 1.5, 0.9),
        ]);
        let result = bt.compute(&closer_pose);
        let hip = result.hip.unwrap();
        assert!((hip.position[2] - 0.5).abs() < 0.01, "closer hip z={}", hip.position[2]);
    }

    // ===================================================================
    // reject_duplicate_pair tests
    // ===================================================================

    #[test]
    fn test_reject_duplicate_pair_close_positions() {
        let mut left = Some(TrackerPose::new([0.1, 0.2, 0.0], [0.0, 0.0, 0.0, 1.0]));
        let mut right = Some(TrackerPose::new([0.12, 0.21, 0.0], [0.0, 0.0, 0.0, 1.0]));
        // distance = sqrt(0.02^2 + 0.01^2) ≈ 0.022 < 0.05
        BodyTracker::reject_duplicate_pair(&mut left, &mut right);
        assert!(left.is_none(), "close pair should be rejected");
        assert!(right.is_none(), "close pair should be rejected");
    }

    #[test]
    fn test_reject_duplicate_pair_distant_positions() {
        let mut left = Some(TrackerPose::new([0.1, 0.2, 0.0], [0.0, 0.0, 0.0, 1.0]));
        let mut right = Some(TrackerPose::new([0.3, 0.4, 0.0], [0.0, 0.0, 0.0, 1.0]));
        // distance = sqrt(0.2^2 + 0.2^2) ≈ 0.283 > 0.05
        BodyTracker::reject_duplicate_pair(&mut left, &mut right);
        assert!(left.is_some(), "distant pair should be kept");
        assert!(right.is_some(), "distant pair should be kept");
    }

    #[test]
    fn test_reject_duplicate_pair_one_none() {
        let mut left = Some(TrackerPose::new([0.1, 0.2, 0.0], [0.0, 0.0, 0.0, 1.0]));
        let mut right: Option<TrackerPose> = None;
        BodyTracker::reject_duplicate_pair(&mut left, &mut right);
        assert!(left.is_some(), "should not touch when one is None");
    }

    // ===================================================================
    // convert_position tests
    // ===================================================================

    #[test]
    fn test_convert_position_uncalibrated() {
        let bt = BodyTracker::new(false, 0.0, 0.0);
        // hip at (0.5, 0.5), point at (0.3, 0.7)
        let pos = bt.convert_position(0.3, 0.7, 0.5, 0.5, 0.0);
        // ref=(0.5,0.5), global=(0.5-0.5,0.5-0.5)=(0,0)
        // pos_x = 0 + (0.5-0.3) = 0.2
        // pos_y = 0 + (0 + (0.5-0.7)) = -0.2
        assert!((pos[0] - 0.2).abs() < 0.001, "x={}", pos[0]);
        assert!((pos[1] - (-0.2)).abs() < 0.001, "y={}", pos[1]);
        assert!((pos[2]).abs() < 0.001, "z={}", pos[2]);
    }

    #[test]
    fn test_convert_position_mirror_x() {
        let bt = BodyTracker::new(true, 0.0, 0.0);
        let pos = bt.convert_position(0.3, 0.7, 0.5, 0.5, 0.0);
        // pos_x before mirror = 0.2, after mirror = -0.2
        assert!((pos[0] - (-0.2)).abs() < 0.001, "mirror x={}", pos[0]);
    }

    // ===================================================================
    // sanitize_pose tests
    // ===================================================================

    #[test]
    fn test_sanitize_pose_boundary_x_low() {
        let pose = make_pose(&[
            (KP_LEFT_ANKLE, 0.01, 0.50, 0.0, 0.9),  // x < 0.02 → zeroed
            (KP_RIGHT_ANKLE, 0.50, 0.50, 0.0, 0.9),  // normal
        ]);
        let s = sanitize_pose(&pose);
        assert_eq!(s.keypoints[KP_LEFT_ANKLE].confidence, 0.0, "left ankle should be zeroed");
        assert_eq!(s.keypoints[KP_RIGHT_ANKLE].confidence, 0.9, "right ankle should be kept");
    }

    #[test]
    fn test_sanitize_pose_boundary_x_high() {
        let pose = make_pose(&[
            (KP_LEFT_SHOULDER, 0.99, 0.50, 0.0, 0.9),  // x > 0.98 → zeroed
        ]);
        let s = sanitize_pose(&pose);
        assert_eq!(s.keypoints[KP_LEFT_SHOULDER].confidence, 0.0);
    }

    #[test]
    fn test_sanitize_pose_boundary_y_low() {
        let pose = make_pose(&[
            (KP_LEFT_KNEE, 0.50, 0.01, 0.0, 0.9),  // y < 0.02 → zeroed (full boundary)
        ]);
        let s = sanitize_pose(&pose);
        assert_eq!(s.keypoints[KP_LEFT_KNEE].confidence, 0.0);
    }

    #[test]
    fn test_sanitize_pose_boundary_y_high() {
        let pose = make_pose(&[
            (KP_RIGHT_KNEE, 0.50, 0.99, 0.0, 0.9),  // y > 0.98 → zeroed (full boundary)
        ]);
        let s = sanitize_pose(&pose);
        assert_eq!(s.keypoints[KP_RIGHT_KNEE].confidence, 0.0);
    }

    #[test]
    fn test_sanitize_pose_hip_x_only_boundary() {
        // Hips use X-only boundary: Y edges should NOT zero confidence
        let pose = make_pose(&[
            (KP_LEFT_HIP, 0.50, 0.01, 0.0, 0.9),   // y=0.01 but hip is X-only → kept
            (KP_RIGHT_HIP, 0.01, 0.50, 0.0, 0.9),   // x=0.01 → zeroed
        ]);
        let s = sanitize_pose(&pose);
        assert_eq!(s.keypoints[KP_LEFT_HIP].confidence, 0.9, "hip y-edge should be kept");
        assert_eq!(s.keypoints[KP_RIGHT_HIP].confidence, 0.0, "hip x-edge should be zeroed");
    }

    #[test]
    fn test_sanitize_pose_inside_boundary_kept() {
        let pose = make_pose(&[
            (KP_LEFT_ANKLE,  0.10, 0.50, 0.0, 0.9),
            (KP_RIGHT_ANKLE, 0.90, 0.50, 0.0, 0.9),
            (KP_LEFT_HIP,   0.30, 0.50, 0.0, 0.9),
            (KP_RIGHT_HIP,  0.70, 0.50, 0.0, 0.9),
        ]);
        let s = sanitize_pose(&pose);
        assert_eq!(s.keypoints[KP_LEFT_ANKLE].confidence, 0.9);
        assert_eq!(s.keypoints[KP_RIGHT_ANKLE].confidence, 0.9);
        assert_eq!(s.keypoints[KP_LEFT_HIP].confidence, 0.9);
        assert_eq!(s.keypoints[KP_RIGHT_HIP].confidence, 0.9);
    }

    // ===================================================================
    // yaw_to_quaternion / yaw_pitch_to_quaternion tests
    // ===================================================================

    #[test]
    fn test_yaw_to_quaternion_zero() {
        let q = BodyTracker::yaw_to_quaternion(0.0);
        // [0, sin(0), 0, cos(0)] = [0, 0, 0, 1]
        assert!((q[0]).abs() < 0.001);
        assert!((q[1]).abs() < 0.001);
        assert!((q[2]).abs() < 0.001);
        assert!((q[3] - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_yaw_to_quaternion_90deg() {
        let q = BodyTracker::yaw_to_quaternion(std::f32::consts::FRAC_PI_2);
        // [0, sin(pi/4), 0, cos(pi/4)] ≈ [0, 0.707, 0, 0.707]
        assert!((q[1] - 0.707).abs() < 0.01, "q[1]={}", q[1]);
        assert!((q[3] - 0.707).abs() < 0.01, "q[3]={}", q[3]);
    }

    #[test]
    fn test_yaw_pitch_to_quaternion_zero() {
        let q = BodyTracker::yaw_pitch_to_quaternion(0.0, 0.0);
        assert!((q[3] - 1.0).abs() < 0.001, "identity quaternion w={}", q[3]);
    }

    // ===================================================================
    // compute_tilt_angle / rotate_pose_yz tests
    // ===================================================================

    #[test]
    fn test_compute_tilt_angle_flat() {
        // Hips and ankles at same z → tilt ≈ 0
        let pose = make_pose(&[
            (KP_LEFT_HIP,   0.45, 0.50, 1.0, 0.9),
            (KP_RIGHT_HIP,  0.55, 0.50, 1.0, 0.9),
            (KP_LEFT_ANKLE,  0.43, 0.80, 1.0, 0.9),
            (KP_RIGHT_ANKLE, 0.57, 0.80, 1.0, 0.9),
        ]);
        let tilt = BodyTracker::compute_tilt_angle(&pose);
        assert!(tilt.abs() < 0.01, "flat tilt={}", tilt);
    }

    #[test]
    fn test_compute_tilt_angle_tilted() {
        // Ankles further in z than hips → nonzero tilt
        let pose = make_pose(&[
            (KP_LEFT_HIP,   0.45, 0.50, 1.0, 0.9),
            (KP_RIGHT_HIP,  0.55, 0.50, 1.0, 0.9),
            (KP_LEFT_ANKLE,  0.43, 0.80, 2.0, 0.9),
            (KP_RIGHT_ANKLE, 0.57, 0.80, 2.0, 0.9),
        ]);
        let tilt = BodyTracker::compute_tilt_angle(&pose);
        // atan2(2.0 - 1.0, 0.8 - 0.5) = atan2(1.0, 0.3) ≈ 1.28
        assert!(tilt > 0.5, "tilted tilt={}", tilt);
    }
}
