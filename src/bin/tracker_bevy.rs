use anyhow::Result;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Mutex};
use std::sync::Arc;
use std::time::{Duration, Instant};

use bevy::app::{App, ScheduleRunnerPlugin, Update};
use bevy::ecs::prelude::*;
use opencv::core::Mat;

use talava_tracker::calibration::load_calibration;
use talava_tracker::camera::ThreadedCamera;
use talava_tracker::config::Config;
use talava_tracker::pose::{
    bbox_from_keypoints, crop_for_pose, preprocess_for_movenet, preprocess_for_spinepose,
    preprocess_for_rtmw3d, remap_pose, CropRegion, ModelType, PersonDetector, PoseDetector, Pose,
};
use talava_tracker::tracker::{BodyTracker, Extrapolator, Lerper, PoseFilter};
use talava_tracker::triangulation::{triangulate_poses, CameraParams};
use talava_tracker::vmt::{TrackerPose, VmtClient};

const CONFIG_PATH: &str = "config.toml";
const TRACKER_COUNT: usize = 6;
const TRACKER_INDICES: [i32; TRACKER_COUNT] = [0, 1, 2, 3, 4, 5];
const CONFIDENCE_THRESHOLD: f32 = 0.3;

// --- Inference thread types ---

struct InferenceRequest {
    camera_index: usize,
    frame: Mat,
    prev_pose: Option<Pose>,
    width: u32,
    height: u32,
}

struct InferenceResult {
    camera_index: usize,
    pose: Pose,
}

// --- Bevy Resources ---

#[derive(Resource)]
struct CameraInputs {
    cameras: Vec<ThreadedCamera>,
    params: Vec<CameraParams>,
    last_frame_ids: Vec<u64>,
    widths: Vec<u32>,
    heights: Vec<u32>,
}

#[derive(Resource)]
struct InferenceTx(mpsc::SyncSender<InferenceRequest>);

#[derive(Resource)]
struct InferenceRx(Mutex<mpsc::Receiver<InferenceResult>>);

#[derive(Resource)]
struct PoseState {
    poses_2d: Vec<Option<Pose>>,
    prev_poses: Vec<Option<Pose>>,
    pose_3d: Option<Pose>,
    multi_camera: bool,
}

#[derive(Resource)]
struct TrackerState {
    body_tracker: BodyTracker,
    filters: [PoseFilter; TRACKER_COUNT],
    extrapolators: [Extrapolator; TRACKER_COUNT],
    lerpers: [Lerper; TRACKER_COUNT],
    last_poses: [Option<TrackerPose>; TRACKER_COUNT],
    last_update_times: [Instant; TRACKER_COUNT],
    last_inference_time: Instant,
    interp_mode: String,
}

#[derive(Resource)]
struct VmtSender(VmtClient);

#[derive(Resource)]
struct CalibrationState {
    deadline: Option<Instant>,
    console_flag: Arc<AtomicBool>,
}

#[derive(Resource)]
struct FpsCounter {
    frame_count: u32,
    inference_count: u32,
    timer: Instant,
}

fn main() -> Result<()> {
    let config = Config::load(CONFIG_PATH)?;

    println!("Tracker Bevy - Multi-camera triangulation");
    println!("VMT target: {}", config.vmt.addr);
    println!("Target FPS: {}", config.app.target_fps);
    println!("Interpolation: {}", config.interpolation.mode);

    // カメラ起動: calibration.jsonがあればそちらから、なければconfig.tomlの設定を使用
    let mut cameras = Vec::new();
    let mut params = Vec::new();
    let mut widths = Vec::new();
    let mut heights = Vec::new();

    if let Some(ref cal_path) = config.calibration_file {
        // キャリブレーションファイルからカメラ起動
        match load_calibration(cal_path) {
            Ok(cal) => {
                println!("Calibration: {}", cal_path);
                println!("Cameras: {}", cal.cameras.len());
                for cam_cal in &cal.cameras {
                    let cam = ThreadedCamera::start(
                        cam_cal.camera_index,
                        Some(cam_cal.width),
                        Some(cam_cal.height),
                    )?;
                    let (w, h) = cam.resolution();
                    println!("Camera {}: {}x{}", cam_cal.camera_index, w, h);
                    params.push(CameraParams::from_calibration(
                        &cam_cal.intrinsic_matrix,
                        &cam_cal.rvec,
                        &cam_cal.tvec,
                    ));
                    widths.push(w);
                    heights.push(h);
                    cameras.push(cam);
                }
            }
            Err(e) => {
                eprintln!("Calibration load failed: {}. Falling back to config.", e);
                // フォールバック: config.toml
                let entries = config.camera_entries();
                for entry in &entries {
                    let cam = ThreadedCamera::start(
                        entry.index,
                        Some(entry.width),
                        Some(entry.height),
                    )?;
                    let (w, h) = cam.resolution();
                    println!("Camera {}: {}x{}", entry.index, w, h);
                    if entries.len() >= 2 {
                        params.push(CameraParams::from_config(
                            entry.fov_v, w, h, entry.position, entry.rotation,
                        ));
                    }
                    widths.push(w);
                    heights.push(h);
                    cameras.push(cam);
                }
            }
        }
    } else {
        // config.tomlのカメラ設定を使用
        let entries = config.camera_entries();
        println!("Cameras: {}", entries.len());
        for entry in &entries {
            let cam = ThreadedCamera::start(
                entry.index,
                Some(entry.width),
                Some(entry.height),
            )?;
            let (w, h) = cam.resolution();
            println!("Camera {}: {}x{}", entry.index, w, h);
            if entries.len() >= 2 {
                params.push(CameraParams::from_config(
                    entry.fov_v, w, h, entry.position, entry.rotation,
                ));
            }
            widths.push(w);
            heights.push(h);
            cameras.push(cam);
        }
    }

    let num_cameras = cameras.len();
    let multi_camera = num_cameras >= 2;
    println!("Mode: {}", if multi_camera { "triangulation" } else { "single camera" });
    println!();

    // 推論スレッド
    let (frame_tx, frame_rx) = mpsc::sync_channel::<InferenceRequest>(num_cameras);
    let (result_tx, result_rx) = mpsc::channel::<InferenceResult>();

    let (model_path, model_type): (&str, ModelType) = match config.app.model.as_str() {
        "movenet" => ("models/movenet_lightning.onnx", ModelType::MoveNet),
        "spinepose_small" => ("models/spinepose_small.onnx", ModelType::SpinePose),
        "spinepose_medium" => ("models/spinepose_medium.onnx", ModelType::SpinePose),
        "rtmw3d" => ("models/rtmw3d-x.onnx", ModelType::RTMW3D),
        other => { eprintln!("Unknown model: {}", other); std::process::exit(1); }
    };
    let detector_mode = config.app.detector.clone();
    println!("Model: {}", config.app.model);
    println!("Detector: {}", config.app.detector);

    std::thread::spawn(move || {
        let mut detector = PoseDetector::new(model_path, model_type)
            .expect("Failed to load pose model");
        println!("Inference thread: model loaded");

        let mut person_detector = match detector_mode.as_str() {
            "yolo" => {
                let pd = PersonDetector::new("models/yolov8n.onnx", 160)
                    .expect("Failed to load person detector");
                println!("Inference thread: person detector loaded");
                Some(pd)
            }
            _ => None,
        };

        while let Ok(req) = frame_rx.recv() {
            // 人物検出 → クロップ
            let (input_frame, crop_region) = match detector_mode.as_str() {
                "keypoint" => {
                    match req.prev_pose.as_ref()
                        .and_then(|p| bbox_from_keypoints(p, req.width, req.height, CONFIDENCE_THRESHOLD))
                    {
                        Some(bbox) => match crop_for_pose(&req.frame, &bbox, req.width, req.height) {
                            Ok(v) => v,
                            Err(_) => (req.frame.clone(), CropRegion::full()),
                        },
                        None => (req.frame.clone(), CropRegion::full()),
                    }
                }
                "yolo" => {
                    match person_detector.as_mut().unwrap().detect(&req.frame) {
                        Ok(Some(bbox)) => match crop_for_pose(&req.frame, &bbox, req.width, req.height) {
                            Ok(v) => v,
                            Err(_) => (req.frame.clone(), CropRegion::full()),
                        },
                        _ => (req.frame.clone(), CropRegion::full()),
                    }
                }
                _ => (req.frame.clone(), CropRegion::full()),
            };

            // 前処理
            let input = match model_type {
                ModelType::MoveNet => preprocess_for_movenet(&input_frame),
                ModelType::SpinePose => preprocess_for_spinepose(&input_frame),
                ModelType::RTMW3D => preprocess_for_rtmw3d(&input_frame),
            };
            let input = match input {
                Ok(v) => v,
                Err(e) => { eprintln!("preprocess error: {}", e); continue; }
            };

            // 推論
            let mut pose = match detector.detect(input) {
                Ok(v) => v,
                Err(e) => { eprintln!("inference error: {}", e); continue; }
            };

            if !crop_region.is_full() {
                pose = remap_pose(&pose, &crop_region);
            }

            let _ = result_tx.send(InferenceResult {
                camera_index: req.camera_index,
                pose,
            });
        }
    });

    // Body tracker
    let fov_v = if multi_camera { 0.0 } else { config.camera.fov_v };
    let body_tracker = BodyTracker::new(&config.tracker, fov_v);
    let filters: [PoseFilter; TRACKER_COUNT] =
        std::array::from_fn(|_| PoseFilter::from_config(&config.filter));
    let extrapolators: [Extrapolator; TRACKER_COUNT] =
        std::array::from_fn(|_| Extrapolator::new());
    let lerpers: [Lerper; TRACKER_COUNT] =
        std::array::from_fn(|_| Lerper::new());

    let vmt = VmtClient::new(&config.vmt.addr)?;
    println!("VMT client ready");

    // コンソール入力
    let console_flag = Arc::new(AtomicBool::new(false));
    {
        let flag = console_flag.clone();
        std::thread::spawn(move || {
            let stdin = std::io::stdin();
            let mut line = String::new();
            loop {
                line.clear();
                if stdin.read_line(&mut line).is_ok() && line.trim().eq_ignore_ascii_case("c") {
                    flag.store(true, Ordering::Release);
                }
            }
        });
    }

    println!();
    println!("操作: [C + Enter] キャリブレーション  [Ctrl+C] 終了");
    println!();

    let frame_duration = Duration::from_secs_f64(1.0 / config.app.target_fps as f64);

    App::new()
        .add_plugins(ScheduleRunnerPlugin::run_loop(frame_duration))
        .insert_resource(CameraInputs {
            cameras,
            params,
            last_frame_ids: vec![0; num_cameras],
            widths,
            heights,
        })
        .insert_resource(InferenceTx(frame_tx))
        .insert_resource(InferenceRx(Mutex::new(result_rx)))
        .insert_resource(PoseState {
            poses_2d: vec![None; num_cameras],
            prev_poses: vec![None; num_cameras],
            pose_3d: None,
            multi_camera,
        })
        .insert_resource(TrackerState {
            body_tracker,
            filters,
            extrapolators,
            lerpers,
            last_poses: [None; TRACKER_COUNT],
            last_update_times: [Instant::now(); TRACKER_COUNT],
            last_inference_time: Instant::now(),
            interp_mode: config.interpolation.mode.clone(),
        })
        .insert_resource(VmtSender(vmt))
        .insert_resource(CalibrationState {
            deadline: None,
            console_flag,
        })
        .insert_resource(FpsCounter {
            frame_count: 0,
            inference_count: 0,
            timer: Instant::now(),
        })
        .add_systems(Update, (
            (send_frames_system, receive_results_system, triangulate_system).chain(),
            (calibration_system, compute_trackers_system, send_vmt_system, fps_system).chain(),
        ).chain())
        .run();

    println!("Shutting down...");
    Ok(())
}

// --- Systems ---

fn send_frames_system(
    mut cam_inputs: ResMut<CameraInputs>,
    tx: Res<InferenceTx>,
    pose_state: Res<PoseState>,
) {
    for i in 0..cam_inputs.cameras.len() {
        let fid = cam_inputs.cameras[i].frame_id();
        if fid != cam_inputs.last_frame_ids[i] {
            if let Some(frame) = cam_inputs.cameras[i].get_frame() {
                let req = InferenceRequest {
                    camera_index: i,
                    frame,
                    prev_pose: pose_state.prev_poses[i].clone(),
                    width: cam_inputs.widths[i],
                    height: cam_inputs.heights[i],
                };
                match tx.0.try_send(req) {
                    Ok(()) => { cam_inputs.last_frame_ids[i] = fid; }
                    Err(mpsc::TrySendError::Full(_)) => {}
                    Err(mpsc::TrySendError::Disconnected(_)) => {
                        eprintln!("Inference thread disconnected");
                    }
                }
            }
        }
    }
}

fn receive_results_system(
    mut pose_state: ResMut<PoseState>,
    rx: Res<InferenceRx>,
    mut fps: ResMut<FpsCounter>,
) {
    let rx = rx.0.lock().unwrap();
    while let Ok(result) = rx.try_recv() {
        let idx = result.camera_index;
        if idx < pose_state.poses_2d.len() {
            pose_state.prev_poses[idx] = Some(result.pose.clone());
            pose_state.poses_2d[idx] = Some(result.pose);
            fps.inference_count += 1;
        }
    }
}

fn triangulate_system(
    cam_inputs: Res<CameraInputs>,
    mut pose_state: ResMut<PoseState>,
) {
    if !pose_state.multi_camera {
        // 単眼モード: 最初のカメラのPoseをそのまま使用
        // clone()で毎フレーム供給することでOne Euroフィルタが高頻度(75Hz)で
        // 動作し、安定した平滑化を実現する
        pose_state.pose_3d = pose_state.poses_2d[0].clone();
        return;
    }

    // 複数カメラ: 全カメラのPoseが揃っているか確認
    let all_ready = pose_state.poses_2d.iter().all(|p| p.is_some());
    if !all_ready {
        return;
    }

    let poses: Vec<&Pose> = pose_state.poses_2d.iter()
        .filter_map(|p| p.as_ref())
        .collect();
    let cam_params: Vec<&CameraParams> = cam_inputs.params.iter().collect();

    let pose_3d = triangulate_poses(&cam_params, &poses, CONFIDENCE_THRESHOLD);
    pose_state.pose_3d = Some(pose_3d);

    // 消費: 次の三角測量まで待機
    for p in pose_state.poses_2d.iter_mut() {
        *p = None;
    }
}

fn calibration_system(
    mut cal: ResMut<CalibrationState>,
    mut tracker_state: ResMut<TrackerState>,
    pose_state: Res<PoseState>,
) {
    if cal.console_flag.swap(false, Ordering::AcqRel) && cal.deadline.is_none() {
        cal.deadline = Some(Instant::now() + Duration::from_secs(5));
        println!("Calibration in 5s... 基準位置に立ってください");
    }

    if let Some(deadline) = cal.deadline {
        if deadline.saturating_duration_since(Instant::now()).is_zero() {
            if let Some(ref pose) = pose_state.pose_3d {
                if tracker_state.body_tracker.calibrate(pose) {
                    tracker_state.filters = std::array::from_fn(|_| {
                        PoseFilter::new(1.5, 0.01, 1.0, 0.01)
                    });
                    tracker_state.extrapolators = std::array::from_fn(|_| Extrapolator::new());
                    tracker_state.lerpers = std::array::from_fn(|_| Lerper::new());
                    tracker_state.last_poses = [None; TRACKER_COUNT];
                    println!("Calibrated!");
                } else {
                    println!("Calibration failed: hip not detected");
                }
            }
            cal.deadline = None;
        }
    }
}

fn compute_trackers_system(
    pose_state: Res<PoseState>,
    mut tracker_state: ResMut<TrackerState>,
) {
    let pose = match &pose_state.pose_3d {
        Some(p) => p,
        None => return,
    };

    let body_poses = tracker_state.body_tracker.compute(pose);
    let poses = [
        body_poses.hip, body_poses.left_foot, body_poses.right_foot,
        body_poses.chest, body_poses.left_knee, body_poses.right_knee,
    ];

    let dt = tracker_state.last_inference_time.elapsed().as_secs_f32();
    let lerp_t = (dt / (1.0 / 30.0)).min(1.0);

    let now = Instant::now();
    // 1フレームあたりの最大移動距離（VR空間単位）
    // 人間の最大移動速度 ~5m/s × scale ~3.4 ÷ 25Hz ≈ 0.68 → 余裕を持って1.0
    const MAX_DISPLACEMENT: f32 = 1.0;
    // 四肢からhipまでの最大距離（VR空間単位）
    // 安定期の実測値: 足0.7, 膝0.5-0.8, 胸0.4 → 余裕を持って1.5
    const MAX_LIMB_DIST: f32 = 1.5;

    for i in 0..TRACKER_COUNT {
        let mut accepted = false;
        if let Some(p) = poses[i] {
            // 速度ベースの外れ値除去: 前フレームから離れすぎた検出はスキップ
            let velocity_ok = match tracker_state.last_poses[i].as_ref() {
                Some(last) => {
                    let dx = p.position[0] - last.position[0];
                    let dy = p.position[1] - last.position[1];
                    let dz = p.position[2] - last.position[2];
                    (dx * dx + dy * dy + dz * dz).sqrt() <= MAX_DISPLACEMENT
                }
                None => true,
            };
            // 四肢のhipからの距離チェック: 解剖学的にありえない位置を除去
            let limb_ok = if i > 0 {
                match tracker_state.last_poses[0].as_ref() {
                    Some(hip) => {
                        let dx = p.position[0] - hip.position[0];
                        let dy = p.position[1] - hip.position[1];
                        (dx * dx + dy * dy).sqrt() <= MAX_LIMB_DIST
                    }
                    None => true,
                }
            } else {
                true
            };
            if velocity_ok && limb_ok {
                let smoothed = tracker_state.filters[i].apply(p);
                tracker_state.extrapolators[i].update(smoothed);
                tracker_state.lerpers[i].update(smoothed, lerp_t);
                tracker_state.last_poses[i] = Some(smoothed);
                tracker_state.last_update_times[i] = now;
                accepted = true;
            }
        }
        // 0.3秒以上更新なし → ステールデータをクリア
        // velocity/limb checkで拒否された場合もタイムアウトが発動し、
        // 次の検出で参照なしからリスタートできる
        if !accepted
            && tracker_state.last_poses[i].is_some()
            && now.duration_since(tracker_state.last_update_times[i]).as_secs_f32() > 0.3
        {
            tracker_state.last_poses[i] = None;
            tracker_state.filters[i].reset();
            tracker_state.extrapolators[i] = Extrapolator::new();
            tracker_state.lerpers[i] = Lerper::new();
        }
    }

    tracker_state.last_inference_time = now;
}

fn send_vmt_system(
    tracker_state: Res<TrackerState>,
    vmt: Res<VmtSender>,
    pose_state: Res<PoseState>,
) {
    if pose_state.pose_3d.is_some() {
        for i in 0..TRACKER_COUNT {
            if let Some(ref p) = tracker_state.last_poses[i] {
                let _ = vmt.0.send(TRACKER_INDICES[i], 1, p);
            }
        }
    } else {
        let dt = tracker_state.last_inference_time.elapsed().as_secs_f32();
        match tracker_state.interp_mode.as_str() {
            "extrapolate" => {
                for i in 0..TRACKER_COUNT {
                    if let Some(predicted) = tracker_state.extrapolators[i].predict(dt) {
                        let _ = vmt.0.send(TRACKER_INDICES[i], 1, &predicted);
                    }
                }
            }
            "lerp" => {
                let t = (dt / (1.0 / 30.0)).min(1.0);
                for i in 0..TRACKER_COUNT {
                    if let Some(interpolated) = tracker_state.lerpers[i].interpolate(t) {
                        let _ = vmt.0.send(TRACKER_INDICES[i], 1, &interpolated);
                    }
                }
            }
            _ => {
                for i in 0..TRACKER_COUNT {
                    if let Some(ref p) = tracker_state.last_poses[i] {
                        let _ = vmt.0.send(TRACKER_INDICES[i], 1, p);
                    }
                }
            }
        }
    }
}

fn fps_system(mut fps: ResMut<FpsCounter>, tracker_state: Res<TrackerState>) {
    fps.frame_count += 1;
    let elapsed = fps.timer.elapsed().as_secs_f32();
    if elapsed >= 1.0 {
        let names = ["hip", "L_foot", "R_foot", "chest", "L_knee", "R_knee"];
        let mut parts = Vec::new();
        for i in 0..TRACKER_COUNT {
            if let Some(ref p) = tracker_state.last_poses[i] {
                parts.push(format!(
                    "{}({:.2},{:.2},{:.2})",
                    names[i], p.position[0], p.position[1], p.position[2]
                ));
            }
        }
        println!(
            "FPS: {:.1} (infer: {}) | {}",
            fps.frame_count as f32 / elapsed,
            fps.inference_count,
            if parts.is_empty() { "no pose".to_string() } else { parts.join(" ") },
        );
        fps.frame_count = 0;
        fps.inference_count = 0;
        fps.timer = Instant::now();
    }
}
