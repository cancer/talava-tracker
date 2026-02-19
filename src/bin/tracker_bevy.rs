use anyhow::Result;
use std::fs;
use std::io::Write;
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
    preprocess_for_rtmw3d, remap_pose, unletterbox_pose, CropRegion, LetterboxInfo, ModelType,
    PersonDetector, PoseDetector, Pose,
};
use talava_tracker::render::MinifbRenderer;
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
    timestamp: Instant,
}

struct InferenceResult {
    camera_index: usize,
    pose: Pose,
    timestamp: Instant,
}

// --- Bevy Resources ---

#[derive(Resource)]
struct CameraInputs {
    cameras: Vec<ThreadedCamera>,
    params: Vec<CameraParams>,
    last_frame_ids: Vec<u64>,
    widths: Vec<u32>,
    heights: Vec<u32>,
    in_flight: Vec<bool>,
}

#[derive(Resource)]
struct InferenceTx(mpsc::SyncSender<InferenceRequest>);

#[derive(Resource)]
struct InferenceRx(Mutex<mpsc::Receiver<InferenceResult>>);

#[derive(Resource)]
struct PoseState {
    poses_2d: Vec<Option<Pose>>,
    prev_poses: Vec<Option<Pose>>,
    pose_timestamps: Vec<Option<Instant>>,
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
    stale: [bool; TRACKER_COUNT],
    reject_count: [u32; TRACKER_COUNT], // velocity_ok連続拒否回数
    last_inference_time: Instant,
    interp_mode: String,
}

#[derive(Resource)]
struct VmtSender(VmtClient);

#[derive(Resource)]
struct CalibrationState {
    deadline: Option<Instant>,
    console_flag: Arc<AtomicBool>,
    /// 起動後の自動キャリブレーション: hip検出開始からの経過時間で発動
    auto_cal_triggered: bool,
    first_hip_time: Option<Instant>,
    last_hip_time: Option<Instant>,
    /// 自動キャリブレーション用: 直近のhip z値を蓄積し安定性を判定
    recent_hip_z: Vec<f32>,
}

#[derive(Resource)]
struct FpsCounter {
    frame_count: u32,
    inference_count: u32,
    timer: Instant,
}

#[derive(Resource)]
struct LogFileRes(LogFile);

struct DebugView {
    renderer: MinifbRenderer,
    view_w: usize,
    view_h: usize,
    num_cameras: usize,
}

type LogFile = Arc<Mutex<std::io::BufWriter<std::fs::File>>>;

fn open_log_file() -> Result<LogFile> {
    fs::create_dir_all("logs")?;
    let ts = chrono::Local::now().format("%Y%m%d_%H%M%S");
    let path = format!("logs/tracker_{}.log", ts);
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

fn main() -> Result<()> {
    // macOSスリープ防止: プロセス終了時に自動解除
    let _caffeinate = std::process::Command::new("caffeinate")
        .args(["-i", "-w", &std::process::id().to_string()])
        .spawn()
        .ok();

    let config = Config::load(CONFIG_PATH)?;
    let logfile = open_log_file()?;

    log!(logfile, "Tracker Bevy - Multi-camera triangulation");
    log!(logfile, "VMT target: {}", config.vmt.addr);
    log!(logfile, "Target FPS: {}", config.app.target_fps);
    log!(logfile, "Interpolation: {}", config.interpolation.mode);

    // カメラ起動: calibration.jsonがあればそちらから、なければconfig.tomlの設定を使用
    let mut cameras = Vec::new();
    let mut params = Vec::new();
    let mut widths = Vec::new();
    let mut heights = Vec::new();

    if let Some(ref cal_path) = config.calibration_file {
        // キャリブレーションファイルからカメラ起動
        match load_calibration(cal_path) {
            Ok(cal) => {
                log!(logfile, "Calibration: {}", cal_path);
                log!(logfile, "Cameras: {}", cal.cameras.len());
                for cam_cal in &cal.cameras {
                    let cam = ThreadedCamera::start(
                        cam_cal.camera_index,
                        Some(cam_cal.width),
                        Some(cam_cal.height),
                    )?;
                    let (w, h) = cam.resolution();
                    log!(logfile, "Camera {}: {}x{}", cam_cal.camera_index, w, h);
                    // 歪み係数は無効化: キャリブレーションの歪み係数が異常に大きい場合
                    // (k2=-100等)、undistort_point()が収束せずリプロジェクション比較が破綻する。
                    // 歪み補正なしでも内部/外部パラメータによる三角測量は機能する。
                    let dc: [f64; 5] = [0.0; 5];
                    params.push(CameraParams::from_calibration(
                        &cam_cal.intrinsic_matrix,
                        &dc,
                        &cam_cal.rvec,
                        &cam_cal.tvec,
                        cam_cal.width,
                        cam_cal.height,
                    ));
                    // 実際の解像度がキャリブレーション解像度と異なる場合、
                    // 内部パラメータをリスケール
                    if w != cam_cal.width || h != cam_cal.height {
                        let last = params.last_mut().unwrap();
                        if last.rescale_to(w, h) {
                            log!(logfile, "  -> Rescaled intrinsics: {}x{} -> {}x{}", cam_cal.width, cam_cal.height, w, h);
                        } else {
                            log!(logfile, "  WARNING: Aspect ratio mismatch! cal={}x{} actual={}x{}. Triangulation may be inaccurate.",
                                cam_cal.width, cam_cal.height, w, h);
                        }
                    }
                    widths.push(w);
                    heights.push(h);
                    cameras.push(cam);
                }
            }
            Err(e) => {
                log!(logfile, "Calibration load failed: {}. Falling back to config.", e);
                // フォールバック: config.toml
                let entries = config.camera_entries();
                for entry in &entries {
                    let cam = ThreadedCamera::start(
                        entry.index,
                        Some(entry.width),
                        Some(entry.height),
                    )?;
                    let (w, h) = cam.resolution();
                    log!(logfile, "Camera {}: {}x{}", entry.index, w, h);
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
        log!(logfile, "Cameras: {}", entries.len());
        for entry in &entries {
            let cam = ThreadedCamera::start(
                entry.index,
                Some(entry.width),
                Some(entry.height),
            )?;
            let (w, h) = cam.resolution();
            log!(logfile, "Camera {}: {}x{}", entry.index, w, h);
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
    log!(logfile, "Mode: {}", if multi_camera { "triangulation" } else { "single camera" });
    log!(logfile, "");

    // 推論スレッド
    let (frame_tx, frame_rx) = mpsc::sync_channel::<InferenceRequest>(num_cameras);
    let (result_tx, result_rx) = mpsc::channel::<InferenceResult>();

    let (model_path, model_type): (&str, ModelType) = match config.app.model.as_str() {
        "movenet" => ("models/movenet_lightning.onnx", ModelType::MoveNet),
        "spinepose_small" => ("models/spinepose_small.onnx", ModelType::SpinePose),
        "spinepose_medium" => ("models/spinepose_medium.onnx", ModelType::SpinePose),
        "rtmw3d" => ("models/rtmw3d-x.onnx", ModelType::RTMW3D),
        other => { log!(logfile, "Unknown model: {}", other); std::process::exit(1); }
    };
    let detector_mode = config.app.detector.clone();
    log!(logfile, "Model: {}", config.app.model);
    log!(logfile, "Detector: {}", config.app.detector);

    let logfile_infer = logfile.clone();
    std::thread::spawn(move || {
        let mut detector = PoseDetector::new(model_path, model_type)
            .expect("Failed to load pose model");
        log!(logfile_infer, "Inference thread: model loaded");

        let mut person_detector = match detector_mode.as_str() {
            "yolo" => {
                let pd = PersonDetector::new("models/yolov8n.onnx", 160)
                    .expect("Failed to load person detector");
                log!(logfile_infer, "Inference thread: person detector loaded");
                Some(pd)
            }
            _ => None,
        };

        while let Ok(req) = frame_rx.recv() {
            // キューに溜まったリクエストをドレインし、カメラ毎に最新のみ保持
            let mut latest: std::collections::HashMap<usize, InferenceRequest> = std::collections::HashMap::new();
            latest.insert(req.camera_index, req);
            while let Ok(queued) = frame_rx.try_recv() {
                latest.insert(queued.camera_index, queued);
            }

            // カメラインデックス順に処理（安定した処理順序）
            let mut cam_indices: Vec<usize> = latest.keys().copied().collect();
            cam_indices.sort();

            for cam_idx in cam_indices {
                let req = latest.remove(&cam_idx).unwrap();

                // タイムスタンプが古すぎるリクエストはスキップ（500ms以上前）
                if req.timestamp.elapsed().as_secs_f32() > 0.5 {
                    continue;
                }

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
            let (input, letterbox) = match model_type {
                ModelType::MoveNet => {
                    match preprocess_for_movenet(&input_frame) {
                        Ok(v) => (v, LetterboxInfo::identity()),
                        Err(e) => { log!(logfile_infer, "preprocess error: {}", e); continue; }
                    }
                }
                ModelType::SpinePose => {
                    match preprocess_for_spinepose(&input_frame) {
                        Ok(v) => v,
                        Err(e) => { log!(logfile_infer, "preprocess error: {}", e); continue; }
                    }
                }
                ModelType::RTMW3D => {
                    match preprocess_for_rtmw3d(&input_frame) {
                        Ok(v) => v,
                        Err(e) => { log!(logfile_infer, "preprocess error: {}", e); continue; }
                    }
                }
            };

            // 推論
            let mut pose = match detector.detect(input) {
                Ok(v) => v,
                Err(e) => { log!(logfile_infer, "inference error: {}", e); continue; }
            };

            // レターボックス座標を元の画像座標に変換
            pose = unletterbox_pose(&pose, &letterbox);

            if !crop_region.is_full() {
                pose = remap_pose(&pose, &crop_region);
            }

            let _ = result_tx.send(InferenceResult {
                camera_index: req.camera_index,
                pose,
                timestamp: req.timestamp,
            });
            } // for cam_idx
        }
    });

    // Body tracker
    let fov_v = if multi_camera { 0.0 } else { config.camera.fov_v };
    let body_tracker = BodyTracker::new(&config.tracker, fov_v, multi_camera);
    let filters: [PoseFilter; TRACKER_COUNT] =
        std::array::from_fn(|_| PoseFilter::from_config(&config.filter));
    let extrapolators: [Extrapolator; TRACKER_COUNT] =
        std::array::from_fn(|_| Extrapolator::new());
    let lerpers: [Lerper; TRACKER_COUNT] =
        std::array::from_fn(|_| Lerper::new());

    let vmt = VmtClient::new(&config.vmt.addr)?;
    log!(logfile, "VMT client ready");

    // キャリブレーショントリガー: SIGUSR1シグナルまたはコンソール入力 "c"
    let console_flag = Arc::new(AtomicBool::new(false));
    {
        let flag = console_flag.clone();
        // SIGUSR1でキャリブレーション発動
        signal_hook::flag::register(signal_hook::consts::SIGUSR1, flag.clone())
            .expect("failed to register SIGUSR1 handler");
        // コンソール入力も残す（直接実行時用）
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

    // デバッグビュー
    let debug_view = if config.debug.view {
        let view_w = 640;
        let view_h = 360;
        let total_w = view_w * num_cameras;
        match MinifbRenderer::new("tracker_bevy debug", total_w, view_h) {
            Ok(renderer) => {
                log!(logfile, "Debug view: {}x{} ({}cameras)", total_w, view_h, num_cameras);
                Some(DebugView { renderer, view_w, view_h, num_cameras })
            }
            Err(e) => { log!(logfile, "Debug view failed: {}", e); None }
        }
    } else {
        None
    };

    log!(logfile, "");
    log!(logfile, "操作: [C + Enter] キャリブレーション  [Ctrl+C] 終了");
    log!(logfile, "");

    let frame_duration = Duration::from_secs_f64(1.0 / config.app.target_fps as f64);

    let mut app = App::new();
    app.add_plugins(ScheduleRunnerPlugin::run_loop(frame_duration))
        .insert_resource(CameraInputs {
            cameras,
            params,
            last_frame_ids: vec![0; num_cameras],
            widths,
            heights,
            in_flight: vec![false; num_cameras],
        })
        .insert_resource(InferenceTx(frame_tx))
        .insert_resource(InferenceRx(Mutex::new(result_rx)))
        .insert_resource(PoseState {
            poses_2d: vec![None; num_cameras],
            prev_poses: vec![None; num_cameras],
            pose_timestamps: vec![None; num_cameras],
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
            stale: [false; TRACKER_COUNT],
            reject_count: [0; TRACKER_COUNT],
            last_inference_time: Instant::now(),
            interp_mode: config.interpolation.mode.clone(),
        })
        .insert_resource(VmtSender(vmt))
        .insert_resource(CalibrationState {
            deadline: None,
            console_flag,
            auto_cal_triggered: false,
            first_hip_time: None,
            last_hip_time: None,
            recent_hip_z: Vec::new(),
        })
        .insert_resource(FpsCounter {
            frame_count: 0,
            inference_count: 0,
            timer: Instant::now(),
        })
        .insert_resource(LogFileRes(logfile.clone()));

    if let Some(dv) = debug_view {
        app.insert_non_send_resource(dv)
            .add_systems(Update, (
                (send_frames_system, receive_results_system, triangulate_system).chain(),
                (calibration_system, compute_trackers_system, send_vmt_system, fps_system, debug_view_system).chain(),
            ).chain());
    } else {
        app.add_systems(Update, (
            (send_frames_system, receive_results_system, triangulate_system).chain(),
            (calibration_system, compute_trackers_system, send_vmt_system, fps_system).chain(),
        ).chain());
    }

    app.run();

    log!(logfile, "Shutting down...");
    Ok(())
}

// --- Systems ---

fn send_frames_system(
    mut cam_inputs: ResMut<CameraInputs>,
    tx: Res<InferenceTx>,
    pose_state: Res<PoseState>,
) {
    for i in 0..cam_inputs.cameras.len() {
        if cam_inputs.in_flight[i] { continue; }
        let fid = cam_inputs.cameras[i].frame_id();
        if fid != cam_inputs.last_frame_ids[i] {
            if let Some(frame) = cam_inputs.cameras[i].get_frame() {
                let req = InferenceRequest {
                    camera_index: i,
                    frame,
                    prev_pose: pose_state.prev_poses[i].clone(),
                    width: cam_inputs.widths[i],
                    height: cam_inputs.heights[i],
                    timestamp: Instant::now(),
                };
                match tx.0.try_send(req) {
                    Ok(()) => {
                        cam_inputs.last_frame_ids[i] = fid;
                        cam_inputs.in_flight[i] = true;
                    }
                    Err(mpsc::TrySendError::Full(_)) => {}
                    Err(mpsc::TrySendError::Disconnected(_)) => {}
                    // Inference thread disconnected は無視（終了時に発生）
                }
            }
        }
    }
}

fn receive_results_system(
    mut pose_state: ResMut<PoseState>,
    mut cam_inputs: ResMut<CameraInputs>,
    rx: Res<InferenceRx>,
    mut fps: ResMut<FpsCounter>,
) {
    let rx = rx.0.lock().unwrap();
    while let Ok(result) = rx.try_recv() {
        let idx = result.camera_index;
        if idx < cam_inputs.in_flight.len() {
            cam_inputs.in_flight[idx] = false;
        }
        if idx < pose_state.poses_2d.len() {
            // 古すぎる推論結果は破棄（300ms以上前のフレーム）
            if result.timestamp.elapsed().as_secs_f32() > 0.3 {
                continue;
            }
            pose_state.prev_poses[idx] = Some(result.pose.clone());
            pose_state.poses_2d[idx] = Some(result.pose);
            pose_state.pose_timestamps[idx] = Some(result.timestamp);
            fps.inference_count += 1;
        }
    }
}

/// 三角測量の最大待機時間: これを超えたら揃ったカメラのみで三角測量する
const TRIANGULATION_TIMEOUT_MS: f32 = 100.0;

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

    // 複数カメラ: 全カメラのPoseが揃っているか、またはタイムアウトで部分更新
    let all_ready = pose_state.poses_2d.iter().all(|p| p.is_some());
    let ready_count = pose_state.poses_2d.iter().filter(|p| p.is_some()).count();

    if !all_ready && ready_count >= 2 {
        // 2台以上揃っている場合、最古のポーズのタイムスタンプからの経過時間を確認
        let oldest_ts = pose_state.pose_timestamps.iter()
            .filter_map(|t| *t)
            .min();
        let timed_out = oldest_ts
            .map(|ts| ts.elapsed().as_secs_f32() * 1000.0 > TRIANGULATION_TIMEOUT_MS)
            .unwrap_or(false);
        if !timed_out {
            return; // まだ待てる
        }
        // タイムアウト: 揃っているカメラだけで三角測量する（フォールスルー）
    } else if !all_ready {
        return; // 2台未満 → 待機
    }

    // 利用可能なカメラのみでインデックスを収集
    let mut available_poses: Vec<&Pose> = Vec::new();
    let mut available_params: Vec<&CameraParams> = Vec::new();
    for i in 0..pose_state.poses_2d.len() {
        if let Some(ref pose) = pose_state.poses_2d[i] {
            available_poses.push(pose);
            available_params.push(&cam_inputs.params[i]);
        }
    }

    if available_poses.len() < 2 {
        return;
    }

    let mut pose_3d = triangulate_poses(&available_params, &available_poses, CONFIDENCE_THRESHOLD);

    // 三角測量の各キーポイントの境界チェック: カメラ座標系で妥当な範囲外を除去
    for kp in pose_3d.keypoints.iter_mut() {
        if kp.confidence > 0.0
            && (kp.x.abs() > 10.0 || kp.y.abs() > 10.0 || kp.z.abs() > 10.0 || kp.z < 0.0)
        {
            kp.confidence = 0.0;
        }
    }

    pose_state.pose_3d = Some(pose_3d);

    // 消費: 次の三角測量まで待機
    for p in pose_state.poses_2d.iter_mut() {
        *p = None;
    }
    for t in pose_state.pose_timestamps.iter_mut() {
        *t = None;
    }
}

fn calibration_system(
    mut cal: ResMut<CalibrationState>,
    mut tracker_state: ResMut<TrackerState>,
    pose_state: Res<PoseState>,
    lf: Res<LogFileRes>,
) {
    // 手動キャリブレーション（コンソール入力 or SIGUSR1）
    if cal.console_flag.swap(false, Ordering::AcqRel) && cal.deadline.is_none() {
        cal.deadline = Some(Instant::now() + Duration::from_secs(5));
        log!(lf.0, "Calibration in 5s... 基準位置に立ってください");
    }

    // 自動キャリブレーション: 未キャリブ時、hipを3秒間安定検出したら自動発動
    if !cal.auto_cal_triggered && !tracker_state.body_tracker.is_calibrated() && cal.deadline.is_none() {
        use talava_tracker::pose::KeypointIndex;
        let hip_z = pose_state.pose_3d.as_ref().and_then(|p| {
            let lh = p.get(KeypointIndex::LeftHip);
            let rh = p.get(KeypointIndex::RightHip);
            if lh.is_valid(0.2) && rh.is_valid(0.2) {
                Some((lh.z + rh.z) / 2.0)
            } else {
                None
            }
        });
        if let Some(z) = hip_z {
            let now = Instant::now();
            if cal.first_hip_time.is_none() {
                cal.first_hip_time = Some(now);
                cal.recent_hip_z.clear();
            }
            cal.last_hip_time = Some(now);
            cal.recent_hip_z.push(z);
            if now.duration_since(cal.first_hip_time.unwrap()).as_secs_f32() > 3.0 {
                // z値の安定性チェック: 標準偏差が0.3m未満なら安定とみなす
                let n = cal.recent_hip_z.len() as f32;
                let mean = cal.recent_hip_z.iter().sum::<f32>() / n;
                let variance = cal.recent_hip_z.iter()
                    .map(|z| (z - mean).powi(2))
                    .sum::<f32>() / n;
                let std_dev = variance.sqrt();
                if std_dev < 0.3 {
                    cal.auto_cal_triggered = true;
                    cal.deadline = Some(now);
                    log!(lf.0, "Auto-calibration triggered (hip z stable: mean={:.2}, std={:.2}, n={})",
                        mean, std_dev, cal.recent_hip_z.len());
                } else {
                    log!(lf.0, "Auto-cal rejected: hip z unstable (mean={:.2}, std={:.2}, n={})",
                        mean, std_dev, cal.recent_hip_z.len());
                    cal.first_hip_time = None;
                    cal.recent_hip_z.clear();
                }
            }
        } else {
            // hip未検出でも0.5秒以内なら蓄積を維持（no poseの散発的挿入を許容）
            let gap = cal.last_hip_time.map_or(f32::MAX, |t| Instant::now().duration_since(t).as_secs_f32());
            if gap > 0.5 {
                cal.first_hip_time = None;
                cal.last_hip_time = None;
                cal.recent_hip_z.clear();
            }
        }
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
                    log!(lf.0, "Calibrated!");
                } else {
                    log!(lf.0, "Calibration failed: hip not detected");
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

    // 三角測量の出力が妥当な範囲か検証（カメラから±10m以内）
    if pose_state.multi_camera {
        use talava_tracker::pose::KeypointIndex;
        let lh = pose.get(KeypointIndex::LeftHip);
        let rh = pose.get(KeypointIndex::RightHip);
        if lh.is_valid(0.2) && rh.is_valid(0.2) {
            let hip_x = (lh.x + rh.x) / 2.0;
            let hip_y = (lh.y + rh.y) / 2.0;
            let hip_z = (lh.z + rh.z) / 2.0;
            if hip_x.abs() > 10.0 || hip_y.abs() > 10.0 || hip_z.abs() > 10.0 || hip_z < 0.0 {
                return; // カメラ座標系で妥当な範囲外 → スキップ
            }
        }
    }

    let body_poses = tracker_state.body_tracker.compute(pose);
    let poses = [
        body_poses.hip, body_poses.left_foot, body_poses.right_foot,
        body_poses.chest, body_poses.left_knee, body_poses.right_knee,
    ];

    let dt = tracker_state.last_inference_time.elapsed().as_secs_f32();
    let lerp_t = (dt / (1.0 / 30.0)).min(1.0);

    let now = Instant::now();
    // VR空間スケールに応じた閾値
    // 単眼モード(scale~1): 1.0/1.5、複眼モード(scale~3.4): スケール比例
    let scale_factor = if pose_state.multi_camera {
        let cfg = &tracker_state.body_tracker;
        cfg.max_scale()
    } else {
        1.0
    };
    let max_displacement = 0.4 * scale_factor;
    let max_limb_dist = 1.5 * scale_factor;

    for i in 0..TRACKER_COUNT {
        let mut accepted = false;
        if let Some(p) = poses[i] {
            // 速度ベースの外れ値除去: 前フレームから離れすぎた検出はスキップ
            // ただし連続5回拒否（≈0.2s）されたらlast_posesをクリアして受け入れ
            // （正当な急速移動が速度チェックで引っかかった場合の復帰用）
            let velocity_ok = match tracker_state.last_poses[i].as_ref() {
                Some(last) => {
                    let dx = p.position[0] - last.position[0];
                    let dy = p.position[1] - last.position[1];
                    let dz = p.position[2] - last.position[2];
                    (dx * dx + dy * dy + dz * dz).sqrt() <= max_displacement
                }
                None => true,
            };
            // 四肢のhipからの距離チェック: 解剖学的にありえない位置を除去
            let limb_ok = if i > 0 {
                match tracker_state.last_poses[0].as_ref() {
                    Some(hip) => {
                        let dx = p.position[0] - hip.position[0];
                        let dy = p.position[1] - hip.position[1];
                        let dz = p.position[2] - hip.position[2];
                        let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                        // 解剖学的制約: 足・膝はhipより下(y < hip.y)
                        let anatomy_ok = match i {
                            1 | 2 | 4 | 5 => p.position[1] < hip.position[1],
                            _ => true,
                        };
                        dist <= max_limb_dist && anatomy_ok
                    }
                    None => false, // hip基準がない場合は四肢を受け入れない
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
                tracker_state.stale[i] = false;
                tracker_state.reject_count[i] = 0;
                accepted = true;
            } else if !velocity_ok && limb_ok {
                // velocity_ok失敗: 連続拒否をカウント
                tracker_state.reject_count[i] += 1;
                if tracker_state.reject_count[i] >= 5 {
                    // 連続5回拒否 → 正当な急速移動と判断
                    // フィルタリセットして新位置を受け入れ
                    tracker_state.filters[i].reset();
                    tracker_state.extrapolators[i] = Extrapolator::new();
                    tracker_state.lerpers[i] = Lerper::new();
                    let smoothed = tracker_state.filters[i].apply(p);
                    tracker_state.extrapolators[i].update(smoothed);
                    tracker_state.lerpers[i].update(smoothed, lerp_t);
                    tracker_state.last_poses[i] = Some(smoothed);
                    tracker_state.last_update_times[i] = now;
                    tracker_state.stale[i] = false;
                    tracker_state.reject_count[i] = 0;
                    accepted = true;
                }
            }
        }
        // ステールタイムアウト: 2段階
        // Phase 1 (0.3s): 送信停止・フィルタリセット（速度参照は維持してゴースト除去）
        // Phase 2 (3.0s): 速度参照もクリア（人が移動した場合に再検出を許可）
        if !accepted && tracker_state.last_poses[i].is_some() {
            let stale_time = now.duration_since(tracker_state.last_update_times[i]).as_secs_f32();
            if stale_time > 3.0 {
                tracker_state.last_poses[i] = None;
                tracker_state.stale[i] = false;
            } else if stale_time > 0.3 && !tracker_state.stale[i] {
                tracker_state.stale[i] = true;
                tracker_state.filters[i].reset();
                tracker_state.extrapolators[i] = Extrapolator::new();
                tracker_state.lerpers[i] = Lerper::new();
            }
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
            if !tracker_state.stale[i] {
                if let Some(ref p) = tracker_state.last_poses[i] {
                    let _ = vmt.0.send(TRACKER_INDICES[i], 1, p);
                }
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

fn debug_view_system(
    cam_inputs: Res<CameraInputs>,
    pose_state: Res<PoseState>,
    mut debug_view: NonSendMut<DebugView>,
) {
    use opencv::imgproc;
    use opencv::core::Size;
    use opencv::prelude::*;

    let vw = debug_view.view_w;
    let vh = debug_view.view_h;
    let nc = debug_view.num_cameras;

    for cam_idx in 0..nc {
        if let Some(frame) = cam_inputs.cameras[cam_idx].get_frame() {
            // フレームをview_w x view_hにリサイズ
            let mut resized = Mat::default();
            let _ = imgproc::resize(&frame, &mut resized, Size::new(vw as i32, vh as i32), 0.0, 0.0, imgproc::INTER_LINEAR);

            // ポーズのキーポイントを描画
            if let Some(ref pose) = pose_state.prev_poses[cam_idx] {
                for kp in pose.keypoints.iter() {
                    if kp.confidence >= 0.2 {
                        let px = (kp.x * vw as f32) as i32;
                        let py = (kp.y * vh as f32) as i32;
                        let _ = imgproc::circle(
                            &mut resized,
                            opencv::core::Point::new(px, py),
                            3,
                            opencv::core::Scalar::new(0.0, 255.0, 0.0, 0.0),
                            -1, imgproc::LINE_8, 0,
                        );
                    }
                }
            }

            // バッファに描画（カメラ毎にオフセット）
            let x_offset = cam_idx * vw;
            if let Ok(data) = resized.data_bytes() {
                let channels = resized.channels() as usize;
                let step = resized.mat_step().get(0) as usize;
                let buf = &mut debug_view.renderer;
                for y in 0..vh {
                    for x in 0..vw {
                        let px = y * step + x * channels;
                        if px + 2 < data.len() {
                            let b = data[px] as u32;
                            let g = data[px + 1] as u32;
                            let r = data[px + 2] as u32;
                            let buf_idx = y * (vw * nc) + x_offset + x;
                            buf.set_pixel_raw(buf_idx, (r << 16) | (g << 8) | b);
                        }
                    }
                }
            }
        }
    }

    let _ = debug_view.renderer.update();
}

fn fps_system(mut fps: ResMut<FpsCounter>, tracker_state: Res<TrackerState>, pose_state: Res<PoseState>, lf: Res<LogFileRes>) {
    fps.frame_count += 1;
    let elapsed = fps.timer.elapsed().as_secs_f32();
    if elapsed >= 1.0 {
        let names = ["hip", "L_foot", "R_foot", "chest", "L_knee", "R_knee"];
        let mut parts = Vec::new();
        for i in 0..TRACKER_COUNT {
            if !tracker_state.stale[i] {
                if let Some(ref p) = tracker_state.last_poses[i] {
                    parts.push(format!(
                        "{}({:.2},{:.2},{:.2})",
                        names[i], p.position[0], p.position[1], p.position[2]
                    ));
                }
            }
        }
        // 各カメラの下半身キーポイントconfidence（消失原因の診断用）
        use talava_tracker::pose::KeypointIndex;
        let diag_kps = [
            ("LA", KeypointIndex::LeftAnkle),
            ("RA", KeypointIndex::RightAnkle),
            ("LK", KeypointIndex::LeftKnee),
            ("RK", KeypointIndex::RightKnee),
        ];
        let mut cam_diag = String::new();
        for (cam_idx, prev) in pose_state.prev_poses.iter().enumerate() {
            if let Some(ref p) = prev {
                let confs: Vec<String> = diag_kps.iter().map(|(name, idx)| {
                    format!("{}={:.2}", name, p.get(*idx).confidence)
                }).collect();
                cam_diag.push_str(&format!(" cam{}[{}]", cam_idx, confs.join(",")));
            }
        }
        log!(lf.0,
            "FPS: {:.1} (infer: {}) | {}{}",
            fps.frame_count as f32 / elapsed,
            fps.inference_count,
            if parts.is_empty() { "no pose".to_string() } else { parts.join(" ") },
            cam_diag,
        );
        fps.frame_count = 0;
        fps.inference_count = 0;
        fps.timer = Instant::now();
    }
}
