use anyhow::Result;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::time::{Duration, Instant};

use opencv::core::Mat;

use talava_tracker::camera::ThreadedCamera;
use talava_tracker::config::Config;
use talava_tracker::pose::{
    bbox_from_keypoints, crop_for_pose, preprocess_for_movenet, preprocess_for_spinepose,
    remap_pose, CropRegion, ModelType, PersonDetector, Pose, PoseDetector,
};
use talava_tracker::render::{Key, MinifbRenderer};
use talava_tracker::tracker::{BodyTracker, Extrapolator, Lerper, PoseFilter};
use talava_tracker::vmt::{TrackerPose, VmtClient};

const CONFIG_PATH: &str = "config.toml";
const HIP_INDEX: i32 = 0;
const LEFT_FOOT_INDEX: i32 = 1;
const RIGHT_FOOT_INDEX: i32 = 2;
const CHEST_INDEX: i32 = 3;
const CONFIDENCE_THRESHOLD: f32 = 0.3;
const TRACKER_COUNT: usize = 4;
const TRACKER_INDICES: [i32; TRACKER_COUNT] = [HIP_INDEX, LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX, CHEST_INDEX];

struct InferenceRequest {
    frame: Mat,
    prev_pose: Option<Pose>,
}

struct InferenceResult {
    pose: Pose,
    frame: Mat,
    t_detect_ms: f64,
    t_preprocess_ms: f64,
    t_inference_ms: f64,
}

fn main() -> Result<()> {
    let config = Config::load(CONFIG_PATH)?;

    let (model_path, model_type): (&str, ModelType) = match config.app.model.as_str() {
        "movenet" => ("models/movenet_lightning.onnx", ModelType::MoveNet),
        "spinepose_small" => ("models/spinepose_small.onnx", ModelType::SpinePose),
        "spinepose_medium" => ("models/spinepose_medium.onnx", ModelType::SpinePose),
        other => anyhow::bail!("Unknown model: {}", other),
    };

    println!("Tracker Sender - Phase 6 (async inference)");
    println!("Model: {}", config.app.model);
    println!("Detector: {}", config.app.detector);
    println!("VMT target: {}", config.vmt.addr);
    println!("Target FPS: {}", config.app.target_fps);
    println!("Interpolation: {}", config.interpolation.mode);
    println!("Debug view: {}", if config.debug.view { "ON" } else { "OFF" });
    println!("Tracker: scale=({}, {}), body_scale={}, leg_scale={}, depth_scale={}, mirror_x={}, offset_y={}",
        config.tracker.scale_x, config.tracker.scale_y,
        config.tracker.body_scale, config.tracker.leg_scale, config.tracker.depth_scale,
        config.tracker.mirror_x, config.tracker.offset_y);
    println!("Filter: pos(cutoff={}, beta={}) rot(cutoff={}, beta={})",
        config.filter.position_min_cutoff, config.filter.position_beta,
        config.filter.rotation_min_cutoff, config.filter.rotation_beta);
    println!();
    println!("操作: [C] キャリブレーション  [Esc] 終了  (コンソール: c + Enter)");
    println!();

    let camera = ThreadedCamera::start(
        config.camera.index,
        Some(config.camera.width),
        Some(config.camera.height),
    )?;
    let (width, height) = camera.resolution();
    println!("Camera: {}x{}", width, height);

    let mut body_tracker = BodyTracker::from_config(&config.tracker);
    let mut filters: [PoseFilter; TRACKER_COUNT] =
        std::array::from_fn(|_| PoseFilter::from_config(&config.filter));
    let mut extrapolators: [Extrapolator; TRACKER_COUNT] =
        std::array::from_fn(|_| Extrapolator::new());
    let mut lerpers: [Lerper; TRACKER_COUNT] =
        std::array::from_fn(|_| Lerper::new());
    let interp_mode = config.interpolation.mode.clone();

    let vmt = VmtClient::new(&config.vmt.addr)?;
    println!("VMT client ready");

    let mut renderer = if config.debug.view {
        Some(MinifbRenderer::new("Tracker Debug", width as usize, height as usize)?)
    } else {
        None
    };

    let frame_duration = Duration::from_secs_f64(1.0 / config.app.target_fps as f64);

    // FPS計測
    let mut frame_count = 0u32;
    let mut inference_count = 0u32;
    let mut fps_timer = Instant::now();
    let mut t_detect_sum = 0.0f64;
    let mut t_preprocess_sum = 0.0f64;
    let mut t_inference_sum = 0.0f64;
    let mut t_render = 0.0f64;
    let mut t_tracker = 0.0f64;

    // キャリブレーション
    let mut calibration_deadline: Option<Instant> = None;
    const CALIBRATION_DELAY_SECS: u64 = 5;

    // コンソール入力スレッド
    let console_calibrate = Arc::new(AtomicBool::new(false));
    {
        let flag = console_calibrate.clone();
        std::thread::spawn(move || {
            let stdin = std::io::stdin();
            let mut line = String::new();
            loop {
                line.clear();
                if stdin.read_line(&mut line).is_ok() {
                    if line.trim().eq_ignore_ascii_case("c") {
                        flag.store(true, Ordering::Release);
                    }
                }
            }
        });
    }

    // 推論スレッドのセットアップ
    let (frame_tx, frame_rx) = mpsc::sync_channel::<InferenceRequest>(1);
    let (result_tx, result_rx) = mpsc::channel::<InferenceResult>();

    let detector_mode = config.app.detector.clone();
    {
        let detector_mode = detector_mode.clone();
        std::thread::spawn(move || {
            let mut detector = PoseDetector::new(model_path, model_type)
                .expect("Failed to load pose model in inference thread");
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
                let t0 = Instant::now();

                // 人物検出 → クロップ
                let detect_result: Result<(Mat, CropRegion)> = match detector_mode.as_str() {
                    "keypoint" => {
                        match req.prev_pose.as_ref()
                            .and_then(|p| bbox_from_keypoints(p, width, height, CONFIDENCE_THRESHOLD))
                        {
                            Some(bbox) => crop_for_pose(&req.frame, &bbox, width, height),
                            None => Ok((req.frame.clone(), CropRegion::full())),
                        }
                    }
                    "yolo" => {
                        match person_detector.as_mut().unwrap().detect(&req.frame) {
                            Ok(Some(bbox)) => crop_for_pose(&req.frame, &bbox, width, height),
                            Ok(None) => Ok((req.frame.clone(), CropRegion::full())),
                            Err(e) => Err(e),
                        }
                    }
                    _ => Ok((req.frame.clone(), CropRegion::full())),
                };

                let (input_frame, crop_region) = match detect_result {
                    Ok(v) => v,
                    Err(e) => {
                        eprintln!("Inference thread: detect error: {}", e);
                        continue;
                    }
                };
                let t1 = Instant::now();

                // 前処理
                let input = match model_type {
                    ModelType::MoveNet => preprocess_for_movenet(&input_frame),
                    ModelType::SpinePose => preprocess_for_spinepose(&input_frame),
                };
                let input = match input {
                    Ok(v) => v,
                    Err(e) => {
                        eprintln!("Inference thread: preprocess error: {}", e);
                        continue;
                    }
                };
                let t2 = Instant::now();

                // 推論
                let mut pose = match detector.detect(input) {
                    Ok(v) => v,
                    Err(e) => {
                        eprintln!("Inference thread: inference error: {}", e);
                        continue;
                    }
                };

                // クロップ座標→フレーム座標にリマップ
                if !crop_region.is_full() {
                    pose = remap_pose(&pose, &crop_region);
                }
                let t3 = Instant::now();

                let _ = result_tx.send(InferenceResult {
                    pose,
                    frame: req.frame,
                    t_detect_ms: (t1 - t0).as_secs_f64() * 1000.0,
                    t_preprocess_ms: (t2 - t1).as_secs_f64() * 1000.0,
                    t_inference_ms: (t3 - t2).as_secs_f64() * 1000.0,
                });
            }
            println!("Inference thread: shutting down");
        });
    }

    // フレーム追跡
    let mut last_sent_frame_id: u64 = 0;
    let mut last_inference_time = Instant::now();
    let mut last_poses: [Option<TrackerPose>; TRACKER_COUNT] = [None; TRACKER_COUNT];
    let mut prev_pose: Option<Pose> = None;

    loop {
        let loop_start = Instant::now();

        if let Some(ref r) = renderer {
            if !r.is_open() {
                break;
            }
        }

        // キャリブレーション要求検知（コンソール or デバッグウィンドウ）
        let cal_from_console = console_calibrate.swap(false, Ordering::AcqRel);
        let cal_from_renderer = renderer.as_ref().map_or(false, |r| r.is_key_pressed(Key::C));
        if (cal_from_console || cal_from_renderer) && calibration_deadline.is_none() {
            let deadline = Instant::now() + Duration::from_secs(CALIBRATION_DELAY_SECS);
            calibration_deadline = Some(deadline);
            println!("Calibration in {}s... 基準位置に立ってください", CALIBRATION_DELAY_SECS);
        }

        // 新フレームがあれば推論スレッドに送る（ノンブロッキング）
        let current_frame_id = camera.frame_id();
        if current_frame_id != last_sent_frame_id {
            if let Some(frame) = camera.get_frame() {
                let req = InferenceRequest {
                    frame,
                    prev_pose: prev_pose.clone(),
                };
                // sync_channel(1): 推論スレッドがビジーならスキップ
                // 次のループで新しいフレームを送る
                match frame_tx.try_send(req) {
                    Ok(()) => {}
                    Err(mpsc::TrySendError::Full(_)) => {}
                    Err(mpsc::TrySendError::Disconnected(_)) => {
                        eprintln!("Inference thread disconnected");
                        break;
                    }
                }
                last_sent_frame_id = current_frame_id;
            }
        }

        // 推論結果をノンブロッキングで受け取る
        if let Ok(result) = result_rx.try_recv() {
            prev_pose = Some(result.pose.clone());

            // キャリブレーション実行（カウントダウン完了時）
            if let Some(deadline) = calibration_deadline {
                if deadline.saturating_duration_since(Instant::now()).is_zero() {
                    if body_tracker.calibrate(&result.pose) {
                        for f in &mut filters { f.reset(); }
                        extrapolators = std::array::from_fn(|_| Extrapolator::new());
                        lerpers = std::array::from_fn(|_| Lerper::new());
                        last_poses = [None; TRACKER_COUNT];
                        println!("Calibrated!");
                    } else {
                        println!("Calibration failed: hip not detected");
                    }
                    calibration_deadline = None;
                }
            }

            // デバッグ描画
            let t_render_start = Instant::now();
            if let Some(ref mut r) = renderer {
                let _ = r.draw_frame(&result.frame);
                r.draw_pose(&result.pose, CONFIDENCE_THRESHOLD);
                let _ = r.update();
            }
            let t_render_end = Instant::now();

            // トラッカー変換 & 補間器更新 & 平滑化 & 送信
            let t_tracker_start = Instant::now();
            let body_poses = body_tracker.compute(&result.pose);
            let poses = [body_poses.hip, body_poses.left_foot, body_poses.right_foot, body_poses.chest];

            let dt_since_last = last_inference_time.elapsed().as_secs_f32();
            let camera_interval = 1.0 / 30.0;
            let lerp_t = (dt_since_last / camera_interval).min(1.0);

            for i in 0..TRACKER_COUNT {
                if let Some(p) = poses[i] {
                    extrapolators[i].update(p);
                    lerpers[i].update(p, lerp_t);
                    let smoothed = filters[i].apply(p);
                    let _ = vmt.send(TRACKER_INDICES[i], 1, &smoothed);
                    last_poses[i] = Some(smoothed);
                }
            }

            last_inference_time = Instant::now();
            let t_tracker_end = Instant::now();

            // ログ（1秒に1回、最初の推論結果で）
            if inference_count == 0 {
                let pose = &result.pose;
                let lh = pose.get(talava_tracker::pose::KeypointIndex::LeftHip);
                let rh = pose.get(talava_tracker::pose::KeypointIndex::RightHip);
                let ls = pose.get(talava_tracker::pose::KeypointIndex::LeftShoulder);
                let rs = pose.get(talava_tracker::pose::KeypointIndex::RightShoulder);
                let sp = pose.get(talava_tracker::pose::KeypointIndex::Spine03);
                eprintln!("KP | Hip L({:.3},{:.3}) R({:.3},{:.3}) | Sh L({:.3},{:.3}) R({:.3},{:.3}) | Sp03({:.3},{:.3}) c={:.2}",
                    lh.x, lh.y, rh.x, rh.y, ls.x, ls.y, rs.x, rs.y, sp.x, sp.y, sp.confidence);
                if let Some(ref hip) = poses[0] {
                    eprint!("Hip: [{:.2}, {:.2}, {:.2}]", hip.position[0], hip.position[1], hip.position[2]);
                }
                let count = poses.iter().filter(|p| p.is_some()).count();
                eprintln!(" Active: {}/4{}", count, if body_tracker.is_calibrated() { " [CAL]" } else { "" });
            }

            // 計測集計
            t_detect_sum += result.t_detect_ms;
            t_preprocess_sum += result.t_preprocess_ms;
            t_inference_sum += result.t_inference_ms;
            t_render += (t_render_end - t_render_start).as_secs_f64() * 1000.0;
            t_tracker += (t_tracker_end - t_tracker_start).as_secs_f64() * 1000.0;
            inference_count += 1;
        } else {
            // === 推論結果なし: 補間 ===
            let dt = last_inference_time.elapsed().as_secs_f32();

            match interp_mode.as_str() {
                "extrapolate" => {
                    for i in 0..TRACKER_COUNT {
                        if let Some(predicted) = extrapolators[i].predict(dt) {
                            let smoothed = filters[i].apply(predicted);
                            let _ = vmt.send(TRACKER_INDICES[i], 1, &smoothed);
                            last_poses[i] = Some(smoothed);
                        }
                    }
                }
                "lerp" => {
                    let camera_interval = 1.0 / 30.0;
                    let t = (dt / camera_interval).min(1.0);
                    for i in 0..TRACKER_COUNT {
                        if let Some(interpolated) = lerpers[i].interpolate(t) {
                            let smoothed = filters[i].apply(interpolated);
                            let _ = vmt.send(TRACKER_INDICES[i], 1, &smoothed);
                            last_poses[i] = Some(smoothed);
                        }
                    }
                }
                _ => {
                    // "none": 最後のポーズを再送信
                    for i in 0..TRACKER_COUNT {
                        if let Some(ref p) = last_poses[i] {
                            let _ = vmt.send(TRACKER_INDICES[i], 1, p);
                        }
                    }
                }
            }
        }

        // FPS表示
        frame_count += 1;
        let elapsed = fps_timer.elapsed().as_secs_f32();
        if elapsed >= 1.0 {
            if inference_count > 0 {
                let n = inference_count as f64;
                println!("FPS: {:.1} (infer: {}) | detect {:.1}ms  preprocess {:.1}ms  inference {:.1}ms  render {:.1}ms  tracker {:.1}ms",
                    frame_count as f32 / elapsed,
                    inference_count,
                    t_detect_sum / n, t_preprocess_sum / n, t_inference_sum / n, t_render / n, t_tracker / n);
            } else {
                println!("FPS: {:.1} (infer: 0)", frame_count as f32 / elapsed);
            }
            frame_count = 0;
            inference_count = 0;
            fps_timer = Instant::now();
            t_detect_sum = 0.0;
            t_preprocess_sum = 0.0;
            t_inference_sum = 0.0;
            t_render = 0.0;
            t_tracker = 0.0;
        }

        // FPS上限制御（spin wait for precision）
        while loop_start.elapsed() < frame_duration {
            std::hint::spin_loop();
        }
    }

    println!("Shutting down...");
    Ok(())
}
