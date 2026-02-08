use anyhow::Result;
use std::time::{Duration, Instant};

use talava_tracker::camera::ThreadedCamera;
use talava_tracker::config::Config;
use talava_tracker::pose::{preprocess_for_movenet, PoseDetector};
use talava_tracker::render::{Key, MinifbRenderer};
use talava_tracker::tracker::{BodyTracker, Extrapolator, Lerper, Smoother};
use talava_tracker::vmt::{TrackerPose, VmtClient};

const MODEL_PATH: &str = "models/movenet_lightning.onnx";
const CONFIG_PATH: &str = "config.toml";
const HIP_INDEX: i32 = 0;
const LEFT_FOOT_INDEX: i32 = 1;
const RIGHT_FOOT_INDEX: i32 = 2;
const CHEST_INDEX: i32 = 3;
const CONFIDENCE_THRESHOLD: f32 = 0.3;
const TRACKER_COUNT: usize = 4;
const TRACKER_INDICES: [i32; TRACKER_COUNT] = [HIP_INDEX, LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX, CHEST_INDEX];

fn main() -> Result<()> {
    let config = Config::load_or_default(CONFIG_PATH);

    println!("Tracker Sender - Phase 6");
    println!("VMT target: {}", config.vmt.addr);
    println!("Target FPS: {}", config.app.target_fps);
    println!("Interpolation: {}", config.interpolation.mode);
    println!("Debug view: {}", if config.debug.view { "ON" } else { "OFF" });
    println!("Tracker: scale=({}, {}), mirror_x={}, offset_y={}",
        config.tracker.scale_x, config.tracker.scale_y,
        config.tracker.mirror_x, config.tracker.offset_y);
    println!("Smooth: position={}, rotation={}",
        config.smooth.position, config.smooth.rotation);
    println!();
    println!("操作: [C] キャリブレーション  [Esc] 終了");
    println!();

    let camera = ThreadedCamera::start(
        config.camera.index,
        Some(config.camera.width),
        Some(config.camera.height),
    )?;
    let (width, height) = camera.resolution();
    println!("Camera: {}x{}", width, height);

    let mut detector = PoseDetector::new(MODEL_PATH)?;
    println!("Model loaded");

    let mut body_tracker = BodyTracker::from_config(&config.tracker);
    let mut smoothers: [Smoother; TRACKER_COUNT] =
        std::array::from_fn(|_| Smoother::from_config(&config.smooth));
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
    let mut t_clone = 0.0f64;
    let mut t_preprocess = 0.0f64;
    let mut t_inference = 0.0f64;
    let mut t_render = 0.0f64;
    let mut t_tracker = 0.0f64;

    // キャリブレーション
    let mut calibration_deadline: Option<Instant> = None;
    const CALIBRATION_DELAY_SECS: u64 = 5;

    // フレーム追跡
    let mut last_frame_id: u64 = 0;
    let mut last_inference_time = Instant::now();
    let mut last_poses: [Option<TrackerPose>; TRACKER_COUNT] = [None; TRACKER_COUNT];

    loop {
        let loop_start = Instant::now();

        if let Some(ref r) = renderer {
            if !r.is_open() {
                break;
            }
        }

        let current_frame_id = camera.frame_id();
        let is_new_frame = current_frame_id != last_frame_id;

        if is_new_frame {
            // === 新フレーム: 推論実行 ===
            let t0 = Instant::now();
            let frame = match camera.get_frame() {
                Some(f) => f,
                None => {
                    std::thread::sleep(Duration::from_millis(1));
                    continue;
                }
            };
            let t1 = Instant::now();
            let input = preprocess_for_movenet(&frame)?;
            let t2 = Instant::now();
            let pose = detector.detect(input)?;
            let t3 = Instant::now();

            // キャリブレーション
            if let Some(ref r) = renderer {
                if r.is_key_pressed(Key::C) && calibration_deadline.is_none() {
                    let deadline = Instant::now() + Duration::from_secs(CALIBRATION_DELAY_SECS);
                    calibration_deadline = Some(deadline);
                    println!("Calibration in {}s... 基準位置に立ってください", CALIBRATION_DELAY_SECS);
                }
            }
            if let Some(deadline) = calibration_deadline {
                if deadline.saturating_duration_since(Instant::now()).is_zero() {
                    if body_tracker.calibrate(&pose) {
                        for s in &mut smoothers { s.reset(); }
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
            if let Some(ref mut r) = renderer {
                r.draw_frame(&frame)?;
                r.draw_pose(&pose, CONFIDENCE_THRESHOLD);
                r.update()?;
            }
            let t4 = Instant::now();

            // トラッカー変換 & 補間器更新 & 平滑化 & 送信
            let body_poses = body_tracker.compute(&pose);
            let poses = [body_poses.hip, body_poses.left_foot, body_poses.right_foot, body_poses.chest];

            let dt_since_last = last_inference_time.elapsed().as_secs_f32();
            let camera_interval = 1.0 / 30.0;
            let lerp_t = (dt_since_last / camera_interval).min(1.0);

            for i in 0..TRACKER_COUNT {
                if let Some(p) = poses[i] {
                    extrapolators[i].update(p);
                    lerpers[i].update(p, lerp_t);
                    let smoothed = smoothers[i].apply(p);
                    vmt.send(TRACKER_INDICES[i], 1, &smoothed)?;
                    last_poses[i] = Some(smoothed);
                }
            }

            last_frame_id = current_frame_id;
            last_inference_time = Instant::now();

            // ログ（1秒に1回）
            if frame_count == 0 {
                if let Some(ref hip) = poses[0] {
                    eprint!("Hip: [{:.2}, {:.2}, {:.2}]", hip.position[0], hip.position[1], hip.position[2]);
                }
                let count = poses.iter().filter(|p| p.is_some()).count();
                eprintln!(" Active: {}/4{}", count, if body_tracker.is_calibrated() { " [CAL]" } else { "" });
            }
            let t5 = Instant::now();

            t_clone += (t1 - t0).as_secs_f64() * 1000.0;
            t_preprocess += (t2 - t1).as_secs_f64() * 1000.0;
            t_inference += (t3 - t2).as_secs_f64() * 1000.0;
            t_render += (t4 - t3).as_secs_f64() * 1000.0;
            t_tracker += (t5 - t4).as_secs_f64() * 1000.0;
            inference_count += 1;
        } else {
            // === フレーム未更新: 補間 ===
            let dt = last_inference_time.elapsed().as_secs_f32();

            match interp_mode.as_str() {
                "extrapolate" => {
                    for i in 0..TRACKER_COUNT {
                        if let Some(predicted) = extrapolators[i].predict(dt) {
                            let smoothed = smoothers[i].apply(predicted);
                            vmt.send(TRACKER_INDICES[i], 1, &smoothed)?;
                            last_poses[i] = Some(smoothed);
                        }
                    }
                }
                "lerp" => {
                    let camera_interval = 1.0 / 30.0;
                    let t = (dt / camera_interval).min(1.0);
                    for i in 0..TRACKER_COUNT {
                        if let Some(interpolated) = lerpers[i].interpolate(t) {
                            let smoothed = smoothers[i].apply(interpolated);
                            vmt.send(TRACKER_INDICES[i], 1, &smoothed)?;
                            last_poses[i] = Some(smoothed);
                        }
                    }
                }
                _ => {
                    // "none": 最後のポーズを再送信
                    for i in 0..TRACKER_COUNT {
                        if let Some(ref p) = last_poses[i] {
                            vmt.send(TRACKER_INDICES[i], 1, p)?;
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
                println!("FPS: {:.1} (infer: {}) | clone {:.1}ms  preprocess {:.1}ms  inference {:.1}ms  render {:.1}ms  tracker {:.1}ms",
                    frame_count as f32 / elapsed,
                    inference_count,
                    t_clone / n, t_preprocess / n, t_inference / n, t_render / n, t_tracker / n);
            } else {
                println!("FPS: {:.1} (infer: 0)", frame_count as f32 / elapsed);
            }
            frame_count = 0;
            inference_count = 0;
            fps_timer = Instant::now();
            t_clone = 0.0;
            t_preprocess = 0.0;
            t_inference = 0.0;
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
