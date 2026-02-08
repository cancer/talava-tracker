use anyhow::Result;
use std::time::Instant;

use talava_tracker::camera::OpenCvCamera;
use talava_tracker::config::Config;
use talava_tracker::pose::{preprocess_for_movenet, PoseDetector};
use talava_tracker::render::{Key, MinifbRenderer};
use talava_tracker::tracker::{BodyTracker, Smoother};
use talava_tracker::vmt::VmtClient;

const MODEL_PATH: &str = "models/movenet_lightning.onnx";
const CONFIG_PATH: &str = "config.toml";
const HIP_INDEX: i32 = 0;
const LEFT_FOOT_INDEX: i32 = 1;
const RIGHT_FOOT_INDEX: i32 = 2;
const CHEST_INDEX: i32 = 3;
const CONFIDENCE_THRESHOLD: f32 = 0.3;

fn main() -> Result<()> {
    // 設定読み込み
    let config = Config::load_or_default(CONFIG_PATH);

    println!("Tracker Sender - Phase 5");
    println!("VMT target: {}", config.vmt.addr);
    println!("Debug view: {}", if config.debug.view { "ON" } else { "OFF" });
    println!("Tracker: scale=({}, {}), mirror_x={}, offset_y={}",
        config.tracker.scale_x, config.tracker.scale_y,
        config.tracker.mirror_x, config.tracker.offset_y);
    println!("Smooth: position={}, rotation={}",
        config.smooth.position, config.smooth.rotation);
    println!();
    println!("操作: [C] キャリブレーション  [Esc] 終了");
    println!();

    // 初期化
    let mut camera = OpenCvCamera::open_with_resolution(
        config.camera.index,
        Some(config.camera.width),
        Some(config.camera.height),
    )?;
    let (width, height) = camera.resolution();
    println!("Camera: {}x{}", width, height);

    let mut detector = PoseDetector::new(MODEL_PATH)?;
    println!("Model loaded");

    let mut body_tracker = BodyTracker::from_config(&config.tracker);
    let mut smoothers = [
        Smoother::from_config(&config.smooth),
        Smoother::from_config(&config.smooth),
        Smoother::from_config(&config.smooth),
        Smoother::from_config(&config.smooth),
    ];
    let vmt = VmtClient::new(&config.vmt.addr)?;
    println!("VMT client ready");

    // デバッグ表示用
    let mut renderer = if config.debug.view {
        Some(MinifbRenderer::new("Tracker Debug", width as usize, height as usize)?)
    } else {
        None
    };

    // FPS計測用
    let mut frame_count = 0u32;
    let mut fps_timer = Instant::now();
    let mut send_count = 0u32;

    // 各段階の累積時間（ms）
    let mut t_camera = 0.0f64;
    let mut t_preprocess = 0.0f64;
    let mut t_inference = 0.0f64;
    let mut t_render = 0.0f64;
    let mut t_tracker = 0.0f64;

    // キャリブレーション用カウントダウン
    let mut calibration_deadline: Option<Instant> = None;
    const CALIBRATION_DELAY_SECS: u64 = 5;

    loop {
        // デバッグウィンドウが閉じられたら終了
        if let Some(ref r) = renderer {
            if !r.is_open() {
                break;
            }
        }

        // フレーム取得
        let t0 = Instant::now();
        let frame = match camera.read_frame() {
            Ok(f) => f,
            Err(e) => {
                eprintln!("Frame error: {}", e);
                continue;
            }
        };
        let t1 = Instant::now();

        // 前処理
        let input = preprocess_for_movenet(&frame)?;
        let t2 = Instant::now();

        // 推論
        let pose = detector.detect(input)?;
        let t3 = Instant::now();

        // キャリブレーション（Cキーで5秒カウントダウン開始）
        if let Some(ref r) = renderer {
            if r.is_key_pressed(Key::C) {
                let deadline = Instant::now() + std::time::Duration::from_secs(CALIBRATION_DELAY_SECS);
                calibration_deadline = Some(deadline);
                println!("Calibration in {}s... 基準位置に立ってください", CALIBRATION_DELAY_SECS);
            }
        }

        // カウントダウン中 → 時間が来たら実行
        if let Some(deadline) = calibration_deadline {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                if body_tracker.calibrate(&pose) {
                    for s in &mut smoothers {
                        s.reset();
                    }
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

        // トラッカー変換 & 平滑化 & 送信
        let body_poses = body_tracker.compute(&pose);
        let tracker_data = [
            (HIP_INDEX, body_poses.hip),
            (LEFT_FOOT_INDEX, body_poses.left_foot),
            (RIGHT_FOOT_INDEX, body_poses.right_foot),
            (CHEST_INDEX, body_poses.chest),
        ];
        for (i, (index, pose_opt)) in tracker_data.iter().enumerate() {
            if let Some(p) = pose_opt {
                let smoothed = smoothers[i].apply(*p);
                vmt.send(*index, 1, &smoothed)?;
                send_count += 1;
            }
        }
        if frame_count == 0 {
            if let Some(ref hip) = body_poses.hip {
                eprint!("Hip: [{:.2}, {:.2}, {:.2}]", hip.position[0], hip.position[1], hip.position[2]);
            }
            let active = [body_poses.hip.is_some(), body_poses.left_foot.is_some(), body_poses.right_foot.is_some(), body_poses.chest.is_some()];
            let count = active.iter().filter(|&&x| x).count();
            eprintln!(" Active: {}/4{}", count, if body_tracker.is_calibrated() { " [CAL]" } else { "" });
        }

        let t5 = Instant::now();

        // 累積
        t_camera += (t1 - t0).as_secs_f64() * 1000.0;
        t_preprocess += (t2 - t1).as_secs_f64() * 1000.0;
        t_inference += (t3 - t2).as_secs_f64() * 1000.0;
        t_render += (t4 - t3).as_secs_f64() * 1000.0;
        t_tracker += (t5 - t4).as_secs_f64() * 1000.0;

        // FPS表示
        frame_count += 1;
        let elapsed = fps_timer.elapsed().as_secs_f32();
        if elapsed >= 1.0 {
            let n = frame_count as f64;
            println!("FPS: {:.1} | camera {:.1}ms  preprocess {:.1}ms  inference {:.1}ms  render {:.1}ms  tracker {:.1}ms",
                frame_count as f32 / elapsed,
                t_camera / n, t_preprocess / n, t_inference / n, t_render / n, t_tracker / n);
            frame_count = 0;
            send_count = 0;
            fps_timer = Instant::now();
            t_camera = 0.0;
            t_preprocess = 0.0;
            t_inference = 0.0;
            t_render = 0.0;
            t_tracker = 0.0;
        }
    }

    println!("Shutting down...");
    Ok(())
}
