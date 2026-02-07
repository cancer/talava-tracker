use anyhow::Result;
use std::time::Instant;

use talava_tracker::camera::OpenCvCamera;
use talava_tracker::config::Config;
use talava_tracker::pose::{self, preprocess_for_movenet, PoseDetector};
use talava_tracker::render::{Key, MinifbRenderer};
use talava_tracker::tracker::HipTracker;
use talava_tracker::vmt::VmtClient;

const MODEL_PATH: &str = "models/movenet_lightning.onnx";
const CONFIG_PATH: &str = "config.toml";
const HIP_TRACKER_INDEX: i32 = 0;
const CONFIDENCE_THRESHOLD: f32 = 0.3;

fn main() -> Result<()> {
    // 設定読み込み
    let config = Config::load_or_default(CONFIG_PATH);

    println!("Tracker Sender - Phase 3");
    println!("VMT target: {}", config.vmt.addr);
    println!("Debug view: {}", if config.debug.view { "ON" } else { "OFF" });
    println!("Tracker: scale=({}, {}), mirror_x={}, offset_y={}",
        config.tracker.scale_x, config.tracker.scale_y,
        config.tracker.mirror_x, config.tracker.offset_y);
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

    let mut hip_tracker = HipTracker::from_config(&config.tracker);
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
        let frame = match camera.read_frame() {
            Ok(f) => f,
            Err(e) => {
                eprintln!("Frame error: {}", e);
                continue;
            }
        };

        // 推論
        let input = preprocess_for_movenet(&frame)?;
        let pose = detector.detect(input)?;

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
                if hip_tracker.calibrate(&pose) {
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

        // トラッカー変換 & 送信
        if let Some(tracker_pose) = hip_tracker.compute(&pose) {
            if frame_count == 0 {
                eprintln!("Pos: [{:.2}, {:.2}, {:.2}] Rot: [{:.2}, {:.2}, {:.2}, {:.2}]{}",
                    tracker_pose.position[0], tracker_pose.position[1], tracker_pose.position[2],
                    tracker_pose.rotation[0], tracker_pose.rotation[1], tracker_pose.rotation[2], tracker_pose.rotation[3],
                    if hip_tracker.is_calibrated() { " [CAL]" } else { "" });
            }
            vmt.send(HIP_TRACKER_INDEX, 1, &tracker_pose)?;
            send_count += 1;
        } else if frame_count == 0 {
            let lh = pose.get(pose::KeypointIndex::LeftHip);
            let rh = pose.get(pose::KeypointIndex::RightHip);
            eprintln!("Hip confidence: L={:.2} R={:.2}", lh.confidence, rh.confidence);
        }

        // FPS表示
        frame_count += 1;
        let elapsed = fps_timer.elapsed().as_secs_f32();
        if elapsed >= 1.0 {
            println!("FPS: {:.1}, Sent: {}", frame_count as f32 / elapsed, send_count);
            frame_count = 0;
            send_count = 0;
            fps_timer = Instant::now();
        }
    }

    println!("Shutting down...");
    Ok(())
}
