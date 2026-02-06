use anyhow::Result;
use std::time::Instant;

use talava_tracker::camera::OpenCvCamera;
use talava_tracker::pose::{preprocess_for_movenet, PoseDetector};
use talava_tracker::render::MinifbRenderer;

/// 信頼度の閾値
const CONFIDENCE_THRESHOLD: f32 = 0.3;

/// モデルファイルのパス
const MODEL_PATH: &str = "models/movenet_lightning.onnx";

fn main() -> Result<()> {
    println!("Pose Viewer - Phase 2");
    println!("Press ESC to exit");

    // カメラを開く（640x480で高速化）
    println!("Opening camera...");
    let mut camera = OpenCvCamera::open_with_resolution(0, Some(640), Some(480))?;
    let (width, height) = camera.resolution();
    println!("Camera resolution: {}x{}", width, height);

    // モデルを読み込む
    println!("Loading model from {}...", MODEL_PATH);
    let mut detector = PoseDetector::new(MODEL_PATH)?;
    println!("Model loaded");

    // ウィンドウを作成
    let mut renderer = MinifbRenderer::new("Pose Viewer", width as usize, height as usize)?;

    // FPS計測用
    let mut frame_count = 0u32;
    let mut fps_timer = Instant::now();

    // メインループ
    while renderer.is_open() {
        let frame_start = Instant::now();

        // フレームを取得
        let frame = match camera.read_frame() {
            Ok(f) => f,
            Err(e) => {
                eprintln!("Frame capture error: {}", e);
                continue;
            }
        };

        // 前処理
        let input = preprocess_for_movenet(&frame)?;

        // 推論
        let pose = detector.detect(input)?;

        // 描画
        renderer.draw_frame(&frame)?;
        renderer.draw_pose(&pose, CONFIDENCE_THRESHOLD);
        renderer.update()?;

        // FPS計算
        frame_count += 1;
        let elapsed = fps_timer.elapsed().as_secs_f32();
        if elapsed >= 1.0 {
            let fps = frame_count as f32 / elapsed;
            println!(
                "FPS: {:.1}, Avg confidence: {:.2}",
                fps,
                pose.average_confidence()
            );
            frame_count = 0;
            fps_timer = Instant::now();
        }

        // フレームレート制限なし（最大速度で実行）
        let _ = frame_start;
    }

    println!("Shutting down...");
    Ok(())
}
