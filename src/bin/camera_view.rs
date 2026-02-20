use anyhow::Result;
use opencv::core::{Mat, Point, Scalar, Size};
use opencv::imgproc;
use opencv::prelude::VideoCaptureTraitConst;
use opencv::videoio::{VideoCapture, VideoCaptureAPIs};
use std::process::Command;
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use talava_tracker::camera::capture::ThreadedCamera;
use talava_tracker::render::window::MinifbRenderer;

fn main() -> Result<()> {
    println!("=== カメラビュー（画角調整モード） ===");
    println!("カメラ検出中...");

    let device_names = get_camera_names();
    let indices = probe_cameras(10, Duration::from_secs(3));
    if indices.is_empty() {
        println!("カメラが見つかりません");
        return Ok(());
    }

    // コンソールにindex + 機種名を表示
    for &idx in &indices {
        let name = device_names
            .get(idx as usize)
            .map(|s| s.as_str())
            .unwrap_or("(unknown)");
        println!("  index {}: {}", idx, name);
    }

    // カメラ起動
    let mut cameras = Vec::new();
    let mut cam_indices = Vec::new();
    for &idx in &indices {
        match ThreadedCamera::start(idx, Some(1280), Some(960)) {
            Ok(cam) => {
                let (w, h) = cam.resolution();
                println!("cam {}: {}x{}", idx, w, h);
                cameras.push(cam);
                cam_indices.push(idx);
            }
            Err(e) => {
                println!("cam {} 起動失敗: {}", idx, e);
            }
        }
    }
    if cameras.is_empty() {
        println!("起動できたカメラがありません");
        return Ok(());
    }

    let n = cameras.len();
    let cell_w: usize = 640;
    let cell_h: usize = 480;
    let (cols, rows) = grid_layout(n);
    let win_w = cell_w * cols;
    let win_h = cell_h * rows;

    println!("レイアウト: {}x{} ({}台, {}列x{}行)", win_w, win_h, n, cols, rows);

    let mut renderer = MinifbRenderer::new("camera_view", win_w, win_h)?;

    // 初回フレーム待ち
    thread::sleep(Duration::from_millis(500));

    while renderer.is_open() {
        for (i, cam) in cameras.iter().enumerate() {
            if let Some(frame) = cam.get_frame() {
                let col = i % cols;
                let row = i / cols;
                let x_off = col * cell_w;
                let y_off = row * cell_h;

                let mut resized = Mat::default();
                imgproc::resize(
                    &frame,
                    &mut resized,
                    Size::new(cell_w as i32, cell_h as i32),
                    0.0,
                    0.0,
                    imgproc::INTER_LINEAR,
                )?;

                // index をフレームに描画
                let label = format!("cam {}", cam_indices[i]);
                // 背景（黒）で視認性確保
                imgproc::put_text(
                    &mut resized,
                    &label,
                    Point::new(12, 42),
                    imgproc::FONT_HERSHEY_SIMPLEX,
                    1.2,
                    Scalar::new(0.0, 0.0, 0.0, 0.0),
                    4,
                    imgproc::LINE_8,
                    false,
                )?;
                // 前景（白）
                imgproc::put_text(
                    &mut resized,
                    &label,
                    Point::new(12, 42),
                    imgproc::FONT_HERSHEY_SIMPLEX,
                    1.2,
                    Scalar::new(255.0, 255.0, 255.0, 0.0),
                    2,
                    imgproc::LINE_8,
                    false,
                )?;

                renderer.draw_frame_at(&resized, x_off, y_off)?;
            }
        }
        renderer.update()?;
        thread::sleep(Duration::from_millis(16));
    }

    Ok(())
}

/// system_profiler からカメラ名リストを取得（AVFoundation indexの順序に一致）
fn get_camera_names() -> Vec<String> {
    let output = match Command::new("system_profiler")
        .args(["SPCameraDataType"])
        .output()
    {
        Ok(o) => o,
        Err(_) => return Vec::new(),
    };

    let text = String::from_utf8_lossy(&output.stdout);
    let mut names = Vec::new();

    for line in text.lines() {
        let trimmed = line.trim();
        // "Camera:" の次のインデント4つのレベルがデバイス名
        // 形式: "    DeviceName:" (4スペース + 名前 + コロン)
        // "Model ID:", "Unique ID:" 等のプロパティは6スペース以上
        if trimmed.ends_with(':')
            && !trimmed.starts_with("Camera")
            && !trimmed.starts_with("Model ID")
            && !trimmed.starts_with("Unique ID")
            && line.starts_with("    ")
            && !line.starts_with("      ")
        {
            let name = trimmed.trim_end_matches(':').trim();
            // Unicode制御文字を除去（iPhoneデバイス名にLRM等が含まれる）
            let clean: String = name.chars().filter(|c| !c.is_control() && *c != '\u{200e}' && *c != '\u{200f}').collect();
            names.push(clean);
        }
    }
    names
}

/// 全indexを並列プローブしてタイムアウト内に応答したカメラを返す
fn probe_cameras(max_index: i32, timeout: Duration) -> Vec<i32> {
    let (tx, rx) = mpsc::channel();

    for i in 0..max_index {
        let tx = tx.clone();
        thread::spawn(move || {
            match VideoCapture::new(i, VideoCaptureAPIs::CAP_AVFOUNDATION as i32) {
                Ok(cap) => {
                    if cap.is_opened().unwrap_or(false) {
                        let _ = tx.send(i);
                    }
                }
                Err(_) => {}
            }
        });
    }
    drop(tx);

    let deadline = Instant::now() + timeout;
    let mut found = Vec::new();
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            break;
        }
        match rx.recv_timeout(remaining) {
            Ok(idx) => found.push(idx),
            Err(_) => break,
        }
    }
    found.sort();
    found
}

fn grid_layout(n: usize) -> (usize, usize) {
    match n {
        0 | 1 => (1, 1),
        2 => (2, 1),
        3 | 4 => (2, 2),
        5 | 6 => (3, 2),
        _ => {
            let cols = (n as f64).sqrt().ceil() as usize;
            let rows = (n + cols - 1) / cols;
            (cols, rows)
        }
    }
}
