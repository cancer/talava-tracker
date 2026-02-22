//! Camera server: captures frames from local cameras and sends JPEG-compressed
//! frames over TCP to the inference server.
//!
//! Only imports from `talava_tracker::protocol`. All other functionality is inline.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context, Result};
use opencv::core::{AlgorithmHint, Mat, Vector};
use opencv::prelude::*;
use opencv::videoio::{self, VideoCapture, CAP_ANY};
use opencv::{imgcodecs, imgproc};
use serde::Deserialize;

use talava_tracker::protocol::{
    self, CalibrationData, ClientMessage, Frame, MessageStream, ServerMessage,
};

// ---------------------------------------------------------------------------
// Config (inline, reads camera_server.toml)
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct Config {
    calibration_file: String,
    server_addr: String,
    #[serde(default = "default_jpeg_quality")]
    jpeg_quality: i32,
}

fn default_jpeg_quality() -> i32 {
    80
}

// ---------------------------------------------------------------------------
// Camera capture thread
// ---------------------------------------------------------------------------

struct CameraThread {
    camera_index: i32,
    width: u32,
    height: u32,
    latest_frame: Arc<Mutex<Option<Mat>>>,
}

fn spawn_camera_thread(
    camera_index: i32,
    width: u32,
    height: u32,
    running: Arc<AtomicBool>,
) -> Result<CameraThread> {
    let latest_frame: Arc<Mutex<Option<Mat>>> = Arc::new(Mutex::new(None));
    let frame_ref = Arc::clone(&latest_frame);

    // Open camera on the current thread to report errors immediately
    let mut cap = VideoCapture::new(camera_index, CAP_ANY)
        .with_context(|| format!("failed to open camera {camera_index}"))?;
    cap.set(videoio::CAP_PROP_FRAME_WIDTH, width as f64)?;
    cap.set(videoio::CAP_PROP_FRAME_HEIGHT, height as f64)?;

    if !cap.is_opened()? {
        bail!("camera {camera_index} is not opened");
    }

    eprintln!(
        "[cam{}] opened (requested {}x{})",
        camera_index, width, height
    );

    std::thread::spawn(move || {
        let target_interval = Duration::from_millis(33); // ~30fps
        while running.load(Ordering::Relaxed) {
            let start = Instant::now();
            let mut frame = Mat::default();
            match cap.read(&mut frame) {
                Ok(true) if !frame.empty() => {
                    let mut lock = frame_ref.lock().unwrap();
                    *lock = Some(frame);
                }
                Ok(_) => {
                    std::thread::sleep(Duration::from_millis(5));
                }
                Err(e) => {
                    eprintln!("[cam{}] read error: {e}", camera_index);
                    std::thread::sleep(Duration::from_millis(100));
                }
            }
            let elapsed = start.elapsed();
            if elapsed < target_interval {
                std::thread::sleep(target_interval - elapsed);
            }
        }
    });

    Ok(CameraThread {
        camera_index,
        width,
        height,
        latest_frame,
    })
}

// ---------------------------------------------------------------------------
// JPEG compression
// ---------------------------------------------------------------------------

fn jpeg_encode(frame: &Mat, quality: i32) -> Result<Vec<u8>> {
    let params = Vector::from_iter([imgcodecs::IMWRITE_JPEG_QUALITY, quality]);
    let mut buf: Vector<u8> = Vector::new();

    // imencode expects BGR 8UC3; convert BGRA if needed
    let mat = if frame.channels() == 4 {
        let mut bgr = Mat::default();
        imgproc::cvt_color(frame, &mut bgr, imgproc::COLOR_BGRA2BGR, 0, AlgorithmHint::ALGO_HINT_DEFAULT)?;
        bgr
    } else {
        frame.clone()
    };

    imgcodecs::imencode(".jpg", &mat, &mut buf, &params)?;
    Ok(buf.to_vec())
}

// ---------------------------------------------------------------------------
// Frame sync: wait until all cameras have a frame (50ms timeout)
// ---------------------------------------------------------------------------

fn collect_frames(cameras: &[CameraThread], timeout: Duration) -> Option<Vec<(i32, u32, u32, Mat)>> {
    let deadline = Instant::now() + timeout;
    let mut frames: Vec<Option<(i32, u32, u32, Mat)>> = vec![None; cameras.len()];

    loop {
        let mut all_ready = true;
        for (i, cam) in cameras.iter().enumerate() {
            if frames[i].is_none() {
                let lock = cam.latest_frame.lock().unwrap();
                if let Some(ref f) = *lock {
                    frames[i] = Some((cam.camera_index, cam.width, cam.height, f.clone()));
                } else {
                    all_ready = false;
                }
            }
        }
        if all_ready {
            return Some(frames.into_iter().map(|f| f.unwrap()).collect());
        }
        if Instant::now() >= deadline {
            return None;
        }
        std::thread::sleep(Duration::from_millis(2));
    }
}

// ---------------------------------------------------------------------------
// TCP client session
// ---------------------------------------------------------------------------

async fn run_session(
    stream: &mut MessageStream,
    calibration: &CalibrationData,
    cameras: &[CameraThread],
    jpeg_quality: i32,
    trigger_calibration: &AtomicBool,
) -> Result<()> {
    // 1. Send calibration
    protocol::send_message(
        stream,
        &ClientMessage::CameraCalibration {
            data: calibration.clone(),
        },
    )
    .await?;
    eprintln!("[tcp] sent CameraCalibration");

    // 2. Wait for Ack
    let ack: ServerMessage = protocol::recv_message(stream).await?;
    match ack {
        ServerMessage::CameraCalibrationAck { ok: true, .. } => {
            eprintln!("[tcp] calibration acknowledged");
        }
        ServerMessage::CameraCalibrationAck {
            ok: false, error, ..
        } => {
            bail!("server rejected calibration: {}", error.unwrap_or_default());
        }
        other => bail!("expected CameraCalibrationAck, got {other:?}"),
    }

    // 3. Wait for Ready
    let ready: ServerMessage = protocol::recv_message(stream).await?;
    match ready {
        ServerMessage::Ready => eprintln!("[tcp] server ready, starting stream"),
        other => bail!("expected Ready, got {other:?}"),
    }

    // 4. Stream frames
    let mut fps_counter: u32 = 0;
    let mut fps_timer = Instant::now();
    let frame_timeout = Duration::from_millis(50);

    loop {
        // Check pose calibration trigger
        if trigger_calibration.swap(false, Ordering::Relaxed) {
            eprintln!("[tcp] sending TriggerPoseCalibration");
            protocol::send_message(stream, &ClientMessage::TriggerPoseCalibration).await?;
        }

        // Collect frames from all cameras
        let raw_frames = match collect_frames(cameras, frame_timeout) {
            Some(f) => f,
            None => continue,
        };

        let timestamp_us = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64;

        // JPEG encode
        let mut frames = Vec::with_capacity(raw_frames.len());
        for (cam_id, w, h, mat) in &raw_frames {
            let jpeg_data = jpeg_encode(mat, jpeg_quality)?;
            frames.push(Frame {
                camera_id: *cam_id as u8,
                width: *w as u16,
                height: *h as u16,
                jpeg_data,
            });
        }

        protocol::send_message(
            stream,
            &ClientMessage::FrameSet {
                timestamp_us,
                frames,
            },
        )
        .await?;

        fps_counter += 1;
        if fps_timer.elapsed() >= Duration::from_secs(1) {
            eprintln!("[fps] {fps_counter}");
            fps_counter = 0;
            fps_timer = Instant::now();
        }
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> Result<()> {
    // Prevent macOS sleep
    let pid = std::process::id();
    let _ = std::process::Command::new("caffeinate")
        .args(["-d", "-i", "-w", &pid.to_string()])
        .spawn();

    // Load config
    let config_str =
        std::fs::read_to_string("camera_server.toml").context("failed to read camera_server.toml")?;
    let config: Config = toml::from_str(&config_str)?;
    eprintln!(
        "[config] server_addr={}, jpeg_quality={}",
        config.server_addr, config.jpeg_quality
    );

    // Load calibration
    let calib_str = std::fs::read_to_string(&config.calibration_file)
        .with_context(|| format!("failed to read {}", config.calibration_file))?;
    let calibration: CalibrationData = serde_json::from_str(&calib_str)?;
    eprintln!("[calibration] loaded {} cameras", calibration.cameras.len());

    // Running flag for camera threads
    let running = Arc::new(AtomicBool::new(true));

    // Spawn camera threads
    let mut cameras = Vec::new();
    for cam in &calibration.cameras {
        let ct = spawn_camera_thread(cam.camera_index, cam.width, cam.height, Arc::clone(&running))?;
        cameras.push(ct);
    }

    // SIGUSR1 → trigger pose calibration
    let trigger_calibration = Arc::new(AtomicBool::new(false));
    {
        let flag = Arc::clone(&trigger_calibration);
        signal_hook::flag::register(signal_hook::consts::SIGUSR1, flag)?;
    }

    // Console input thread: 'p' + Enter → trigger pose calibration
    {
        let flag = Arc::clone(&trigger_calibration);
        std::thread::spawn(move || {
            let stdin = std::io::stdin();
            let mut line = String::new();
            loop {
                line.clear();
                if stdin.read_line(&mut line).is_ok() && line.trim() == "p" {
                    eprintln!("[input] pose calibration triggered");
                    flag.store(true, Ordering::Relaxed);
                }
            }
        });
    }

    // Main loop: connect, stream, reconnect on error
    loop {
        eprintln!("[tcp] connecting to {}...", config.server_addr);
        match tokio::net::TcpStream::connect(&config.server_addr).await {
            Ok(tcp) => {
                eprintln!("[tcp] connected");
                let mut stream = protocol::message_stream(tcp);
                if let Err(e) = run_session(
                    &mut stream,
                    &calibration,
                    &cameras,
                    config.jpeg_quality,
                    &trigger_calibration,
                )
                .await
                {
                    eprintln!("[tcp] session error: {e:#}");
                }
            }
            Err(e) => {
                eprintln!("[tcp] connection failed: {e}");
            }
        }
        eprintln!("[tcp] reconnecting in 2s...");
        tokio::time::sleep(Duration::from_secs(2)).await;
    }
}
