//! Camera server: captures frames from local cameras and sends JPEG-compressed
//! frames over TCP to the inference server.
//!
//! Only imports from `talava_tracker::protocol`. All other functionality is inline.

use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context, Result};
use opencv::core::{Mat, Vector};
use opencv::prelude::*;
use opencv::videoio::{self, VideoCapture, CAP_ANY};
use opencv::{imgcodecs, imgproc};
use serde::Deserialize;

use futures::StreamExt as _;
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
    #[serde(default)]
    verbose: bool,
}

fn default_jpeg_quality() -> i32 {
    80
}

// ---------------------------------------------------------------------------
// Logging
// ---------------------------------------------------------------------------

type LogFile = Arc<Mutex<std::io::BufWriter<std::fs::File>>>;

fn open_log_file() -> Result<LogFile> {
    std::fs::create_dir_all("logs")?;
    let ts = chrono::Local::now().format("%Y%m%d_%H%M%S");
    let path = format!("logs/camera_{}.log", ts);
    let file = std::fs::File::create(&path)?;
    eprintln!("Log: {}", path);
    Ok(Arc::new(Mutex::new(std::io::BufWriter::new(file))))
}

macro_rules! log {
    ($logfile:expr, $($arg:tt)*) => {{
        let msg = format!($($arg)*);
        eprintln!("{}", msg);
        if let Ok(mut f) = $logfile.lock() {
            let _ = writeln!(f, "{}", msg);
            let _ = f.flush();
        }
    }};
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
        imgproc::cvt_color_def(frame, &mut bgr, imgproc::COLOR_BGRA2BGR)?;
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
                let mut lock = cam.latest_frame.lock().unwrap();
                if let Some(f) = lock.take() {
                    frames[i] = Some((cam.camera_index, cam.width, cam.height, f));
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
    stream: MessageStream,
    calibration: &CalibrationData,
    cameras: &[CameraThread],
    jpeg_quality: i32,
    trigger_calibration: &AtomicBool,
    verbose: bool,
    logfile: &LogFile,
) -> Result<()> {
    let mut stream = stream;

    // 1. Send calibration
    protocol::send_message(
        &mut stream,
        &ClientMessage::CameraCalibration {
            data: calibration.clone(),
        },
    )
    .await?;
    log!(logfile, "[tcp] sent CameraCalibration");

    // 2. Wait for Ack
    let ack: ServerMessage = protocol::recv_message(&mut stream).await?;
    match ack {
        ServerMessage::CameraCalibrationAck { ok: true, .. } => {
            log!(logfile, "[tcp] calibration acknowledged");
        }
        ServerMessage::CameraCalibrationAck {
            ok: false, error, ..
        } => {
            bail!("server rejected calibration: {}", error.unwrap_or_default());
        }
        other => bail!("expected CameraCalibrationAck, got {other:?}"),
    }

    // 3. Wait for Ready
    let ready: ServerMessage = protocol::recv_message(&mut stream).await?;
    match ready {
        ServerMessage::Ready => log!(logfile, "[tcp] server ready, starting stream"),
        other => bail!("expected Ready, got {other:?}"),
    }

    // 4. Split stream: sink for sending frames, reader for receiving server messages
    let (mut sink, mut reader) = stream.split();

    // Spawn reader task to receive ServerMessage (e.g. LogData)
    let reader_logfile = logfile.clone();
    let reader_task = tokio::spawn(async move {
        loop {
            match reader.next().await {
                Some(Ok(bytes)) => {
                    match bincode::deserialize::<ServerMessage>(&bytes) {
                        Ok(ServerMessage::LogData { filename, data }) => {
                            std::fs::create_dir_all("logs").ok();
                            let path = format!("logs/{}", filename);
                            match std::fs::write(&path, &data) {
                                Ok(_) => {
                                    let msg = format!("[tcp] received log: {} ({}KB)", path, data.len() / 1024);
                                    eprintln!("{}", msg);
                                    if let Ok(mut f) = reader_logfile.lock() {
                                        let _ = writeln!(f, "{}", msg);
                                        let _ = f.flush();
                                    }
                                }
                                Err(e) => { eprintln!("[tcp] failed to save log: {}", e); }
                            }
                        }
                        Ok(other) => { eprintln!("[tcp] unexpected server message: {:?}", other); }
                        Err(e) => { eprintln!("[tcp] deserialize error: {}", e); }
                    }
                }
                Some(Err(e)) => { eprintln!("[tcp] reader error: {}", e); break; }
                None => break,
            }
        }
    });

    // 5. Stream frames via sink
    let mut fps_counter: u32 = 0;
    let mut fps_timer = Instant::now();
    let frame_timeout = Duration::from_millis(50);
    let mut sync_sum_ms: f64 = 0.0;
    let mut encode_sum_ms: f64 = 0.0;
    let mut send_sum_ms: f64 = 0.0;

    let result: Result<()> = async {
        loop {
            // Check pose calibration trigger
            if trigger_calibration.swap(false, Ordering::Relaxed) {
                log!(logfile, "[tcp] sending TriggerPoseCalibration");
                protocol::send_to_sink(&mut sink, &ClientMessage::TriggerPoseCalibration).await?;
            }

            // Collect frames from all cameras
            let sync_start = Instant::now();
            let raw_frames = match collect_frames(cameras, frame_timeout) {
                Some(f) => f,
                None => {
                    if verbose { log!(logfile, "[verbose] frame sync timeout ({}ms)", frame_timeout.as_millis()); }
                    continue;
                }
            };
            let sync_elapsed = sync_start.elapsed();

            let timestamp_us = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_micros() as u64;

            // JPEG encode
            let encode_start = Instant::now();
            let mut frames = Vec::with_capacity(raw_frames.len());
            let mut total_payload: usize = 0;
            for (cam_id, w, h, mat) in &raw_frames {
                let t0 = Instant::now();
                let jpeg_data = jpeg_encode(mat, jpeg_quality)?;
                let jpeg_size = jpeg_data.len();
                total_payload += jpeg_size;
                if verbose {
                    let actual_rows = mat.rows();
                    let actual_cols = mat.cols();
                    let channels = mat.channels();
                    log!(logfile,
                        "[verbose] cam{}: mat={}x{}x{} jpeg={}KB ({:.1}ms)",
                        cam_id, actual_cols, actual_rows, channels,
                        jpeg_size / 1024, t0.elapsed().as_secs_f64() * 1000.0,
                    );
                }
                frames.push(Frame {
                    camera_id: *cam_id as u8,
                    width: *w as u16,
                    height: *h as u16,
                    jpeg_data,
                });
            }
            let encode_elapsed = encode_start.elapsed();

            let send_start = Instant::now();
            protocol::send_to_sink(
                &mut sink,
                &ClientMessage::FrameSet {
                    timestamp_us,
                    frames,
                },
            )
            .await?;
            let send_elapsed = send_start.elapsed();

            sync_sum_ms += sync_elapsed.as_secs_f64() * 1000.0;
            encode_sum_ms += encode_elapsed.as_secs_f64() * 1000.0;
            send_sum_ms += send_elapsed.as_secs_f64() * 1000.0;

            if verbose {
                log!(logfile,
                    "[verbose] frameset: sync={:.1}ms encode={:.1}ms send={:.1}ms payload={}KB ts={}",
                    sync_elapsed.as_secs_f64() * 1000.0,
                    encode_elapsed.as_secs_f64() * 1000.0,
                    send_elapsed.as_secs_f64() * 1000.0,
                    total_payload / 1024,
                    timestamp_us,
                );
            }

            fps_counter += 1;
            if fps_timer.elapsed() >= Duration::from_secs(1) {
                let n = fps_counter as f64;
                log!(logfile, "[fps] {} (sync={:.1}ms encode={:.1}ms send={:.1}ms)",
                    fps_counter, sync_sum_ms / n, encode_sum_ms / n, send_sum_ms / n);
                fps_counter = 0;
                fps_timer = Instant::now();
                sync_sum_ms = 0.0;
                encode_sum_ms = 0.0;
                send_sum_ms = 0.0;
            }
        }
    }.await;

    reader_task.abort();
    result
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
    let logfile = open_log_file()?;
    log!(logfile, "Camera Server ({})", env!("GIT_VERSION"));
    log!(logfile,
        "[config] server_addr={}, jpeg_quality={}, verbose={}",
        config.server_addr, config.jpeg_quality, config.verbose
    );

    // Load calibration
    let calib_str = std::fs::read_to_string(&config.calibration_file)
        .with_context(|| format!("failed to read {}", config.calibration_file))?;
    let calibration: CalibrationData = serde_json::from_str(&calib_str)?;
    log!(logfile, "[calibration] loaded {} cameras", calibration.cameras.len());

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
        log!(logfile, "[tcp] connecting to {}...", config.server_addr);
        match tokio::net::TcpStream::connect(&config.server_addr).await {
            Ok(tcp) => {
                tcp.set_nodelay(true)?;
                log!(logfile, "[tcp] connected");
                let stream = protocol::message_stream(tcp);
                if let Err(e) = run_session(
                    stream,
                    &calibration,
                    &cameras,
                    config.jpeg_quality,
                    &trigger_calibration,
                    config.verbose,
                    &logfile,
                )
                .await
                {
                    log!(logfile, "[tcp] session error: {e:#}");
                }
            }
            Err(e) => {
                log!(logfile, "[tcp] connection failed: {e}");
            }
        }
        log!(logfile, "[tcp] reconnecting in 2s...");
        tokio::time::sleep(Duration::from_secs(2)).await;
    }
}
