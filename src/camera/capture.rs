use anyhow::{Context, Result};
use opencv::{
    core::Mat,
    prelude::*,
    videoio::{self, VideoCapture, VideoCaptureAPIs, VideoCaptureTrait},
};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

/// OpenCVを使用したカメラキャプチャ
pub struct OpenCvCamera {
    capture: VideoCapture,
    width: u32,
    height: u32,
}

impl OpenCvCamera {
    /// カメラを開く（デフォルトカメラ: index 0）
    pub fn open(index: i32) -> Result<Self> {
        Self::open_with_resolution(index, None, None)
    }

    /// 解像度とFPSを指定してカメラを開く
    pub fn open_with_resolution(index: i32, width: Option<u32>, height: Option<u32>) -> Result<Self> {
        Self::open_with_config(index, width, height, Some(60))
    }

    /// 解像度とFPSを指定してカメラを開く
    pub fn open_with_config(index: i32, width: Option<u32>, height: Option<u32>, fps: Option<u32>) -> Result<Self> {
        let mut capture =
            VideoCapture::new(index, VideoCaptureAPIs::CAP_ANY as i32).context("Failed to open camera")?;

        if !capture.is_opened()? {
            anyhow::bail!("Camera {} is not available", index);
        }

        // 解像度を設定
        if let Some(w) = width {
            capture.set(videoio::CAP_PROP_FRAME_WIDTH, w as f64)?;
        }
        if let Some(h) = height {
            capture.set(videoio::CAP_PROP_FRAME_HEIGHT, h as f64)?;
        }
        if let Some(f) = fps {
            capture.set(videoio::CAP_PROP_FPS, f as f64)?;
        }
        capture.set(videoio::CAP_PROP_BUFFERSIZE, 1.0)?;

        let actual_width = capture.get(videoio::CAP_PROP_FRAME_WIDTH)? as u32;
        let actual_height = capture.get(videoio::CAP_PROP_FRAME_HEIGHT)? as u32;
        let actual_fps = capture.get(videoio::CAP_PROP_FPS)?;
        println!("Camera FPS: {}", actual_fps);

        Ok(Self {
            capture,
            width: actual_width,
            height: actual_height,
        })
    }

    /// 解像度を取得
    pub fn resolution(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// フレームを読み込む（BGR形式）
    pub fn read_frame(&mut self) -> Result<Mat> {
        let mut frame = Mat::default();
        self.capture
            .read(&mut frame)
            .context("Failed to read frame")?;

        if frame.empty() {
            anyhow::bail!("Empty frame received");
        }

        Ok(frame)
    }
}

/// 別スレッドでカメラキャプチャを行い、最新フレームを提供する
pub struct ThreadedCamera {
    latest: Arc<Mutex<Option<Mat>>>,
    frame_id: Arc<AtomicU64>,
    width: u32,
    height: u32,
    _handle: thread::JoinHandle<()>,
}

impl ThreadedCamera {
    pub fn start(index: i32, width: Option<u32>, height: Option<u32>) -> Result<Self> {
        let mut camera = OpenCvCamera::open_with_resolution(index, width, height)?;
        let (w, h) = camera.resolution();
        let latest = Arc::new(Mutex::new(None::<Mat>));
        let latest_ref = latest.clone();
        let frame_id = Arc::new(AtomicU64::new(0));
        let frame_id_ref = frame_id.clone();

        let handle = thread::spawn(move || {
            loop {
                match camera.read_frame() {
                    Ok(frame) => {
                        *latest_ref.lock().unwrap() = Some(frame);
                        frame_id_ref.fetch_add(1, Ordering::Release);
                    }
                    Err(_) => {}
                }
            }
        });

        Ok(Self {
            latest,
            frame_id,
            width: w,
            height: h,
            _handle: handle,
        })
    }

    pub fn resolution(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// 現在のフレームIDを取得。新フレームが到着するたびにインクリメントされる。
    pub fn frame_id(&self) -> u64 {
        self.frame_id.load(Ordering::Acquire)
    }

    /// 最新フレームを取得。フレームは保持されるので何度でも取得可能。
    /// カメラスレッドが新フレームを書き込むまで同じフレームが返る。
    /// 初回フレーム到着前のみNone。
    pub fn get_frame(&self) -> Option<Mat> {
        let guard = self.latest.lock().unwrap();
        guard.as_ref().map(|m| m.clone())
    }
}
