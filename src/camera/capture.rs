use anyhow::{Context, Result};
use opencv::{
    core::Mat,
    prelude::*,
    videoio::{self, VideoCapture, VideoCaptureAPIs, VideoCaptureTrait},
};

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
