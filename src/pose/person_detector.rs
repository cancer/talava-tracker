use anyhow::{Context, Result};
use ndarray::Array4;
use opencv::{
    core::{Mat, Size, CV_32FC3},
    imgproc,
    prelude::*,
};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;

use super::crop::BBox;

/// 人物検出の信頼度閾値
const PERSON_SCORE_THRESHOLD: f32 = 0.25;

/// YOLOv8nベースの人物検出器
pub struct PersonDetector {
    session: Session,
    input_size: i32,
}

impl PersonDetector {
    /// ONNXモデルを読み込んで初期化
    pub fn new(model_path: &str, input_size: i32) -> Result<Self> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .commit_from_file(model_path)
            .context("Failed to load person detection ONNX model")?;
        Ok(Self {
            session,
            input_size,
        })
    }

    /// フレームから人物を検出し、最もスコアの高い人物のBBoxを返す
    pub fn detect(&mut self, frame: &Mat) -> Result<Option<BBox>> {
        let frame_w = frame.cols();
        let frame_h = frame.rows();
        let input = self.preprocess(frame)?;

        // 推論
        let input_tensor = Tensor::from_array(input)?;
        let outputs = self
            .session
            .run(ort::inputs!["images" => input_tensor])
            .context("Person detection inference failed")?;

        // 出力: [1, 84, N]
        let output: ndarray::ArrayViewD<f32> = outputs["output0"]
            .try_extract_array()
            .context("Failed to extract person detection output")?;

        // 後処理: 最もスコアの高いperson検出を選択
        let n_detections = output.shape()[2];
        let mut best_score: f32 = 0.0;
        let mut best_idx: Option<usize> = None;

        for i in 0..n_detections {
            // class 0 = person
            let person_score = output[[0, 4, i]];
            if person_score > best_score && person_score >= PERSON_SCORE_THRESHOLD {
                best_score = person_score;
                best_idx = Some(i);
            }
        }

        let Some(idx) = best_idx else {
            return Ok(None);
        };

        // 座標変換: 入力サイズ基準 → フレーム座標
        let cx = output[[0, 0, idx]];
        let cy = output[[0, 1, idx]];
        let w = output[[0, 2, idx]];
        let h = output[[0, 3, idx]];

        let scale_x = frame_w as f32 / self.input_size as f32;
        let scale_y = frame_h as f32 / self.input_size as f32;

        Ok(Some(BBox {
            x: (cx - w / 2.0) * scale_x,
            y: (cy - h / 2.0) * scale_y,
            width: w * scale_x,
            height: h * scale_y,
        }))
    }

    /// BGR Mat → NCHW [1, 3, input_size, input_size] テンソルに変換
    fn preprocess(&self, frame: &Mat) -> Result<Array4<f32>> {
        let size = self.input_size;

        // BGR -> RGB
        let mut rgb = Mat::default();
        imgproc::cvt_color_def(frame, &mut rgb, imgproc::COLOR_BGR2RGB)?;

        // input_size x input_size にリサイズ
        let mut resized = Mat::default();
        imgproc::resize(
            &rgb,
            &mut resized,
            Size::new(size, size),
            0.0,
            0.0,
            imgproc::INTER_LINEAR,
        )?;

        // f32 に変換
        let mut float_mat = Mat::default();
        resized.convert_to(&mut float_mat, CV_32FC3, 1.0, 0.0)?;

        // [0, 255] → [0.0, 1.0] 正規化 & NCHW変換 [1, 3, size, size]
        let s = size as usize;
        let mut tensor = Array4::<f32>::zeros((1, 3, s, s));
        let data = float_mat.data_bytes()?;
        let step = float_mat.mat_step().get(0);
        for y in 0..s {
            let row_ptr = unsafe {
                std::slice::from_raw_parts(data.as_ptr().add(y * step) as *const f32, s * 3)
            };
            for x in 0..s {
                for c in 0..3 {
                    tensor[[0, c, y, x]] = row_ptr[x * 3 + c] / 255.0;
                }
            }
        }

        Ok(tensor)
    }
}
