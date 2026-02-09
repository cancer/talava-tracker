use anyhow::{Context, Result};
use ndarray::Array4;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;
use std::path::Path;

use super::keypoint::{Keypoint, KeypointIndex, Pose};

/// モデル種別
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ModelType {
    MoveNet,
    SpinePose,
}

/// 姿勢検出器
pub struct PoseDetector {
    session: Session,
    model_type: ModelType,
}

impl PoseDetector {
    /// ONNXモデルを読み込んで初期化
    pub fn new<P: AsRef<Path>>(model_path: P, model_type: ModelType) -> Result<Self> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .commit_from_file(model_path.as_ref())
            .context("Failed to load ONNX model")?;
        Ok(Self { session, model_type })
    }

    /// 前処理済みテンソルから姿勢を検出
    pub fn detect(&mut self, input: Array4<f32>) -> Result<Pose> {
        match self.model_type {
            ModelType::MoveNet => self.detect_movenet(input),
            ModelType::SpinePose => self.detect_spinepose(input),
        }
    }

    fn detect_movenet(&mut self, input: Array4<f32>) -> Result<Pose> {
        let input_tensor = Tensor::from_array(input)?;
        let outputs = self
            .session
            .run(ort::inputs!["serving_default_input_0" => input_tensor])
            .context("Inference failed")?;

        let output: ndarray::ArrayViewD<f32> = outputs["StatefulPartitionedCall_0"]
            .try_extract_array()
            .context("Failed to extract output tensor")?;

        let mut keypoints = [Keypoint::default(); KeypointIndex::COUNT];

        // MoveNetは17キーポイントのみ、残りはconfidence=0のまま
        for i in 0..17 {
            let y = output[[0, 0, i, 0]];
            let x = output[[0, 0, i, 1]];
            let confidence = output[[0, 0, i, 2]];
            keypoints[i] = Keypoint::new(x, y, confidence);
        }

        Ok(Pose::new(keypoints))
    }

    fn detect_spinepose(&mut self, input: Array4<f32>) -> Result<Pose> {
        let input_tensor = Tensor::from_array(input)?;
        let outputs = self
            .session
            .run(ort::inputs!["input" => input_tensor])
            .context("Inference failed")?;

        // SimCC出力: simcc_x [1, 37, 384], simcc_y [1, 37, 512]
        let simcc_x: ndarray::ArrayViewD<f32> = outputs["simcc_x"]
            .try_extract_array()
            .context("Failed to extract simcc_x")?;
        let simcc_y: ndarray::ArrayViewD<f32> = outputs["simcc_y"]
            .try_extract_array()
            .context("Failed to extract simcc_y")?;

        let mut keypoints = [Keypoint::default(); KeypointIndex::COUNT];

        for i in 0..KeypointIndex::COUNT {
            // argmax for x
            let mut max_x_val = f32::NEG_INFINITY;
            let mut max_x_idx = 0usize;
            for j in 0..384 {
                let v = simcc_x[[0, i, j]];
                if v > max_x_val {
                    max_x_val = v;
                    max_x_idx = j;
                }
            }

            // argmax for y
            let mut max_y_val = f32::NEG_INFINITY;
            let mut max_y_idx = 0usize;
            for j in 0..512 {
                let v = simcc_y[[0, i, j]];
                if v > max_y_val {
                    max_y_val = v;
                    max_y_idx = j;
                }
            }

            // 正規化座標に変換 (SimCCは2倍解像度)
            let x = max_x_idx as f32 / (2.0 * 192.0);
            let y = max_y_idx as f32 / (2.0 * 256.0);

            // confidence = sigmoid(max logit) — xとyのlogitの平均
            let avg_logit = (max_x_val + max_y_val) / 2.0;
            let confidence = 1.0 / (1.0 + (-avg_logit).exp());

            keypoints[i] = Keypoint::new(x, y, confidence);
        }

        Ok(Pose::new(keypoints))
    }
}
