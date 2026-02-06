use anyhow::{Context, Result};
use ndarray::Array4;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;
use std::path::Path;

use super::keypoint::{Keypoint, KeypointIndex, Pose};

/// MoveNet を使用した姿勢検出器
pub struct PoseDetector {
    session: Session,
}

impl PoseDetector {
    /// ONNXモデルを読み込んで初期化
    pub fn new<P: AsRef<Path>>(model_path: P) -> Result<Self> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .commit_from_file(model_path.as_ref())
            .context("Failed to load ONNX model")?;

        Ok(Self { session })
    }

    /// 前処理済みテンソルから姿勢を検出
    ///
    /// 入力: [1, 192, 192, 3] の f32 テンソル
    /// 出力: Pose (17キーポイント)
    pub fn detect(&mut self, input: Array4<f32>) -> Result<Pose> {
        let input_tensor = Tensor::from_array(input)?;
        let outputs = self
            .session
            .run(ort::inputs!["serving_default_input_0" => input_tensor])
            .context("Inference failed")?;

        // MoveNet の出力は [1, 1, 17, 3] (y, x, confidence)
        let output: ndarray::ArrayViewD<f32> = outputs["StatefulPartitionedCall_0"]
            .try_extract_array()
            .context("Failed to extract output tensor")?;

        let mut keypoints = [Keypoint::default(); KeypointIndex::COUNT];

        for i in 0..KeypointIndex::COUNT {
            let y = output[[0, 0, i, 0]];
            let x = output[[0, 0, i, 1]];
            let confidence = output[[0, 0, i, 2]];

            keypoints[i] = Keypoint::new(x, y, confidence);
        }

        Ok(Pose::new(keypoints))
    }
}
