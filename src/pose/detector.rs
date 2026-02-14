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
    RTMW3D,
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
            ModelType::RTMW3D => self.detect_rtmw3d(input),
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

    /// RTMW3D: 133キーポイント(COCO-WholeBody)からCOCO17+足をマッピング
    /// SimCC3D出力: simcc_x [1,133,576], simcc_y [1,133,768], simcc_z [1,133,576]
    ///
    /// RTMW3D出力順: 0-16 body, 17-22 feet, 23-90 face, 91-132 hands
    /// 我々の順序: 0-16 COCO body, 17 Head, 18 Neck, 19 Hip, 20-25 feet, 26-36 spine等
    fn detect_rtmw3d(&mut self, input: Array4<f32>) -> Result<Pose> {
        let input_tensor = Tensor::from_array(input)?;
        let outputs = self
            .session
            .run(ort::inputs!["input" => input_tensor])
            .context("Inference failed")?;

        let simcc_x: ndarray::ArrayViewD<f32> = outputs["output"]
            .try_extract_array()
            .context("Failed to extract simcc_x")?;
        let simcc_y: ndarray::ArrayViewD<f32> = outputs["1554"]
            .try_extract_array()
            .context("Failed to extract simcc_y")?;
        let simcc_z: ndarray::ArrayViewD<f32> = outputs["1556"]
            .try_extract_array()
            .context("Failed to extract simcc_z")?;

        const SIMCC_SPLIT_RATIO: f32 = 2.0;
        const MODEL_W: f32 = 288.0;
        const MODEL_H: f32 = 384.0;
        const Z_RANGE: f32 = 2.1744869;

        // 133キーポイント全てをデコード
        let mut raw = [(0.0f32, 0.0f32, 0.0f32, 0.0f32); 133]; // (x, y, z, confidence)
        for i in 0..133 {
            let mut max_x_val = f32::NEG_INFINITY;
            let mut max_x_idx = 0usize;
            for j in 0..576 {
                let v = simcc_x[[0, i, j]];
                if v > max_x_val {
                    max_x_val = v;
                    max_x_idx = j;
                }
            }

            let mut max_y_val = f32::NEG_INFINITY;
            let mut max_y_idx = 0usize;
            for j in 0..768 {
                let v = simcc_y[[0, i, j]];
                if v > max_y_val {
                    max_y_val = v;
                    max_y_idx = j;
                }
            }

            let mut max_z_idx = 0usize;
            let mut max_z_val = f32::NEG_INFINITY;
            for j in 0..576 {
                let v = simcc_z[[0, i, j]];
                if v > max_z_val {
                    max_z_val = v;
                    max_z_idx = j;
                }
            }

            let x = max_x_idx as f32 / SIMCC_SPLIT_RATIO / MODEL_W;
            let y = max_y_idx as f32 / SIMCC_SPLIT_RATIO / MODEL_H;
            let z_raw = max_z_idx as f32 / SIMCC_SPLIT_RATIO;
            let z = (z_raw / (MODEL_H / 2.0) - 1.0) * Z_RANGE;
            let confidence = max_x_val.min(max_y_val);
            let confidence = if confidence <= 0.0 { 0.0 } else { confidence };

            raw[i] = (x, y, z, confidence);
        }

        let mut keypoints = [Keypoint::default(); KeypointIndex::COUNT];

        // COCO17 body (0-16): 直接マッピング
        for i in 0..17 {
            let (x, y, z, c) = raw[i];
            keypoints[i] = Keypoint::new_3d(x, y, z, c);
        }

        // RTMW3D足キーポイント → 我々の足インデックス
        // RTMW3D: 17=LeftBigToe, 18=LeftSmallToe, 19=LeftHeel
        //         20=RightBigToe, 21=RightSmallToe, 22=RightHeel
        const FOOT_MAP: [(usize, usize); 6] = [
            (17, 20), // RTMW3D LeftBigToe → our LeftBigToe
            (18, 22), // RTMW3D LeftSmallToe → our LeftSmallToe
            (19, 24), // RTMW3D LeftHeel → our LeftHeel
            (20, 21), // RTMW3D RightBigToe → our RightBigToe
            (21, 23), // RTMW3D RightSmallToe → our RightSmallToe
            (22, 25), // RTMW3D RightHeel → our RightHeel
        ];
        for &(src, dst) in &FOOT_MAP {
            let (x, y, z, c) = raw[src];
            keypoints[dst] = Keypoint::new_3d(x, y, z, c);
        }

        // 合成キーポイント: Head = Noseから（COCO17にHeadは無いので）
        let (nx, ny, nz, nc) = raw[0]; // Nose
        keypoints[KeypointIndex::Head as usize] = Keypoint::new_3d(nx, ny, nz, nc);

        // Neck = 肩中点
        let (lsx, lsy, lsz, lsc) = raw[5]; // LeftShoulder
        let (rsx, rsy, rsz, rsc) = raw[6]; // RightShoulder
        if lsc > 0.0 && rsc > 0.0 {
            keypoints[KeypointIndex::Neck as usize] = Keypoint::new_3d(
                (lsx + rsx) / 2.0, (lsy + rsy) / 2.0, (lsz + rsz) / 2.0,
                lsc.min(rsc),
            );
        }

        // Hip = 腰中点
        let (lhx, lhy, lhz, lhc) = raw[11]; // LeftHip
        let (rhx, rhy, rhz, rhc) = raw[12]; // RightHip
        if lhc > 0.0 && rhc > 0.0 {
            keypoints[KeypointIndex::Hip as usize] = Keypoint::new_3d(
                (lhx + rhx) / 2.0, (lhy + rhy) / 2.0, (lhz + rhz) / 2.0,
                lhc.min(rhc),
            );
        }

        // Spine等 (26-36): RTMW3Dには対応なし → default (confidence=0) のまま

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
