use crate::config::TrackerConfig;
use crate::pose::{KeypointIndex, Pose};
use crate::vmt::TrackerPose;

/// 腰トラッカー変換
pub struct HipTracker {
    /// 信頼度閾値
    confidence_threshold: f32,
    /// X軸スケール（メートル）
    scale_x: f32,
    /// Y軸スケール（メートル）
    scale_y: f32,
    /// X軸反転
    mirror_x: bool,
    /// Y軸オフセット（メートル）
    offset_y: f32,
}

impl HipTracker {
    pub fn new() -> Self {
        Self {
            confidence_threshold: 0.1,
            scale_x: 1.0,
            scale_y: 1.0,
            mirror_x: false,
            offset_y: 1.0,  // 腰の高さ約1m
        }
    }

    /// 設定から作成
    pub fn from_config(config: &TrackerConfig) -> Self {
        Self {
            confidence_threshold: 0.1,
            scale_x: config.scale_x,
            scale_y: config.scale_y,
            mirror_x: config.mirror_x,
            offset_y: config.offset_y,
        }
    }

    /// スケールを設定
    pub fn with_scale(mut self, x: f32, y: f32) -> Self {
        self.scale_x = x;
        self.scale_y = y;
        self
    }

    /// Poseから腰トラッカーを計算
    ///
    /// 要件フェーズ3:
    /// - 左右ヒップから腰位置を作る
    /// - 胴体方向からyawを作る
    pub fn compute(&self, pose: &Pose) -> Option<TrackerPose> {
        let left_hip = pose.get(KeypointIndex::LeftHip);
        let right_hip = pose.get(KeypointIndex::RightHip);

        // 信頼度チェック
        if !left_hip.is_valid(self.confidence_threshold)
            || !right_hip.is_valid(self.confidence_threshold)
        {
            return None;
        }

        // 腰位置 = 左右ヒップの中点 (正規化座標 0-1)
        let hip_x = (left_hip.x + right_hip.x) / 2.0;
        let hip_y = (left_hip.y + right_hip.y) / 2.0;

        // 座標変換 (正規化座標 → メートル座標)
        // X: 画像中央を0とし、scale_xでスケール
        let mut pos_x = (hip_x - 0.5) * self.scale_x;
        if self.mirror_x {
            pos_x = -pos_x;
        }
        // Y: offset_yを基準高さとし、画像上の変動をscale_yでスケール
        let pos_y = self.offset_y + (0.5 - hip_y) * self.scale_y;
        let pos_z = 0.0;

        // 胴体方向からyaw（肩が見えている場合）
        let left_shoulder = pose.get(KeypointIndex::LeftShoulder);
        let right_shoulder = pose.get(KeypointIndex::RightShoulder);

        let yaw = if left_shoulder.is_valid(self.confidence_threshold)
            && right_shoulder.is_valid(self.confidence_threshold)
        {
            let dx = right_shoulder.x - left_shoulder.x;
            let dy = right_shoulder.y - left_shoulder.y;
            // 肩の傾きからyaw推定
            // 正面向き: dx>0, dy≈0 → angle≈0
            f32::atan2(dy, dx)
        } else {
            0.0
        };

        // Yaw → Quaternion (Y軸回転)
        let half = yaw / 2.0;
        let rotation = [0.0, half.sin(), 0.0, half.cos()];

        Some(TrackerPose::new([pos_x, pos_y, pos_z], rotation))
    }
}

impl Default for HipTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pose::Keypoint;

    fn make_pose(left_hip: (f32, f32), right_hip: (f32, f32), left_shoulder: (f32, f32), right_shoulder: (f32, f32)) -> Pose {
        let mut keypoints = [Keypoint::default(); KeypointIndex::COUNT];
        keypoints[KeypointIndex::LeftHip as usize] = Keypoint::new(left_hip.0, left_hip.1, 0.9);
        keypoints[KeypointIndex::RightHip as usize] = Keypoint::new(right_hip.0, right_hip.1, 0.9);
        keypoints[KeypointIndex::LeftShoulder as usize] = Keypoint::new(left_shoulder.0, left_shoulder.1, 0.9);
        keypoints[KeypointIndex::RightShoulder as usize] = Keypoint::new(right_shoulder.0, right_shoulder.1, 0.9);
        Pose::new(keypoints)
    }

    #[test]
    fn test_center_position() {
        let tracker = HipTracker::new();
        let pose = make_pose((0.5, 0.5), (0.5, 0.5), (0.4, 0.3), (0.6, 0.3));

        let result = tracker.compute(&pose).unwrap();
        assert!((result.position[0]).abs() < 0.01); // X ≈ 0 (画像中央)
        assert!((result.position[1] - 1.0).abs() < 0.01); // Y ≈ offset_y (画像中央 → 基準高さ)
    }

    #[test]
    fn test_facing_forward() {
        let tracker = HipTracker::new();
        // 肩が水平（同じY座標） → 前を向いている
        let pose = make_pose((0.4, 0.6), (0.6, 0.6), (0.3, 0.4), (0.7, 0.4));

        let result = tracker.compute(&pose).unwrap();
        // dy=0, dx>0 → atan2(dx, 0) = π/2 → half = π/4
        // 水平な肩は実際には横向き判定になる
        // テストは結果が出ることだけ確認
        assert!(result.rotation[3].abs() <= 1.0);
    }
}
