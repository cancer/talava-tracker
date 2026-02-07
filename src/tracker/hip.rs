use crate::config::TrackerConfig;
use crate::pose::{KeypointIndex, Pose};
use crate::vmt::TrackerPose;

/// キャリブレーションデータ
/// 基準位置・基準yawを記録し、以降の計算で差分を取る
struct Calibration {
    /// 基準腰位置X（正規化座標）
    hip_x: f32,
    /// 基準腰位置Y（正規化座標）
    hip_y: f32,
    /// 基準yaw角度
    yaw: f32,
}

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
    /// キャリブレーションデータ
    calibration: Option<Calibration>,
}

impl HipTracker {
    pub fn new() -> Self {
        Self {
            confidence_threshold: 0.1,
            scale_x: 1.0,
            scale_y: 1.0,
            mirror_x: false,
            offset_y: 1.0,
            calibration: None,
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
            calibration: None,
        }
    }

    /// スケールを設定
    pub fn with_scale(mut self, x: f32, y: f32) -> Self {
        self.scale_x = x;
        self.scale_y = y;
        self
    }

    /// キャリブレーション実行
    /// 現在のポーズを基準点として記録する
    /// 成功時: true、失敗時（信頼度不足）: false
    pub fn calibrate(&mut self, pose: &Pose) -> bool {
        let left_hip = pose.get(KeypointIndex::LeftHip);
        let right_hip = pose.get(KeypointIndex::RightHip);

        if !left_hip.is_valid(self.confidence_threshold)
            || !right_hip.is_valid(self.confidence_threshold)
        {
            return false;
        }

        let hip_x = (left_hip.x + right_hip.x) / 2.0;
        let hip_y = (left_hip.y + right_hip.y) / 2.0;

        let yaw = self.compute_raw_yaw(pose);

        self.calibration = Some(Calibration { hip_x, hip_y, yaw });
        true
    }

    /// キャリブレーション済みか
    pub fn is_calibrated(&self) -> bool {
        self.calibration.is_some()
    }

    /// 肩からyaw角度を算出（キャリブレーション補正なし）
    fn compute_raw_yaw(&self, pose: &Pose) -> f32 {
        let left_shoulder = pose.get(KeypointIndex::LeftShoulder);
        let right_shoulder = pose.get(KeypointIndex::RightShoulder);

        if left_shoulder.is_valid(self.confidence_threshold)
            && right_shoulder.is_valid(self.confidence_threshold)
        {
            let dx = right_shoulder.x - left_shoulder.x;
            let dy = right_shoulder.y - left_shoulder.y;
            f32::atan2(dy, dx)
        } else {
            0.0
        }
    }

    /// Poseから腰トラッカーを計算
    ///
    /// 要件フェーズ3:
    /// - 左右ヒップから腰位置を作る
    /// - 胴体方向からyawを作る
    /// - 初期キャリブレーションで座標系を合わせる
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

        // キャリブレーション基準点（未キャリブレーション時は画像中央）
        let (ref_x, ref_y, ref_yaw) = match &self.calibration {
            Some(cal) => (cal.hip_x, cal.hip_y, cal.yaw),
            None => (0.5, 0.5, 0.0),
        };

        // 座標変換: 基準点からの差分をメートルに変換
        let mut pos_x = (hip_x - ref_x) * self.scale_x;
        if self.mirror_x {
            pos_x = -pos_x;
        }
        let pos_y = self.offset_y + (ref_y - hip_y) * self.scale_y;
        let pos_z = 0.0;

        // 胴体方向からyaw（基準角度を引いて補正）
        let raw_yaw = self.compute_raw_yaw(pose);
        let yaw = raw_yaw - ref_yaw;

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
    fn test_center_position_uncalibrated() {
        let tracker = HipTracker::new();
        let pose = make_pose((0.5, 0.5), (0.5, 0.5), (0.4, 0.3), (0.6, 0.3));

        let result = tracker.compute(&pose).unwrap();
        assert!((result.position[0]).abs() < 0.01);
        assert!((result.position[1] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_calibration_zeroes_position() {
        let mut tracker = HipTracker::new();
        // 腰が画像の左上寄り(0.3, 0.3)にいる状態でキャリブレーション
        let cal_pose = make_pose((0.3, 0.3), (0.3, 0.3), (0.2, 0.1), (0.4, 0.1));
        assert!(tracker.calibrate(&cal_pose));

        // 同じ位置で compute → 原点(0, offset_y, 0)になる
        let result = tracker.compute(&cal_pose).unwrap();
        assert!((result.position[0]).abs() < 0.01);
        assert!((result.position[1] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_calibration_offset() {
        let mut tracker = HipTracker::new();
        // 中央(0.5, 0.5)でキャリブレーション
        let cal_pose = make_pose((0.5, 0.5), (0.5, 0.5), (0.4, 0.3), (0.6, 0.3));
        tracker.calibrate(&cal_pose);

        // 右に0.1移動 → pos_x = 0.1 * scale_x
        let moved_pose = make_pose((0.6, 0.5), (0.6, 0.5), (0.5, 0.3), (0.7, 0.3));
        let result = tracker.compute(&moved_pose).unwrap();
        assert!((result.position[0] - 0.1).abs() < 0.01);
        assert!((result.position[1] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_calibration_yaw_zeroed() {
        let mut tracker = HipTracker::new();
        // 肩が少し傾いた状態でキャリブレーション
        let cal_pose = make_pose((0.5, 0.5), (0.5, 0.5), (0.4, 0.35), (0.6, 0.25));
        tracker.calibrate(&cal_pose);

        // 同じ姿勢で compute → yaw=0 → quaternion=(0,0,0,1)
        let result = tracker.compute(&cal_pose).unwrap();
        assert!((result.rotation[1]).abs() < 0.01); // qy ≈ 0
        assert!((result.rotation[3] - 1.0).abs() < 0.01); // qw ≈ 1
    }
}
