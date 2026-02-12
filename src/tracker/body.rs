use crate::config::TrackerConfig;
use crate::pose::{KeypointIndex, Pose};
use crate::vmt::TrackerPose;

/// 全トラッカーの計算結果
pub struct BodyPoses {
    pub hip: Option<TrackerPose>,
    pub left_foot: Option<TrackerPose>,
    pub right_foot: Option<TrackerPose>,
    pub chest: Option<TrackerPose>,
    pub left_knee: Option<TrackerPose>,
    pub right_knee: Option<TrackerPose>,
}

struct Calibration {
    hip_x: f32,
    hip_y: f32,
    torso_height: f32,
    yaw_shoulder: f32,
    yaw_left_foot: f32,
    yaw_right_foot: f32,
    // 腰からの相対オフセット（画像座標）
    left_ankle_offset: Option<(f32, f32)>,
    right_ankle_offset: Option<(f32, f32)>,
    left_knee_offset: Option<(f32, f32)>,
    right_knee_offset: Option<(f32, f32)>,
    yaw_left_knee: f32,
    yaw_right_knee: f32,
}

pub struct BodyTracker {
    confidence_threshold: f32,
    scale_x: f32,
    scale_y: f32,
    body_scale: f32,
    leg_scale: f32,
    depth_scale: f32,
    mirror_x: bool,
    offset_y: f32,
    fov_v_rad: f32,
    calibration: Option<Calibration>,
}

impl BodyTracker {
    pub fn from_config(config: &TrackerConfig) -> Self {
        Self::new(config, 0.0)
    }

    pub fn new(config: &TrackerConfig, fov_v_deg: f32) -> Self {
        Self {
            confidence_threshold: 0.1,
            scale_x: config.scale_x,
            scale_y: config.scale_y,
            body_scale: config.body_scale,
            leg_scale: config.leg_scale,
            depth_scale: config.depth_scale,
            mirror_x: config.mirror_x,
            offset_y: config.offset_y,
            fov_v_rad: fov_v_deg.to_radians(),
            calibration: None,
        }
    }

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
        let torso_height = self.compute_torso_height(pose).unwrap_or(0.0);
        let yaw_shoulder = self.compute_shoulder_yaw(pose);
        let yaw_left_foot =
            self.compute_foot_yaw(pose, KeypointIndex::LeftKnee, KeypointIndex::LeftAnkle);
        let yaw_right_foot =
            self.compute_foot_yaw(pose, KeypointIndex::RightKnee, KeypointIndex::RightAnkle);

        let left_ankle = pose.get(KeypointIndex::LeftAnkle);
        let left_ankle_offset = if left_ankle.is_valid(self.confidence_threshold) {
            Some((left_ankle.x - hip_x, left_ankle.y - hip_y))
        } else {
            None
        };
        let right_ankle = pose.get(KeypointIndex::RightAnkle);
        let right_ankle_offset = if right_ankle.is_valid(self.confidence_threshold) {
            Some((right_ankle.x - hip_x, right_ankle.y - hip_y))
        } else {
            None
        };

        let left_knee = pose.get(KeypointIndex::LeftKnee);
        let left_knee_offset = if left_knee.is_valid(self.confidence_threshold) {
            Some((left_knee.x - hip_x, left_knee.y - hip_y))
        } else {
            None
        };
        let right_knee = pose.get(KeypointIndex::RightKnee);
        let right_knee_offset = if right_knee.is_valid(self.confidence_threshold) {
            Some((right_knee.x - hip_x, right_knee.y - hip_y))
        } else {
            None
        };

        let yaw_left_knee =
            self.compute_knee_yaw(pose, KeypointIndex::LeftHip, KeypointIndex::LeftKnee);
        let yaw_right_knee =
            self.compute_knee_yaw(pose, KeypointIndex::RightHip, KeypointIndex::RightKnee);

        self.calibration = Some(Calibration {
            hip_x,
            hip_y,
            torso_height,
            yaw_shoulder,
            yaw_left_foot,
            yaw_right_foot,
            left_ankle_offset,
            right_ankle_offset,
            left_knee_offset,
            right_knee_offset,
            yaw_left_knee,
            yaw_right_knee,
        });
        true
    }

    pub fn is_calibrated(&self) -> bool {
        self.calibration.is_some()
    }

    pub fn compute(&self, pose: &Pose) -> BodyPoses {
        let left_hip = pose.get(KeypointIndex::LeftHip);
        let right_hip = pose.get(KeypointIndex::RightHip);
        let hip_center = if left_hip.is_valid(self.confidence_threshold)
            && right_hip.is_valid(self.confidence_threshold)
        {
            Some(((left_hip.x + right_hip.x) / 2.0, (left_hip.y + right_hip.y) / 2.0))
        } else {
            None
        };

        let pos_z = self.estimate_depth(pose);
        let body_ratio = self.compute_body_ratio(pose);

        BodyPoses {
            hip: self.compute_hip(pose, hip_center, pos_z, body_ratio),
            left_foot: self.compute_left_foot(pose, hip_center, pos_z, body_ratio),
            right_foot: self.compute_right_foot(pose, hip_center, pos_z, body_ratio),
            chest: self.compute_chest(pose, hip_center, pos_z, body_ratio),
            left_knee: self.compute_left_knee(pose, hip_center, pos_z, body_ratio),
            right_knee: self.compute_right_knee(pose, hip_center, pos_z, body_ratio),
        }
    }

    /// 画像座標yにおけるFOV補正係数
    /// カメラ中心から離れるほど斜め距離が増え、1ピクセルあたりの物理距離が大きくなる
    fn fov_scale(&self, y: f32) -> f32 {
        let half_fov = self.fov_v_rad / 2.0;
        if half_fov < 0.01 {
            return 1.0;
        }
        // 画像y → カメラ中心からの角度
        let theta = ((y - 0.5) * 2.0 * half_fov.tan()).atan();
        // 斜め距離の補正: 1/cos(θ)
        1.0 / theta.cos()
    }

    fn convert_position(&self, x: f32, y: f32, hip_x: f32, hip_y: f32, part_scale: f32, pos_z: f32, body_ratio: f32) -> [f32; 3] {
        let (ref_x, ref_y) = match &self.calibration {
            Some(cal) => (cal.hip_x, cal.hip_y),
            None => (0.5, 0.5),
        };
        // FOV補正: 画像端ほどパーツオフセットを拡大
        let fov = self.fov_scale(y);
        // body_ratioで奥行き正規化
        let mut pos_x = ((ref_x - hip_x) * self.scale_x + (hip_x - x) * part_scale * fov) * body_ratio;
        if self.mirror_x {
            pos_x = -pos_x;
        }
        let pos_y = self.offset_y + ((ref_y - hip_y) * self.scale_y + (hip_y - y) * part_scale * fov) * body_ratio;
        [pos_x, pos_y, pos_z]
    }

    /// 胴体の縦の長さ（肩中点→腰中点の垂直距離）
    fn compute_torso_height(&self, pose: &Pose) -> Option<f32> {
        let ls = pose.get(KeypointIndex::LeftShoulder);
        let rs = pose.get(KeypointIndex::RightShoulder);
        let lh = pose.get(KeypointIndex::LeftHip);
        let rh = pose.get(KeypointIndex::RightHip);

        if !ls.is_valid(self.confidence_threshold)
            || !rs.is_valid(self.confidence_threshold)
            || !lh.is_valid(self.confidence_threshold)
            || !rh.is_valid(self.confidence_threshold)
        {
            return None;
        }

        let shoulder_y = (ls.y + rs.y) / 2.0;
        let hip_y = (lh.y + rh.y) / 2.0;
        Some((hip_y - shoulder_y).abs())
    }

    /// 胴体高さの変化から奥行きを推定
    /// 近づく→体が大きく映る→胴体高さ増→z正
    fn estimate_depth(&self, pose: &Pose) -> f32 {
        let cal = match &self.calibration {
            Some(cal) if cal.torso_height > 0.01 => cal,
            _ => return 0.0,
        };

        let current = match self.compute_torso_height(pose) {
            Some(h) if h > 0.01 => h,
            _ => return 0.0,
        };

        (1.0 - cal.torso_height / current) * self.depth_scale
    }

    /// 奥行き変化による見かけの体型変化を補正する比率
    /// cal_torso / current: 近づく→比率<1→オフセット縮小、離れる→比率>1→オフセット拡大
    fn compute_body_ratio(&self, pose: &Pose) -> f32 {
        let cal = match &self.calibration {
            Some(cal) if cal.torso_height > 0.01 => cal,
            _ => return 1.0,
        };

        let current = match self.compute_torso_height(pose) {
            Some(h) if h > 0.01 => h,
            _ => return 1.0,
        };

        cal.torso_height / current
    }

    fn compute_shoulder_yaw(&self, pose: &Pose) -> f32 {
        let left_shoulder = pose.get(KeypointIndex::LeftShoulder);
        let right_shoulder = pose.get(KeypointIndex::RightShoulder);

        if left_shoulder.is_valid(self.confidence_threshold)
            && right_shoulder.is_valid(self.confidence_threshold)
        {
            let dx = if self.mirror_x {
                right_shoulder.x - left_shoulder.x
            } else {
                left_shoulder.x - right_shoulder.x
            };
            let dy = right_shoulder.y - left_shoulder.y;
            f32::atan2(dy, dx)
        } else {
            0.0
        }
    }

    fn compute_foot_yaw(
        &self,
        pose: &Pose,
        knee_idx: KeypointIndex,
        ankle_idx: KeypointIndex,
    ) -> f32 {
        let knee = pose.get(knee_idx);
        let ankle = pose.get(ankle_idx);

        if knee.is_valid(self.confidence_threshold) && ankle.is_valid(self.confidence_threshold) {
            let raw_dx = ankle.x - knee.x;
            let dx = if self.mirror_x { raw_dx } else { -raw_dx };
            let dy = ankle.y - knee.y;
            f32::atan2(dx, dy)
        } else {
            0.0
        }
    }

    fn yaw_to_quaternion(yaw: f32) -> [f32; 4] {
        let half = yaw / 2.0;
        [0.0, half.sin(), 0.0, half.cos()]
    }

    fn compute_hip(&self, pose: &Pose, hip_center: Option<(f32, f32)>, pos_z: f32, body_ratio: f32) -> Option<TrackerPose> {
        let (hip_x, hip_y) = hip_center?;
        let position = self.convert_position(hip_x, hip_y, hip_x, hip_y, self.body_scale, pos_z, body_ratio);

        let ref_yaw = self.calibration.as_ref().map_or(0.0, |c| c.yaw_shoulder);
        let yaw = self.compute_shoulder_yaw(pose) - ref_yaw;
        let rotation = Self::yaw_to_quaternion(yaw);

        Some(TrackerPose::new(position, rotation))
    }

    fn compute_left_foot(&self, pose: &Pose, hip_center: Option<(f32, f32)>, pos_z: f32, body_ratio: f32) -> Option<TrackerPose> {
        let (hip_x, hip_y) = hip_center?;
        let knee = pose.get(KeypointIndex::LeftKnee);
        let ankle = pose.get(KeypointIndex::LeftAnkle);

        let (ax, ay, has_keypoints) = if knee.is_valid(self.confidence_threshold)
            && ankle.is_valid(self.confidence_threshold)
        {
            (ankle.x, ankle.y, true)
        } else if let Some(cal) = &self.calibration {
            if let Some((ox, oy)) = cal.left_ankle_offset {
                (hip_x + ox, hip_y + oy, false)
            } else {
                return None;
            }
        } else {
            return None;
        };

        let position = self.convert_position(ax, ay, hip_x, hip_y, self.leg_scale, pos_z, body_ratio);

        let yaw = if has_keypoints {
            let ref_yaw = self.calibration.as_ref().map_or(0.0, |c| c.yaw_left_foot);
            self.compute_foot_yaw(pose, KeypointIndex::LeftKnee, KeypointIndex::LeftAnkle) - ref_yaw
        } else {
            0.0
        };
        let rotation = Self::yaw_to_quaternion(yaw);

        Some(TrackerPose::new(position, rotation))
    }

    fn compute_right_foot(&self, pose: &Pose, hip_center: Option<(f32, f32)>, pos_z: f32, body_ratio: f32) -> Option<TrackerPose> {
        let (hip_x, hip_y) = hip_center?;
        let knee = pose.get(KeypointIndex::RightKnee);
        let ankle = pose.get(KeypointIndex::RightAnkle);

        let (ax, ay, has_keypoints) = if knee.is_valid(self.confidence_threshold)
            && ankle.is_valid(self.confidence_threshold)
        {
            (ankle.x, ankle.y, true)
        } else if let Some(cal) = &self.calibration {
            if let Some((ox, oy)) = cal.right_ankle_offset {
                (hip_x + ox, hip_y + oy, false)
            } else {
                return None;
            }
        } else {
            return None;
        };

        let position = self.convert_position(ax, ay, hip_x, hip_y, self.leg_scale, pos_z, body_ratio);

        let yaw = if has_keypoints {
            let ref_yaw = self.calibration.as_ref().map_or(0.0, |c| c.yaw_right_foot);
            self.compute_foot_yaw(pose, KeypointIndex::RightKnee, KeypointIndex::RightAnkle) - ref_yaw
        } else {
            0.0
        };
        let rotation = Self::yaw_to_quaternion(yaw);

        Some(TrackerPose::new(position, rotation))
    }

    fn compute_chest(&self, pose: &Pose, hip_center: Option<(f32, f32)>, pos_z: f32, body_ratio: f32) -> Option<TrackerPose> {
        let (hip_x, hip_y) = hip_center?;
        let left_shoulder = pose.get(KeypointIndex::LeftShoulder);
        let right_shoulder = pose.get(KeypointIndex::RightShoulder);

        if !left_shoulder.is_valid(self.confidence_threshold)
            || !right_shoulder.is_valid(self.confidence_threshold)
        {
            return None;
        }

        // X座標は肩中点
        let x = (left_shoulder.x + right_shoulder.x) / 2.0;

        // Y座標: Spine03があればそれを使用、なければ肩中点
        let spine03 = pose.get(KeypointIndex::Spine03);
        let y = if spine03.is_valid(self.confidence_threshold) {
            spine03.y
        } else {
            (left_shoulder.y + right_shoulder.y) / 2.0
        };

        let position = self.convert_position(x, y, hip_x, hip_y, self.body_scale, pos_z, body_ratio);

        let ref_yaw = self.calibration.as_ref().map_or(0.0, |c| c.yaw_shoulder);
        let yaw = self.compute_shoulder_yaw(pose) - ref_yaw;
        let rotation = Self::yaw_to_quaternion(yaw);

        Some(TrackerPose::new(position, rotation))
    }

    /// 膝のyaw: 腰→膝方向
    fn compute_knee_yaw(&self, pose: &Pose, hip_idx: KeypointIndex, knee_idx: KeypointIndex) -> f32 {
        let hip = pose.get(hip_idx);
        let knee = pose.get(knee_idx);

        if hip.is_valid(self.confidence_threshold) && knee.is_valid(self.confidence_threshold) {
            let raw_dx = knee.x - hip.x;
            let dx = if self.mirror_x { raw_dx } else { -raw_dx };
            let dy = knee.y - hip.y;
            f32::atan2(dx, dy)
        } else {
            0.0
        }
    }

    fn compute_left_knee(&self, pose: &Pose, hip_center: Option<(f32, f32)>, pos_z: f32, body_ratio: f32) -> Option<TrackerPose> {
        let (hip_x, hip_y) = hip_center?;
        let knee = pose.get(KeypointIndex::LeftKnee);

        let (kx, ky, has_keypoint) = if knee.is_valid(self.confidence_threshold) {
            (knee.x, knee.y, true)
        } else if let Some(cal) = &self.calibration {
            if let Some((ox, oy)) = cal.left_knee_offset {
                (hip_x + ox, hip_y + oy, false)
            } else {
                return None;
            }
        } else {
            return None;
        };

        let position = self.convert_position(kx, ky, hip_x, hip_y, self.leg_scale, pos_z, body_ratio);

        let yaw = if has_keypoint {
            let ref_yaw = self.calibration.as_ref().map_or(0.0, |c| c.yaw_left_knee);
            self.compute_knee_yaw(pose, KeypointIndex::LeftHip, KeypointIndex::LeftKnee) - ref_yaw
        } else {
            0.0
        };
        let rotation = Self::yaw_to_quaternion(yaw);

        Some(TrackerPose::new(position, rotation))
    }

    fn compute_right_knee(&self, pose: &Pose, hip_center: Option<(f32, f32)>, pos_z: f32, body_ratio: f32) -> Option<TrackerPose> {
        let (hip_x, hip_y) = hip_center?;
        let knee = pose.get(KeypointIndex::RightKnee);

        let (kx, ky, has_keypoint) = if knee.is_valid(self.confidence_threshold) {
            (knee.x, knee.y, true)
        } else if let Some(cal) = &self.calibration {
            if let Some((ox, oy)) = cal.right_knee_offset {
                (hip_x + ox, hip_y + oy, false)
            } else {
                return None;
            }
        } else {
            return None;
        };

        let position = self.convert_position(kx, ky, hip_x, hip_y, self.leg_scale, pos_z, body_ratio);

        let yaw = if has_keypoint {
            let ref_yaw = self.calibration.as_ref().map_or(0.0, |c| c.yaw_right_knee);
            self.compute_knee_yaw(pose, KeypointIndex::RightHip, KeypointIndex::RightKnee) - ref_yaw
        } else {
            0.0
        };
        let rotation = Self::yaw_to_quaternion(yaw);

        Some(TrackerPose::new(position, rotation))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pose::Keypoint;

    fn make_pose(
        left_hip: (f32, f32),
        right_hip: (f32, f32),
        left_shoulder: (f32, f32),
        right_shoulder: (f32, f32),
        left_knee: (f32, f32),
        right_knee: (f32, f32),
        left_ankle: (f32, f32),
        right_ankle: (f32, f32),
    ) -> Pose {
        let mut keypoints = [Keypoint::default(); KeypointIndex::COUNT];
        keypoints[KeypointIndex::LeftHip as usize] =
            Keypoint::new(left_hip.0, left_hip.1, 0.9);
        keypoints[KeypointIndex::RightHip as usize] =
            Keypoint::new(right_hip.0, right_hip.1, 0.9);
        keypoints[KeypointIndex::LeftShoulder as usize] =
            Keypoint::new(left_shoulder.0, left_shoulder.1, 0.9);
        keypoints[KeypointIndex::RightShoulder as usize] =
            Keypoint::new(right_shoulder.0, right_shoulder.1, 0.9);
        keypoints[KeypointIndex::LeftKnee as usize] =
            Keypoint::new(left_knee.0, left_knee.1, 0.9);
        keypoints[KeypointIndex::RightKnee as usize] =
            Keypoint::new(right_knee.0, right_knee.1, 0.9);
        keypoints[KeypointIndex::LeftAnkle as usize] =
            Keypoint::new(left_ankle.0, left_ankle.1, 0.9);
        keypoints[KeypointIndex::RightAnkle as usize] =
            Keypoint::new(right_ankle.0, right_ankle.1, 0.9);
        Pose::new(keypoints)
    }

    #[test]
    fn test_hip_center_uncalibrated() {
        let config = TrackerConfig::default();
        let tracker = BodyTracker::from_config(&config);
        // Hip at center (0.5, 0.5)
        let pose = make_pose(
            (0.5, 0.5), (0.5, 0.5),   // hips
            (0.4, 0.3), (0.6, 0.3),   // shoulders
            (0.4, 0.7), (0.6, 0.7),   // knees
            (0.4, 0.9), (0.6, 0.9),   // ankles
        );
        let result = tracker.compute(&pose);
        let hip = result.hip.unwrap();
        assert!((hip.position[0]).abs() < 0.01);
        assert!((hip.position[1] - 1.0).abs() < 0.01);
        assert!((hip.position[2]).abs() < 0.01);
    }

    #[test]
    fn test_calibration_zeroes_hip() {
        let config = TrackerConfig::default();
        let mut tracker = BodyTracker::from_config(&config);
        let pose = make_pose(
            (0.3, 0.4), (0.3, 0.4),
            (0.2, 0.2), (0.4, 0.2),
            (0.2, 0.6), (0.4, 0.6),
            (0.2, 0.8), (0.4, 0.8),
        );
        assert!(tracker.calibrate(&pose));

        let result = tracker.compute(&pose);
        let hip = result.hip.unwrap();
        assert!((hip.position[0]).abs() < 0.01);
        assert!((hip.position[1] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_chest_above_hip() {
        let config = TrackerConfig::default();
        let tracker = BodyTracker::from_config(&config);
        // Shoulders at y=0.3 (higher in image = higher in space), hips at y=0.5
        let pose = make_pose(
            (0.4, 0.5), (0.6, 0.5),
            (0.4, 0.3), (0.6, 0.3),
            (0.4, 0.7), (0.6, 0.7),
            (0.4, 0.9), (0.6, 0.9),
        );
        let result = tracker.compute(&pose);
        let hip = result.hip.unwrap();
        let chest = result.chest.unwrap();
        // Chest should have higher pos_y than hip (shoulders are above hips in image)
        assert!(chest.position[1] > hip.position[1]);
    }

    #[test]
    fn test_foot_below_hip() {
        let config = TrackerConfig::default();
        let tracker = BodyTracker::from_config(&config);
        // Ankles at y=0.9 (lower in image = lower in space), hips at y=0.5
        let pose = make_pose(
            (0.4, 0.5), (0.6, 0.5),
            (0.4, 0.3), (0.6, 0.3),
            (0.4, 0.7), (0.6, 0.7),
            (0.4, 0.9), (0.6, 0.9),
        );
        let result = tracker.compute(&pose);
        let hip = result.hip.unwrap();
        let left_foot = result.left_foot.unwrap();
        let right_foot = result.right_foot.unwrap();
        // Feet should have lower pos_y than hip
        assert!(left_foot.position[1] < hip.position[1]);
        assert!(right_foot.position[1] < hip.position[1]);
    }

    #[test]
    fn test_all_trackers_returned() {
        let config = TrackerConfig::default();
        let tracker = BodyTracker::from_config(&config);
        let pose = make_pose(
            (0.4, 0.5), (0.6, 0.5),
            (0.4, 0.3), (0.6, 0.3),
            (0.4, 0.7), (0.6, 0.7),
            (0.4, 0.9), (0.6, 0.9),
        );
        let result = tracker.compute(&pose);
        assert!(result.hip.is_some());
        assert!(result.left_foot.is_some());
        assert!(result.right_foot.is_some());
        assert!(result.chest.is_some());
    }

    #[test]
    fn test_depth_closer_positive_z() {
        let config = TrackerConfig::default();
        let mut tracker = BodyTracker::from_config(&config);

        // キャリブレーション: 肩y=0.3, 腰y=0.5 → 胴体高さ=0.2
        let cal_pose = make_pose(
            (0.4, 0.5), (0.6, 0.5),
            (0.4, 0.3), (0.6, 0.3),
            (0.4, 0.7), (0.6, 0.7),
            (0.4, 0.9), (0.6, 0.9),
        );
        assert!(tracker.calibrate(&cal_pose));

        // 近づく: 胴体が大きく映る（肩y=0.25, 腰y=0.55 → 胴体高さ=0.3）
        let closer_pose = make_pose(
            (0.35, 0.55), (0.65, 0.55),
            (0.35, 0.25), (0.65, 0.25),
            (0.35, 0.75), (0.65, 0.75),
            (0.35, 0.95), (0.65, 0.95),
        );
        let result = tracker.compute(&closer_pose);
        let hip = result.hip.unwrap();
        assert!(hip.position[2] > 0.0, "closer should give positive z, got {}", hip.position[2]);
    }

    #[test]
    fn test_depth_farther_negative_z() {
        let config = TrackerConfig::default();
        let mut tracker = BodyTracker::from_config(&config);

        // キャリブレーション: 胴体高さ=0.2
        let cal_pose = make_pose(
            (0.4, 0.5), (0.6, 0.5),
            (0.4, 0.3), (0.6, 0.3),
            (0.4, 0.7), (0.6, 0.7),
            (0.4, 0.9), (0.6, 0.9),
        );
        assert!(tracker.calibrate(&cal_pose));

        // 離れる: 胴体が小さく映る（肩y=0.35, 腰y=0.5 → 胴体高さ=0.15）
        let farther_pose = make_pose(
            (0.45, 0.5), (0.55, 0.5),
            (0.45, 0.35), (0.55, 0.35),
            (0.45, 0.65), (0.55, 0.65),
            (0.45, 0.8), (0.55, 0.8),
        );
        let result = tracker.compute(&farther_pose);
        let hip = result.hip.unwrap();
        assert!(hip.position[2] < 0.0, "farther should give negative z, got {}", hip.position[2]);
    }

    #[test]
    fn test_body_proportion_stable_at_different_depths() {
        let config = TrackerConfig::default();
        let mut tracker = BodyTracker::from_config(&config);

        // キャリブレーション: 胴体高さ=0.2
        let cal_pose = make_pose(
            (0.4, 0.5), (0.6, 0.5),
            (0.4, 0.3), (0.6, 0.3),
            (0.4, 0.7), (0.6, 0.7),
            (0.4, 0.9), (0.6, 0.9),
        );
        assert!(tracker.calibrate(&cal_pose));
        let cal_result = tracker.compute(&cal_pose);
        let cal_hip_y = cal_result.hip.unwrap().position[1];
        let cal_chest_y = cal_result.chest.unwrap().position[1];
        let cal_diff = cal_chest_y - cal_hip_y;

        // 近づく: 胴体が1.5倍に（胴体高さ0.3）
        let closer_pose = make_pose(
            (0.35, 0.55), (0.65, 0.55),
            (0.35, 0.25), (0.65, 0.25),
            (0.35, 0.75), (0.65, 0.75),
            (0.35, 0.95), (0.65, 0.95),
        );
        let closer_result = tracker.compute(&closer_pose);
        let closer_hip_y = closer_result.hip.unwrap().position[1];
        let closer_chest_y = closer_result.chest.unwrap().position[1];
        let closer_diff = closer_chest_y - closer_hip_y;

        // 胸-腰の距離がキャリブレーション時と近いこと（補正なしだと1.5倍になる）
        let ratio = closer_diff / cal_diff;
        assert!(
            (ratio - 1.0).abs() < 0.15,
            "body proportion should be stable, got ratio={:.2} (cal_diff={:.3}, closer_diff={:.3})",
            ratio, cal_diff, closer_diff
        );
    }

    #[test]
    fn test_depth_zero_at_calibration() {
        let config = TrackerConfig::default();
        let mut tracker = BodyTracker::from_config(&config);

        let pose = make_pose(
            (0.4, 0.5), (0.6, 0.5),
            (0.4, 0.3), (0.6, 0.3),
            (0.4, 0.7), (0.6, 0.7),
            (0.4, 0.9), (0.6, 0.9),
        );
        assert!(tracker.calibrate(&pose));

        let result = tracker.compute(&pose);
        let hip = result.hip.unwrap();
        assert!((hip.position[2]).abs() < 0.01, "same pose as calibration should give z≈0, got {}", hip.position[2]);
    }
}
