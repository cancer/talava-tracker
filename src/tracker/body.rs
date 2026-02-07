use crate::config::TrackerConfig;
use crate::pose::{KeypointIndex, Pose};
use crate::vmt::TrackerPose;

/// 全トラッカーの計算結果
pub struct BodyPoses {
    pub hip: Option<TrackerPose>,
    pub left_foot: Option<TrackerPose>,
    pub right_foot: Option<TrackerPose>,
    pub chest: Option<TrackerPose>,
}

struct Calibration {
    hip_x: f32,
    hip_y: f32,
    yaw_shoulder: f32,
    yaw_left_foot: f32,
    yaw_right_foot: f32,
}

pub struct BodyTracker {
    confidence_threshold: f32,
    scale_x: f32,
    scale_y: f32,
    mirror_x: bool,
    offset_y: f32,
    calibration: Option<Calibration>,
}

impl BodyTracker {
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
        let yaw_shoulder = self.compute_shoulder_yaw(pose);
        let yaw_left_foot =
            self.compute_foot_yaw(pose, KeypointIndex::LeftKnee, KeypointIndex::LeftAnkle);
        let yaw_right_foot =
            self.compute_foot_yaw(pose, KeypointIndex::RightKnee, KeypointIndex::RightAnkle);

        self.calibration = Some(Calibration {
            hip_x,
            hip_y,
            yaw_shoulder,
            yaw_left_foot,
            yaw_right_foot,
        });
        true
    }

    pub fn is_calibrated(&self) -> bool {
        self.calibration.is_some()
    }

    pub fn compute(&self, pose: &Pose) -> BodyPoses {
        BodyPoses {
            hip: self.compute_hip(pose),
            left_foot: self.compute_left_foot(pose),
            right_foot: self.compute_right_foot(pose),
            chest: self.compute_chest(pose),
        }
    }

    fn convert_position(&self, x: f32, y: f32) -> [f32; 3] {
        let (ref_x, ref_y) = match &self.calibration {
            Some(cal) => (cal.hip_x, cal.hip_y),
            None => (0.5, 0.5),
        };
        let mut pos_x = (ref_x - x) * self.scale_x;
        if self.mirror_x {
            pos_x = -pos_x;
        }
        let pos_y = self.offset_y + (ref_y - y) * self.scale_y;
        [pos_x, pos_y, 0.0]
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

    fn compute_hip(&self, pose: &Pose) -> Option<TrackerPose> {
        let left_hip = pose.get(KeypointIndex::LeftHip);
        let right_hip = pose.get(KeypointIndex::RightHip);

        if !left_hip.is_valid(self.confidence_threshold)
            || !right_hip.is_valid(self.confidence_threshold)
        {
            return None;
        }

        let x = (left_hip.x + right_hip.x) / 2.0;
        let y = (left_hip.y + right_hip.y) / 2.0;
        let position = self.convert_position(x, y);

        let ref_yaw = self.calibration.as_ref().map_or(0.0, |c| c.yaw_shoulder);
        let yaw = self.compute_shoulder_yaw(pose) - ref_yaw;
        let rotation = Self::yaw_to_quaternion(yaw);

        Some(TrackerPose::new(position, rotation))
    }

    fn compute_left_foot(&self, pose: &Pose) -> Option<TrackerPose> {
        let knee = pose.get(KeypointIndex::LeftKnee);
        let ankle = pose.get(KeypointIndex::LeftAnkle);

        if !knee.is_valid(self.confidence_threshold)
            || !ankle.is_valid(self.confidence_threshold)
        {
            return None;
        }

        let position = self.convert_position(ankle.x, ankle.y);

        let ref_yaw = self
            .calibration
            .as_ref()
            .map_or(0.0, |c| c.yaw_left_foot);
        let yaw = self.compute_foot_yaw(pose, KeypointIndex::LeftKnee, KeypointIndex::LeftAnkle)
            - ref_yaw;
        let rotation = Self::yaw_to_quaternion(yaw);

        Some(TrackerPose::new(position, rotation))
    }

    fn compute_right_foot(&self, pose: &Pose) -> Option<TrackerPose> {
        let knee = pose.get(KeypointIndex::RightKnee);
        let ankle = pose.get(KeypointIndex::RightAnkle);

        if !knee.is_valid(self.confidence_threshold)
            || !ankle.is_valid(self.confidence_threshold)
        {
            return None;
        }

        let position = self.convert_position(ankle.x, ankle.y);

        let ref_yaw = self
            .calibration
            .as_ref()
            .map_or(0.0, |c| c.yaw_right_foot);
        let yaw = self.compute_foot_yaw(pose, KeypointIndex::RightKnee, KeypointIndex::RightAnkle)
            - ref_yaw;
        let rotation = Self::yaw_to_quaternion(yaw);

        Some(TrackerPose::new(position, rotation))
    }

    fn compute_chest(&self, pose: &Pose) -> Option<TrackerPose> {
        let left_shoulder = pose.get(KeypointIndex::LeftShoulder);
        let right_shoulder = pose.get(KeypointIndex::RightShoulder);

        if !left_shoulder.is_valid(self.confidence_threshold)
            || !right_shoulder.is_valid(self.confidence_threshold)
        {
            return None;
        }

        let x = (left_shoulder.x + right_shoulder.x) / 2.0;
        let y = (left_shoulder.y + right_shoulder.y) / 2.0;
        let position = self.convert_position(x, y);

        let ref_yaw = self.calibration.as_ref().map_or(0.0, |c| c.yaw_shoulder);
        let yaw = self.compute_shoulder_yaw(pose) - ref_yaw;
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
}
