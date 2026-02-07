use crate::config::SmoothConfig;
use crate::vmt::TrackerPose;

/// EMAベースのトラッカーポーズ平滑化フィルタ
///
/// 位置: 成分ごとのEMA
/// 回転: NLERP (Normalized Linear Interpolation)
pub struct Smoother {
    alpha_position: f32,
    alpha_rotation: f32,
    prev: Option<TrackerPose>,
}

impl Smoother {
    pub fn new(alpha_position: f32, alpha_rotation: f32) -> Self {
        Self {
            alpha_position,
            alpha_rotation,
            prev: None,
        }
    }

    pub fn from_config(config: &SmoothConfig) -> Self {
        Self::new(config.position, config.rotation)
    }

    pub fn apply(&mut self, pose: TrackerPose) -> TrackerPose {
        let prev = match self.prev {
            Some(prev) => prev,
            None => {
                self.prev = Some(pose);
                return pose;
            }
        };

        let ap = self.alpha_position;
        let ar = self.alpha_rotation;

        // Position: EMA
        let position = [
            ap * pose.position[0] + (1.0 - ap) * prev.position[0],
            ap * pose.position[1] + (1.0 - ap) * prev.position[1],
            ap * pose.position[2] + (1.0 - ap) * prev.position[2],
        ];

        // Rotation: NLERP
        let mut new_rot = pose.rotation;
        let dot = prev.rotation[0] * new_rot[0]
            + prev.rotation[1] * new_rot[1]
            + prev.rotation[2] * new_rot[2]
            + prev.rotation[3] * new_rot[3];

        if dot < 0.0 {
            new_rot[0] = -new_rot[0];
            new_rot[1] = -new_rot[1];
            new_rot[2] = -new_rot[2];
            new_rot[3] = -new_rot[3];
        }

        let rotation = [
            ar * new_rot[0] + (1.0 - ar) * prev.rotation[0],
            ar * new_rot[1] + (1.0 - ar) * prev.rotation[1],
            ar * new_rot[2] + (1.0 - ar) * prev.rotation[2],
            ar * new_rot[3] + (1.0 - ar) * prev.rotation[3],
        ];

        // Normalize quaternion
        let len = (rotation[0] * rotation[0]
            + rotation[1] * rotation[1]
            + rotation[2] * rotation[2]
            + rotation[3] * rotation[3])
            .sqrt();
        let rotation = [
            rotation[0] / len,
            rotation[1] / len,
            rotation[2] / len,
            rotation[3] / len,
        ];

        let result = TrackerPose::new(position, rotation);
        self.prev = Some(result);
        result
    }

    pub fn reset(&mut self) {
        self.prev = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq_f32(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    fn approx_eq_pose(a: &TrackerPose, b: &TrackerPose, eps: f32) -> bool {
        a.position.iter().zip(b.position.iter()).all(|(x, y)| approx_eq_f32(*x, *y, eps))
            && a.rotation.iter().zip(b.rotation.iter()).all(|(x, y)| approx_eq_f32(*x, *y, eps))
    }

    #[test]
    fn test_first_frame_passthrough() {
        let mut s = Smoother::new(0.5, 0.5);
        let pose = TrackerPose::new([1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 1.0]);
        let result = s.apply(pose);
        assert_eq!(result, pose);
    }

    #[test]
    fn test_no_smoothing() {
        let mut s = Smoother::new(1.0, 1.0);
        let pose1 = TrackerPose::new([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]);
        let pose2 = TrackerPose::new([1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 1.0]);
        s.apply(pose1);
        let result = s.apply(pose2);
        assert!(approx_eq_pose(&result, &pose2, 1e-6));
    }

    #[test]
    fn test_full_smoothing() {
        let mut s = Smoother::new(0.0, 0.0);
        let pose1 = TrackerPose::new([1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 1.0]);
        let pose2 = TrackerPose::new([4.0, 5.0, 6.0], [0.0, 0.0, 0.0, 1.0]);
        s.apply(pose1);
        let result = s.apply(pose2);
        assert!(approx_eq_pose(&result, &pose1, 1e-6));
    }

    #[test]
    fn test_position_smoothing() {
        let mut s = Smoother::new(0.5, 1.0);
        let pose1 = TrackerPose::new([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]);
        let pose2 = TrackerPose::new([2.0, 4.0, 6.0], [0.0, 0.0, 0.0, 1.0]);
        s.apply(pose1);
        let result = s.apply(pose2);
        let expected = TrackerPose::new([1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 1.0]);
        assert!(approx_eq_pose(&result, &expected, 1e-6));
    }

    #[test]
    fn test_rotation_nlerp() {
        let mut s = Smoother::new(1.0, 0.5);
        // identity quaternion
        let pose1 = TrackerPose::new([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]);
        // 90 degrees around Y axis: (0, sin(45), 0, cos(45))
        let half_sqrt2 = std::f32::consts::FRAC_1_SQRT_2;
        let pose2 = TrackerPose::new([0.0, 0.0, 0.0], [0.0, half_sqrt2, 0.0, half_sqrt2]);
        s.apply(pose1);
        let result = s.apply(pose2);

        // NLERP with alpha=0.5: lerp then normalize
        // lerp: (0, 0.5*0.707, 0, 0.5*1.0 + 0.5*0.707) = (0, 0.3535, 0, 0.8535)
        let ly = 0.5 * half_sqrt2;
        let lw = 0.5 * 1.0 + 0.5 * half_sqrt2;
        let len = (ly * ly + lw * lw).sqrt();
        let expected_y = ly / len;
        let expected_w = lw / len;

        assert!(approx_eq_f32(result.rotation[0], 0.0, 1e-6));
        assert!(approx_eq_f32(result.rotation[1], expected_y, 1e-5));
        assert!(approx_eq_f32(result.rotation[2], 0.0, 1e-6));
        assert!(approx_eq_f32(result.rotation[3], expected_w, 1e-5));
    }

    #[test]
    fn test_reset() {
        let mut s = Smoother::new(0.0, 0.0);
        let pose1 = TrackerPose::new([1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 1.0]);
        let pose2 = TrackerPose::new([4.0, 5.0, 6.0], [0.0, 0.0, 0.0, 1.0]);
        s.apply(pose1);
        s.reset();
        // After reset, next frame should pass through
        let result = s.apply(pose2);
        assert_eq!(result, pose2);
    }
}
