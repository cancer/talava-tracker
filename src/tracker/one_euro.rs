use std::time::Instant;

use crate::config::FilterConfig;
use crate::vmt::TrackerPose;

/// Low-pass filter component
struct LowPassFilter {
    prev: Option<f32>,
}

impl LowPassFilter {
    fn new() -> Self {
        Self { prev: None }
    }

    fn filter(&mut self, value: f32, alpha: f32) -> f32 {
        match self.prev {
            Some(prev) => {
                let result = alpha * value + (1.0 - alpha) * prev;
                self.prev = Some(result);
                result
            }
            None => {
                self.prev = Some(value);
                value
            }
        }
    }

    fn reset(&mut self) {
        self.prev = None;
    }
}

/// alpha = 1 / (1 + tau/Te), tau = 1/(2*pi*fc)
fn smoothing_factor(te: f32, cutoff: f32) -> f32 {
    let r = 2.0 * std::f32::consts::PI * cutoff * te;
    r / (r + 1.0)
}

/// One Euro Filter for a single scalar value
struct ScalarFilter {
    min_cutoff: f32,
    beta: f32,
    d_cutoff: f32,
    x_filter: LowPassFilter,
    dx_filter: LowPassFilter,
    prev_value: Option<f32>,
}

impl ScalarFilter {
    fn new(min_cutoff: f32, beta: f32, d_cutoff: f32) -> Self {
        Self {
            min_cutoff,
            beta,
            d_cutoff,
            x_filter: LowPassFilter::new(),
            dx_filter: LowPassFilter::new(),
            prev_value: None,
        }
    }

    fn filter(&mut self, value: f32, dt: f32) -> f32 {
        let dx = match self.prev_value {
            Some(prev) => {
                if dt > 0.0 {
                    (value - prev) / dt
                } else {
                    0.0
                }
            }
            None => 0.0,
        };
        self.prev_value = Some(value);

        let edx = self
            .dx_filter
            .filter(dx, smoothing_factor(dt, self.d_cutoff));
        let cutoff = self.min_cutoff + self.beta * edx.abs();
        self.x_filter.filter(value, smoothing_factor(dt, cutoff))
    }

    fn reset(&mut self) {
        self.x_filter.reset();
        self.dx_filter.reset();
        self.prev_value = None;
    }
}

/// One Euro Filter for TrackerPose (position + rotation)
pub struct PoseFilter {
    position: [ScalarFilter; 3],
    rotation: [ScalarFilter; 4],
    prev_rotation: Option<[f32; 4]>,
    last_time: Option<Instant>,
}

impl PoseFilter {
    pub fn new(
        pos_min_cutoff: f32,
        pos_beta: f32,
        rot_min_cutoff: f32,
        rot_beta: f32,
    ) -> Self {
        let d_cutoff = 1.0;
        Self {
            position: std::array::from_fn(|_| {
                ScalarFilter::new(pos_min_cutoff, pos_beta, d_cutoff)
            }),
            rotation: std::array::from_fn(|_| {
                ScalarFilter::new(rot_min_cutoff, rot_beta, d_cutoff)
            }),
            prev_rotation: None,
            last_time: None,
        }
    }

    pub fn from_config(config: &FilterConfig) -> Self {
        Self::new(
            config.position_min_cutoff,
            config.position_beta,
            config.rotation_min_cutoff,
            config.rotation_beta,
        )
    }

    pub fn apply(&mut self, pose: TrackerPose) -> TrackerPose {
        let now = Instant::now();
        let dt = match self.last_time {
            Some(t) => {
                let d = now.duration_since(t).as_secs_f32();
                if d > 0.0 { d } else { 1.0 / 90.0 }
            }
            None => {
                self.last_time = Some(now);
                self.prev_rotation = Some(pose.rotation);
                return pose;
            }
        };
        self.last_time = Some(now);

        let position = [
            self.position[0].filter(pose.position[0], dt),
            self.position[1].filter(pose.position[1], dt),
            self.position[2].filter(pose.position[2], dt),
        ];

        // Quaternion shortest path
        let mut rot = pose.rotation;
        if let Some(ref prev) = self.prev_rotation {
            let dot = prev[0] * rot[0] + prev[1] * rot[1] + prev[2] * rot[2] + prev[3] * rot[3];
            if dot < 0.0 {
                rot = [-rot[0], -rot[1], -rot[2], -rot[3]];
            }
        }
        self.prev_rotation = Some(rot);

        let mut rotation = [
            self.rotation[0].filter(rot[0], dt),
            self.rotation[1].filter(rot[1], dt),
            self.rotation[2].filter(rot[2], dt),
            self.rotation[3].filter(rot[3], dt),
        ];

        // Normalize quaternion
        let len = (rotation[0] * rotation[0]
            + rotation[1] * rotation[1]
            + rotation[2] * rotation[2]
            + rotation[3] * rotation[3])
            .sqrt();
        if len > 0.0 {
            for v in &mut rotation {
                *v /= len;
            }
        }

        TrackerPose::new(position, rotation)
    }

    pub fn reset(&mut self) {
        for f in &mut self.position {
            f.reset();
        }
        for f in &mut self.rotation {
            f.reset();
        }
        self.prev_rotation = None;
        self.last_time = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_smoothing_factor_bounds() {
        // alpha should be between 0 and 1
        for &cutoff in &[0.1, 1.0, 10.0, 100.0] {
            for &te in &[0.001, 0.01, 0.033, 0.1] {
                let alpha = smoothing_factor(te, cutoff);
                assert!(alpha > 0.0 && alpha < 1.0, "alpha={} for te={}, cutoff={}", alpha, te, cutoff);
            }
        }
    }

    #[test]
    fn test_scalar_filter_passthrough_first() {
        let mut f = ScalarFilter::new(1.0, 0.0, 1.0);
        let result = f.filter(5.0, 0.033);
        assert_eq!(result, 5.0);
    }

    #[test]
    fn test_scalar_filter_smooths() {
        let mut f = ScalarFilter::new(1.0, 0.0, 1.0);
        f.filter(0.0, 0.033);
        let result = f.filter(10.0, 0.033);
        // With min_cutoff=1.0, beta=0, the filter should smooth significantly
        assert!(result < 10.0, "Expected smoothing, got {}", result);
        assert!(result > 0.0, "Expected positive value, got {}", result);
    }

    #[test]
    fn test_scalar_filter_high_beta_responsive() {
        // High beta: fast movements should pass through with less filtering
        let mut f_low_beta = ScalarFilter::new(1.0, 0.0, 1.0);
        let mut f_high_beta = ScalarFilter::new(1.0, 1.0, 1.0);

        f_low_beta.filter(0.0, 0.033);
        f_high_beta.filter(0.0, 0.033);

        let r_low = f_low_beta.filter(10.0, 0.033);
        let r_high = f_high_beta.filter(10.0, 0.033);

        // High beta should be closer to the target (less lag)
        assert!(r_high > r_low, "High beta ({}) should be more responsive than low beta ({})", r_high, r_low);
    }

    #[test]
    fn test_pose_filter_first_frame_passthrough() {
        let mut pf = PoseFilter::new(1.0, 0.01, 1.0, 0.01);
        let pose = TrackerPose::new([1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 1.0]);
        let result = pf.apply(pose);
        assert_eq!(result.position, pose.position);
        assert_eq!(result.rotation, pose.rotation);
    }

    #[test]
    fn test_pose_filter_quaternion_normalized() {
        let mut pf = PoseFilter::new(1.0, 0.01, 1.0, 0.01);
        let p1 = TrackerPose::new([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]);
        pf.apply(p1);
        std::thread::sleep(std::time::Duration::from_millis(10));

        let angle = std::f32::consts::FRAC_PI_4;
        let p2 = TrackerPose::new([1.0, 1.0, 1.0], [0.0, angle.sin(), 0.0, angle.cos()]);
        let result = pf.apply(p2);

        let len = (result.rotation[0] * result.rotation[0]
            + result.rotation[1] * result.rotation[1]
            + result.rotation[2] * result.rotation[2]
            + result.rotation[3] * result.rotation[3])
            .sqrt();
        assert!(
            (len - 1.0).abs() < 1e-5,
            "Quaternion not normalized: len={}",
            len
        );
    }

    #[test]
    fn test_pose_filter_reset() {
        let mut pf = PoseFilter::new(1.0, 0.01, 1.0, 0.01);
        let p1 = TrackerPose::new([1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 1.0]);
        pf.apply(p1);
        pf.reset();

        let p2 = TrackerPose::new([10.0, 20.0, 30.0], [0.0, 0.0, 0.0, 1.0]);
        let result = pf.apply(p2);
        // After reset, first frame should pass through
        assert_eq!(result.position, p2.position);
    }
}
