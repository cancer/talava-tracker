use crate::vmt::TrackerPose;
use std::time::Instant;

/// カメラフレーム間を線形外挿で補間するエクストラポレーター
pub struct Extrapolator {
    prev: Option<TrackerPose>,
    current: Option<TrackerPose>,
    velocity: [f32; 3],
    last_update: Option<Instant>,
}

impl Extrapolator {
    pub fn new() -> Self {
        Self {
            prev: None,
            current: None,
            velocity: [0.0; 3],
            last_update: None,
        }
    }

    /// 新しいカメラフレームの推論結果で更新する
    pub fn update(&mut self, pose: TrackerPose) {
        let now = Instant::now();

        if let (Some(prev_pose), Some(last_t)) = (self.current, self.last_update) {
            let dt = now.duration_since(last_t).as_secs_f32();
            if dt > 0.0 {
                self.velocity = [
                    (pose.position[0] - prev_pose.position[0]) / dt,
                    (pose.position[1] - prev_pose.position[1]) / dt,
                    (pose.position[2] - prev_pose.position[2]) / dt,
                ];
            }
        }

        self.prev = self.current;
        self.current = Some(pose);
        self.last_update = Some(now);
    }

    /// 現在のポーズを dt_secs 秒分だけ外挿して返す。回転は外挿しない。
    pub fn predict(&self, dt_secs: f32) -> Option<TrackerPose> {
        self.current.map(|cur| {
            TrackerPose::new(
                [
                    cur.position[0] + self.velocity[0] * dt_secs,
                    cur.position[1] + self.velocity[1] * dt_secs,
                    cur.position[2] + self.velocity[2] * dt_secs,
                ],
                cur.rotation,
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_predict_before_update() {
        let ext = Extrapolator::new();
        assert!(ext.predict(0.0).is_none());
        assert!(ext.predict(1.0).is_none());
    }

    #[test]
    fn test_predict_after_single_update() {
        let mut ext = Extrapolator::new();
        let pose = TrackerPose::new([1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 1.0]);
        ext.update(pose);

        // velocity is still zero after single update
        let p0 = ext.predict(0.0).unwrap();
        assert_eq!(p0.position, [1.0, 2.0, 3.0]);

        let p1 = ext.predict(0.1).unwrap();
        assert_eq!(p1.position, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_extrapolation_direction() {
        let mut ext = Extrapolator::new();
        // Manually set fields to control velocity without timing dependency
        ext.current = Some(TrackerPose::new([1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 1.0]));
        ext.velocity = [10.0, -5.0, 0.0]; // 10 units/s in x, -5 in y, 0 in z

        let predicted = ext.predict(0.1).unwrap();
        let eps = 1e-6;
        assert!((predicted.position[0] - 2.0).abs() < eps); // 1.0 + 10.0*0.1
        assert!((predicted.position[1] - 1.5).abs() < eps); // 2.0 + (-5.0)*0.1
        assert!((predicted.position[2] - 3.0).abs() < eps); // 3.0 + 0.0*0.1
    }

    #[test]
    fn test_rotation_not_extrapolated() {
        let mut ext = Extrapolator::new();
        let rotation = [0.1, 0.2, 0.3, 0.9];
        ext.current = Some(TrackerPose::new([0.0, 0.0, 0.0], rotation));
        ext.velocity = [1.0, 1.0, 1.0];

        let p = ext.predict(5.0).unwrap();
        assert_eq!(p.rotation, rotation);
    }
}
