use crate::vmt::TrackerPose;

pub struct Lerper {
    start: Option<TrackerPose>,
    end: Option<TrackerPose>,
}

impl Lerper {
    pub fn new() -> Self {
        Self {
            start: None,
            end: None,
        }
    }

    /// 新しいカメラフレームの推論結果で呼ばれる。
    /// current_t: 現在の補間進捗 (0.0..=1.0)
    pub fn update(&mut self, pose: TrackerPose, current_t: f32) {
        match self.start {
            None => {
                // 初回: 両方に同じ値をセット
                self.start = Some(pose);
                self.end = Some(pose);
            }
            Some(_) => {
                // ジャンプ防止: 現在の補間値を新しいstartにする
                self.start = self.interpolate(current_t);
                self.end = Some(pose);
            }
        }
    }

    /// t (0.0..=1.0) に基づいて補間した TrackerPose を返す。
    /// update が一度も呼ばれていなければ None。
    pub fn interpolate(&self, t: f32) -> Option<TrackerPose> {
        let start = self.start?;
        let end = self.end?;
        let t = t.clamp(0.0, 1.0);

        let position = lerp_position(&start.position, &end.position, t);
        let rotation = nlerp(&start.rotation, &end.rotation, t);

        Some(TrackerPose::new(position, rotation))
    }
}

fn lerp_position(a: &[f32; 3], b: &[f32; 3], t: f32) -> [f32; 3] {
    [
        (1.0 - t) * a[0] + t * b[0],
        (1.0 - t) * a[1] + t * b[1],
        (1.0 - t) * a[2] + t * b[2],
    ]
}

fn nlerp(a: &[f32; 4], b: &[f32; 4], t: f32) -> [f32; 4] {
    // shortest path: dot < 0 なら end を反転
    let dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
    let sign = if dot < 0.0 { -1.0 } else { 1.0 };

    let mut result = [
        (1.0 - t) * a[0] + t * sign * b[0],
        (1.0 - t) * a[1] + t * sign * b[1],
        (1.0 - t) * a[2] + t * sign * b[2],
        (1.0 - t) * a[3] + t * sign * b[3],
    ];

    // normalize
    let len = (result[0] * result[0]
        + result[1] * result[1]
        + result[2] * result[2]
        + result[3] * result[3])
        .sqrt();
    if len > 0.0 {
        for v in &mut result {
            *v /= len;
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq_3(a: &[f32; 3], b: &[f32; 3], eps: f32) -> bool {
        (a[0] - b[0]).abs() < eps && (a[1] - b[1]).abs() < eps && (a[2] - b[2]).abs() < eps
    }

    fn quat_length(q: &[f32; 4]) -> f32 {
        (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt()
    }

    #[test]
    fn test_interpolate_before_update() {
        let lerper = Lerper::new();
        assert!(lerper.interpolate(0.5).is_none());
    }

    #[test]
    fn test_interpolate_at_zero() {
        let mut lerper = Lerper::new();
        let start = TrackerPose::new([1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 1.0]);
        let end = TrackerPose::new([4.0, 5.0, 6.0], [0.0, 0.0, 0.0, 1.0]);
        lerper.update(start, 0.0);
        lerper.update(end, 0.0);

        let result = lerper.interpolate(0.0).unwrap();
        assert!(approx_eq_3(&result.position, &start.position, 1e-6));
    }

    #[test]
    fn test_interpolate_at_one() {
        let mut lerper = Lerper::new();
        let start = TrackerPose::new([1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 1.0]);
        let end = TrackerPose::new([4.0, 5.0, 6.0], [0.0, 0.0, 0.0, 1.0]);
        lerper.update(start, 0.0);
        lerper.update(end, 0.0);

        let result = lerper.interpolate(1.0).unwrap();
        assert!(approx_eq_3(&result.position, &end.position, 1e-6));
    }

    #[test]
    fn test_interpolate_midpoint() {
        let mut lerper = Lerper::new();
        let start = TrackerPose::new([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]);
        let end = TrackerPose::new([2.0, 4.0, 6.0], [0.0, 0.0, 0.0, 1.0]);
        lerper.update(start, 0.0);
        lerper.update(end, 0.0);

        let result = lerper.interpolate(0.5).unwrap();
        let expected = [1.0, 2.0, 3.0];
        assert!(approx_eq_3(&result.position, &expected, 1e-6));
    }

    #[test]
    fn test_nlerp_normalized() {
        let mut lerper = Lerper::new();
        // 異なる回転のクォータニオン
        let a = TrackerPose::new([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]);
        let angle = std::f32::consts::FRAC_PI_4; // 45度
        let b = TrackerPose::new([1.0, 1.0, 1.0], [0.0, angle.sin(), 0.0, angle.cos()]);
        lerper.update(a, 0.0);
        lerper.update(b, 0.0);

        for &t in &[0.0, 0.25, 0.5, 0.75, 1.0] {
            let result = lerper.interpolate(t).unwrap();
            let len = quat_length(&result.rotation);
            assert!(
                (len - 1.0).abs() < 1e-5,
                "t={}: quaternion length {} is not unit",
                t,
                len
            );
        }
    }

    #[test]
    fn test_clamp_t() {
        let mut lerper = Lerper::new();
        let start = TrackerPose::new([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]);
        let end = TrackerPose::new([10.0, 10.0, 10.0], [0.0, 0.0, 0.0, 1.0]);
        lerper.update(start, 0.0);
        lerper.update(end, 0.0);

        // t < 0.0 は 0.0 にクランプ
        let result_neg = lerper.interpolate(-1.0).unwrap();
        let result_zero = lerper.interpolate(0.0).unwrap();
        assert!(approx_eq_3(
            &result_neg.position,
            &result_zero.position,
            1e-6
        ));

        // t > 1.0 は 1.0 にクランプ
        let result_over = lerper.interpolate(2.0).unwrap();
        let result_one = lerper.interpolate(1.0).unwrap();
        assert!(approx_eq_3(
            &result_over.position,
            &result_one.position,
            1e-6
        ));
    }
}
