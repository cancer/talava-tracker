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
    hip_z: f32, // 3Dモデルの基準z座標（キャリブレーション時の腰z）
    yaw_shoulder: f32,
    yaw_left_foot: f32,
    yaw_right_foot: f32,
    // 腰からの相対オフセット（画像座標）
    left_knee_offset: Option<(f32, f32)>,
    right_knee_offset: Option<(f32, f32)>,
    yaw_left_knee: f32,
    yaw_right_knee: f32,
    pitch_left_knee: f32,
    pitch_right_knee: f32,
    tilt_angle: f32, // カメラチルト角（ラジアン）: YZ平面の回転補正用
}

pub struct BodyTracker {
    confidence_threshold: f32,
    mirror_x: bool,
    offset_y: f32,
    foot_y_offset: f32,
    calibration: Option<Calibration>,
}

impl BodyTracker {
    pub fn new(config: &TrackerConfig) -> Self {
        Self {
            confidence_threshold: 0.2,
            mirror_x: config.mirror_x,
            offset_y: config.offset_y,
            foot_y_offset: config.foot_y_offset,
            calibration: None,
        }
    }

    /// カメラのチルト角を推定（ラジアン）
    /// hip中点とankle中点（fallback: knee中点）のYZ差分からX軸まわりの回転角を算出
    /// カメラが下向きに設置されている場合、実世界の鉛直方向（重力）とカメラY軸がずれている
    /// 立っている人の体幹は鉛直なので、hip→ankleのベクトルから傾きを推定できる
    fn compute_tilt_angle(pose: &Pose, confidence_threshold: f32) -> f32 {
        let left_hip = pose.get(KeypointIndex::LeftHip);
        let right_hip = pose.get(KeypointIndex::RightHip);
        if !left_hip.is_valid(confidence_threshold) || !right_hip.is_valid(confidence_threshold) {
            return 0.0;
        }
        let hip_y = (left_hip.y + right_hip.y) / 2.0;
        let hip_z = (left_hip.z + right_hip.z) / 2.0;

        // ankle優先、fallback: knee
        let left_ankle = pose.get(KeypointIndex::LeftAnkle);
        let right_ankle = pose.get(KeypointIndex::RightAnkle);
        let left_knee = pose.get(KeypointIndex::LeftKnee);
        let right_knee = pose.get(KeypointIndex::RightKnee);

        let (lower_y, lower_z) = if left_ankle.is_valid(confidence_threshold)
            && right_ankle.is_valid(confidence_threshold)
        {
            (
                (left_ankle.y + right_ankle.y) / 2.0,
                (left_ankle.z + right_ankle.z) / 2.0,
            )
        } else if left_knee.is_valid(confidence_threshold)
            && right_knee.is_valid(confidence_threshold)
        {
            (
                (left_knee.y + right_knee.y) / 2.0,
                (left_knee.z + right_knee.z) / 2.0,
            )
        } else {
            return 0.0;
        };

        let delta_y = lower_y - hip_y; // カメラY: 下が正
        let delta_z = lower_z - hip_z; // カメラZ: 前方が正

        // 鉛直ベクトル（重力方向）はカメラ座標系でY+Z成分を持つ
        // チルト角 = atan2(delta_z, delta_y)
        // 立っている人のhip→ankleは鉛直なので、このベクトルがカメラのY軸からどれだけ傾いているかがチルト角
        f32::atan2(delta_z, delta_y)
    }

    /// PoseのYZ座標をチルト角で回転補正
    /// カメラ座標系を重力基準に変換する
    /// Y軸: hip中点のzを全keypointの回転に使用（per-keypoint z三角測量ノイズ→Y漏れ防止）
    /// Z軸: 各keypointの個別z値を使用（正確な奥行き保持）
    fn rotate_pose_yz(pose: &Pose, tilt: f32, confidence_threshold: f32) -> Pose {
        let (sin_t, cos_t) = tilt.sin_cos();

        // hip中点zを基準にY回転（全keypointで共通）
        // 前後移動による均一なz変化が各keypoint Yに同じ補正を与える
        // per-keypoint zノイズがY軸に漏れるのを防ぐ
        let left_hip = pose.get(KeypointIndex::LeftHip);
        let right_hip = pose.get(KeypointIndex::RightHip);
        let ref_z = if left_hip.is_valid(confidence_threshold)
            && right_hip.is_valid(confidence_threshold)
        {
            (left_hip.z + right_hip.z) / 2.0
        } else {
            0.0
        };

        let mut rotated = pose.clone();
        for kp in rotated.keypoints.iter_mut() {
            let y = kp.y;
            let z = kp.z;
            kp.y = y * cos_t + ref_z * sin_t;
            kp.z = -y * sin_t + z * cos_t;
        }
        rotated
    }

    pub fn calibrate(&mut self, pose: &Pose) -> bool {
        let tilt_angle = Self::compute_tilt_angle(pose, self.confidence_threshold);

        let rotated = Self::rotate_pose_yz(pose, tilt_angle, self.confidence_threshold);
        let pose = &rotated;

        let left_hip = pose.get(KeypointIndex::LeftHip);
        let right_hip = pose.get(KeypointIndex::RightHip);

        if !left_hip.is_valid(self.confidence_threshold)
            || !right_hip.is_valid(self.confidence_threshold)
        {
            return false;
        }

        let hip_x = (left_hip.x + right_hip.x) / 2.0;
        let hip_y = (left_hip.y + right_hip.y) / 2.0;

        // 3Dモデルのz座標を保存（キャリブレーション基準点）
        let hip_z = (left_hip.z + right_hip.z) / 2.0;

        let yaw_shoulder = self.compute_shoulder_yaw(pose);
        let yaw_left_foot =
            self.compute_foot_yaw(pose, KeypointIndex::LeftKnee, KeypointIndex::LeftAnkle);
        let yaw_right_foot =
            self.compute_foot_yaw(pose, KeypointIndex::RightKnee, KeypointIndex::RightAnkle);

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
        let pitch_left_knee =
            self.compute_knee_pitch(pose, KeypointIndex::LeftKnee, KeypointIndex::LeftAnkle);
        let pitch_right_knee =
            self.compute_knee_pitch(pose, KeypointIndex::RightKnee, KeypointIndex::RightAnkle);

        self.calibration = Some(Calibration {
            hip_x,
            hip_y,
            hip_z,
            yaw_shoulder,
            yaw_left_foot,
            yaw_right_foot,
            left_knee_offset,
            right_knee_offset,
            yaw_left_knee,
            yaw_right_knee,
            pitch_left_knee,
            pitch_right_knee,
            tilt_angle,
        });
        true
    }

    pub fn is_calibrated(&self) -> bool {
        self.calibration.is_some()
    }

    pub fn compute(&self, pose: &Pose) -> BodyPoses {
        // キャリブレーション済みならチルト補正を適用
        let rotated;
        let pose = {
            let tilt = self.calibration.as_ref().map_or(0.0, |c| c.tilt_angle);
            if tilt.abs() > 0.001 {
                rotated = Self::rotate_pose_yz(pose, tilt, self.confidence_threshold);
                &rotated
            } else {
                pose
            }
        };
        let left_hip = pose.get(KeypointIndex::LeftHip);
        let right_hip = pose.get(KeypointIndex::RightHip);
        let hip_center = if left_hip.is_valid(self.confidence_threshold)
            && right_hip.is_valid(self.confidence_threshold)
        {
            Some(((left_hip.x + right_hip.x) / 2.0, (left_hip.y + right_hip.y) / 2.0))
        } else {
            None
        };

        let hip_z = self.estimate_depth(pose);

        // Per-keypoint z: 各キーポイントの三角測量z値を使用
        let lf_z = self.keypoint_depth(pose, KeypointIndex::LeftAnkle, hip_z);
        let rf_z = self.keypoint_depth(pose, KeypointIndex::RightAnkle, hip_z);
        let ch_z = hip_z; // 胸は肩-腰間の内挿位置、直接のキーポイントなし
        let lk_z = self.keypoint_depth(pose, KeypointIndex::LeftKnee, hip_z);
        let rk_z = self.keypoint_depth(pose, KeypointIndex::RightKnee, hip_z);

        let mut left_foot = self.compute_left_foot(pose, hip_center, lf_z);
        let mut right_foot = self.compute_right_foot(pose, hip_center, rf_z);
        let mut left_knee = self.compute_left_knee(pose, hip_center, lk_z);
        let mut right_knee = self.compute_right_knee(pose, hip_center, rk_z);

        // 左右同一位置の検出: モデルが片側のみ検出時にもう片方を複製するアーティファクト
        Self::reject_duplicate_pair(&mut left_foot, &mut right_foot);
        Self::reject_duplicate_pair(&mut left_knee, &mut right_knee);

        BodyPoses {
            hip: self.compute_hip(pose, hip_center, hip_z),
            left_foot,
            right_foot,
            chest: self.compute_chest(pose, hip_center, ch_z),
            left_knee,
            right_knee,
        }
    }

    fn convert_position(&self, x: f32, y: f32, hip_x: f32, hip_y: f32, pos_z: f32) -> [f32; 3] {
        let (ref_x, ref_y) = match &self.calibration {
            Some(cal) => (cal.hip_x, cal.hip_y),
            None => (0.5, 0.5),
        };

        let global_x = ref_x - hip_x;
        let global_y = ref_y - hip_y;

        let mut pos_x = global_x + (hip_x - x);
        if self.mirror_x {
            pos_x = -pos_x;
        }
        let pos_y = self.offset_y + global_y + (hip_y - y);
        [pos_x, pos_y, pos_z]
    }

    fn estimate_depth(&self, pose: &Pose) -> f32 {
        let left_hip = pose.get(KeypointIndex::LeftHip);
        let right_hip = pose.get(KeypointIndex::RightHip);
        if left_hip.is_valid(self.confidence_threshold)
            && right_hip.is_valid(self.confidence_threshold)
            && (left_hip.z.abs() > 0.001 || right_hip.z.abs() > 0.001)
        {
            let hip_z = (left_hip.z + right_hip.z) / 2.0;
            // キャリブレーション済みなら基準z値からの差分を使用
            // カメラ座標系(z=奥)→VR座標系(z=前)の変換: 符号反転
            let ref_z = self.calibration.as_ref().map_or(0.0, |c| c.hip_z);
            return ref_z - hip_z;
        }

        0.0
    }

    /// 個別キーポイントの奥行き（VR z座標）
    /// 三角測量のz値からキャリブレーション基準z値を引いてVR座標に変換
    /// キーポイントが無効な場合はfallback（通常はhip_z）を返す
    fn keypoint_depth(&self, pose: &Pose, kp_idx: KeypointIndex, fallback: f32) -> f32 {
        let kp = pose.get(kp_idx);
        if kp.is_valid(self.confidence_threshold) && kp.z.abs() > 0.001 {
            let ref_z = self.calibration.as_ref().map_or(0.0, |c| c.hip_z);
            ref_z - kp.z
        } else {
            fallback
        }
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

    /// 左右の同一位置検出: 距離が0.05未満の場合、両方をNoneにする
    fn reject_duplicate_pair(left: &mut Option<TrackerPose>, right: &mut Option<TrackerPose>) {
        if let (Some(l), Some(r)) = (left.as_ref(), right.as_ref()) {
            let dx = l.position[0] - r.position[0];
            let dy = l.position[1] - r.position[1];
            let dist = (dx * dx + dy * dy).sqrt();
            if dist < 0.05 {
                *left = None;
                *right = None;
            }
        }
    }

    fn yaw_to_quaternion(yaw: f32) -> [f32; 4] {
        let half = yaw / 2.0;
        [0.0, half.sin(), 0.0, half.cos()]
    }

    /// yaw(Y軸回転) + pitch(X軸回転) から quaternion を生成
    /// q = q_yaw * q_pitch
    fn yaw_pitch_to_quaternion(yaw: f32, pitch: f32) -> [f32; 4] {
        let (sy, cy) = (yaw / 2.0).sin_cos();
        let (sp, cp) = (pitch / 2.0).sin_cos();
        [
            cy * sp,       // x
            sy * cp,       // y
            -sy * sp,      // z
            cy * cp,       // w
        ]
    }

    fn compute_hip(&self, pose: &Pose, hip_center: Option<(f32, f32)>, pos_z: f32) -> Option<TrackerPose> {
        let (hip_x, hip_y) = hip_center?;
        let position = self.convert_position(hip_x, hip_y, hip_x, hip_y, pos_z);

        let ref_yaw = self.calibration.as_ref().map_or(0.0, |c| c.yaw_shoulder);
        let yaw = self.compute_shoulder_yaw(pose) - ref_yaw;
        let rotation = Self::yaw_to_quaternion(yaw);

        Some(TrackerPose::new(position, rotation))
    }

    fn compute_left_foot(&self, pose: &Pose, hip_center: Option<(f32, f32)>, pos_z: f32) -> Option<TrackerPose> {
        let (hip_x, hip_y) = hip_center?;
        let knee = pose.get(KeypointIndex::LeftKnee);
        let ankle = pose.get(KeypointIndex::LeftAnkle);

        let ankle_valid = ankle.is_valid(self.confidence_threshold);
        let knee_valid = knee.is_valid(self.confidence_threshold);

        // 足の位置: ankle優先、未検出時はknee代用、両方なしはNone
        let (ax, ay) = if ankle_valid {
            (ankle.x, ankle.y)
        } else if knee_valid {
            (knee.x, knee.y)
        } else {
            return None;
        };

        let mut position = self.convert_position(ax, ay, hip_x, hip_y, pos_z);
        position[1] += self.foot_y_offset;

        // yaw: knee+ankle両方有効な場合のみ計算、それ以外は0
        let yaw = if knee_valid && ankle_valid {
            let ref_yaw = self.calibration.as_ref().map_or(0.0, |c| c.yaw_left_foot);
            self.compute_foot_yaw(pose, KeypointIndex::LeftKnee, KeypointIndex::LeftAnkle) - ref_yaw
        } else {
            0.0
        };
        let rotation = Self::yaw_to_quaternion(yaw);

        Some(TrackerPose::new(position, rotation))
    }

    fn compute_right_foot(&self, pose: &Pose, hip_center: Option<(f32, f32)>, pos_z: f32) -> Option<TrackerPose> {
        let (hip_x, hip_y) = hip_center?;
        let knee = pose.get(KeypointIndex::RightKnee);
        let ankle = pose.get(KeypointIndex::RightAnkle);

        let ankle_valid = ankle.is_valid(self.confidence_threshold);
        let knee_valid = knee.is_valid(self.confidence_threshold);

        // 足の位置: ankle優先、未検出時はknee代用、両方なしはNone
        let (ax, ay) = if ankle_valid {
            (ankle.x, ankle.y)
        } else if knee_valid {
            (knee.x, knee.y)
        } else {
            return None;
        };

        let mut position = self.convert_position(ax, ay, hip_x, hip_y, pos_z);
        position[1] += self.foot_y_offset;

        // yaw: knee+ankle両方有効な場合のみ計算、それ以外は0
        let yaw = if knee_valid && ankle_valid {
            let ref_yaw = self.calibration.as_ref().map_or(0.0, |c| c.yaw_right_foot);
            self.compute_foot_yaw(pose, KeypointIndex::RightKnee, KeypointIndex::RightAnkle) - ref_yaw
        } else {
            0.0
        };
        let rotation = Self::yaw_to_quaternion(yaw);

        Some(TrackerPose::new(position, rotation))
    }

    fn compute_chest(&self, pose: &Pose, hip_center: Option<(f32, f32)>, pos_z: f32) -> Option<TrackerPose> {
        let (hip_x, hip_y) = hip_center?;
        let left_shoulder = pose.get(KeypointIndex::LeftShoulder);
        let right_shoulder = pose.get(KeypointIndex::RightShoulder);
        let ls_valid = left_shoulder.is_valid(self.confidence_threshold);
        let rs_valid = right_shoulder.is_valid(self.confidence_threshold);

        if !ls_valid && !rs_valid {
            return None;
        }

        // 肩のY座標: 片肩のみ有効な場合はその肩を使用
        let shoulder_y = match (ls_valid, rs_valid) {
            (true, true) => (left_shoulder.y + right_shoulder.y) / 2.0,
            (true, false) => left_shoulder.y,
            (false, true) => right_shoulder.y,
            _ => unreachable!(),
        };
        // X座標: 両肩検出時は肩中点（傾き追従）、片肩のみはhip_x（片肩xは偏るため）
        let x = match (ls_valid, rs_valid) {
            (true, true) => (left_shoulder.x + right_shoulder.x) / 2.0,
            _ => hip_x,
        };
        // Y座標: 肩と腰中点を内挿して胸骨付近を推定
        let y = shoulder_y + (hip_y - shoulder_y) * 0.35;

        let position = self.convert_position(x, y, hip_x, hip_y, pos_z);

        let ref_yaw = self.calibration.as_ref().map_or(0.0, |c| c.yaw_shoulder);
        let yaw = self.compute_shoulder_yaw(pose) - ref_yaw;
        let rotation = Self::yaw_to_quaternion(yaw);

        Some(TrackerPose::new(position, rotation))
    }

    /// 膝のpitch: 膝→足首ベクトルの鉛直方向からの傾き
    /// 現在はz値が歪み補正なしで不正確なため無効化（常に0を返す）
    /// TODO: カメラキャリブレーションのdist_coeffs問題を解決後に再有効化
    fn compute_knee_pitch(&self, _pose: &Pose, _knee_idx: KeypointIndex, _ankle_idx: KeypointIndex) -> f32 {
        0.0
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

    fn compute_left_knee(&self, pose: &Pose, hip_center: Option<(f32, f32)>, pos_z: f32) -> Option<TrackerPose> {
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

        let position = self.convert_position(kx, ky, hip_x, hip_y, pos_z);

        let yaw = if has_keypoint {
            let ref_yaw = self.calibration.as_ref().map_or(0.0, |c| c.yaw_left_knee);
            self.compute_knee_yaw(pose, KeypointIndex::LeftHip, KeypointIndex::LeftKnee) - ref_yaw
        } else {
            0.0
        };
        let pitch = if has_keypoint {
            let ref_pitch = self.calibration.as_ref().map_or(0.0, |c| c.pitch_left_knee);
            self.compute_knee_pitch(pose, KeypointIndex::LeftKnee, KeypointIndex::LeftAnkle) - ref_pitch
        } else {
            0.0
        };
        let rotation = Self::yaw_pitch_to_quaternion(yaw, pitch);

        Some(TrackerPose::new(position, rotation))
    }

    fn compute_right_knee(&self, pose: &Pose, hip_center: Option<(f32, f32)>, pos_z: f32) -> Option<TrackerPose> {
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

        let position = self.convert_position(kx, ky, hip_x, hip_y, pos_z);

        let yaw = if has_keypoint {
            let ref_yaw = self.calibration.as_ref().map_or(0.0, |c| c.yaw_right_knee);
            self.compute_knee_yaw(pose, KeypointIndex::RightHip, KeypointIndex::RightKnee) - ref_yaw
        } else {
            0.0
        };
        let pitch = if has_keypoint {
            let ref_pitch = self.calibration.as_ref().map_or(0.0, |c| c.pitch_right_knee);
            self.compute_knee_pitch(pose, KeypointIndex::RightKnee, KeypointIndex::RightAnkle) - ref_pitch
        } else {
            0.0
        };
        let rotation = Self::yaw_pitch_to_quaternion(yaw, pitch);

        Some(TrackerPose::new(position, rotation))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pose::Keypoint;

    /// カメラ座標系（X右, Y下, Z前方）メートル単位のPoseを生成
    /// カメラ高さ1.0m、人物2.0m前方を想定
    fn make_world_pose(
        left_hip: (f32, f32, f32),
        right_hip: (f32, f32, f32),
        left_shoulder: (f32, f32, f32),
        right_shoulder: (f32, f32, f32),
        left_knee: (f32, f32, f32),
        right_knee: (f32, f32, f32),
        left_ankle: (f32, f32, f32),
        right_ankle: (f32, f32, f32),
    ) -> Pose {
        let mut keypoints = [Keypoint::default(); KeypointIndex::COUNT];
        keypoints[KeypointIndex::LeftHip as usize] =
            Keypoint::new_3d(left_hip.0, left_hip.1, left_hip.2, 0.9);
        keypoints[KeypointIndex::RightHip as usize] =
            Keypoint::new_3d(right_hip.0, right_hip.1, right_hip.2, 0.9);
        keypoints[KeypointIndex::LeftShoulder as usize] =
            Keypoint::new_3d(left_shoulder.0, left_shoulder.1, left_shoulder.2, 0.9);
        keypoints[KeypointIndex::RightShoulder as usize] =
            Keypoint::new_3d(right_shoulder.0, right_shoulder.1, right_shoulder.2, 0.9);
        keypoints[KeypointIndex::LeftKnee as usize] =
            Keypoint::new_3d(left_knee.0, left_knee.1, left_knee.2, 0.9);
        keypoints[KeypointIndex::RightKnee as usize] =
            Keypoint::new_3d(right_knee.0, right_knee.1, right_knee.2, 0.9);
        keypoints[KeypointIndex::LeftAnkle as usize] =
            Keypoint::new_3d(left_ankle.0, left_ankle.1, left_ankle.2, 0.9);
        keypoints[KeypointIndex::RightAnkle as usize] =
            Keypoint::new_3d(right_ankle.0, right_ankle.1, right_ankle.2, 0.9);
        Pose::new(keypoints)
    }

    /// カメラ座標: 身長1.7mの人, カメラ高さ1.0m, 2m前方
    /// Y下方向: 肩y=-0.4(カメラより上), 腰y=0.1, 膝y=0.5, 足首y=0.9
    fn standing_world_pose() -> Pose {
        make_world_pose(
            (0.15, 0.1, 2.0), (-0.15, 0.1, 2.0),     // hips
            (0.2, -0.4, 2.0), (-0.2, -0.4, 2.0),     // shoulders
            (0.15, 0.5, 2.0), (-0.15, 0.5, 2.0),     // knees
            (0.15, 0.9, 2.0), (-0.15, 0.9, 2.0),     // ankles
        )
    }

    #[test]
    fn test_world_coords_calibration_succeeds() {
        let config = TrackerConfig::default();
        let mut tracker = BodyTracker::new(&config);
        let pose = standing_world_pose();
        assert!(tracker.calibrate(&pose), "calibration should succeed with world coords");
    }

    #[test]
    fn test_world_coords_all_trackers_returned() {
        let config = TrackerConfig::default();
        let mut tracker = BodyTracker::new(&config);
        let pose = standing_world_pose();
        tracker.calibrate(&pose);

        let result = tracker.compute(&pose);
        assert!(result.hip.is_some(), "hip");
        assert!(result.left_foot.is_some(), "left_foot");
        assert!(result.right_foot.is_some(), "right_foot");
        assert!(result.chest.is_some(), "chest");
        assert!(result.left_knee.is_some(), "left_knee");
        assert!(result.right_knee.is_some(), "right_knee");
    }

    #[test]
    fn test_world_coords_anatomy() {
        let config = TrackerConfig::default();
        let mut tracker = BodyTracker::new(&config);
        let pose = standing_world_pose();
        tracker.calibrate(&pose);

        let result = tracker.compute(&pose);
        let hip = result.hip.unwrap();
        let chest = result.chest.unwrap();
        let left_foot = result.left_foot.unwrap();
        let right_foot = result.right_foot.unwrap();
        let left_knee = result.left_knee.unwrap();
        let right_knee = result.right_knee.unwrap();

        // 胸 > 腰 > 膝 > 足（VR Y軸: 上が正）
        assert!(chest.position[1] > hip.position[1],
            "chest.y({:.3}) > hip.y({:.3})", chest.position[1], hip.position[1]);
        assert!(hip.position[1] > left_knee.position[1],
            "hip.y({:.3}) > left_knee.y({:.3})", hip.position[1], left_knee.position[1]);
        assert!(left_knee.position[1] > left_foot.position[1],
            "left_knee.y({:.3}) > left_foot.y({:.3})", left_knee.position[1], left_foot.position[1]);
        assert!(hip.position[1] > right_knee.position[1],
            "hip.y({:.3}) > right_knee.y({:.3})", hip.position[1], right_knee.position[1]);
        assert!(right_knee.position[1] > right_foot.position[1],
            "right_knee.y({:.3}) > right_foot.y({:.3})", right_knee.position[1], right_foot.position[1]);
    }

    #[test]
    fn test_world_coords_calibration_zeroes_hip() {
        let config = TrackerConfig::default();
        let mut tracker = BodyTracker::new(&config);
        let pose = standing_world_pose();
        tracker.calibrate(&pose);

        let result = tracker.compute(&pose);
        let hip = result.hip.unwrap();
        // キャリブレーション位置でhip.x≈0, hip.z≈0
        assert!((hip.position[0]).abs() < 0.01,
            "hip.x should be ~0 at calibration, got {}", hip.position[0]);
        assert!((hip.position[2]).abs() < 0.01,
            "hip.z should be ~0 at calibration, got {}", hip.position[2]);
    }

    #[test]
    fn test_world_coords_lateral_movement() {
        let config = TrackerConfig::default();
        let mut tracker = BodyTracker::new(&config);
        let pose = standing_world_pose();
        tracker.calibrate(&pose);

        // 人物が0.3m右に移動（カメラのX+方向）
        let moved_pose = make_world_pose(
            (0.45, 0.1, 2.0), (0.15, 0.1, 2.0),
            (0.50, -0.4, 2.0), (0.10, -0.4, 2.0),
            (0.45, 0.5, 2.0), (0.15, 0.5, 2.0),
            (0.45, 0.9, 2.0), (0.15, 0.9, 2.0),
        );
        let result = tracker.compute(&moved_pose);
        let hip = result.hip.unwrap();
        // カメラX+方向の移動 → VR pos_xが変化する（方向はmirror_x依存）
        assert!(hip.position[0].abs() > 0.1,
            "hip.x should reflect lateral movement, got {}", hip.position[0]);
    }

    #[test]
    fn test_world_coords_depth_movement() {
        let config = TrackerConfig::default();
        let mut tracker = BodyTracker::new(&config);
        let pose = standing_world_pose();
        tracker.calibrate(&pose);

        // 人物が0.5mカメラに近づく（z: 2.0 → 1.5）
        let closer_pose = make_world_pose(
            (0.15, 0.1, 1.5), (-0.15, 0.1, 1.5),
            (0.2, -0.4, 1.5), (-0.2, -0.4, 1.5),
            (0.15, 0.5, 1.5), (-0.15, 0.5, 1.5),
            (0.15, 0.9, 1.5), (-0.15, 0.9, 1.5),
        );
        let result = tracker.compute(&closer_pose);
        let hip = result.hip.unwrap();
        // カメラに近づく→VR前方=正のz: (2.0 - 1.5) * 1.0 = 0.5
        assert!(hip.position[2] > 0.0,
            "closer to camera should give positive z (forward in VR), got {}", hip.position[2]);
    }

    #[test]
    fn test_world_coords_squat() {
        let config = TrackerConfig::default();
        let mut tracker = BodyTracker::new(&config);
        let pose = standing_world_pose();
        tracker.calibrate(&pose);

        // しゃがむ: 腰がY+方向（下）に移動
        let squat_pose = make_world_pose(
            (0.15, 0.5, 2.0), (-0.15, 0.5, 2.0),     // hip下がる
            (0.2, 0.0, 2.0), (-0.2, 0.0, 2.0),       // 肩も下がる
            (0.25, 0.6, 2.0), (-0.25, 0.6, 2.0),     // 膝は少し下
            (0.15, 0.9, 2.0), (-0.15, 0.9, 2.0),     // 足首は同じ
        );
        let cal_result = tracker.compute(&standing_world_pose());
        let squat_result = tracker.compute(&squat_pose);

        let cal_hip_y = cal_result.hip.unwrap().position[1];
        let squat_hip_y = squat_result.hip.unwrap().position[1];
        // しゃがむとVR Y座標が下がる
        assert!(squat_hip_y < cal_hip_y,
            "squat hip.y({:.3}) should be lower than standing({:.3})", squat_hip_y, cal_hip_y);
    }

    /// チルト角30°のカメラで前後移動した場合、足のY座標が変化しないことを確認
    /// カメラが30°下向き: 前後移動がカメラYとZ両方に成分を持つ
    /// チルト補正により、VR Y軸にはZ成分が漏れないはず
    #[test]
    fn test_tilt_correction_z_does_not_leak_into_y() {
        let config = TrackerConfig::default();
        let mut tracker = BodyTracker::new(&config);

        // カメラが30°下向き: 立っている人のhip→ankleが(Δy=0.69, Δz=0.40)に見える
        // 真の鉛直距離0.8m, tilt=30° → Δy=0.8*cos(30°)≈0.69, Δz=0.8*sin(30°)=0.40
        let tilt = 30.0_f32.to_radians();
        let (sin_t, cos_t) = tilt.sin_cos();

        // 真の鉛直座標（重力基準）で人物を定義してからカメラ座標に変換
        // 真のY: hip=0.0, ankle=0.8（下が正）
        // 真のZ: 全部2.0m前方
        let hip_true_y = 0.0_f32;
        let ankle_true_y = 0.8_f32;
        let shoulder_true_y = -0.5_f32;
        let knee_true_y = 0.4_f32;
        let true_z = 2.0_f32;

        // カメラ座標に変換: y_cam = y_true * cos(θ) - z_true * sin(θ)  [逆回転]
        //                   z_cam = y_true * sin(θ) + z_true * cos(θ)
        // ※ rotate_pose_yzは順回転なので、逆変換は-θで行う
        let to_cam = |y: f32, z: f32| -> (f32, f32) {
            (y * cos_t - z * sin_t, y * sin_t + z * cos_t)
        };

        let (hip_cy, hip_cz) = to_cam(hip_true_y, true_z);
        let (shoulder_cy, shoulder_cz) = to_cam(shoulder_true_y, true_z);
        let (knee_cy, knee_cz) = to_cam(knee_true_y, true_z);
        let (ankle_cy, ankle_cz) = to_cam(ankle_true_y, true_z);

        let cal_pose = make_world_pose(
            (0.15, hip_cy, hip_cz), (-0.15, hip_cy, hip_cz),
            (0.2, shoulder_cy, shoulder_cz), (-0.2, shoulder_cy, shoulder_cz),
            (0.15, knee_cy, knee_cz), (-0.15, knee_cy, knee_cz),
            (0.15, ankle_cy, ankle_cz), (-0.15, ankle_cy, ankle_cz),
        );
        assert!(tracker.calibrate(&cal_pose));

        // 前後移動: 真のZ方向に0.5m前進（真のYは変化しない）
        let moved_z = 1.5_f32; // 2.0 → 1.5（カメラに近づく）
        let (hip_my, hip_mz) = to_cam(hip_true_y, moved_z);
        let (shoulder_my, shoulder_mz) = to_cam(shoulder_true_y, moved_z);
        let (knee_my, knee_mz) = to_cam(knee_true_y, moved_z);
        let (ankle_my, ankle_mz) = to_cam(ankle_true_y, moved_z);

        let moved_pose = make_world_pose(
            (0.15, hip_my, hip_mz), (-0.15, hip_my, hip_mz),
            (0.2, shoulder_my, shoulder_mz), (-0.2, shoulder_my, shoulder_mz),
            (0.15, knee_my, knee_mz), (-0.15, knee_my, knee_mz),
            (0.15, ankle_my, ankle_mz), (-0.15, ankle_my, ankle_mz),
        );

        let cal_result = tracker.compute(&cal_pose);
        let moved_result = tracker.compute(&moved_pose);

        let cal_foot_y = cal_result.left_foot.unwrap().position[1];
        let moved_foot_y = moved_result.left_foot.unwrap().position[1];

        // 前後移動ではY座標がほぼ変化しないこと（漏れが0.05m以下）
        let y_diff = (moved_foot_y - cal_foot_y).abs();
        assert!(y_diff < 0.05,
            "forward movement should not leak into Y: cal_foot.y={:.3}, moved_foot.y={:.3}, diff={:.3}",
            cal_foot_y, moved_foot_y, y_diff);

        // 一方でZ座標は変化すること（前進）
        let cal_hip_z = cal_result.hip.unwrap().position[2];
        let moved_hip_z = moved_result.hip.unwrap().position[2];
        assert!(moved_hip_z > cal_hip_z + 0.1,
            "forward movement should increase z: cal_hip.z={:.3}, moved_hip.z={:.3}",
            cal_hip_z, moved_hip_z);
    }

}
