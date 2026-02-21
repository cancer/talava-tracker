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
    shoulder_y: f32,
    torso_height: f32,
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
    scale_x: f32,
    scale_y: f32,
    body_scale: f32,
    leg_scale: f32,
    depth_scale: f32,
    mirror_x: bool,
    offset_y: f32,
    foot_y_offset: f32,
    fov_v_rad: f32,
    world_coords: bool, // 三角測量モード: キーポイントがワールド座標（メートル）
    calibration: Option<Calibration>,
}

impl BodyTracker {
    pub fn from_config(config: &TrackerConfig) -> Self {
        Self::new(config, 0.0, false)
    }

    pub fn new(config: &TrackerConfig, fov_v_deg: f32, world_coords: bool) -> Self {
        // world_coordsモード: 三角測量がメートル単位を出力するためスケール不要
        let (scale_x, scale_y, body_scale, leg_scale, depth_scale) = if world_coords {
            (1.0, 1.0, 1.0, 1.0, 1.0)
        } else {
            (config.scale_x, config.scale_y, config.body_scale, config.leg_scale, config.depth_scale)
        };
        Self {
            confidence_threshold: 0.2,
            scale_x,
            scale_y,
            body_scale,
            leg_scale,
            depth_scale,
            mirror_x: config.mirror_x,
            offset_y: config.offset_y,
            foot_y_offset: config.foot_y_offset,
            fov_v_rad: fov_v_deg.to_radians(),
            world_coords,
            calibration: None,
        }
    }

    /// フレーム境界の四肢キーポイントを無効化したPoseを返す
    /// 画像端のキーポイントはモデルの受容野が切れるため位置が不正確
    /// 体幹（腰・肩）は画像端でもゴーストになりにくいため対象外
    fn sanitize_pose(pose: &Pose) -> Pose {
        use KeypointIndex::*;
        // 四肢・肩: X/Y両方の境界チェック
        let full_boundary: &[KeypointIndex] = &[
            LeftShoulder, RightShoulder,
            LeftKnee, RightKnee, LeftAnkle, RightAnkle,
            LeftBigToe, RightBigToe, LeftSmallToe, RightSmallToe,
            LeftHeel, RightHeel,
        ];
        // 腰: X方向のみ境界チェック（Y方向は下半身がフレーム外でも許容）
        let x_only_boundary: &[KeypointIndex] = &[
            LeftHip, RightHip,
        ];
        let mut sanitized = pose.clone();
        for &idx in full_boundary {
            let kp = &mut sanitized.keypoints[idx as usize];
            if kp.x <= 0.02 || kp.x >= 0.98 || kp.y <= 0.02 || kp.y >= 0.98 {
                kp.confidence = 0.0;
            }
        }
        for &idx in x_only_boundary {
            let kp = &mut sanitized.keypoints[idx as usize];
            if kp.x <= 0.02 || kp.x >= 0.98 {
                kp.confidence = 0.0;
            }
        }
        sanitized
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
        // world_coordsモード: 生poseからチルト角を計算し、補正済みposeでキャリブレーション
        let tilt_angle = if self.world_coords {
            Self::compute_tilt_angle(pose, self.confidence_threshold)
        } else {
            0.0
        };

        let sanitized;
        let rotated;
        let pose = if self.world_coords {
            rotated = Self::rotate_pose_yz(pose, tilt_angle, self.confidence_threshold);
            &rotated
        } else {
            sanitized = Self::sanitize_pose(pose);
            &sanitized
        };
        let left_hip = pose.get(KeypointIndex::LeftHip);
        let right_hip = pose.get(KeypointIndex::RightHip);

        if !left_hip.is_valid(self.confidence_threshold)
            || !right_hip.is_valid(self.confidence_threshold)
        {
            return false;
        }

        let hip_x = (left_hip.x + right_hip.x) / 2.0;
        let hip_y = (left_hip.y + right_hip.y) / 2.0;

        let left_shoulder = pose.get(KeypointIndex::LeftShoulder);
        let right_shoulder = pose.get(KeypointIndex::RightShoulder);
        let shoulder_y = if left_shoulder.is_valid(self.confidence_threshold)
            && right_shoulder.is_valid(self.confidence_threshold)
        {
            (left_shoulder.y + right_shoulder.y) / 2.0
        } else {
            0.5
        };

        let torso_height = self.compute_torso_height(pose).unwrap_or(0.0);

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
            shoulder_y,
            torso_height,
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

    /// VR空間での最大スケール係数（閾値のスケーリング用）
    pub fn max_scale(&self) -> f32 {
        self.scale_x
            .max(self.scale_y)
            .max(self.body_scale)
            .max(self.leg_scale)
    }

    pub fn compute(&self, pose: &Pose) -> BodyPoses {
        // ワールド座標モードでは境界チェック不要（座標が0-1正規化ではなくメートル単位）
        // キャリブレーション済みならチルト補正を適用
        let sanitized;
        let rotated;
        let pose = if self.world_coords {
            let tilt = self.calibration.as_ref().map_or(0.0, |c| c.tilt_angle);
            if tilt.abs() > 0.001 {
                rotated = Self::rotate_pose_yz(pose, tilt, self.confidence_threshold);
                &rotated
            } else {
                pose
            }
        } else {
            sanitized = Self::sanitize_pose(pose);
            &sanitized
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
        let body_ratio = if self.world_coords { 1.0 } else { self.compute_body_ratio(pose) };

        // Per-keypoint z: world_coordsモードでは各キーポイントの三角測量z値を使用
        // 三角測量で同一カメラペア（hip基準）が強制されるため、z値の一貫性が保証される
        // 非world_coordsモードではhip推定深度を全部位に使用（2Dポーズにはz情報なし）
        let (lf_z, rf_z, ch_z, lk_z, rk_z) = if self.world_coords {
            (
                self.keypoint_depth(pose, KeypointIndex::LeftAnkle, hip_z),
                self.keypoint_depth(pose, KeypointIndex::RightAnkle, hip_z),
                hip_z, // 胸は肩-腰間の内挿位置、直接のキーポイントなし
                self.keypoint_depth(pose, KeypointIndex::LeftKnee, hip_z),
                self.keypoint_depth(pose, KeypointIndex::RightKnee, hip_z),
            )
        } else {
            (hip_z, hip_z, hip_z, hip_z, hip_z)
        };

        let mut left_foot = self.compute_left_foot(pose, hip_center, lf_z, body_ratio);
        let mut right_foot = self.compute_right_foot(pose, hip_center, rf_z, body_ratio);
        let mut left_knee = self.compute_left_knee(pose, hip_center, lk_z, body_ratio);
        let mut right_knee = self.compute_right_knee(pose, hip_center, rk_z, body_ratio);

        // 左右同一位置の検出: モデルが片側のみ検出時にもう片方を複製するアーティファクト
        Self::reject_duplicate_pair(&mut left_foot, &mut right_foot);
        Self::reject_duplicate_pair(&mut left_knee, &mut right_knee);

        BodyPoses {
            hip: self.compute_hip(pose, hip_center, hip_z, body_ratio),
            left_foot,
            right_foot,
            chest: self.compute_chest(pose, hip_center, ch_z, body_ratio),
            left_knee,
            right_knee,
        }
    }

    /// 画像座標yにおけるFOV補正係数
    /// カメラ中心から離れるほど斜め距離が増え、1ピクセルあたりの物理距離が大きくなる
    fn fov_scale(&self, y: f32) -> f32 {
        let half_fov = self.fov_v_rad / 2.0;
        if half_fov < 0.01 {
            return 1.0;
        }
        // カメラ高さ基準: 肩位置からの角度で補正
        // カメラが肩の高さにある場合、肩から離れるほど斜め距離が増える
        let camera_y = match &self.calibration {
            Some(cal) => cal.shoulder_y,
            None => 0.5,
        };
        let theta = ((y - camera_y) * 2.0 * half_fov.tan()).atan();
        1.0 / theta.cos()
    }

    fn convert_position(&self, x: f32, y: f32, hip_x: f32, hip_y: f32, part_scale: f32, pos_z: f32, body_ratio: f32) -> [f32; 3] {
        let (ref_x, ref_y) = match &self.calibration {
            Some(cal) => (cal.hip_x, cal.hip_y),
            None => (0.5, 0.5),
        };
        let fov = self.fov_scale(y);

        // パースペクティブ補正: 奥行き変化による見かけの腰位置シフトを除去
        // ピンホールカメラモデルでは画像中心(0.5)からのオフセットが深度に反比例する
        // 前後移動がVR空間の上下移動に化けるのを防ぐ
        // ワールド座標モードでは不要（三角測量が真の3D位置を算出済み）
        let (perspective_x_shift, perspective_y_shift) = if self.world_coords {
            (0.0, 0.0)
        } else {
            (
                (ref_x - 0.5) * (1.0 / body_ratio - 1.0),
                (ref_y - 0.5) * (1.0 / body_ratio - 1.0),
            )
        };

        let global_x = (ref_x - hip_x + perspective_x_shift) * self.scale_x;
        let global_y = (ref_y - hip_y + perspective_y_shift) * self.scale_y;

        let mut pos_x = (global_x + (hip_x - x) * part_scale * fov) * body_ratio;
        if self.mirror_x {
            pos_x = -pos_x;
        }
        let pos_y = self.offset_y + (global_y + (hip_y - y) * part_scale * fov) * body_ratio;
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

    fn estimate_depth(&self, pose: &Pose) -> f32 {
        // 3Dモデルのz座標がある場合
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
            return (ref_z - hip_z) * self.depth_scale;
        }

        // フォールバック: 胴体高さ比率ベース推定
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

    /// 個別キーポイントの奥行き（VR z座標）
    /// 三角測量のz値からキャリブレーション基準z値を引いてVR座標に変換
    /// キーポイントが無効な場合はfallback（通常はhip_z）を返す
    fn keypoint_depth(&self, pose: &Pose, kp_idx: KeypointIndex, fallback: f32) -> f32 {
        let kp = pose.get(kp_idx);
        if kp.is_valid(self.confidence_threshold) && kp.z.abs() > 0.001 {
            let ref_z = self.calibration.as_ref().map_or(0.0, |c| c.hip_z);
            (ref_z - kp.z) * self.depth_scale
        } else {
            fallback
        }
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

        let mut position = self.convert_position(ax, ay, hip_x, hip_y, self.leg_scale, pos_z, body_ratio);
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

    fn compute_right_foot(&self, pose: &Pose, hip_center: Option<(f32, f32)>, pos_z: f32, body_ratio: f32) -> Option<TrackerPose> {
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

        let mut position = self.convert_position(ax, ay, hip_x, hip_y, self.leg_scale, pos_z, body_ratio);
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

    fn compute_chest(&self, pose: &Pose, hip_center: Option<(f32, f32)>, pos_z: f32, body_ratio: f32) -> Option<TrackerPose> {
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

        let position = self.convert_position(x, y, hip_x, hip_y, self.body_scale, pos_z, body_ratio);

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
        let pitch = if has_keypoint {
            let ref_pitch = self.calibration.as_ref().map_or(0.0, |c| c.pitch_left_knee);
            self.compute_knee_pitch(pose, KeypointIndex::LeftKnee, KeypointIndex::LeftAnkle) - ref_pitch
        } else {
            0.0
        };
        let rotation = Self::yaw_pitch_to_quaternion(yaw, pitch);

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

    // --- ワールド座標（三角測量）モードのテスト ---

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
        let mut tracker = BodyTracker::new(&config, 0.0, true);
        let pose = standing_world_pose();
        // ワールド座標でもキャリブレーション成功すること
        assert!(tracker.calibrate(&pose), "calibration should succeed with world coords");
    }

    #[test]
    fn test_world_coords_all_trackers_returned() {
        let config = TrackerConfig::default();
        let mut tracker = BodyTracker::new(&config, 0.0, true);
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
        let mut tracker = BodyTracker::new(&config, 0.0, true);
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
        let mut tracker = BodyTracker::new(&config, 0.0, true);
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
        let mut tracker = BodyTracker::new(&config, 0.0, true);
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
        let mut tracker = BodyTracker::new(&config, 0.0, true);
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
        let mut tracker = BodyTracker::new(&config, 0.0, true);
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
        let mut tracker = BodyTracker::new(&config, 0.0, true);

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
