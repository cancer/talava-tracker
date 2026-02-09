/// 37 キーポイントインデックス
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(usize)]
pub enum KeypointIndex {
    Nose = 0,
    LeftEye = 1,
    RightEye = 2,
    LeftEar = 3,
    RightEar = 4,
    LeftShoulder = 5,
    RightShoulder = 6,
    LeftElbow = 7,
    RightElbow = 8,
    LeftWrist = 9,
    RightWrist = 10,
    LeftHip = 11,
    RightHip = 12,
    LeftKnee = 13,
    RightKnee = 14,
    LeftAnkle = 15,
    RightAnkle = 16,
    Head = 17,
    Neck = 18,
    Hip = 19,
    LeftBigToe = 20,
    RightBigToe = 21,
    LeftSmallToe = 22,
    RightSmallToe = 23,
    LeftHeel = 24,
    RightHeel = 25,
    Spine01 = 26,
    Spine02 = 27,
    Spine03 = 28,
    Spine04 = 29,
    Spine05 = 30,
    LeftLatissimus = 31,
    RightLatissimus = 32,
    LeftClavicle = 33,
    RightClavicle = 34,
    Neck02 = 35,
    Neck03 = 36,
}

impl KeypointIndex {
    pub const COUNT: usize = 37;

    pub fn from_index(index: usize) -> Option<Self> {
        match index {
            0 => Some(Self::Nose),
            1 => Some(Self::LeftEye),
            2 => Some(Self::RightEye),
            3 => Some(Self::LeftEar),
            4 => Some(Self::RightEar),
            5 => Some(Self::LeftShoulder),
            6 => Some(Self::RightShoulder),
            7 => Some(Self::LeftElbow),
            8 => Some(Self::RightElbow),
            9 => Some(Self::LeftWrist),
            10 => Some(Self::RightWrist),
            11 => Some(Self::LeftHip),
            12 => Some(Self::RightHip),
            13 => Some(Self::LeftKnee),
            14 => Some(Self::RightKnee),
            15 => Some(Self::LeftAnkle),
            16 => Some(Self::RightAnkle),
            17 => Some(Self::Head),
            18 => Some(Self::Neck),
            19 => Some(Self::Hip),
            20 => Some(Self::LeftBigToe),
            21 => Some(Self::RightBigToe),
            22 => Some(Self::LeftSmallToe),
            23 => Some(Self::RightSmallToe),
            24 => Some(Self::LeftHeel),
            25 => Some(Self::RightHeel),
            26 => Some(Self::Spine01),
            27 => Some(Self::Spine02),
            28 => Some(Self::Spine03),
            29 => Some(Self::Spine04),
            30 => Some(Self::Spine05),
            31 => Some(Self::LeftLatissimus),
            32 => Some(Self::RightLatissimus),
            33 => Some(Self::LeftClavicle),
            34 => Some(Self::RightClavicle),
            35 => Some(Self::Neck02),
            36 => Some(Self::Neck03),
            _ => None,
        }
    }
}

/// 単一キーポイント
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Keypoint {
    /// 正規化されたX座標 (0.0〜1.0)
    pub x: f32,
    /// 正規化されたY座標 (0.0〜1.0)
    pub y: f32,
    /// 信頼度スコア (0.0〜1.0)
    pub confidence: f32,
}

impl Keypoint {
    pub fn new(x: f32, y: f32, confidence: f32) -> Self {
        Self { x, y, confidence }
    }

    /// 信頼度が閾値以上か
    pub fn is_valid(&self, threshold: f32) -> bool {
        self.confidence >= threshold
    }

    /// ピクセル座標に変換
    pub fn to_pixel(&self, width: u32, height: u32) -> (i32, i32) {
        let px = (self.x * width as f32) as i32;
        let py = (self.y * height as f32) as i32;
        (px, py)
    }
}

impl Default for Keypoint {
    fn default() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            confidence: 0.0,
        }
    }
}

/// 37キーポイントからなる姿勢
#[derive(Debug, Clone)]
pub struct Pose {
    pub keypoints: [Keypoint; KeypointIndex::COUNT],
}

impl Pose {
    pub fn new(keypoints: [Keypoint; KeypointIndex::COUNT]) -> Self {
        Self { keypoints }
    }

    /// インデックスでキーポイントを取得
    pub fn get(&self, index: KeypointIndex) -> &Keypoint {
        &self.keypoints[index as usize]
    }

    /// 全キーポイントの平均信頼度
    pub fn average_confidence(&self) -> f32 {
        let sum: f32 = self.keypoints.iter().map(|k| k.confidence).sum();
        sum / KeypointIndex::COUNT as f32
    }
}

impl Default for Pose {
    fn default() -> Self {
        Self {
            keypoints: [Keypoint::default(); KeypointIndex::COUNT],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_keypoint_index_count() {
        assert_eq!(KeypointIndex::COUNT, 37);
    }

    #[test]
    fn test_keypoint_index_from_index() {
        assert_eq!(KeypointIndex::from_index(0), Some(KeypointIndex::Nose));
        assert_eq!(KeypointIndex::from_index(16), Some(KeypointIndex::RightAnkle));
        assert_eq!(KeypointIndex::from_index(17), Some(KeypointIndex::Head));
        assert_eq!(KeypointIndex::from_index(37), None);
    }

    #[test]
    fn test_keypoint_is_valid() {
        let kp = Keypoint::new(0.5, 0.5, 0.7);
        assert!(kp.is_valid(0.5));
        assert!(!kp.is_valid(0.8));
    }

    #[test]
    fn test_keypoint_to_pixel() {
        let kp = Keypoint::new(0.5, 0.25, 1.0);
        let (px, py) = kp.to_pixel(640, 480);
        assert_eq!(px, 320);
        assert_eq!(py, 120);
    }

    #[test]
    fn test_pose_get() {
        let mut keypoints = [Keypoint::default(); KeypointIndex::COUNT];
        keypoints[KeypointIndex::Nose as usize] = Keypoint::new(0.5, 0.3, 0.9);

        let pose = Pose::new(keypoints);
        let nose = pose.get(KeypointIndex::Nose);
        assert_eq!(nose.x, 0.5);
        assert_eq!(nose.y, 0.3);
        assert_eq!(nose.confidence, 0.9);
    }

    #[test]
    fn test_pose_average_confidence() {
        let keypoints = [Keypoint::new(0.0, 0.0, 0.5); KeypointIndex::COUNT];
        let pose = Pose::new(keypoints);
        assert!((pose.average_confidence() - 0.5).abs() < 0.001);
    }
}
