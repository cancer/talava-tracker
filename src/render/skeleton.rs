use crate::pose::KeypointIndex;

/// 骨格の接続定義 (開始キーポイント, 終了キーポイント)
pub const SKELETON_CONNECTIONS: [(KeypointIndex, KeypointIndex); 16] = [
    // 顔
    (KeypointIndex::LeftEar, KeypointIndex::LeftEye),
    (KeypointIndex::LeftEye, KeypointIndex::Nose),
    (KeypointIndex::Nose, KeypointIndex::RightEye),
    (KeypointIndex::RightEye, KeypointIndex::RightEar),
    // 上半身
    (KeypointIndex::LeftShoulder, KeypointIndex::RightShoulder),
    (KeypointIndex::LeftShoulder, KeypointIndex::LeftElbow),
    (KeypointIndex::LeftElbow, KeypointIndex::LeftWrist),
    (KeypointIndex::RightShoulder, KeypointIndex::RightElbow),
    (KeypointIndex::RightElbow, KeypointIndex::RightWrist),
    // 胴体
    (KeypointIndex::LeftShoulder, KeypointIndex::LeftHip),
    (KeypointIndex::RightShoulder, KeypointIndex::RightHip),
    (KeypointIndex::LeftHip, KeypointIndex::RightHip),
    // 下半身
    (KeypointIndex::LeftHip, KeypointIndex::LeftKnee),
    (KeypointIndex::LeftKnee, KeypointIndex::LeftAnkle),
    (KeypointIndex::RightHip, KeypointIndex::RightKnee),
    (KeypointIndex::RightKnee, KeypointIndex::RightAnkle),
];

/// キーポイントの色 (RGB)
pub const KEYPOINT_COLOR: u32 = 0x00FF00; // 緑

/// 骨格線の色 (RGB)
pub const SKELETON_COLOR: u32 = 0xFFFF00; // 黄色

/// 信頼度が低いキーポイントの色 (RGB)
pub const LOW_CONFIDENCE_COLOR: u32 = 0xFF0000; // 赤
