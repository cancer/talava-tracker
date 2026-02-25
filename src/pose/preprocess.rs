use super::keypoint::{Keypoint, KeypointIndex, Pose};

/// レターボックス情報（推論後にキーポイント座標を元の画像空間に戻すために使用）
#[derive(Debug, Clone, Copy)]
pub struct LetterboxInfo {
    /// コンテンツ領域の左端（モデル入力幅に対する正規化座標 0.0-1.0）
    pub pad_left: f32,
    /// コンテンツ領域の上端（モデル入力高さに対する正規化座標 0.0-1.0）
    pub pad_top: f32,
    /// コンテンツ幅 / モデル入力幅（0.0-1.0）
    pub content_width: f32,
    /// コンテンツ高さ / モデル入力高さ（0.0-1.0）
    pub content_height: f32,
}

impl LetterboxInfo {
    pub fn identity() -> Self {
        Self {
            pad_left: 0.0,
            pad_top: 0.0,
            content_width: 1.0,
            content_height: 1.0,
        }
    }
}

/// レターボックス座標のキーポイントを元の画像座標に変換
pub fn unletterbox_pose(pose: &Pose, info: &LetterboxInfo) -> Pose {
    let mut keypoints = [Keypoint::default(); KeypointIndex::COUNT];
    for i in 0..KeypointIndex::COUNT {
        let kp = &pose.keypoints[i];
        keypoints[i] = Keypoint {
            x: (kp.x - info.pad_left) / info.content_width,
            y: (kp.y - info.pad_top) / info.content_height,
            z: kp.z,
            confidence: kp.confidence,
        };
    }
    Pose::new(keypoints)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_letterbox_info_identity() {
        let info = LetterboxInfo::identity();
        assert_eq!(info.pad_left, 0.0);
        assert_eq!(info.pad_top, 0.0);
        assert_eq!(info.content_width, 1.0);
        assert_eq!(info.content_height, 1.0);
    }

    #[test]
    fn test_unletterbox_center() {
        // ポートレートカメラ（9:16）→ モデル入力（3:4）のケース
        // 720x1280 → 192x256: scale=0.2 → 144x256, pad_left=24
        let info = LetterboxInfo {
            pad_left: 24.0 / 192.0,  // 0.125
            pad_top: 0.0,
            content_width: 144.0 / 192.0,  // 0.75
            content_height: 1.0,
        };

        let mut keypoints = [Keypoint::default(); KeypointIndex::COUNT];
        // モデル出力: 中心 (0.5, 0.5)
        keypoints[KeypointIndex::Nose as usize] = Keypoint::new(0.5, 0.5, 0.9);
        let pose = Pose::new(keypoints);

        let result = unletterbox_pose(&pose, &info);
        let nose = result.get(KeypointIndex::Nose);
        // (0.5 - 0.125) / 0.75 = 0.5
        assert!((nose.x - 0.5).abs() < 1e-4);
        assert!((nose.y - 0.5).abs() < 1e-4);
    }

    #[test]
    fn test_unletterbox_edge() {
        // pad_left=0.125, content_width=0.75のケース
        let info = LetterboxInfo {
            pad_left: 0.125,
            pad_top: 0.0,
            content_width: 0.75,
            content_height: 1.0,
        };

        let mut keypoints = [Keypoint::default(); KeypointIndex::COUNT];
        // パディング境界（コンテンツ左端）
        keypoints[KeypointIndex::Nose as usize] = Keypoint::new(0.125, 0.3, 0.9);
        let pose = Pose::new(keypoints);

        let result = unletterbox_pose(&pose, &info);
        let nose = result.get(KeypointIndex::Nose);
        // (0.125 - 0.125) / 0.75 = 0.0
        assert!((nose.x - 0.0).abs() < 1e-4);
    }

    #[test]
    fn test_unletterbox_landscape_padding() {
        // ランドスケープ（4:3）→ モデル入力（3:4）のケース
        // 640x480 → 192x256: scale=0.3 → 192x144, pad_top=56
        let info = LetterboxInfo {
            pad_left: 0.0,
            pad_top: 56.0 / 256.0,   // 0.21875
            content_width: 1.0,
            content_height: 144.0 / 256.0, // 0.5625
        };

        let mut keypoints = [Keypoint::default(); KeypointIndex::COUNT];
        keypoints[KeypointIndex::Nose as usize] = Keypoint::new(0.5, 0.5, 0.9);
        let pose = Pose::new(keypoints);

        let result = unletterbox_pose(&pose, &info);
        let nose = result.get(KeypointIndex::Nose);
        assert!((nose.x - 0.5).abs() < 1e-4);
        // (0.5 - 0.21875) / 0.5625 = 0.5
        assert!((nose.y - 0.5).abs() < 1e-4);
    }
}
