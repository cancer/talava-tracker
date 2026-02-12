use anyhow::Result;
use opencv::{
    core::{Mat, Rect},
    prelude::*,
};

use super::keypoint::{Keypoint, KeypointIndex, Pose};

/// クロップ領域（正規化座標 0.0〜1.0）
#[derive(Debug, Clone, Copy)]
pub struct CropRegion {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl CropRegion {
    pub fn full() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            width: 1.0,
            height: 1.0,
        }
    }

    pub fn is_full(&self) -> bool {
        self.width >= 1.0 && self.height >= 1.0
    }
}

/// BBox（ピクセル座標）
#[derive(Debug, Clone, Copy)]
pub struct BBox {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

/// 前フレームのPoseからBBoxを推定（KP-based）
///
/// confidence_threshold以上のキーポイントのmin/maxからBBoxを返す。
/// 有効なキーポイントが2個未満ならNone。
pub fn bbox_from_keypoints(
    pose: &Pose,
    frame_w: u32,
    frame_h: u32,
    confidence_threshold: f32,
) -> Option<BBox> {
    let mut min_x = f32::MAX;
    let mut min_y = f32::MAX;
    let mut max_x = f32::MIN;
    let mut max_y = f32::MIN;
    let mut count = 0u32;

    for kp in &pose.keypoints {
        if kp.confidence >= confidence_threshold {
            let px = kp.x * frame_w as f32;
            let py = kp.y * frame_h as f32;
            min_x = min_x.min(px);
            min_y = min_y.min(py);
            max_x = max_x.max(px);
            max_y = max_y.max(py);
            count += 1;
        }
    }

    if count < 2 {
        return None;
    }

    Some(BBox {
        x: min_x,
        y: min_y,
        width: max_x - min_x,
        height: max_y - min_y,
    })
}

/// BBoxからクロップ領域を計算し、フレームからクロップ
///
/// - BBoxを1.25倍に拡張（中心を保持）
/// - 4:3アスペクト比（h:w = 256:192）に調整
/// - フレーム境界にクリップ
pub fn crop_for_pose(
    frame: &Mat,
    bbox: &BBox,
    frame_w: u32,
    frame_h: u32,
) -> Result<(Mat, CropRegion)> {
    // 1.25倍に拡張（中心を保持）
    let expand = 1.25;
    let cx = bbox.x + bbox.width / 2.0;
    let cy = bbox.y + bbox.height / 2.0;
    let mut w = bbox.width * expand;
    let mut h = bbox.height * expand;

    // 4:3アスペクト比に調整（h:w = 4:3）
    let target_ratio = 4.0 / 3.0; // height / width
    let current_ratio = h / w;
    if current_ratio < target_ratio {
        // 横長 → 高さを増やす
        h = w * target_ratio;
    } else {
        // 縦長 → 幅を増やす
        w = h / target_ratio;
    }

    // 左上座標
    let mut x = cx - w / 2.0;
    let mut y = cy - h / 2.0;

    // フレーム境界にクリップ
    let fw = frame_w as f32;
    let fh = frame_h as f32;
    x = x.max(0.0);
    y = y.max(0.0);
    w = w.min(fw - x);
    h = h.min(fh - y);

    // 整数座標に変換（Rect用）
    let rx = x as i32;
    let ry = y as i32;
    let rw = (w as i32).max(1);
    let rh = (h as i32).max(1);

    let roi = Rect::new(rx, ry, rw, rh);
    let cropped = Mat::roi(frame, roi)?;

    let crop_region = CropRegion {
        x: x / fw,
        y: y / fh,
        width: w / fw,
        height: h / fh,
    };

    Ok((cropped.try_clone()?, crop_region))
}

/// SpinePose出力座標（クロップ画像内の正規化座標）をフレーム全体の正規化座標に変換
pub fn remap_pose(pose: &Pose, crop: &CropRegion) -> Pose {
    let mut keypoints = [Keypoint::default(); KeypointIndex::COUNT];
    for i in 0..KeypointIndex::COUNT {
        let kp = &pose.keypoints[i];
        keypoints[i] = Keypoint {
            x: crop.x + kp.x * crop.width,
            y: crop.y + kp.y * crop.height,
            confidence: kp.confidence,
        };
    }
    Pose::new(keypoints)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crop_region_full() {
        let region = CropRegion::full();
        assert!(region.is_full());
        assert_eq!(region.x, 0.0);
        assert_eq!(region.y, 0.0);
    }

    #[test]
    fn test_bbox_from_keypoints_basic() {
        let mut keypoints = [Keypoint::default(); KeypointIndex::COUNT];
        // 2つの有効なキーポイントを設定（正規化座標）
        keypoints[KeypointIndex::LeftShoulder as usize] =
            Keypoint::new(0.3, 0.2, 0.9);
        keypoints[KeypointIndex::RightShoulder as usize] =
            Keypoint::new(0.7, 0.4, 0.8);

        let pose = Pose::new(keypoints);
        let bbox = bbox_from_keypoints(&pose, 640, 480, 0.5);
        assert!(bbox.is_some());

        let bbox = bbox.unwrap();
        // 0.3 * 640 = 192, 0.7 * 640 = 448
        assert!((bbox.x - 192.0).abs() < 1.0);
        // 0.2 * 480 = 96, 0.4 * 480 = 192
        assert!((bbox.y - 96.0).abs() < 1.0);
        assert!((bbox.width - 256.0).abs() < 1.0);
        assert!((bbox.height - 96.0).abs() < 1.0);
    }

    #[test]
    fn test_bbox_from_keypoints_low_confidence() {
        let mut keypoints = [Keypoint::default(); KeypointIndex::COUNT];
        // 1つだけ有効 → 2個未満なのでNone
        keypoints[KeypointIndex::Nose as usize] = Keypoint::new(0.5, 0.5, 0.9);

        let pose = Pose::new(keypoints);
        let bbox = bbox_from_keypoints(&pose, 640, 480, 0.5);
        assert!(bbox.is_none());
    }

    #[test]
    fn test_remap_pose() {
        let crop = CropRegion {
            x: 0.25,
            y: 0.1,
            width: 0.5,
            height: 0.8,
        };

        let mut keypoints = [Keypoint::default(); KeypointIndex::COUNT];
        keypoints[KeypointIndex::Nose as usize] = Keypoint::new(0.5, 0.5, 0.9);

        let pose = Pose::new(keypoints);
        let remapped = remap_pose(&pose, &crop);

        let nose = remapped.get(KeypointIndex::Nose);
        // new_x = 0.25 + 0.5 * 0.5 = 0.5
        assert!((nose.x - 0.5).abs() < 1e-6);
        // new_y = 0.1 + 0.5 * 0.8 = 0.5
        assert!((nose.y - 0.5).abs() < 1e-6);
        assert_eq!(nose.confidence, 0.9);

        // デフォルト（0,0）のキーポイントはcropの左上にマッピング
        let default_kp = remapped.get(KeypointIndex::LeftEye);
        assert!((default_kp.x - 0.25).abs() < 1e-6);
        assert!((default_kp.y - 0.1).abs() < 1e-6);
    }
}
