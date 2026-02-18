use anyhow::Result;
use ndarray::Array4;
use opencv::{
    core::{AlgorithmHint, Mat, Scalar, Size, CV_32FC3},
    imgproc,
    prelude::*,
};

use super::keypoint::{Keypoint, KeypointIndex, Pose};

/// MoveNet用の入力サイズ
pub const MOVENET_INPUT_SIZE: i32 = 192;

/// OpenCV Mat を MoveNet用の入力テンソルに変換
///
/// - BGR -> RGB
/// - 192x192 にリサイズ
/// - [1, 192, 192, 3] の f32 テンソルに変換 (0.0-255.0)
pub fn preprocess_for_movenet(frame: &Mat) -> Result<Array4<f32>> {
    // BGR -> RGB
    let mut rgb = Mat::default();
    imgproc::cvt_color(frame, &mut rgb, imgproc::COLOR_BGR2RGB, 0, AlgorithmHint::ALGO_HINT_DEFAULT)?;

    // 192x192 にリサイズ
    let mut resized = Mat::default();
    imgproc::resize(
        &rgb,
        &mut resized,
        Size::new(MOVENET_INPUT_SIZE, MOVENET_INPUT_SIZE),
        0.0,
        0.0,
        imgproc::INTER_LINEAR,
    )?;

    // f32 に変換
    let mut float_mat = Mat::default();
    resized.convert_to(&mut float_mat, CV_32FC3, 1.0, 0.0)?;

    // ndarray に変換 [1, 192, 192, 3]
    let h = MOVENET_INPUT_SIZE as usize;
    let w = MOVENET_INPUT_SIZE as usize;
    let mut tensor = Array4::<f32>::zeros((1, h, w, 3));
    let dst = tensor.as_slice_mut().unwrap();
    let row_floats = w * 3;
    let data = float_mat.data_bytes()?;
    let step = float_mat.mat_step().get(0); // 行あたりのバイト数（パディング含む）
    for y in 0..h {
        let src = unsafe {
            std::slice::from_raw_parts(
                data.as_ptr().add(y * step) as *const f32,
                row_floats,
            )
        };
        dst[y * row_floats..(y + 1) * row_floats].copy_from_slice(src);
    }

    Ok(tensor)
}

/// SpinePose入力サイズ
pub const SPINEPOSE_INPUT_HEIGHT: i32 = 256;
pub const SPINEPOSE_INPUT_WIDTH: i32 = 192;

/// RTMW3D入力サイズ
pub const RTMW3D_INPUT_HEIGHT: i32 = 384;
pub const RTMW3D_INPUT_WIDTH: i32 = 288;

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

/// ImageNet正規化 + NCHW変換の共通処理（レターボックス対応）
///
/// アスペクト比を保持してリサイズし、余白をImageNet平均値で埋める。
fn preprocess_imagenet_nchw(frame: &Mat, width: i32, height: i32) -> Result<(Array4<f32>, LetterboxInfo)> {
    let frame_w = frame.cols() as f32;
    let frame_h = frame.rows() as f32;
    let target_w = width as f32;
    let target_h = height as f32;

    // アスペクト比を保持するスケールを計算
    let scale = (target_w / frame_w).min(target_h / frame_h);
    let new_w = (frame_w * scale).round() as i32;
    let new_h = (frame_h * scale).round() as i32;

    // パディング量
    let pad_left = (width - new_w) / 2;
    let pad_top = (height - new_h) / 2;

    // BGR -> RGB
    let mut rgb = Mat::default();
    imgproc::cvt_color(frame, &mut rgb, imgproc::COLOR_BGR2RGB, 0, AlgorithmHint::ALGO_HINT_DEFAULT)?;

    // アスペクト比保持リサイズ
    let mut resized = Mat::default();
    imgproc::resize(
        &rgb,
        &mut resized,
        Size::new(new_w, new_h),
        0.0,
        0.0,
        imgproc::INTER_LINEAR,
    )?;

    // ImageNet平均値(RGB)でパディング
    let mean_rgb = Scalar::new(123.675, 116.28, 103.53, 0.0);
    let mut padded = Mat::default();
    let pad_right = width - new_w - pad_left;
    let pad_bottom = height - new_h - pad_top;
    opencv::core::copy_make_border(
        &resized,
        &mut padded,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        opencv::core::BORDER_CONSTANT,
        mean_rgb,
    )?;

    // f32 に変換
    let mut float_mat = Mat::default();
    padded.convert_to(&mut float_mat, CV_32FC3, 1.0, 0.0)?;

    // ImageNet正規化 & NCHW変換
    let h = height as usize;
    let w = width as usize;
    let mean = [123.675f32, 116.28, 103.53];
    let inv_std = [1.0 / 58.395f32, 1.0 / 57.12, 1.0 / 57.375];
    let mut tensor = Array4::<f32>::zeros((1, 3, h, w));
    let dst = tensor.as_slice_mut().unwrap();
    let ch_stride = h * w;
    let data = float_mat.data_bytes()?;
    let step = float_mat.mat_step().get(0);
    for y_idx in 0..h {
        let row_ptr = unsafe {
            std::slice::from_raw_parts(
                data.as_ptr().add(y_idx * step) as *const f32,
                w * 3,
            )
        };
        let row_offset = y_idx * w;
        for x_idx in 0..w {
            let src_base = x_idx * 3;
            let dst_base = row_offset + x_idx;
            dst[dst_base] = (row_ptr[src_base] - mean[0]) * inv_std[0];
            dst[ch_stride + dst_base] = (row_ptr[src_base + 1] - mean[1]) * inv_std[1];
            dst[2 * ch_stride + dst_base] = (row_ptr[src_base + 2] - mean[2]) * inv_std[2];
        }
    }

    let letterbox = LetterboxInfo {
        pad_left: pad_left as f32 / target_w,
        pad_top: pad_top as f32 / target_h,
        content_width: new_w as f32 / target_w,
        content_height: new_h as f32 / target_h,
    };

    Ok((tensor, letterbox))
}

/// SpinePose用の入力テンソルに変換
///
/// - BGR -> RGB -> アスペクト比保持リサイズ+レターボックス -> ImageNet正規化 -> NCHW [1, 3, 256, 192]
pub fn preprocess_for_spinepose(frame: &Mat) -> Result<(Array4<f32>, LetterboxInfo)> {
    preprocess_imagenet_nchw(frame, SPINEPOSE_INPUT_WIDTH, SPINEPOSE_INPUT_HEIGHT)
}

/// RTMW3D用の入力テンソルに変換
///
/// - BGR -> RGB -> アスペクト比保持リサイズ+レターボックス -> ImageNet正規化 -> NCHW [1, 3, 384, 288]
pub fn preprocess_for_rtmw3d(frame: &Mat) -> Result<(Array4<f32>, LetterboxInfo)> {
    preprocess_imagenet_nchw(frame, RTMW3D_INPUT_WIDTH, RTMW3D_INPUT_HEIGHT)
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
