use anyhow::Result;
use ndarray::Array4;
use opencv::{
    core::{AlgorithmHint, Mat, Size, CV_32FC3},
    imgproc,
    prelude::*,
};

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

/// ImageNet正規化 + NCHW変換の共通処理
fn preprocess_imagenet_nchw(frame: &Mat, width: i32, height: i32) -> Result<Array4<f32>> {
    // BGR -> RGB
    let mut rgb = Mat::default();
    imgproc::cvt_color(frame, &mut rgb, imgproc::COLOR_BGR2RGB, 0, AlgorithmHint::ALGO_HINT_DEFAULT)?;

    // リサイズ
    let mut resized = Mat::default();
    imgproc::resize(
        &rgb,
        &mut resized,
        Size::new(width, height),
        0.0,
        0.0,
        imgproc::INTER_LINEAR,
    )?;

    // f32 に変換
    let mut float_mat = Mat::default();
    resized.convert_to(&mut float_mat, CV_32FC3, 1.0, 0.0)?;

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

    Ok(tensor)
}

/// SpinePose用の入力テンソルに変換
///
/// - BGR -> RGB -> 256x192 にリサイズ -> ImageNet正規化 -> NCHW [1, 3, 256, 192]
pub fn preprocess_for_spinepose(frame: &Mat) -> Result<Array4<f32>> {
    preprocess_imagenet_nchw(frame, SPINEPOSE_INPUT_WIDTH, SPINEPOSE_INPUT_HEIGHT)
}

/// RTMW3D用の入力テンソルに変換
///
/// - BGR -> RGB -> 384x288 にリサイズ -> ImageNet正規化 -> NCHW [1, 3, 384, 288]
pub fn preprocess_for_rtmw3d(frame: &Mat) -> Result<Array4<f32>> {
    preprocess_imagenet_nchw(frame, RTMW3D_INPUT_WIDTH, RTMW3D_INPUT_HEIGHT)
}
