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

/// SpinePose用の入力テンソルに変換
///
/// - BGR -> RGB
/// - 256x192 にリサイズ
/// - ImageNet正規化: mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]
/// - [1, 3, 256, 192] の f32 NCHW テンソルに変換
pub fn preprocess_for_spinepose(frame: &Mat) -> Result<Array4<f32>> {
    // BGR -> RGB
    let mut rgb = Mat::default();
    imgproc::cvt_color(frame, &mut rgb, imgproc::COLOR_BGR2RGB, 0, AlgorithmHint::ALGO_HINT_DEFAULT)?;

    // 256x192 にリサイズ
    let mut resized = Mat::default();
    imgproc::resize(
        &rgb,
        &mut resized,
        Size::new(SPINEPOSE_INPUT_WIDTH, SPINEPOSE_INPUT_HEIGHT),
        0.0,
        0.0,
        imgproc::INTER_LINEAR,
    )?;

    // f32 に変換
    let mut float_mat = Mat::default();
    resized.convert_to(&mut float_mat, CV_32FC3, 1.0, 0.0)?;

    // ImageNet正規化 & NCHW変換 [1, 3, 256, 192]
    let h = SPINEPOSE_INPUT_HEIGHT as usize;
    let w = SPINEPOSE_INPUT_WIDTH as usize;
    let mean = [123.675f32, 116.28, 103.53];
    let std_dev = [58.395f32, 57.12, 57.375];
    let mut tensor = Array4::<f32>::zeros((1, 3, h, w));
    let data = float_mat.data_bytes()?;
    let step = float_mat.mat_step().get(0);
    for y_idx in 0..h {
        let row_ptr = unsafe {
            std::slice::from_raw_parts(
                data.as_ptr().add(y_idx * step) as *const f32,
                w * 3,
            )
        };
        for x_idx in 0..w {
            for c in 0..3 {
                let val = row_ptr[x_idx * 3 + c];
                tensor[[0, c, y_idx, x_idx]] = (val - mean[c]) / std_dev[c];
            }
        }
    }

    Ok(tensor)
}
