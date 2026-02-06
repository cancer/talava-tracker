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
    let mut tensor = Array4::<f32>::zeros((1, MOVENET_INPUT_SIZE as usize, MOVENET_INPUT_SIZE as usize, 3));

    for y in 0..MOVENET_INPUT_SIZE {
        for x in 0..MOVENET_INPUT_SIZE {
            let pixel = float_mat.at_2d::<opencv::core::Vec3f>(y, x)?;
            tensor[[0, y as usize, x as usize, 0]] = pixel[0];
            tensor[[0, y as usize, x as usize, 1]] = pixel[1];
            tensor[[0, y as usize, x as usize, 2]] = pixel[2];
        }
    }

    Ok(tensor)
}
