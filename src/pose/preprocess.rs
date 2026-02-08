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
