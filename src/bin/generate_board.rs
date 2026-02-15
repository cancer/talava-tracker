use anyhow::Result;
use opencv::core::{Mat, Point, Scalar, Size};
use opencv::imgcodecs;
use opencv::imgproc;
use opencv::prelude::*;

use talava_tracker::calibration::create_board;
use talava_tracker::config::Config;

const CONFIG_PATH: &str = "config.toml";

fn main() -> Result<()> {
    let config = Config::load(CONFIG_PATH)?;
    let cal = &config.calibration;

    let board = create_board(cal)?;

    // ボード画像生成（余白なし）
    let board_margin = 0;
    let border_bits = 1;

    // マス1辺のピクセルサイズを決めてボードサイズを算出
    let px_per_square = 120;
    let board_w = cal.squares_x * px_per_square;
    let board_h = cal.squares_y * px_per_square;
    let board_size = Size::new(board_w, board_h);

    let mut board_img = Mat::default();
    board.generate_image(board_size, &mut board_img, board_margin, border_bits)?;

    // ルーラー領域のサイズ
    let ruler_h = 80; // 下部ルーラー高さ
    let ruler_w = 60; // 左部ルーラー幅
    let pad = 20; // ボード周囲の余白
    let text_h = 40; // 下部テキスト高さ

    let total_w = ruler_w + pad + board_w + pad;
    let total_h = pad + board_h + pad + ruler_h + text_h;

    // 白背景キャンバス
    let mut canvas = Mat::new_rows_cols_with_default(total_h, total_w, opencv::core::CV_8UC1, Scalar::all(255.0))?;

    // ボード画像をキャンバスに配置
    let board_x = ruler_w + pad;
    let board_y = pad;
    let mut roi = Mat::roi_mut(
        &mut canvas,
        opencv::core::Rect::new(board_x, board_y, board_w, board_h),
    )?;
    board_img.copy_to(&mut roi)?;

    // --- 下部ルーラー（横方向、マス幅を計測用） ---
    let ruler_y_start = board_y + board_h + pad;
    let ruler_y_end = ruler_y_start + 30;
    let black = Scalar::all(0.0);
    // ルーラーベースライン
    imgproc::line(
        &mut canvas,
        Point::new(board_x, ruler_y_start),
        Point::new(board_x + board_w, ruler_y_start),
        black, 2, imgproc::LINE_8, 0,
    )?;

    // マスごとの目盛り
    for i in 0..=cal.squares_x {
        let x = board_x + i * px_per_square;
        let tick_len = 20;
        imgproc::line(
            &mut canvas,
            Point::new(x, ruler_y_start),
            Point::new(x, ruler_y_start + tick_len),
            black, 2, imgproc::LINE_8, 0,
        )?;

        // 数字ラベル
        imgproc::put_text(
            &mut canvas,
            &format!("{}", i),
            Point::new(x - 5, ruler_y_start + tick_len + 18),
            imgproc::FONT_HERSHEY_SIMPLEX, 0.5, black, 1, imgproc::LINE_8, false,
        )?;
    }

    // "1 square" 矢印範囲
    let arrow_y = ruler_y_end + 22;
    let arr_x0 = board_x;
    let arr_x1 = board_x + px_per_square;
    imgproc::arrowed_line(
        &mut canvas,
        Point::new(arr_x0, arrow_y),
        Point::new(arr_x1 - 2, arrow_y),
        black, 1, imgproc::LINE_8, 0, 0.15,
    )?;
    imgproc::arrowed_line(
        &mut canvas,
        Point::new(arr_x1, arrow_y),
        Point::new(arr_x0 + 2, arrow_y),
        black, 1, imgproc::LINE_8, 0, 0.15,
    )?;

    // 説明テキスト
    let text_y = ruler_y_end + 55;
    imgproc::put_text(
        &mut canvas,
        "measure 1 square -> config.toml square_length (m)",
        Point::new(board_x, text_y),
        imgproc::FONT_HERSHEY_SIMPLEX, 0.5, black, 1, imgproc::LINE_8, false,
    )?;

    // --- 左部ルーラー（縦方向） ---
    let ruler_x_start = ruler_w - 5;

    // ベースライン
    imgproc::line(
        &mut canvas,
        Point::new(ruler_x_start, board_y),
        Point::new(ruler_x_start, board_y + board_h),
        black, 2, imgproc::LINE_8, 0,
    )?;

    // マスごとの目盛り
    for i in 0..=cal.squares_y {
        let y = board_y + i * px_per_square;
        let tick_len = 15;
        imgproc::line(
            &mut canvas,
            Point::new(ruler_x_start - tick_len, y),
            Point::new(ruler_x_start, y),
            black, 2, imgproc::LINE_8, 0,
        )?;

        imgproc::put_text(
            &mut canvas,
            &format!("{}", i),
            Point::new(ruler_x_start - tick_len - 18, y + 5),
            imgproc::FONT_HERSHEY_SIMPLEX, 0.5, black, 1, imgproc::LINE_8, false,
        )?;
    }

    let filename = format!(
        "charuco_{}x{}_{}.png",
        cal.squares_x, cal.squares_y, cal.dictionary
    );
    imgcodecs::imwrite(&filename, &canvas, &opencv::core::Vector::new())?;
    println!("保存: {}", filename);
    println!("  マス数: {}x{}", cal.squares_x, cal.squares_y);
    println!("  辞書: {}", cal.dictionary);
    println!("  モニターに表示し、1マスの幅を物理的に測ってconfig.tomlのsquare_lengthに入力");
    println!("  marker_length = square_length * 0.75 が目安");

    Ok(())
}
