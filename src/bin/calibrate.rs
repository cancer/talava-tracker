use anyhow::{bail, Result};
use minifb::Key;
use opencv::core::{Mat, Size, Scalar};
use opencv::{calib3d, prelude::*};
use std::io::Write;
use std::time::Instant;

use talava_tracker::calibration::{
    BoardParams, CameraCalibration, MultiCameraCalibration,
    create_board, create_detector, detect_charuco_corners,
    calibrate_intrinsics, estimate_pose, compute_relative_poses,
    mat3x3_to_array, mat_to_vec, save_calibration,
};
use talava_tracker::camera::{OpenCvCamera, ThreadedCamera, detect_cameras};
use talava_tracker::config::Config;
use talava_tracker::render::MinifbRenderer;

const CONFIG_PATH: &str = "config.toml";

fn parse_camera_args() -> Option<Vec<i32>> {
    let args: Vec<String> = std::env::args().collect();
    // Usage: calibrate [camera_index ...]
    // e.g.  calibrate 0 2
    if args.len() <= 1 {
        return None;
    }
    let indices: Vec<i32> = args[1..]
        .iter()
        .filter_map(|s| s.parse().ok())
        .collect();
    if indices.is_empty() { None } else { Some(indices) }
}

fn main() -> Result<()> {
    let config = Config::load(CONFIG_PATH)?;
    let cal_config = &config.calibration;

    println!("=== ChArUco キャリブレーションツール ===");
    println!();
    println!("ボード設定:");
    println!("  辞書: {}", cal_config.dictionary);
    println!("  マス数: {}x{}", cal_config.squares_x, cal_config.squares_y);
    println!("  マス辺長: {}m", cal_config.square_length);
    println!("  マーカー辺長: {}m", cal_config.marker_length);
    println!("  出力先: {}", cal_config.output_path);
    println!("  内部パラメータ用フレーム数: {}", cal_config.intrinsic_frames);
    println!();

    // [1/4] カメラ検出・選択
    let camera_indices = if let Some(manual) = parse_camera_args() {
        println!("[1/4] カメラ指定: {:?}", manual);
        manual
    } else {
        println!("[1/4] カメラ検出中...");
        let found = detect_cameras(10);
        if found.is_empty() {
            bail!("カメラが見つかりません");
        }
        println!("  検出: {:?}", found);

        if found.len() == 1 {
            found
        } else {
            // プレビュー付きカメラ選択
            // 各ウィンドウで映像を確認し、そのウィンドウでSpaceを押して選択/解除
            println!();
            println!("  カメラ選択:");
            println!("  [Space] そのウィンドウのカメラを選択/解除");
            println!("  [Enter] 確定  [Q] 中止");
            println!();

            let mut cameras: Vec<OpenCvCamera> = Vec::new();
            let mut renderers: Vec<MinifbRenderer> = Vec::new();
            let mut selected = vec![false; found.len()];

            for &idx in &found {
                let mut cam = OpenCvCamera::open_with_resolution(idx, Some(1280), Some(960))?;
                // 実フレームサイズを取得してウィンドウを合わせる
                let first_frame = cam.read_frame()?;
                let fw = first_frame.cols() as usize;
                let fh = first_frame.rows() as usize;
                renderers.push(MinifbRenderer::new(
                    &format!("Camera {}", idx),
                    fw,
                    fh,
                )?);
                cameras.push(cam);
            }

            let mut confirmed = false;
            while !confirmed && renderers.iter().any(|r| r.is_open()) {
                for (i, cam) in cameras.iter_mut().enumerate() {
                    let frame = match cam.read_frame() {
                        Ok(f) => f,
                        Err(_) => continue,
                    };

                    let mut display = frame.clone();

                    // 選択状態のオーバーレイ
                    if selected[i] {
                        // 緑枠
                        let h = display.rows();
                        let w = display.cols();
                        let green = Scalar::new(0.0, 255.0, 0.0, 0.0);
                        let thickness = 6;
                        opencv::imgproc::rectangle(
                            &mut display,
                            opencv::core::Rect::new(0, 0, w, h),
                            green, thickness, opencv::imgproc::LINE_8, 0,
                        )?;
                        opencv::imgproc::put_text(
                            &mut display, "SELECTED",
                            opencv::core::Point::new(10, 30),
                            opencv::imgproc::FONT_HERSHEY_SIMPLEX, 0.8,
                            green, 2, opencv::imgproc::LINE_8, false,
                        )?;
                    } else {
                        opencv::imgproc::put_text(
                            &mut display, "Space: select",
                            opencv::core::Point::new(10, 30),
                            opencv::imgproc::FONT_HERSHEY_SIMPLEX, 0.7,
                            Scalar::new(200.0, 200.0, 200.0, 0.0),
                            2, opencv::imgproc::LINE_8, false,
                        )?;
                    }

                    renderers[i].draw_frame(&display)?;
                    renderers[i].update()?;

                    // そのウィンドウでSpaceが押されたらトグル
                    if renderers[i].is_key_pressed(Key::Space) {
                        selected[i] = !selected[i];
                    }
                }

                if renderers.iter().any(|r| r.is_key_pressed(Key::Enter)) {
                    if selected.iter().any(|&s| s) {
                        confirmed = true;
                    } else {
                        println!("  1台以上選択してください");
                    }
                }
                if renderers.iter().any(|r| r.is_key_pressed(Key::Q)) {
                    bail!("ユーザーにより中止されました");
                }
            }

            if !confirmed {
                bail!("カメラ選択が完了しませんでした");
            }

            let chosen: Vec<i32> = found.iter().zip(selected.iter())
                .filter(|(_, &sel)| sel)
                .map(|(&idx, _)| idx)
                .collect();
            println!("  選択: {:?}", chosen);
            chosen
        }
    };
    println!();

    // ボード作成
    let board = create_board(cal_config)?;
    let detector = create_detector(&board)?;

    let num_cameras = camera_indices.len();

    // [2/4] 内部パラメータキャリブレーション
    println!("[2/4] 内部パラメータキャリブレーション");
    println!("  各カメラでChArUcoボードを様々な角度から見せてください");
    println!("  6コーナー以上検出されると自動キャプチャします（1秒間隔）");
    println!("  [Q] 中止");
    println!();

    let capture_interval = std::time::Duration::from_millis(1000);

    let mut camera_matrices: Vec<Mat> = Vec::new();
    let mut dist_coeffs_vec: Vec<Mat> = Vec::new();
    let mut image_sizes: Vec<Size> = Vec::new();
    let mut reprojection_errors: Vec<f64> = Vec::new();

    // ログ用データ蓄積
    let mut log_lines: Vec<String> = Vec::new();
    log_lines.push(format!("=== キャリブレーションログ ==="));
    log_lines.push(format!("日時: {}", chrono::Local::now().format("%Y-%m-%d %H:%M:%S")));
    log_lines.push(format!("ボード: {} {}x{} square={:.3}m marker={:.3}m",
        cal_config.dictionary, cal_config.squares_x, cal_config.squares_y,
        cal_config.square_length, cal_config.marker_length));
    log_lines.push(format!("カメラ: {:?}", camera_indices));
    log_lines.push(String::new());

    for (cam_idx, &cam_id) in camera_indices.iter().enumerate() {
        println!("--- カメラ {} ({}/{}) ---", cam_id, cam_idx + 1, num_cameras);

        let camera = ThreadedCamera::start(cam_id, Some(1280), Some(960))?;
        let (w, h) = camera.resolution();
        let image_size = Size::new(w as i32, h as i32);
        image_sizes.push(image_size);

        let mut renderer = MinifbRenderer::new(
            &format!("Calibrate - Camera {}", cam_id),
            w as usize,
            h as usize,
        )?;

        // k3異常値検出時にリトライするための外側ループ
        let (k_result, dist_result, error_result, capture_stats) = loop {
            let mut all_corners: Vec<Mat> = Vec::new();
            let mut all_ids: Vec<Mat> = Vec::new();
            let mut capture_stats: Vec<(i32, f64, f64, f64, f64, f64)> = Vec::new(); // (n_corners, area_ratio, min_x, min_y, max_x, max_y)
            let mut last_capture = Instant::now() - capture_interval;

            println!("  キャプチャ: 0/{}", cal_config.intrinsic_frames);

            while all_corners.len() < cal_config.intrinsic_frames && renderer.is_open() {
                let frame = match camera.get_frame() {
                    Some(f) => f,
                    None => continue,
                };

                // ChArUcoコーナー検出（処理に時間がかかる）
                let (corners, ids, n_corners) = detect_charuco_corners(&detector, &frame)?;

                // 自動キャプチャ
                let captured_now = if n_corners >= 6 && last_capture.elapsed() >= capture_interval {
                    // コーナーのバウンディングボックスと面積比を計算
                    let mut min_x = f64::MAX;
                    let mut min_y = f64::MAX;
                    let mut max_x = f64::MIN;
                    let mut max_y = f64::MIN;
                    for i in 0..n_corners {
                        let pt = corners.at_2d::<opencv::core::Point2f>(i, 0)?;
                        min_x = min_x.min(pt.x as f64);
                        min_y = min_y.min(pt.y as f64);
                        max_x = max_x.max(pt.x as f64);
                        max_y = max_y.max(pt.y as f64);
                    }
                    let bbox_area = (max_x - min_x) * (max_y - min_y);
                    let image_area = w as f64 * h as f64;
                    let area_ratio = bbox_area / image_area;
                    capture_stats.push((n_corners, area_ratio, min_x, min_y, max_x, max_y));

                    all_corners.push(corners.clone());
                    all_ids.push(ids);
                    last_capture = Instant::now();
                    println!("  キャプチャ: {}/{} (corners={}, area={:.1}%)",
                        all_corners.len(), cal_config.intrinsic_frames, n_corners, area_ratio * 100.0);
                    true
                } else {
                    false
                };

                // 検出後に最新フレームを取得して表示（遅延解消）
                let mut display = camera.get_frame().unwrap_or(frame);

                if n_corners > 0 {
                    for i in 0..n_corners {
                        let pt = corners.at_2d::<opencv::core::Point2f>(i, 0)?;
                        let color = if n_corners >= 6 {
                            Scalar::new(0.0, 255.0, 0.0, 0.0)
                        } else {
                            Scalar::new(0.0, 255.0, 255.0, 0.0)
                        };
                        opencv::imgproc::circle(
                            &mut display,
                            opencv::core::Point::new(pt.x as i32, pt.y as i32),
                            5, color, -1, opencv::imgproc::LINE_8, 0,
                        )?;
                    }
                }

                // ステータス表示
                let (status, color) = if captured_now {
                    (format!("CAPTURED! | {}/{}", all_corners.len(), cal_config.intrinsic_frames),
                     Scalar::new(255.0, 255.0, 255.0, 0.0))
                } else if n_corners >= 6 {
                    (format!("Corners: {} OK | {}/{}", n_corners, all_corners.len(), cal_config.intrinsic_frames),
                     Scalar::new(0.0, 255.0, 0.0, 0.0))
                } else if n_corners > 0 {
                    (format!("Corners: {} (need 6+) | {}/{}", n_corners, all_corners.len(), cal_config.intrinsic_frames),
                     Scalar::new(0.0, 255.0, 255.0, 0.0))
                } else {
                    (format!("No corners | {}/{}", all_corners.len(), cal_config.intrinsic_frames),
                     Scalar::new(0.0, 0.0, 255.0, 0.0))
                };
                opencv::imgproc::put_text(
                    &mut display, &status,
                    opencv::core::Point::new(10, 30),
                    opencv::imgproc::FONT_HERSHEY_SIMPLEX, 0.7,
                    color, 2, opencv::imgproc::LINE_8, false,
                )?;

                renderer.draw_frame(&display)?;
                renderer.update()?;

                if renderer.is_key_pressed(Key::Q) {
                    bail!("ユーザーにより中止されました");
                }
            }

            if all_corners.len() < 3 {
                println!("  フレーム不足（{}枚、最低3枚必要）。デフォルト値を使用します", all_corners.len());
                let fy = h as f64 / (2.0 * (55.0f64.to_radians() / 2.0).tan());
                let mut k = Mat::eye(3, 3, opencv::core::CV_64F)?.to_mat()?;
                *k.at_2d_mut::<f64>(0, 0)? = fy;
                *k.at_2d_mut::<f64>(1, 1)? = fy;
                *k.at_2d_mut::<f64>(0, 2)? = w as f64 / 2.0;
                *k.at_2d_mut::<f64>(1, 2)? = h as f64 / 2.0;
                break (k, Mat::zeros(5, 1, opencv::core::CV_64F)?.to_mat()?, -1.0, capture_stats);
            }

            println!("  キャリブレーション中...");
            let (k, dist, error) = calibrate_intrinsics(&all_corners, &all_ids, &board, image_size)?;
            println!("  再投影誤差: {:.4} px", error);

            // k3異常値検出: 画像境界8点でradial factor R を検証
            let fx = *k.at_2d::<f64>(0, 0)?;
            let fy_val = *k.at_2d::<f64>(1, 1)?;
            let cx = *k.at_2d::<f64>(0, 2)?;
            let cy = *k.at_2d::<f64>(1, 2)?;
            let k1 = *dist.at::<f64>(0)?;
            let k2 = *dist.at::<f64>(1)?;
            let k3 = if dist.rows() >= 5 { *dist.at::<f64>(4)? } else { 0.0 };

            let w_f = w as f64;
            let h_f = h as f64;
            let boundary_points: [(f64, f64, &str); 8] = [
                (0.0,       0.0,       "top-left"),
                (w_f,       0.0,       "top-right"),
                (0.0,       h_f,       "bottom-left"),
                (w_f,       h_f,       "bottom-right"),
                (w_f / 2.0, 0.0,       "top-center"),
                (w_f / 2.0, h_f,       "bottom-center"),
                (0.0,       h_f / 2.0, "left-center"),
                (w_f,       h_f / 2.0, "right-center"),
            ];

            let mut has_negative = false;
            for &(u, v, label) in &boundary_points {
                let x = (u - cx) / fx;
                let y = (v - cy) / fy_val;
                let r2 = x * x + y * y;
                let r4 = r2 * r2;
                let r6 = r4 * r2;
                let radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;
                if radial <= 0.0 {
                    println!("  警告: {} (u={:.0}, v={:.0}) で R={:.4} ≤ 0 (r2={:.4}, k1={:.4}, k2={:.4}, k3={:.4})",
                        label, u, v, radial, r2, k1, k2, k3);
                    has_negative = true;
                }
            }

            if has_negative {
                println!("  歪みモデル破綻を検出。キャプチャをやり直します...");
                println!();
                continue; // 外側loopの先頭に戻り、キャプチャからやり直し
            }

            break (k, dist, error, capture_stats);
        };

        // 内部パラメータのログ記録
        {
            let fx = *k_result.at_2d::<f64>(0, 0)?;
            let fy = *k_result.at_2d::<f64>(1, 1)?;
            let cx = *k_result.at_2d::<f64>(0, 2)?;
            let cy = *k_result.at_2d::<f64>(1, 2)?;
            let dist_vals: Vec<f64> = (0..dist_result.rows().max(dist_result.cols()))
                .map(|i| {
                    if dist_result.rows() == 1 {
                        *dist_result.at_2d::<f64>(0, i).unwrap()
                    } else {
                        *dist_result.at_2d::<f64>(i, 0).unwrap()
                    }
                })
                .collect();
            log_lines.push(format!("[内部パラメータ] カメラ{} ({}x{})", cam_id, image_sizes[cam_idx].width, image_sizes[cam_idx].height));
            log_lines.push(format!("  fx={:.4} fy={:.4} cx={:.4} cy={:.4}", fx, fy, cx, cy));
            log_lines.push(format!("  fx/fy={:.4}", if fy.abs() > 1e-10 { fx / fy } else { f64::NAN }));
            log_lines.push(format!("  dist_coeffs={:?}", dist_vals));
            log_lines.push(format!("  reproj_error={:.4} px", error_result));
            log_lines.push(format!("  キャプチャフレーム: {}枚", capture_stats.len()));
            for (fi, (nc, area, x0, y0, x1, y1)) in capture_stats.iter().enumerate() {
                log_lines.push(format!("    frame {}: corners={} area={:.1}% bbox=({:.0},{:.0})-({:.0},{:.0})",
                    fi, nc, area * 100.0, x0, y0, x1, y1));
            }
            // カバー範囲の集計: 全フレームの全コーナーが画像のどの領域をカバーしているか
            if !capture_stats.is_empty() {
                let global_min_x = capture_stats.iter().map(|s| s.2).fold(f64::MAX, f64::min);
                let global_min_y = capture_stats.iter().map(|s| s.3).fold(f64::MAX, f64::min);
                let global_max_x = capture_stats.iter().map(|s| s.4).fold(f64::MIN, f64::max);
                let global_max_y = capture_stats.iter().map(|s| s.5).fold(f64::MIN, f64::max);
                let avg_area: f64 = capture_stats.iter().map(|s| s.1).sum::<f64>() / capture_stats.len() as f64;
                log_lines.push(format!("  全体カバー範囲: ({:.0},{:.0})-({:.0},{:.0}) = {:.0}x{:.0}px",
                    global_min_x, global_min_y, global_max_x, global_max_y,
                    global_max_x - global_min_x, global_max_y - global_min_y));
                log_lines.push(format!("  平均ボード面積比: {:.1}%", avg_area * 100.0));
            }
            log_lines.push(String::new());
        }

        camera_matrices.push(k_result);
        dist_coeffs_vec.push(dist_result);
        reprojection_errors.push(error_result);
    }

    // [3/4] 外部パラメータキャリブレーション
    if num_cameras >= 2 {
        println!();
        println!("[3/4] 外部パラメータキャリブレーション");
        println!("  全カメラからボードが見える位置にボードを置いてください");
        println!("  全カメラで6コーナー以上検出されると自動キャプチャします");
        println!("  [Q] 中止");
        println!();

        // 全カメラを開く + 1ウィンドウにタイル表示
        let mut cameras: Vec<ThreadedCamera> = Vec::new();
        let mut cam_sizes: Vec<(usize, usize)> = Vec::new();
        for &cam_id in &camera_indices {
            let cam = ThreadedCamera::start(cam_id, Some(1280), Some(960))?;
            let (w, h) = cam.resolution();
            cam_sizes.push((w as usize, h as usize));
            cameras.push(cam);
        }

        // タイルレイアウト: 合計幅1280px以内に収める
        let cols = if num_cameras <= 3 { num_cameras } else { 2 };
        let rows = (num_cameras + cols - 1) / cols;
        let max_window_w: usize = 1280;
        let raw_w = cam_sizes[0].0 * cols;
        let scale = if raw_w > max_window_w {
            max_window_w as f64 / raw_w as f64
        } else {
            1.0
        };
        let tile_w = (cam_sizes[0].0 as f64 * scale) as usize;
        let tile_h = (cam_sizes[0].1 as f64 * scale) as usize;
        let combined_w = tile_w * cols;
        let combined_h = tile_h * rows;

        let mut renderer = MinifbRenderer::new(
            "Calibrate - All Cameras",
            combined_w,
            combined_h,
        )?;

        // 歪み補正済み画像で検出するため、solvePnPには歪み=0を渡す
        let zero_dist = Mat::zeros(5, 1, opencv::core::CV_64F)?.to_mat()?;

        let extrinsic_frames = cal_config.extrinsic_frames;
        // 各フレームごとの全カメラポーズを蓄積: [frame][camera] = (rvec, tvec)
        let mut pose_samples: Vec<Vec<([f64; 3], [f64; 3])>> = Vec::new();
        let mut last_extrinsic_capture = Instant::now() - capture_interval;

        println!("  キャプチャ: 0/{}", extrinsic_frames);

        while pose_samples.len() < extrinsic_frames && renderer.is_open() {
            // 全カメラからフレーム取得・検出・表示
            let mut frames: Vec<Mat> = Vec::new();
            let mut corner_data: Vec<(Mat, Mat, i32)> = Vec::new();

            for (i, cam) in cameras.iter().enumerate() {
                let frame = match cam.get_frame() {
                    Some(f) => f,
                    None => {
                        frames.push(Mat::default());
                        corner_data.push((Mat::default(), Mat::default(), 0));
                        continue;
                    }
                };

                // 歪み補正してから検出（広角カメラ対応）
                let mut undistorted = Mat::default();
                calib3d::undistort(
                    &frame, &mut undistorted,
                    &camera_matrices[i], &dist_coeffs_vec[i],
                    &camera_matrices[i],
                )?;

                let (corners, ids, n) = detect_charuco_corners(&detector, &undistorted)?;

                // 表示は生画像（検出のみundistort）
                let mut display = cam.get_frame().unwrap_or(frame.clone());

                if n > 0 {
                    for j in 0..n {
                        let pt = corners.at_2d::<opencv::core::Point2f>(j, 0)?;
                        let color = if n >= 6 {
                            Scalar::new(0.0, 255.0, 0.0, 0.0)
                        } else {
                            Scalar::new(0.0, 255.0, 255.0, 0.0)
                        };
                        opencv::imgproc::circle(
                            &mut display,
                            opencv::core::Point::new(pt.x as i32, pt.y as i32),
                            5, color, -1, opencv::imgproc::LINE_8, 0,
                        )?;
                    }
                }

                let (status, color) = if n >= 6 {
                    (format!("Cam {} | Corners: {} | {}/{}",
                        camera_indices[i], n, pose_samples.len(), extrinsic_frames),
                     Scalar::new(0.0, 255.0, 0.0, 0.0))
                } else if n > 0 {
                    (format!("Cam {} | Corners: {} (need 6+) | {}/{}",
                        camera_indices[i], n, pose_samples.len(), extrinsic_frames),
                     Scalar::new(0.0, 255.0, 255.0, 0.0))
                } else {
                    (format!("Cam {} | No corners | {}/{}",
                        camera_indices[i], pose_samples.len(), extrinsic_frames),
                     Scalar::new(0.0, 0.0, 255.0, 0.0))
                };
                opencv::imgproc::put_text(
                    &mut display, &status,
                    opencv::core::Point::new(10, 30),
                    opencv::imgproc::FONT_HERSHEY_SIMPLEX, 0.7,
                    color, 2, opencv::imgproc::LINE_8, false,
                )?;

                // カメラ解像度がタイルサイズと異なる場合はリサイズ
                let display_final = if display.cols() as usize != tile_w || display.rows() as usize != tile_h {
                    let mut resized = Mat::default();
                    opencv::imgproc::resize(
                        &display, &mut resized,
                        Size::new(tile_w as i32, tile_h as i32),
                        0.0, 0.0, opencv::imgproc::INTER_LINEAR,
                    )?;
                    resized
                } else {
                    display
                };

                let col = i % cols;
                let row = i / cols;
                renderer.draw_frame_at(&display_final, col * tile_w, row * tile_h)?;

                frames.push(frame);
                corner_data.push((corners, ids, n));
            }

            renderer.update()?;

            // 全カメラでコーナーが十分かつキャプチャ間隔を満たしたら自動キャプチャ
            let all_have_enough = corner_data.iter().all(|(_, _, n)| *n >= 6);

            if all_have_enough && last_extrinsic_capture.elapsed() >= capture_interval {
                let mut frame_poses = Vec::new();
                let mut all_valid = true;

                for (i, (corners, ids, _)) in corner_data.iter().enumerate() {
                    match estimate_pose(
                        corners, ids, &board,
                        &camera_matrices[i], &zero_dist,
                    )? {
                        Some((rvec, tvec)) => {
                            let rv = [
                                *rvec.at_2d::<f64>(0, 0)?,
                                *rvec.at_2d::<f64>(1, 0)?,
                                *rvec.at_2d::<f64>(2, 0)?,
                            ];
                            let tv = [
                                *tvec.at_2d::<f64>(0, 0)?,
                                *tvec.at_2d::<f64>(1, 0)?,
                                *tvec.at_2d::<f64>(2, 0)?,
                            ];
                            frame_poses.push((rv, tv));
                        }
                        None => {
                            println!("  カメラ{}: ポーズ推定失敗（スキップ）", camera_indices[i]);
                            all_valid = false;
                            break;
                        }
                    }
                }

                if all_valid && frame_poses.len() == num_cameras {
                    pose_samples.push(frame_poses);
                    last_extrinsic_capture = Instant::now();
                    println!("  キャプチャ: {}/{}", pose_samples.len(), extrinsic_frames);
                }
            }

            if renderer.is_key_pressed(Key::Q) {
                bail!("ユーザーにより中止されました");
            }
        }

        if pose_samples.is_empty() {
            bail!("外部パラメータキャリブレーションが完了しませんでした");
        }

        // 各カメラのrvec/tvecについて中央値を算出
        fn median(values: &mut Vec<f64>) -> f64 {
            values.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let n = values.len();
            if n % 2 == 0 {
                (values[n / 2 - 1] + values[n / 2]) / 2.0
            } else {
                values[n / 2]
            }
        }

        println!();
        println!("  外部パラメータ集計（{}フレーム）:", pose_samples.len());

        log_lines.push(format!("[外部パラメータ] {}フレーム", pose_samples.len()));

        // 全フレームの生データをログに記録
        for (frame_idx, sample) in pose_samples.iter().enumerate() {
            log_lines.push(format!("  フレーム {}:", frame_idx));
            for (cam_idx, (rv, tv)) in sample.iter().enumerate() {
                log_lines.push(format!("    カメラ{}: rvec=[{:.6}, {:.6}, {:.6}] tvec=[{:.6}, {:.6}, {:.6}]",
                    camera_indices[cam_idx], rv[0], rv[1], rv[2], tv[0], tv[1], tv[2]));
            }
        }
        log_lines.push(String::new());

        let mut all_poses: Vec<(Mat, Mat)> = Vec::new();
        for cam_idx in 0..num_cameras {
            let mut rvecs: [Vec<f64>; 3] = [Vec::new(), Vec::new(), Vec::new()];
            let mut tvecs: [Vec<f64>; 3] = [Vec::new(), Vec::new(), Vec::new()];

            for sample in &pose_samples {
                let (rv, tv) = &sample[cam_idx];
                for k in 0..3 {
                    rvecs[k].push(rv[k]);
                    tvecs[k].push(tv[k]);
                }
            }

            // 中央値
            let rv_med: [f64; 3] = [
                median(&mut rvecs[0]),
                median(&mut rvecs[1]),
                median(&mut rvecs[2]),
            ];
            let tv_med: [f64; 3] = [
                median(&mut tvecs[0]),
                median(&mut tvecs[1]),
                median(&mut tvecs[2]),
            ];

            // 標準偏差
            let rv_std: [f64; 3] = if pose_samples.len() >= 2 {
                let s: Vec<f64> = (0..3).map(|k| {
                    let mean = rvecs[k].iter().sum::<f64>() / rvecs[k].len() as f64;
                    let var = rvecs[k].iter().map(|v| (v - mean).powi(2)).sum::<f64>() / rvecs[k].len() as f64;
                    var.sqrt()
                }).collect();
                [s[0], s[1], s[2]]
            } else {
                [0.0; 3]
            };
            let tv_std: [f64; 3] = if pose_samples.len() >= 2 {
                let s: Vec<f64> = (0..3).map(|k| {
                    let mean = tvecs[k].iter().sum::<f64>() / tvecs[k].len() as f64;
                    let var = tvecs[k].iter().map(|v| (v - mean).powi(2)).sum::<f64>() / tvecs[k].len() as f64;
                    var.sqrt()
                }).collect();
                [s[0], s[1], s[2]]
            } else {
                [0.0; 3]
            };

            println!("  カメラ{}: rvec median=[{:.4}, {:.4}, {:.4}] std=[{:.4}, {:.4}, {:.4}]",
                camera_indices[cam_idx], rv_med[0], rv_med[1], rv_med[2], rv_std[0], rv_std[1], rv_std[2]);
            println!("           tvec median=[{:.4}, {:.4}, {:.4}] std=[{:.4}, {:.4}, {:.4}]",
                tv_med[0], tv_med[1], tv_med[2], tv_std[0], tv_std[1], tv_std[2]);

            log_lines.push(format!("  [集計] カメラ{}:", camera_indices[cam_idx]));
            log_lines.push(format!("    rvec median=[{:.6}, {:.6}, {:.6}] std=[{:.6}, {:.6}, {:.6}]",
                rv_med[0], rv_med[1], rv_med[2], rv_std[0], rv_std[1], rv_std[2]));
            log_lines.push(format!("    tvec median=[{:.6}, {:.6}, {:.6}] std=[{:.6}, {:.6}, {:.6}]",
                tv_med[0], tv_med[1], tv_med[2], tv_std[0], tv_std[1], tv_std[2]));
            log_lines.push(String::new());

            // 中央値からMat復元（compute_relative_posesに渡すため）
            let mut rvec_mat = Mat::zeros(3, 1, opencv::core::CV_64F)?.to_mat()?;
            let mut tvec_mat = Mat::zeros(3, 1, opencv::core::CV_64F)?.to_mat()?;
            for k in 0..3 {
                *rvec_mat.at_2d_mut::<f64>(k as i32, 0)? = rv_med[k];
                *tvec_mat.at_2d_mut::<f64>(k as i32, 0)? = tv_med[k];
            }
            all_poses.push((rvec_mat, tvec_mat));
        }

        // 相対ポーズ計算
        let relative_poses = compute_relative_poses(&all_poses)?;

        // [4/4] 保存
        println!();
        println!("[4/4] キャリブレーション結果を保存中...");

        let mut camera_cals = Vec::new();
        for (i, &cam_id) in camera_indices.iter().enumerate() {
            let (rv, tv) = &relative_poses[i];
            let intrinsic = mat3x3_to_array(&camera_matrices[i])?;
            let dist = mat_to_vec(&dist_coeffs_vec[i])?;

            camera_cals.push(CameraCalibration {
                camera_index: cam_id,
                width: image_sizes[i].width as u32,
                height: image_sizes[i].height as u32,
                intrinsic_matrix: intrinsic,
                dist_coeffs: dist,
                rvec: *rv,
                tvec: *tv,
                reprojection_error: reprojection_errors[i],
            });
        }

        let result = MultiCameraCalibration {
            board: BoardParams {
                dictionary: cal_config.dictionary.clone(),
                squares_x: cal_config.squares_x,
                squares_y: cal_config.squares_y,
                square_length: cal_config.square_length,
                marker_length: cal_config.marker_length,
            },
            cameras: camera_cals,
        };

        save_calibration(&cal_config.output_path, &result)?;
        println!("保存完了: {}", cal_config.output_path);
    } else {
        // 単一カメラ: 内部パラメータのみ保存
        println!();
        println!("[3/4] 単一カメラのため外部パラメータキャリブレーションをスキップ");
        println!();
        println!("[4/4] キャリブレーション結果を保存中...");

        let intrinsic = mat3x3_to_array(&camera_matrices[0])?;
        let dist = mat_to_vec(&dist_coeffs_vec[0])?;

        let result = MultiCameraCalibration {
            board: BoardParams {
                dictionary: cal_config.dictionary.clone(),
                squares_x: cal_config.squares_x,
                squares_y: cal_config.squares_y,
                square_length: cal_config.square_length,
                marker_length: cal_config.marker_length,
            },
            cameras: vec![CameraCalibration {
                camera_index: camera_indices[0],
                width: image_sizes[0].width as u32,
                height: image_sizes[0].height as u32,
                intrinsic_matrix: intrinsic,
                dist_coeffs: dist,
                rvec: [0.0, 0.0, 0.0],
                tvec: [0.0, 0.0, 0.0],
                reprojection_error: reprojection_errors[0],
            }],
        };

        save_calibration(&cal_config.output_path, &result)?;
        println!("保存完了: {}", cal_config.output_path);
    }

    // ログファイル保存
    let log_path = format!("calibration_{}.log",
        chrono::Local::now().format("%Y%m%d_%H%M%S"));
    let mut log_file = std::fs::File::create(&log_path)?;
    for line in &log_lines {
        writeln!(log_file, "{}", line)?;
    }
    println!("ログ保存: {}", log_path);

    println!();
    println!("=== キャリブレーション完了 ===");
    Ok(())
}
