use anyhow::{bail, Result};
use minifb::Key;
use opencv::core::{Mat, Size, Scalar};
use opencv::{calib3d, prelude::*};
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
    println!("  6コーナー以上検出されると自動キャプチャします（2秒間隔）");
    println!("  [Q] 中止");
    println!();

    let capture_interval = std::time::Duration::from_millis(500);

    let mut camera_matrices: Vec<Mat> = Vec::new();
    let mut dist_coeffs_vec: Vec<Mat> = Vec::new();
    let mut image_sizes: Vec<Size> = Vec::new();
    let mut reprojection_errors: Vec<f64> = Vec::new();

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

        let mut all_corners: Vec<Mat> = Vec::new();
        let mut all_ids: Vec<Mat> = Vec::new();
        let mut last_capture = Instant::now() - capture_interval; // 初回は即キャプチャ可能

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
                all_corners.push(corners.clone());
                all_ids.push(ids);
                last_capture = Instant::now();
                println!("  キャプチャ: {}/{}", all_corners.len(), cal_config.intrinsic_frames);
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
            // デフォルトの内部パラメータ（FOV 55度相当）
            let fy = h as f64 / (2.0 * (55.0f64.to_radians() / 2.0).tan());
            let mut k = Mat::eye(3, 3, opencv::core::CV_64F)?.to_mat()?;
            *k.at_2d_mut::<f64>(0, 0)? = fy;
            *k.at_2d_mut::<f64>(1, 1)? = fy;
            *k.at_2d_mut::<f64>(0, 2)? = w as f64 / 2.0;
            *k.at_2d_mut::<f64>(1, 2)? = h as f64 / 2.0;
            camera_matrices.push(k);
            dist_coeffs_vec.push(Mat::zeros(5, 1, opencv::core::CV_64F)?.to_mat()?);
            reprojection_errors.push(-1.0);
        } else {
            println!("  キャリブレーション中...");
            let (k, dist, error) = calibrate_intrinsics(&all_corners, &all_ids, &board, image_size)?;
            println!("  再投影誤差: {:.4} px", error);
            camera_matrices.push(k);
            dist_coeffs_vec.push(dist);
            reprojection_errors.push(error);
        }
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

        let mut extrinsic_done = false;
        let mut all_poses: Vec<(Mat, Mat)> = Vec::new();

        while !extrinsic_done && renderer.is_open() {
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
                    (format!("Cam {} | Corners: {}", camera_indices[i], n),
                     Scalar::new(0.0, 255.0, 0.0, 0.0))
                } else if n > 0 {
                    (format!("Cam {} | Corners: {} (need 6+)", camera_indices[i], n),
                     Scalar::new(0.0, 255.0, 255.0, 0.0))
                } else {
                    (format!("Cam {} | No corners", camera_indices[i]),
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

            // 全カメラでコーナーが十分なら自動キャプチャ
            let all_have_enough = corner_data.iter().all(|(_, _, n)| *n >= 6);

            if all_have_enough {
                all_poses.clear();
                let mut all_valid = true;

                for (i, (corners, ids, _)) in corner_data.iter().enumerate() {
                    match estimate_pose(
                        corners, ids, &board,
                        &camera_matrices[i], &zero_dist,
                    )? {
                        Some((rvec, tvec)) => {
                            all_poses.push((rvec, tvec));
                        }
                        None => {
                            println!("  カメラ{}: ポーズ推定失敗", camera_indices[i]);
                            all_valid = false;
                            break;
                        }
                    }
                }

                if all_valid && all_poses.len() == num_cameras {
                    println!("  全カメラでポーズ推定成功！");
                    extrinsic_done = true;
                }
            }

            if renderer.is_key_pressed(Key::Q) {
                bail!("ユーザーにより中止されました");
            }
        }

        if !extrinsic_done {
            bail!("外部パラメータキャリブレーションが完了しませんでした");
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

    println!();
    println!("=== キャリブレーション完了 ===");
    Ok(())
}
