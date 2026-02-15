use opencv::prelude::*;
use opencv::videoio::{self, VideoCapture, VideoCaptureAPIs};
use opencv::imgcodecs;
use opencv::core::Mat;
use std::thread;
use std::time::Duration;

fn main() {
    println!("=== カメラプローブ ===");
    println!();

    for index in 0..5 {
        print!("index {}: ", index);
        let mut cap = match VideoCapture::new(index, VideoCaptureAPIs::CAP_AVFOUNDATION as i32) {
            Ok(c) => c,
            Err(_) => { println!("open failed"); break; }
        };
        if !cap.is_opened().unwrap_or(false) {
            println!("not available");
            break;
        }

        let prop_w = cap.get(videoio::CAP_PROP_FRAME_WIDTH).unwrap_or(0.0);
        let prop_h = cap.get(videoio::CAP_PROP_FRAME_HEIGHT).unwrap_or(0.0);
        let prop_fps = cap.get(videoio::CAP_PROP_FPS).unwrap_or(0.0);
        let prop_fmt = cap.get(videoio::CAP_PROP_FORMAT).unwrap_or(-1.0);
        let prop_mode = cap.get(videoio::CAP_PROP_MODE).unwrap_or(-1.0);
        let backend = cap.get_backend_name().unwrap_or_default();

        println!("prop: {}x{} fps={} fmt={} mode={} backend={}",
            prop_w, prop_h, prop_fps, prop_fmt, prop_mode, backend);

        // フレーム読み取り
        thread::sleep(Duration::from_millis(500));
        let mut frame = Mat::default();
        match cap.read(&mut frame) {
            Ok(true) if !frame.empty() => {
                let ch = frame.channels();
                let depth = frame.depth();
                let mat_type = frame.typ();
                println!("       frame: {}x{} ch={} depth={} type={} step={}",
                    frame.cols(), frame.rows(), ch, depth, mat_type,
                    frame.mat_step().get(0));

                // フレームを保存
                let filename = format!("probe_cam{}.png", index);
                match imgcodecs::imwrite(&filename, &frame, &opencv::core::Vector::new()) {
                    Ok(_) => println!("       saved: {}", filename),
                    Err(e) => println!("       save err: {}", e),
                }
            }
            Ok(_) => println!("       frame: EMPTY"),
            Err(e) => println!("       frame err: {}", e),
        }
        println!();
    }
}
