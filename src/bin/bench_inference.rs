use std::time::Instant;
use talava_tracker::camera::OpenCvCamera;
use anyhow::Result;

fn main() -> Result<()> {
    let mut camera = OpenCvCamera::open_with_resolution(0, Some(640), Some(480))?;
    
    // カメラのみを100フレーム計測
    let start = Instant::now();
    let iterations = 100;
    for _ in 0..iterations {
        let _ = camera.read_frame()?;
    }
    let elapsed = start.elapsed();
    
    let avg_ms = elapsed.as_millis() as f64 / iterations as f64;
    let actual_fps = 1000.0 / avg_ms;
    
    println!("Camera capture: {:.2}ms/frame = {:.1} FPS (actual)", avg_ms, actual_fps);
    
    Ok(())
}
