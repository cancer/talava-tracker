use anyhow::Result;
use minifb::{Key, Window, WindowOptions};
use opencv::core::Mat;
use opencv::prelude::*;

use crate::pose::Pose;
use crate::render::skeleton::{KEYPOINT_COLOR, LOW_CONFIDENCE_COLOR, SKELETON_COLOR, SKELETON_CONNECTIONS};

/// minifbを使用したレンダラー
pub struct MinifbRenderer {
    window: Window,
    buffer: Vec<u32>,
    width: usize,
    height: usize,
}

impl MinifbRenderer {
    /// ウィンドウを作成
    pub fn new(title: &str, width: usize, height: usize) -> Result<Self> {
        let window = Window::new(
            title,
            width,
            height,
            WindowOptions {
                resize: false,
                ..WindowOptions::default()
            },
        )?;

        let buffer = vec![0u32; width * height];

        Ok(Self {
            window,
            buffer,
            width,
            height,
        })
    }

    /// ウィンドウが開いているか
    pub fn is_open(&self) -> bool {
        self.window.is_open() && !self.window.is_key_down(Key::Escape)
    }

    /// BGR Mat をバッファにコピー
    pub fn draw_frame(&mut self, frame: &Mat) -> Result<()> {
        let frame_width = frame.cols() as usize;
        let frame_height = frame.rows() as usize;

        // サイズが異なる場合はリサイズが必要だが、ここではシンプルにクロップ/パディング
        for y in 0..self.height.min(frame_height) {
            for x in 0..self.width.min(frame_width) {
                let pixel = frame.at_2d::<opencv::core::Vec3b>(y as i32, x as i32)?;
                // BGR -> RGB -> u32
                let r = pixel[2] as u32;
                let g = pixel[1] as u32;
                let b = pixel[0] as u32;
                self.buffer[y * self.width + x] = (r << 16) | (g << 8) | b;
            }
        }

        Ok(())
    }

    /// 姿勢を描画
    pub fn draw_pose(&mut self, pose: &Pose, confidence_threshold: f32) {
        let w = self.width as u32;
        let h = self.height as u32;

        // 骨格線を描画
        for (start_idx, end_idx) in SKELETON_CONNECTIONS.iter() {
            let start = pose.get(*start_idx);
            let end = pose.get(*end_idx);

            if start.is_valid(confidence_threshold) && end.is_valid(confidence_threshold) {
                let (x1, y1) = start.to_pixel(w, h);
                let (x2, y2) = end.to_pixel(w, h);
                self.draw_line(x1, y1, x2, y2, SKELETON_COLOR);
            }
        }

        // キーポイントを描画
        for kp in pose.keypoints.iter() {
            let (px, py) = kp.to_pixel(w, h);
            let color = if kp.is_valid(confidence_threshold) {
                KEYPOINT_COLOR
            } else {
                LOW_CONFIDENCE_COLOR
            };
            self.draw_circle(px, py, 4, color);
        }
    }

    /// バッファをウィンドウに表示
    pub fn update(&mut self) -> Result<()> {
        self.window
            .update_with_buffer(&self.buffer, self.width, self.height)?;
        Ok(())
    }

    /// Bresenhamのアルゴリズムで線を描画
    fn draw_line(&mut self, x0: i32, y0: i32, x1: i32, y1: i32, color: u32) {
        let dx = (x1 - x0).abs();
        let dy = -(y1 - y0).abs();
        let sx = if x0 < x1 { 1 } else { -1 };
        let sy = if y0 < y1 { 1 } else { -1 };
        let mut err = dx + dy;

        let mut x = x0;
        let mut y = y0;

        loop {
            self.set_pixel(x, y, color);

            if x == x1 && y == y1 {
                break;
            }

            let e2 = 2 * err;
            if e2 >= dy {
                err += dy;
                x += sx;
            }
            if e2 <= dx {
                err += dx;
                y += sy;
            }
        }
    }

    /// 円を描画（塗りつぶし）
    fn draw_circle(&mut self, cx: i32, cy: i32, radius: i32, color: u32) {
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                if dx * dx + dy * dy <= radius * radius {
                    self.set_pixel(cx + dx, cy + dy, color);
                }
            }
        }
    }

    /// ピクセルをセット（境界チェック付き）
    fn set_pixel(&mut self, x: i32, y: i32, color: u32) {
        if x >= 0 && x < self.width as i32 && y >= 0 && y < self.height as i32 {
            self.buffer[y as usize * self.width + x as usize] = color;
        }
    }
}
