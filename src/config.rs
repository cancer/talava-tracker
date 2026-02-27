use anyhow::Result;
use serde::Deserialize;
use std::fs;
use std::path::Path;

#[derive(Debug, Deserialize)]
pub struct Config {
    #[serde(default)]
    pub calibration: CalibrationConfig,
}

#[derive(Debug, Deserialize, Clone)]
pub struct CalibrationConfig {
    /// ChArUco辞書タイプ (e.g. "DICT_4X4_50")
    #[serde(default = "default_dictionary")]
    pub dictionary: String,
    /// 横マス数
    #[serde(default = "default_squares_x")]
    pub squares_x: i32,
    /// 縦マス数
    #[serde(default = "default_squares_y")]
    pub squares_y: i32,
    /// マス辺長（メートル）
    #[serde(default = "default_square_length")]
    pub square_length: f32,
    /// マーカー辺長（メートル）
    #[serde(default = "default_marker_length")]
    pub marker_length: f32,
    /// 保存先パス
    #[serde(default = "default_calibration_output")]
    pub output_path: String,
    /// 内部パラメータ用キャプチャフレーム数
    #[serde(default = "default_intrinsic_frames")]
    pub intrinsic_frames: usize,
}

fn default_dictionary() -> String { "DICT_4X4_50".to_string() }
fn default_squares_x() -> i32 { 5 }
fn default_squares_y() -> i32 { 4 }
fn default_square_length() -> f32 { 0.04 }
fn default_marker_length() -> f32 { 0.03 }
fn default_calibration_output() -> String { "calibration.json".to_string() }
fn default_intrinsic_frames() -> usize { 15 }

impl Default for CalibrationConfig {
    fn default() -> Self {
        Self {
            dictionary: default_dictionary(),
            squares_x: default_squares_x(),
            squares_y: default_squares_y(),
            square_length: default_square_length(),
            marker_length: default_marker_length(),
            output_path: default_calibration_output(),
            intrinsic_frames: default_intrinsic_frames(),
        }
    }
}

impl Config {
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = fs::read_to_string(path)?;
        let config: Config = toml::from_str(&content)?;
        Ok(config)
    }
}
