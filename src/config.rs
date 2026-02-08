use anyhow::Result;
use serde::Deserialize;
use std::fs;
use std::path::Path;

#[derive(Debug, Deserialize)]
pub struct Config {
    #[serde(default)]
    pub vmt: VmtConfig,
    #[serde(default)]
    pub tracker: TrackerConfig,
    #[serde(default)]
    pub camera: CameraConfig,
    #[serde(default)]
    pub debug: DebugConfig,
    #[serde(default)]
    pub smooth: SmoothConfig,
    #[serde(default)]
    pub app: AppConfig,
}

#[derive(Debug, Deserialize)]
pub struct VmtConfig {
    #[serde(default = "default_vmt_addr")]
    pub addr: String,
}

#[derive(Debug, Deserialize)]
pub struct TrackerConfig {
    #[serde(default = "default_scale")]
    pub scale_x: f32,
    #[serde(default = "default_scale")]
    pub scale_y: f32,
    #[serde(default)]
    pub mirror_x: bool,
    #[serde(default = "default_offset_y")]
    pub offset_y: f32,
}

#[derive(Debug, Deserialize)]
pub struct CameraConfig {
    #[serde(default)]
    pub index: i32,
    #[serde(default = "default_width")]
    pub width: u32,
    #[serde(default = "default_height")]
    pub height: u32,
}

#[derive(Debug, Deserialize)]
pub struct DebugConfig {
    #[serde(default)]
    pub view: bool,
}

#[derive(Debug, Deserialize)]
pub struct SmoothConfig {
    #[serde(default = "default_smooth_position")]
    pub position: f32,
    #[serde(default = "default_smooth_rotation")]
    pub rotation: f32,
}

#[derive(Debug, Deserialize)]
pub struct AppConfig {
    #[serde(default = "default_target_fps")]
    pub target_fps: u32,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self { target_fps: default_target_fps() }
    }
}

fn default_target_fps() -> u32 { 30 }
fn default_vmt_addr() -> String { "127.0.0.1:39570".to_string() }
fn default_scale() -> f32 { 1.0 }
fn default_offset_y() -> f32 { 1.0 }
fn default_width() -> u32 { 640 }
fn default_height() -> u32 { 480 }
fn default_smooth_position() -> f32 { 0.5 }
fn default_smooth_rotation() -> f32 { 0.3 }

impl Default for VmtConfig {
    fn default() -> Self {
        Self { addr: default_vmt_addr() }
    }
}

impl Default for TrackerConfig {
    fn default() -> Self {
        Self {
            scale_x: default_scale(),
            scale_y: default_scale(),
            mirror_x: false,
            offset_y: default_offset_y(),
        }
    }
}

impl Default for CameraConfig {
    fn default() -> Self {
        Self {
            index: 0,
            width: default_width(),
            height: default_height(),
        }
    }
}

impl Default for DebugConfig {
    fn default() -> Self {
        Self { view: false }
    }
}

impl Default for SmoothConfig {
    fn default() -> Self {
        Self {
            position: default_smooth_position(),
            rotation: default_smooth_rotation(),
        }
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            vmt: VmtConfig::default(),
            tracker: TrackerConfig::default(),
            camera: CameraConfig::default(),
            debug: DebugConfig::default(),
            smooth: SmoothConfig::default(),
            app: AppConfig::default(),
        }
    }
}

impl Config {
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = fs::read_to_string(path)?;
        let config: Config = toml::from_str(&content)?;
        Ok(config)
    }

    pub fn load_or_default<P: AsRef<Path>>(path: P) -> Self {
        Self::load(path).unwrap_or_default()
    }
}
