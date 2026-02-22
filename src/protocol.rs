//! TCP protocol for camera-server ↔ inference-server communication.
//!
//! Self-contained: no imports from other talava_tracker modules.

use bytes::Bytes;
use futures::{SinkExt, StreamExt};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use tokio::net::TcpStream;
use tokio_util::codec::{Framed, LengthDelimitedCodec};

// --- Calibration data types (mirrors calibration.json format) ---

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoardParams {
    pub dictionary: String,
    pub squares_x: i32,
    pub squares_y: i32,
    pub square_length: f32,
    pub marker_length: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraCalibrationData {
    pub camera_index: i32,
    pub width: u32,
    pub height: u32,
    /// Intrinsic matrix K (row-major 3x3)
    pub intrinsic_matrix: [f64; 9],
    /// Distortion coefficients [k1, k2, p1, p2, k3]
    pub dist_coeffs: Vec<f64>,
    /// Rotation vector (Rodrigues)
    pub rvec: [f64; 3],
    /// Translation vector
    pub tvec: [f64; 3],
    pub reprojection_error: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationData {
    pub board: BoardParams,
    pub cameras: Vec<CameraCalibrationData>,
}

// --- Message types ---

/// Mac → Win
#[derive(Serialize, Deserialize, Debug)]
pub enum ClientMessage {
    CameraCalibration { data: CalibrationData },
    FrameSet { timestamp_us: u64, frames: Vec<Frame> },
    TriggerPoseCalibration,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Frame {
    pub camera_id: u8,
    pub width: u16,
    pub height: u16,
    pub jpeg_data: Vec<u8>,
}

/// Win → Mac
#[derive(Serialize, Deserialize, Debug)]
pub enum ServerMessage {
    CameraCalibrationAck { ok: bool, error: Option<String> },
    Ready,
}

// --- TCP codec helpers ---

pub type MessageStream = Framed<TcpStream, LengthDelimitedCodec>;

/// Create a framed message stream with length-delimited framing.
pub fn message_stream(stream: TcpStream) -> MessageStream {
    let codec = LengthDelimitedCodec::builder()
        .max_frame_length(16 * 1024 * 1024) // 16MB
        .new_codec();
    Framed::new(stream, codec)
}

/// Send a serializable message (bincode + length prefix).
pub async fn send_message<T: Serialize>(
    stream: &mut MessageStream,
    msg: &T,
) -> anyhow::Result<()> {
    let data = bincode::serialize(msg)?;
    stream.send(Bytes::from(data)).await?;
    Ok(())
}

/// Receive and deserialize a message.
pub async fn recv_message<T: DeserializeOwned>(
    stream: &mut MessageStream,
) -> anyhow::Result<T> {
    match stream.next().await {
        Some(Ok(bytes)) => Ok(bincode::deserialize(&bytes)?),
        Some(Err(e)) => Err(e.into()),
        None => Err(anyhow::anyhow!("connection closed")),
    }
}
