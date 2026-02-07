use anyhow::Result;
use rosc::{encoder, OscMessage, OscPacket, OscType};
use std::net::UdpSocket;

/// VMTのデフォルトアドレス
pub const VMT_DEFAULT_ADDR: &str = "127.0.0.1:39570";

/// トラッカーの位置と回転
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrackerPose {
    /// 位置 (x, y, z)
    pub position: [f32; 3],
    /// 回転 (クォータニオン: x, y, z, w)
    pub rotation: [f32; 4],
}

impl TrackerPose {
    pub fn new(position: [f32; 3], rotation: [f32; 4]) -> Self {
        Self { position, rotation }
    }

    /// 原点、回転なし
    pub fn identity() -> Self {
        Self {
            position: [0.0, 0.0, 0.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
        }
    }
}

/// VMTへ送信するOSCメッセージを構築
/// 引数: index, enable, timeoffset, x, y, z, qx, qy, qz, qw
/// enable: 0=無効, 1=トラッカー, 7=トラッカー(VIVE互換モード)
pub fn build_osc_message(index: i32, enable: i32, pose: &TrackerPose) -> OscMessage {
    OscMessage {
        addr: "/VMT/Room/Unity".to_string(),
        args: vec![
            OscType::Int(index),
            OscType::Int(enable),
            OscType::Float(0.0), // timeoffset
            OscType::Float(pose.position[0]),
            OscType::Float(pose.position[1]),
            OscType::Float(pose.position[2]),
            OscType::Float(pose.rotation[0]),
            OscType::Float(pose.rotation[1]),
            OscType::Float(pose.rotation[2]),
            OscType::Float(pose.rotation[3]),
        ],
    }
}

/// OSCメッセージをバイト列にエンコード
pub fn encode_osc_message(msg: &OscMessage) -> Result<Vec<u8>> {
    let packet = OscPacket::Message(msg.clone());
    let encoded = encoder::encode(&packet)?;
    Ok(encoded)
}

/// VMTクライアント
pub struct VmtClient {
    socket: UdpSocket,
    target_addr: String,
}

impl VmtClient {
    /// 新しいVMTクライアントを作成
    pub fn new(target_addr: &str) -> Result<Self> {
        let socket = UdpSocket::bind("0.0.0.0:0")?;
        Ok(Self {
            socket,
            target_addr: target_addr.to_string(),
        })
    }

    /// デフォルトアドレス(127.0.0.1:39570)で作成
    pub fn default() -> Result<Self> {
        Self::new(VMT_DEFAULT_ADDR)
    }

    /// トラッカーの位置・回転を送信
    /// enable: 0=無効, 1=トラッカー, 7=トラッカー(VIVE互換モード)
    pub fn send(&self, index: i32, enable: i32, pose: &TrackerPose) -> Result<()> {
        let msg = build_osc_message(index, enable, pose);
        let data = encode_osc_message(&msg)?;
        self.socket.send_to(&data, &self.target_addr)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tracker_pose_identity() {
        let pose = TrackerPose::identity();
        assert_eq!(pose.position, [0.0, 0.0, 0.0]);
        assert_eq!(pose.rotation, [0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_build_osc_message_address() {
        let pose = TrackerPose::identity();
        let msg = build_osc_message(0, 7, &pose);
        assert_eq!(msg.addr, "/VMT/Room/Unity");
    }

    #[test]
    fn test_build_osc_message_args() {
        let pose = TrackerPose::new([1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 1.0]);
        let msg = build_osc_message(1, 7, &pose);

        // 引数: index, enable, timeoffset, x, y, z, qx, qy, qz, qw
        assert_eq!(msg.args.len(), 10);

        // index
        assert_eq!(msg.args[0], OscType::Int(1));
        // enable (VIVE互換モード)
        assert_eq!(msg.args[1], OscType::Int(7));
        // timeoffset
        assert_eq!(msg.args[2], OscType::Float(0.0));
        // position
        assert_eq!(msg.args[3], OscType::Float(1.0));
        assert_eq!(msg.args[4], OscType::Float(2.0));
        assert_eq!(msg.args[5], OscType::Float(3.0));
        // rotation (quaternion)
        assert_eq!(msg.args[6], OscType::Float(0.0));
        assert_eq!(msg.args[7], OscType::Float(0.0));
        assert_eq!(msg.args[8], OscType::Float(0.0));
        assert_eq!(msg.args[9], OscType::Float(1.0));
    }

    #[test]
    fn test_build_osc_message_disabled() {
        let pose = TrackerPose::identity();
        let msg = build_osc_message(0, 0, &pose);
        assert_eq!(msg.args[1], OscType::Int(0));
    }

    #[test]
    fn test_encode_osc_message() {
        let pose = TrackerPose::identity();
        let msg = build_osc_message(0, 7, &pose);
        let encoded = encode_osc_message(&msg).unwrap();
        assert!(!encoded.is_empty());
    }
}
