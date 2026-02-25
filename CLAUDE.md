# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test Commands

```bash
cargo build          # ビルド
cargo test           # 全テスト実行
cargo test test_name # 単一テスト実行

# 分離構成（Mac + Win）
cargo run --bin camera_server     # Mac側: カメラキャプチャ + TCP送信
cargo run --bin inference_server  # Win側: 推論 + 三角測量 + VMT送信

# ツール
cargo run --bin calibrate    # カメラキャリブレーション
cargo run --bin camera_view  # 画角調整（カメラ映像グリッド表示）
```

## What This Project Does

Beat Saber撮影用のSteamVR仮想トラッカーシステム。Mac側でカメラ映像をキャプチャし、TCP経由でWindows側に送信。Windows側でONNX推論・DLT三角測量を行い、腰・足・胸・膝のトラッカーデータをOSC経由でVMTに送信する。

## Architecture

```
Mac (camera_server) --TCP:9000--> Windows (inference_server) --OSC/UDP:39570--> VMT -> SteamVR
```

詳細は `docs/architecture.md` 参照。

### Modules

- `protocol` - camera_server ↔ inference_server 間のTCPプロトコル（bincode + LengthDelimitedCodec）
- `vmt` - VMTへのOSC送信。`/VMT/Room/Unity`アドレスでトラッカーの位置(x,y,z)と回転(quaternion)を送信
- `calibration` - ChArUcoボードによるカメラキャリブレーション
- `triangulation` - DLT三角測量による3Dポーズ再構成
- `tracker/body` - 骨格キーポイントからトラッカー位置・回転を算出
- `camera` - ThreadedCameraによるバックグラウンドフレーム取り込み
- `pose` - ONNX推論、前処理、人物検出

### Data Flow

```
[Mac] カメラ映像 → JPEG圧縮 → TCP送信
  ↓
[Win] TCP受信 → JPEG展開 → 人物検出 → 前処理 → ONNX推論 → 三角測量 → トラッカー算出 → 平滑化 → OSC送信
```

### Tracker Index

| index | 部位 | ソースキーポイント |
|-------|------|-------------------|
| 0     | 腰   | LeftHip/RightHip中点 |
| 1     | 左足 | LeftAnkle |
| 2     | 右足 | RightAnkle |
| 3     | 胸   | 肩-腰内挿(0.35) |
| 4     | 左膝 | LeftKnee |
| 5     | 右膝 | RightKnee |

## Environment

- Mac (camera_server): カメラキャプチャ。設定は `camera_server.toml`
- Windows (inference_server): 推論 + VMT送信。設定は `inference_server.toml`
- VMT送信先: inference_server.toml の `vmt_addr`（デフォルト: 127.0.0.1:39570）
