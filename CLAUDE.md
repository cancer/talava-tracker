# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test Commands

```bash
cargo build          # ビルド
cargo run            # 実行（tracker_bevy、default-run）
cargo test           # 全テスト実行
cargo test test_name # 単一テスト実行
cargo run --bin calibrate    # カメラキャリブレーション
cargo run --bin camera_view  # 画角調整（カメラ映像グリッド表示）
```

## What This Project Does

Beat Saber撮影用のSteamVR仮想トラッカーシステム。複数カメラ三角測量による3Dポーズ推定で、腰・足・胸・膝のトラッカーデータをOSC経由でVMTに送信する。

## Architecture

```
Mac (talava-tracker)  --OSC/UDP:39570-->  Windows (VMT -> SteamVR -> Beat Saber)
```

詳細は `docs/architecture.md` 参照。

### Modules

- `vmt` - VMTへのOSC送信。`/VMT/Room/Unity`アドレスでトラッカーの位置(x,y,z)と回転(quaternion)を送信
- `calibration` - ChArUcoボードによるカメラキャリブレーション
- `triangulation` - DLT三角測量による3Dポーズ再構成
- `tracker/body` - 骨格キーポイントからトラッカー位置・回転を算出
- `camera` - ThreadedCameraによるバックグラウンドフレーム取り込み
- `pose` - ONNX推論、前処理、人物検出

### Data Flow

カメラ映像 → ThreadedCamera → 人物検出 → 前処理 → ONNX推論 → 三角測量 → トラッカー算出 → 平滑化 → OSC送信

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

- macOS環境で動作
- VMT送信先: 192.168.10.35:39570
- `VMT_ADDR`環境変数で送信先を上書き可能
