# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test Commands

```bash
cargo build          # ビルド
cargo run            # 実行（インタラクティブCLI）
cargo test           # 全テスト実行
cargo test test_name # 単一テスト実行
```

## What This Project Does

Beat Saber撮影用のSteamVR仮想トラッカーシステム。カメラ1台から人体姿勢を推定し、腰・足・胸のトラッカーデータをOSC経由でVMTに送信する。

## Architecture

```
Mac (talava-tracker)  --OSC/UDP:39570-->  Windows (VMT -> SteamVR -> Beat Saber)
```

### Modules

- `vmt` - VMTへのOSC送信。`/VMT/Room/Unity`アドレスでトラッカーの位置(x,y,z)と回転(quaternion)を送信

### Data Flow (将来の完成形)

カメラ映像 → 人体姿勢推定(ONNX) → 座標変換 → トラッカー算出 → 平滑化 → OSC送信

### Tracker Index

| index | 部位 |
|-------|------|
| 0     | 腰   |
| 1     | 左足 |
| 2     | 右足 |
| 3     | 胸   |

## Development Phases

詳細は `docs/requirement.md` 参照。

1. **フェーズ1** (現在): OSC出力経路の完成
2. **フェーズ2**: 人体姿勢推定の安定化
3. **フェーズ3**: 腰トラッカー変換
4. **フェーズ4**: 平滑化・ジッタ対策
5. **フェーズ5**: 足・胸トラッカー追加
6. **フェーズ6**: Beat Saber実運用調整

## Environment

- WSL2環境ではWindowsホストIPを自動検出
- `VMT_ADDR`環境変数で送信先を上書き可能
