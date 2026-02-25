# talava-tracker 要求仕様書

本ディレクトリはtalava-trackerシステムに対する要求を体系的に整理したものである。
ソースコード、設定ファイル、コミット履歴、開発セッション記録、外部仕様書から網羅的に収集した。

## ドキュメント構成

| ファイル | 内容 |
|----------|------|
| [functional.md](functional.md) | 機能要件 |
| [non-functional.md](non-functional.md) | 非機能要件（性能・精度・安定性・運用性） |
| [external-interfaces.md](external-interfaces.md) | 外部インターフェース仕様（VMT OSC、SteamVR座標系、ONNX等） |
| [constraints.md](constraints.md) | 制約条件（ハードウェア・環境・技術） |
| [decisions.md](decisions.md) | 技術的意思決定ログ |
| [history.md](history.md) | 開発経緯・要求変遷の時系列 |

## 関連ドキュメント

- [../requirement.md](../requirement.md) - 初期フェーズ計画（Phase 1-7）
- [../architecture.md](../architecture.md) - システムアーキテクチャ
- [../tracker-spec.md](../tracker-spec.md) - BodyTracker詳細仕様
- [../phase2-design.md](../phase2-design.md) - Phase 2設計書
- [../vmc-setup.md](../vmc-setup.md) - VMCセットアップ手順

## システム概要

Beat Saber撮影用のSteamVR仮想トラッカーシステム。Mac上でカメラ映像をキャプチャし（camera_server）、TCP経由でWindows側（inference_server）に送信。Windows側でONNX推論（GPU）・DLT三角測量を行い、6つのトラッカー（腰・左足・右足・胸・左膝・右膝）の位置・回転データをOSC/UDP経由でVMT(Virtual Motion Tracker)に送信する。最終的なアバター表示先はVMC（バーチャルモーションキャプチャー）。

```
Mac (camera_server: カメラ + JPEG圧縮) → TCP → Windows (inference_server: 推論 + 三角測量 + トラッカー) → OSC/UDP → VMT → SteamVR → VMC
```
