# マルチカメラ三角測量 引き継ぎ資料

## 絶対ルール
- **撮影対象はユーザー本人。カメラの前に立ってもらわないとポーズ検出できない**
- **手順: コード修正・ビルド等を全部終わらせる → 準備完了後に「カメラの前に立ってください」と報告 → ユーザーが立ってからcargo run**
- **ユーザーは暇ではない。準備が終わる前に呼ぶな。報告なしに起動するな**
- **被写体が不要なcargo run(ビルド確認、cargo test等)では呼ぶな。呼ぶのはポーズ検出テスト時のみ**

## アーキテクチャ

camera_server（Mac）+ inference_server（Win）の分離構成。

```
Mac (camera_server) --TCP:9000--> Win (inference_server) --OSC/UDP:39570--> VMT → SteamVR
```

- **camera_server**: カメラキャプチャ → JPEG圧縮 → TCP送信。設定は `camera_server.toml`
- **inference_server**: TCP受信 → JPEG展開 → ONNX推論 → 三角測量 → トラッカー算出 → フィルタ → VMT送信。設定は `inference_server.toml`

## ファイル構成

| ファイル | 内容 |
|---------|------|
| `src/bin/camera_server.rs` | Mac側バイナリ: カメラキャプチャ + TCP送信 |
| `src/bin/inference_server.rs` | Win側バイナリ: 推論 + 三角測量 + トラッカー + VMT送信 |
| `src/protocol.rs` | TCP通信プロトコル（ClientMessage/ServerMessage） |
| `src/triangulation.rs` | DLT三角測量、歪み補正、再投影チェック |
| `src/tracker/body.rs` | キーポイント→トラッカー位置変換、キャリブレーション |
| `calibration.json` | カメラ内部/外部パラメータ、歪み係数 |
| `camera_server.toml` | Mac側設定（カメラ、接続先、JPEG品質） |
| `inference_server.toml` | Win側設定（VMT、モデル、フィルタ） |

## 未解決の重大問題

### 1. レンズ歪み補正

全カメラでdist_coeffs=[0;5]になっている問題。cam2の係数が極端（k2=11.5, k3=-49.9）でundistort_point()が収束しないため全カメラで無効化されている。

### 2. z-jump棄却

prev_hip_zが更新されず、以降全フレーム棄却される永久ループ問題。

### 3. R_kneeの非対称

hip→L_knee=0.59, hip→R_knee=1.33（scale後）。右膝が異常に遠い。

## calibration.jsonの歪み係数

- Camera 0: k1=0.22（穏当）
- Camera 2: k2=11.5, k3=-49.9（極端）
- Camera 3: k1=0.26（穏当）

## 過去の実行結果の場所

- `/private/tmp/claude-501/-Users-cancer-repos-talava-tracker/tasks/bc947e5.output` - 歪み補正あり、キャリブレーション成功→zドリフト
- `/private/tmp/claude-501/-Users-cancer-repos-talava-tracker/tasks/b0280aa.output` - 歪み補正あり、キャリブレーション前のデータ（hip安定）
- `/private/tmp/claude-501/-Users-cancer-repos-talava-tracker/tasks/bada63f.output` - キャリブレーション失敗（hip不検出タイミング）
