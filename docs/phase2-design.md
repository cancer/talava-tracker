# フェーズ2: 人体姿勢推定の安定化

## ゴール
骨格ランドマークが安定して出続ける状態を実現する（SteamVRにはまだ流さない）

## 技術選定

| 項目 | 選定 | 理由 |
|------|------|------|
| モデル | MoveNet Lightning | 前処理・後処理が最もシンプル、192x192入力、17キーポイント |
| 推論 | ort (ONNX Runtime) | 積極的メンテナンス、ドキュメント充実 |
| カメラ | opencv-rust | 実績豊富、画像処理機能も利用可能 |
| 描画 | minifb | 学習コスト低、シンプルで十分 |

## モジュール構成

```
src/
  lib.rs              # pub mod vmt; pub mod pose; pub mod camera; pub mod render;
  pose/
    mod.rs
    keypoint.rs       # Keypoint, Pose, KeypointIndex
    detector.rs       # PoseDetector (ONNX推論)
    preprocess.rs     # 画像前処理
  camera/
    mod.rs
    capture.rs        # OpenCvCamera
  render/
    mod.rs
    skeleton.rs       # 骨格接続定義
    window.rs         # MinifbRenderer

  bin/
    pose_viewer.rs    # フェーズ2用バイナリ
```

## 依存関係追加 (Cargo.toml)

```toml
ort = "2.0.0-rc.9"
ndarray = "0.16"
opencv = { version = "0.93", default-features = false, features = ["videoio", "imgproc"] }
minifb = "0.27"
```

## 実装ステップ

### Step 1: データ型
- `pose/keypoint.rs` - KeypointIndex enum (17種), Keypoint struct, Pose struct
- ユニットテスト

### Step 2: カメラキャプチャ
- `camera/capture.rs` - OpenCvCamera (read_frame, resolution)
- 動作確認用の簡易テスト

### Step 3: 描画
- `render/skeleton.rs` - SKELETON_CONNECTIONS定義
- `render/window.rs` - MinifbRenderer (カメラ映像 + 骨格描画)

### Step 4: ONNX推論
- MoveNet Lightning ONNXモデル取得・配置
- `pose/preprocess.rs` - BGR→RGB、リサイズ、テンソル変換
- `pose/detector.rs` - PoseDetector::detect()

### Step 5: パイプライン統合
- `bin/pose_viewer.rs` - メインループ
- カメラ→前処理→推論→描画の統合
- FPS計測

### Step 6: 安定性検証
- 信頼度閾値調整
- 腕振りテスト
- 30fps維持確認

## 検証方法

```bash
# ビルド
cargo build --release

# 実行
cargo run --release --bin pose_viewer

# 確認項目
# - ウィンドウにカメラ映像と骨格が表示される
# - 30fps前後で動作
# - 腕振りしても大破綻しない
# - 脚・腰が消えすぎない
```

## モデル管理
- `models/` ディレクトリに外部ファイルとして配置
- `.gitignore` に `models/*.onnx` を追加
- README にダウンロード手順を記載

## 重要ファイル
- `src/pose/keypoint.rs` - 中核データ型
- `src/pose/detector.rs` - ONNX推論ラッパー
- `src/bin/pose_viewer.rs` - メインループ
- `models/movenet_lightning.onnx` - モデルファイル（git管理外）
