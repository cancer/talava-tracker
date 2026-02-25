# 外部インターフェース仕様

## EI-01: VMT (Virtual Motion Tracker) OSCプロトコル

### 概要

VMTはOSCプロトコルでポーズデータを受信し、SteamVR上の仮想トラッカーとして動作するOpenVRドライバ。

- 公式ドキュメント: https://gpsnmeajp.github.io/VirtualMotionTrackerDocument/api/
- GitHub: https://github.com/gpsnmeajp/VirtualMotionTracker

### 本プロジェクトが使用するOSCメッセージ

**アドレス**: `/VMT/Room/Unity`
**プロトコル**: UDP
**送信先**: config.toml `vmt.addr`（デフォルト: 127.0.0.1:39570、本番: 192.168.10.35:39570）

| 引数 | 型 | 意味 | 値 |
|------|-----|------|-----|
| 0 | Int | トラッカーindex (0-57) | 0-5 |
| 1 | Int | enable | 1（トラッカー） |
| 2 | Float | timeoffset | 0.0（固定） |
| 3 | Float | x (メートル) | position[0] |
| 4 | Float | y (メートル) | position[1] |
| 5 | Float | z (メートル) | position[2] |
| 6 | Float | qx | rotation[0] |
| 7 | Float | qy | rotation[1] |
| 8 | Float | qz | rotation[2] |
| 9 | Float | qw | rotation[3] |

### enableの全バリエーション

| 値 | 意味 | 使用状況 |
|----|------|---------|
| 0 | 無効 | 未使用 |
| 1 | トラッカー | **本プロジェクトで使用** |
| 2 | 左コントローラー | 未使用 |
| 3 | 右コントローラー | 未使用 |
| 4 | Tracking Reference | 未使用 |
| 5-6 | Index互換 | 未使用 |
| 7 | VIVE互換モード | テストコードのみ |

### 座標系

`/VMT/Room/Unity`はUnity左手座標系（X=右、Y=上、Z=前方）で送信する。VMTが内部でOpenVR右手座標系に自動変換する。

### その他のVMT APIアドレス（未使用だが参考）

- `/VMT/Room/Driver` - OpenVR右手座標系、クォータニオン
- `/VMT/Room/UEuler` - Unity左手座標系、オイラー角
- `/VMT/Joint/*` - デバイス相対座標
- `/VMT/Follow/*` - デバイス相対位置、ルーム空間回転
- `/VMT/Reset` - 全トラッカー無効化
- `/VMT/Skeleton/*` - ハンドトラッキング

## EI-02: 座標系

### カメラ座標系

```
X: 画像の右方向
Y: 画像の下方向
Z: カメラから被写体への奥行き（前方正）
原点: カメラレンズ中心
```

### VR/SteamVR座標系（OpenVR）

```
X: ユーザーの右方向
Y: 上方向
Z: ユーザーの後方（背面方向） ※右手座標系
原点: プレイエリア中心の床
```

### Unity座標系（VMT Room/Unityで使用）

```
X: 右方向
Y: 上方向
Z: 前方 ※左手座標系、OpenVRのZ反転
```

### キーポイント正規化座標

```
x, y ∈ [0.0, 1.0]
(0, 0) = 画像左上
(1, 1) = 画像右下
z = メートル（3D推定時のみ有効）
```

### 座標変換（本プロジェクト）

カメラ座標系→VR空間:
- z軸反転: `VR_z = ref_z - camera_z`（カメラに近づく = VR前方 = 正のz）
- 位置: 正規化座標 → FOV補正 → パースペクティブ補正 → スケール変換

## EI-03: ONNX Runtime

### 使用クレート

```toml
ort = { version = "2.0.0-rc.9", features = ["coreml"] }
```

### セッション作成

```rust
Session::builder()?
    .with_optimization_level(GraphOptimizationLevel::Level3)?
    .commit_from_file(model_path)?
```

### 推論パターン

```rust
let input_tensor = Tensor::from_array(input)?;
let outputs = session.run(ort::inputs!["input" => input_tensor])?;
let output: ArrayViewD<f32> = outputs["output"].try_extract_array()?;
```

### 対応モデル

| モデル | ファイル | 入力 | 出力形式 |
|--------|---------|------|---------|
| MoveNet Lightning | movenet_lightning.onnx | [1,192,192,3] NHWC, 0-255 | [1,1,17,3] (y,x,conf) |
| SpinePose Small | spinepose_small.onnx | [1,3,256,192] NCHW, ImageNet正規化 | SimCC: simcc_x[1,37,384], simcc_y[1,37,512] |
| SpinePose Medium | spinepose_medium.onnx | 同上 | 同上 |
| RTMW3D-X | rtmw3d-x.onnx | [1,3,384,288] NCHW, ImageNet正規化 | SimCC3D: simcc_x[1,133,576], simcc_y[1,133,768], simcc_z[1,133,576] |

### ImageNet正規化パラメータ

```
平均: [123.675, 116.28, 103.53] (RGB)
標準偏差: [58.395, 57.12, 57.375] (RGB)
正規化: (pixel - mean) / std
```

### CoreML

macOS上でApple Neural Engine/GPUを使用した推論高速化。`coreml`フィーチャーを有効化。ただし効果は限定的（6ms→3msの改善のみ）。

## EI-04: OpenCV

### 使用クレート

```toml
opencv = { version = "0.93", features = ["videoio", "imgproc", "imgcodecs", "aruco", "calib3d", "objdetect"] }
```

### カメラキャプチャ (videoio) — camera_server

- バックエンド: `CAP_AVFOUNDATION`（macOS専用）
- 設定: FPS=60, バッファサイズ=1
- 解像度: カメラ依存（1760x1328 / 1920x1440 / 1920x1080）

### キャリブレーション (aruco + calib3d)

- ChArUcoボード: `CharucoBoard::new_def()`で作成
- コーナー検出: `CharucoDetector`（カスタムパラメータ: polygonal_approx_accuracy_rate=0.08等）
- 内部パラメータ: `calib3d::calibrate_camera()`
- 外部パラメータ: `calib3d::solve_pnp(SOLVEPNP_ITERATIVE)`
- 歪み補正: radial(k1,k2,k3) + tangential(p1,p2)

## EI-05: TCPプロトコル（camera_server ↔ inference_server）

### 使用クレート

```toml
tokio = { version = "1", features = ["full"] }
tokio-util = { version = "0.7", features = ["codec"] }
bincode = "1"
bytes = "1"
futures = "0.3"
```

### フレーミング

`tokio_util::codec::LengthDelimitedCodec` で長さプレフィクスフレーミング。最大フレーム長16MB。

### メッセージ型

```rust
/// Mac → Win
enum ClientMessage {
    CameraCalibration { data: CalibrationData },
    FrameSet { timestamp_us: u64, frames: Vec<Frame> },
    TriggerPoseCalibration,
}

/// Win → Mac
enum ServerMessage {
    CameraCalibrationAck { ok: bool, error: Option<String> },
    Ready,
    LogData { filename: String, data: Vec<u8> },
}
```

### 接続シーケンス

1. Win: TCPサーバー起動（デフォルト 0.0.0.0:9000）
2. Mac→Win: TCP接続
3. Mac→Win: `CameraCalibration` 送信
4. Win→Mac: `CameraCalibrationAck` 応答
5. Win→Mac: `Ready` 送信
6. Mac: フレームストリーミング開始

## EI-06: One Euro Filter

### 原論文

Casiez, Roussel, Vogel (2012): "1€ Filter: A Simple Speed-based Low-pass Filter for Noisy Input in Interactive Systems"

### アルゴリズム

速度適応型ローパスフィルタ。低速時は強い平滑化（ジッター抑制）、高速時は弱い平滑化（ラグ軽減）。

```
cutoff = min_cutoff + beta * |filtered_velocity|
alpha = 1 / (1 + tau/Te), tau = 1/(2*pi*cutoff)
filtered = alpha * current + (1-alpha) * previous
```

### パラメータ

| パラメータ | 用途 | 現在値 |
|-----------|------|--------|
| position_min_cutoff | 位置の最小カットオフHz（低い=安定） | 1.5 |
| position_beta | 位置の速度追従係数（高い=機敏） | 0.3 |
| rotation_min_cutoff | 回転の最小カットオフHz | 1.0 |
| rotation_beta | 回転の速度追従係数 | 0.01 |

## EI-07: キャリブレーション結果ファイル (calibration.json)

### フォーマット

```json
{
  "board": {
    "dictionary": "DICT_4X4_50",
    "squares_x": 5,
    "squares_y": 4,
    "square_length": 0.04,
    "marker_length": 0.03
  },
  "cameras": [
    {
      "camera_index": 0,
      "width": 640,
      "height": 480,
      "intrinsic_matrix": [/* 9個のf64, row-major 3x3 */],
      "dist_coeffs": [/* k1, k2, p1, p2, k3 */],
      "rvec": [/* 3個のf64, Rodrigues形式 */],
      "tvec": [/* 3個のf64, メートル */],
      "reprojection_error": 0.5
    }
  ]
}
```

## EI-08: 使用クレート一覧

| クレート | バージョン | 用途 |
|---------|-----------|------|
| rosc | 0.10 | OSCプロトコル（VMT連携） |
| anyhow | 1.0 | エラーハンドリング |
| ort | 2.0.0-rc.9 | ONNX Runtime推論 |
| ndarray | 0.17 | テンソル操作 |
| opencv | 0.93 | カメラ・画像処理・キャリブレーション（camera_server） |
| image | 0.25 | JPEG展開（inference_server） |
| nalgebra | 0.33 | 線形代数（三角測量SVD） |
| tokio | 1 | 非同期ランタイム（TCP通信） |
| tokio-util | 0.7 | LengthDelimitedCodec |
| bincode | 1 | バイナリシリアライズ |
| bytes | 1 | バイトバッファ |
| futures | 0.3 | 非同期ストリーム |
| toml | 0.9.11 | 設定ファイル解析 |
| serde | 1.0.228 | シリアライズ |
| serde_json | 1.0 | JSON処理 |
| chrono | 0.4.43 | 日時操作 |
