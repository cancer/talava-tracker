# Talava Tracker アーキテクチャ

## 概要

カメラ映像から人体姿勢を推定し、SteamVRの仮想トラッカーとして出力するシステム。
単眼モードと複眼（三角測量）モードを切り替え可能。

## 用途

Beat Saber撮影（操作用途ではない）

## システム構成

```
┌───────────────────────────────────────────┐     ┌──────────────────────────────┐
│                   Mac                      │     │          Windows             │
│                                            │     │                              │
│  ┌──────────┐                              │     │                              │
│  │ カメラ 0 │──┐   ┌───────────────────┐  │     │  ┌─────┐   ┌─────────┐      │
│  └──────────┘  ├──→│  tracker_bevy     │  │ OSC │  │ VMT │──→│ SteamVR │      │
│  ┌──────────┐  │   │                   │────────→│     │   │         │      │
│  │ カメラ 2 │──┘   │ ・姿勢推定(ONNX) │  │ UDP │  └─────┘   │ ┌──────────┐  │
│  └──────────┘      │ ・三角測量(DLT)   │  │     │            │ │トラッカー│  │
│                    │ ・座標変換       │  │     │            │ │ 0:腰     │  │
│                    │ ・平滑化         │  │     │            │ │ 1:左足   │  │
│                    │ ・OSC送信        │  │     │            │ │ 2:右足   │  │
│                    └───────────────────┘  │     │            │ │ 3:胸     │  │
│                                            │     │            │ │ 4:左膝   │  │
└───────────────────────────────────────────┘     │            │ │ 5:右膝   │  │
                                                   │            │ └──────────┘  │
                                                   │            └─────────┘      │
                                                   └──────────────────────────────┘
```

- **VMT送信先**: 192.168.10.35:39570
- **バイナリ**: `cargo run --bin tracker_bevy`（Cargo.tomlの`default-run`）

## 動作モード

### 単眼モード

`config.toml`で`calibration_file`が未設定、かつ`[[cameras]]`が1台以下の場合。

- `[camera]`セクションの設定を使用
- 2Dポーズをそのままbody trackerに渡す
- FOV補正による奥行き推定（`fov_v`設定）
- One Euroフィルタが毎フレーム`.clone()`で供給され、高頻度（~75Hz）で安定動作

### 複眼モード（三角測量）

`config.toml`で`calibration_file = "calibration.json"`を設定した場合。

- `calibration.json`からカメラパラメータ（内部行列、歪み係数、外部パラメータ）を読み込み
- 各カメラで独立に2Dポーズ推定 → DLT三角測量で3Dポーズを再構成
- FOV補正は無効（`fov_v = 0.0`）、三角測量が直接3D座標を算出

## データフロー（複眼モード）

```
カメラ0映像 ─┐
             ├→ ThreadedCamera (バックグラウンド取り込み)
カメラ2映像 ─┘
                    │
                    ▼
         ┌─────────────────────┐
         │ send_frames_system  │  新フレームがあれば推論スレッドに送信
         └─────────────────────┘  (sync_channel, try_send)
                    │
                    ▼
         ┌─────────────────────┐
         │ 推論スレッド (1本)  │  人物検出 → クロップ → 前処理 → ONNX推論
         │ ・keypoint/yolo検出 │  全カメラを逐次処理
         │ ・spinepose_medium  │  (~335ms/フレーム)
         └─────────────────────┘
                    │
                    ▼
         ┌─────────────────────┐
         │receive_results_system│  推論結果をposes_2d[cam_idx]に格納
         └─────────────────────┘
                    │
                    ▼
         ┌─────────────────────┐
         │ triangulate_system  │  全カメラのposeが揃ったらDLT三角測量
         │                     │  → pose_3dに3Dポーズを格納
         │                     │  → poses_2dをクリア（次の三角測量待ち）
         └─────────────────────┘
                    │
                    ▼
         ┌─────────────────────┐
         │calibration_system   │  [C+Enter]で5秒後にbody trackerの基準設定
         └─────────────────────┘
                    │
                    ▼
         ┌─────────────────────┐
         │compute_trackers_system│  pose_3d → BodyTracker.compute()
         │  ・sanitize_pose     │  → 6トラッカーの位置・回転算出
         │  ・外れ値除去        │  → One Euroフィルタ → last_poses更新
         │  ・ステールデータ検出│
         └─────────────────────┘
                    │
                    ▼
         ┌─────────────────────┐
         │ send_vmt_system     │  OSC/UDPでVMTに送信
         │  ・補間/外挿        │  pose_3dがないフレームは補間モードで補完
         └─────────────────────┘
```

## スレッド構成

| スレッド | 役割 |
|----------|------|
| メイン (Bevy) | ScheduleRunnerPlugin (target_fps: 90Hz) でシステムチェーン実行 |
| 推論スレッド | 全カメラのフレームを逐次推論（1本で共有） |
| カメラスレッド × N | ThreadedCamera: バックグラウンドでフレーム取り込み |
| コンソールスレッド | stdin監視、キャリブレーション指示 |

## キャリブレーション

### カメラキャリブレーション（`cargo run --bin calibrate`）

ChArUcoボードを使用してカメラの内部・外部パラメータを推定する。

1. **内部パラメータ推定**: 各カメラで個別にChArUcoボードを撮影（自動キャプチャ、0.5s間隔）
2. **外部パラメータ推定**: 全カメラで同時にボードを撮影し、カメラ間の相対姿勢を算出
3. 結果は`calibration.json`に保存

#### calibration.json の構造

```json
{
  "board": {
    "dictionary": "DICT_4X4_50",
    "squares_x": 5, "squares_y": 4,
    "square_length": 0.035, "marker_length": 0.025
  },
  "cameras": [
    {
      "camera_index": 0,
      "width": 1920, "height": 1080,
      "intrinsic_matrix": [/* 3x3 row-major */],
      "dist_coeffs": [/* 5要素 */],
      "rvec": [0.0, 0.0, 0.0],   // 基準カメラ: 原点
      "tvec": [0.0, 0.0, 0.0],
      "reprojection_error": 0.147
    },
    {
      "camera_index": 2,
      "width": 1920, "height": 1080,
      "intrinsic_matrix": [/* 3x3 row-major */],
      "dist_coeffs": [/* 5要素 */],
      "rvec": [-0.400, 0.394, 0.050],  // カメラ0からの相対回転
      "tvec": [-0.843, -0.957, 0.820], // カメラ0からの相対並進
      "reprojection_error": 0.154
    }
  ]
}
```

- 最初のカメラが基準座標系（rvec/tvec = 0）
- 2台目以降は基準カメラからの相対姿勢

### ランタイムキャリブレーション（[C + Enter]）

BodyTrackerの基準姿勢設定。5秒のカウントダウン後に現在のポーズを基準として記録する。

- 腰の基準位置（hip_x, hip_y, hip_z）
- 肩のyaw基準角度
- 胴体高さ（奥行き・体型比率の基準値）
- 各四肢の腰からの相対オフセット

## 三角測量

### DLT (Direct Linear Transform)

各キーポイントについて、N台のカメラの2D観測から`x × (P · X) = 0`の方程式系を構築し、`A^T A`の最小固有値固有ベクトルとして3D座標を推定。

- confidence_threshold（0.3）以上のカメラのみ使用
- 2台以上の有効観測があるキーポイントのみ三角測量
- 1台以下の観測 → confidence=0（トラッカー算出で無視される）
- 2D座標は正規化座標（0-1）をピクセル座標に変換してから使用

### 射影行列の構築

`CameraParams::from_calibration()`で`P = K[R|t]`を構築。

- K: 内部パラメータ行列（calibration.jsonの`intrinsic_matrix`）
- R: Rodriguesベクトルから回転行列に変換
- t: 並進ベクトル

## トラッカー

### index割り当て

| index | 部位 | ソースキーポイント |
|-------|------|-------------------|
| 0 | 腰 | LeftHip/RightHip中点 |
| 1 | 左足 | LeftAnkle |
| 2 | 右足 | RightAnkle |
| 3 | 胸 | 肩中点-腰中点の内挿 (0.35) |
| 4 | 左膝 | LeftKnee |
| 5 | 右膝 | RightKnee |

### 品質保証フィルタ（compute_trackers_system）

| フィルタ | 閾値 | 対象 |
|----------|------|------|
| sanitize_pose | 境界2% | 画像端のキーポイントをconfidence=0に（四肢XY, 腰Xのみ） |
| 速度外れ値除去 | MAX_DISPLACEMENT=1.0 | 前フレームからの3D距離 |
| hip-四肢距離 | MAX_LIMB_DIST=1.5 | 四肢のhipからの2D距離 |
| 左右同一位置 | dist<0.05 | 左右が同位置なら両方None |
| ステールデータ | 0.3秒 | 更新なしでlast_posesクリア＋フィルタリセット |
| One Euro Filter | config.filter | 位置・回転の平滑化 |

### 補間モード（pose_3dがないフレーム）

`config.toml`の`[interpolation] mode`で設定。

| モード | 動作 |
|--------|------|
| `extrapolate` | 速度ベースの線形外挿 |
| `lerp` | 前回値への線形補間 |
| `none` | 最後の値をホールド |

## 通信プロトコル

### OSC (Open Sound Control)

- **プロトコル**: UDP
- **ポート**: 39570（VMTデフォルト）
- **アドレス**: `/VMT/Room/Unity`
- **引数**:
  | 位置 | 型     | 内容          |
  |------|--------|---------------|
  | 0    | int32  | index         |
  | 1    | int32  | enable (0/1)  |
  | 2    | float  | timeoffset    |
  | 3    | float  | x (位置)      |
  | 4    | float  | y (位置)      |
  | 5    | float  | z (位置)      |
  | 6    | float  | qx (回転)     |
  | 7    | float  | qy (回転)     |
  | 8    | float  | qz (回転)     |
  | 9    | float  | qw (回転)     |

## 技術スタック

### Mac側 (talava-tracker)

| 機能 | ライブラリ |
|------|-----------|
| ECSフレームワーク | bevy |
| カメラ取り込み | opencv |
| 姿勢推定 | ort (ONNX Runtime) |
| 座標計算・三角測量 | nalgebra |
| OSC送信 | rosc |
| キャリブレーション | opencv (ChArUco) |

### Windows側

| ソフトウェア | 役割 |
|-------------|------|
| VMT | OSC → SteamVRトラッカー |
| SteamVR | VRランタイム |
| バーチャルモーションキャプチャー | アバター表示 |
| Beat Saber | ゲーム |

## 推論モデル

`config.toml`の`[app] model`で設定。

| モデル名 | ファイル | キーポイント数 | 特徴 |
|----------|---------|---------------|------|
| movenet | movenet_lightning.onnx | 17 | 高速・低精度 |
| spinepose_small | spinepose_small.onnx | 37 | 中速・中精度 |
| spinepose_medium | spinepose_medium.onnx | 37 | 低速・高精度（現在使用） |
| rtmw3d | rtmw3d-x.onnx | 37 | 3D対応 |

## 現在のカメラ構成

- カメラ0: index=0, 1920x1080（基準カメラ）
- カメラ2: index=2, 1920x1080（相対姿勢あり）
- 再投影誤差: 0.147 / 0.154 ピクセル

## 制約・前提

- 脚の動きは機敏（Beat Saberのプレイ中の動きを追従する必要がある）
- 編集前提の品質で許容
- 推論スレッドは1本で全カメラを逐次処理（カメラ数×推論時間がレイテンシに直結）
