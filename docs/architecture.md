# Talava Tracker アーキテクチャ

## 概要

カメラ映像から人体姿勢を推定し、SteamVRの仮想トラッカーとして出力するシステム。
Mac側でカメラキャプチャ、Windows側でGPU推論・三角測量・VMT送信を行う分離構成。

## 用途

Beat Saber撮影（操作用途ではない）

## システム構成

```
┌────────────────────────────────┐     ┌─────────────────────────────────────┐
│          Mac (camera_server)   │     │       Windows (inference_server)    │
│                                │     │                                     │
│  ┌──────────┐                  │     │  ┌──────────────────────────────┐   │
│  │ カメラ 0 │──┐  ┌─────────┐ │ TCP │  │ TCP受信 → JPEG展開          │   │
│  └──────────┘  ├─→│ フレーム │ │────→│  │ → 人物検出 → 前処理        │   │
│  ┌──────────┐  │  │ 同期    │ │ :9000│  │ → ONNX推論(GPU)           │   │
│  │ カメラ 2 │──┤  │ + JPEG  │ │     │  │ → 三角測量(DLT)           │   │
│  └──────────┘  │  │ 圧縮    │ │     │  │ → トラッカー算出          │   │  ┌──────────────────┐
│  ┌──────────┐  │  │ + TCP   │ │     │  │ → フィルタ → VMT送信 ─────│───│→│ VMT → SteamVR    │
│  │ カメラ 3 │──┘  │ 送信    │ │     │  └──────────────────────────────┘   │  │  ┌──────────┐    │
│  └──────────┘     └─────────┘ │     │                                     │  │  │トラッカー│    │
│                                │     │                                     │  │  │ 0:腰     │    │
│  ┌──────────────────────────┐  │     │                                     │  │  │ 1:左足   │    │
│  │ キャリブレーション(Mac内) │  │     │                                     │  │  │ 2:右足   │    │
│  │ ChArUco検出・計算        │  │     │                                     │  │  │ 3:胸     │    │
│  └──────────────────────────┘  │     │                                     │  │  │ 4:左膝   │    │
└────────────────────────────────┘     └─────────────────────────────────────┘  │  │ 5:右膝   │    │
                                                                  OSC/UDP:39570│  └──────────┘    │
                                                                               └──────────────────┘
```

- **設定ファイル**: Mac側 `camera_server.toml` / Win側 `inference_server.toml`
- **バイナリ**: Mac `cargo run --bin camera_server` / Win `cargo run --bin inference_server`

## データフロー

### Mac側 (camera_server)

```
カメラ0映像 ─┐
カメラ2映像 ─┼→ ThreadedCamera (バックグラウンド取り込み)
カメラ3映像 ─┘
                    │
                    ▼
         ┌─────────────────────┐
         │ フレーム同期        │  全カメラのフレームが揃うまで待機
         └─────────────────────┘
                    │
                    ▼
         ┌─────────────────────┐
         │ JPEG圧縮 + TCP送信  │  FrameSet（timestamp_us + Vec<Frame>）
         └─────────────────────┘
```

### Win側 (inference_server)

```
         ┌─────────────────────┐
         │ TCP受信スレッド      │  FrameSet受信 → JPEG展開
         └─────────────────────┘  → channelでメインへ
                    │
                    ▼
         ┌─────────────────────┐
         │ メインスレッド       │  人物検出 → クロップ → 前処理 → ONNX推論
         │ ・keypoint/yolo検出 │  全カメラを逐次処理
         │ ・spinepose_medium  │
         └─────────────────────┘
                    │
                    ▼
         ┌─────────────────────┐
         │ 三角測量(DLT)       │  全カメラのposeからDLT三角測量
         └─────────────────────┘  → 3Dポーズを算出
                    │
                    ▼
         ┌─────────────────────┐
         │ トラッカー算出      │  pose_3d → BodyTracker.compute()
         │  ・外れ値除去       │  → 6トラッカーの位置・回転算出
         │  ・ステールデータ検出│  → One Euroフィルタ → last_poses更新
         └─────────────────────┘
                    │
                    ▼
         ┌─────────────────────┐
         │ VMT送信             │  OSC/UDPでVMTに送信
         │  ・補間/外挿        │  pose_3dがないフレームは補間モードで補完
         └─────────────────────┘
```

## スレッド構成

### Mac側 (camera_server)

| スレッド | 役割 |
|----------|------|
| メイン (tokio) | TCP接続管理、フレーム送信、自動再接続 |
| カメラスレッド × N | ThreadedCamera: バックグラウンドでフレーム取り込み |

### Win側 (inference_server)

| スレッド | 役割 |
|----------|------|
| メイン | channel受信 → 推論 → 三角測量 → トラッカー → VMT送信 |
| TCP受信スレッド | フレームセット受信 → JPEG展開 → channelでメインへ |

## TCPプロトコル

### 接続方向

Mac（クライアント）→ Win（サーバー）。MacはWinのIPを知っている。

### フレーミング

素のTCPに `tokio_util::codec::LengthDelimitedCodec` で長さプレフィクスフレーミング。
serde + bincode でシリアライズ。

### メッセージタイプ

**Mac → Win (ClientMessage):**

| タイプ | 用途 |
|--------|------|
| `CameraCalibration` | キャリブレーションデータ送信（接続時 + 再キャリブ時） |
| `FrameSet` | 同期済みフレーム群（timestamp_us + Vec\<Frame\>） |
| `TriggerPoseCalibration` | VR原点設定の実行指示 |

**Win → Mac (ServerMessage):**

| タイプ | 用途 |
|--------|------|
| `CameraCalibrationAck` | 受信完了（OK/Error） |
| `Ready` | フレーム受信準備完了 |
| `LogData` | ログデータの送信（ファイル名 + データ） |

### 接続シーケンス

1. Win: TCPサーバー起動、接続待ち
2. Mac→Win: TCP接続
3. Mac→Win: `CameraCalibration` 送信
4. Win→Mac: `CameraCalibrationAck` 応答
5. Win→Mac: `Ready` 送信
6. Mac: フレームストリーミング開始

### 帯域見積もり（JPEG Q80、3カメラ、30fps）

| リサイズ | 1フレームセット | 30fps時 |
|----------|----------------|---------|
| そのまま | ~800KB | ~24MB/s (~190Mbps) |
| 半分 | ~250KB | ~7.5MB/s (~60Mbps) |

有線LAN (Gigabit) なら元サイズで余裕。

## キャリブレーション

### カメラキャリブレーション（`cargo run --bin calibrate`）

ChArUcoボードを使用してカメラの内部・外部パラメータを推定する。Mac内部で完結。

1. **内部パラメータ推定**: 各カメラで個別にChArUcoボードを撮影（自動キャプチャ、0.5s間隔）
2. **外部パラメータ推定**: 全カメラで同時にボードを撮影し、カメラ間の相対姿勢を算出
3. 結果は`calibration.json`に保存
4. camera_server起動時に読み込み、`CameraCalibration`メッセージでWin側に送信

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
      "rvec": [0.0, 0.0, 0.0],
      "tvec": [0.0, 0.0, 0.0],
      "reprojection_error": 0.147
    }
  ]
}
```

- 最初のカメラが基準座標系（rvec/tvec = 0）
- 2台目以降は基準カメラからの相対姿勢

### ポーズキャリブレーション

BodyTrackerの基準姿勢設定。Win側で実行（三角測量後の3D座標が必要なため）。

- Mac側から`TriggerPoseCalibration`メッセージで発動、またはWin側コンソール入力
- 自動: hipが3秒間安定で自動発動
- 保存項目: 腰の基準位置、肩のyaw基準角度、胴体高さ、各四肢の腰からの相対オフセット
- キャリブレーション後、フィルタ・補間器はリセット

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

### 品質保証フィルタ

| フィルタ | 閾値 | 対象 |
|----------|------|------|
| 速度外れ値除去 | MAX_DISPLACEMENT | 前フレームからの3D距離 |
| hip-四肢距離 | MAX_LIMB_DIST | 四肢のhipからの距離 |
| 左右同一位置 | dist<0.05 | 左右が同位置なら両方None |
| ステールデータ | 0.3秒 | 更新なしでlast_posesクリア＋フィルタリセット |
| One Euro Filter | config | 位置・回転の平滑化 |

### 補間モード（pose_3dがないフレーム）

inference_server.toml の `interpolation_mode` で設定。

| モード | 動作 |
|--------|------|
| `extrapolate` | 速度ベースの線形外挿 |
| `lerp` | 前回値への線形補間 |
| `none` | 最後の値をホールド |

## 通信プロトコル（VMT）

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

### Mac側 (camera_server)

| 機能 | ライブラリ |
|------|-----------|
| カメラ取り込み | opencv |
| JPEG圧縮 | opencv (imgcodecs) |
| TCP通信 | tokio, tokio-util |
| シリアライズ | serde, bincode |
| キャリブレーション | opencv (ChArUco) |

### Win側 (inference_server)

| 機能 | ライブラリ |
|------|-----------|
| TCP通信 | tokio, tokio-util |
| シリアライズ | serde, bincode |
| JPEG展開 | image crate |
| 姿勢推定 | ort (ONNX Runtime, DirectML) |
| 座標計算・三角測量 | nalgebra |
| OSC送信 | rosc |

### Windows側ソフトウェア

| ソフトウェア | 役割 |
|-------------|------|
| VMT | OSC → SteamVRトラッカー |
| SteamVR | VRランタイム |
| バーチャルモーションキャプチャー | アバター表示 |
| Beat Saber | ゲーム |

## 推論モデル

inference_server.toml の `model` で設定。

| モデル名 | ファイル | キーポイント数 | 特徴 |
|----------|---------|---------------|------|
| movenet | movenet_lightning.onnx | 17 | 高速・低精度 |
| spinepose_small | spinepose_small.onnx | 37 | 中速・中精度 |
| spinepose_medium | spinepose_medium.onnx | 37 | 低速・高精度（デフォルト） |
| rtmw3d | rtmw3d-x.onnx | 37 | 3D対応 |

## 現在のカメラ構成

- カメラ0: index=0, 1760x1328（基準カメラ）
- カメラ2: index=2, 1920x1440
- カメラ3: index=3, 1920x1080（上方から見下ろし角度）

## 制約・前提

- 脚の動きは機敏（Beat Saberのプレイ中の動きを追従する必要がある）
- 編集前提の品質で許容
- 推論はWin側のGPU (DirectML) で実行
- Mac→Win間はGigabit有線LAN前提
