# BodyTracker 詳細仕様

## 概要

複数カメラからSpinePose（37キーポイント）で人体姿勢を推定し、DLT三角測量で3D座標を復元。6つの仮想トラッカーの位置(x, y, z)と回転をOSC経由でVMTに送信する。

```
Mac (camera_server: カメラキャプチャ + TCP送信)
  ↓ TCP :9000
Windows (inference_server: 推論 + 三角測量 + トラッカー + VMT送信)
  ↓ OSC/UDP :39570
Windows (VMT → SteamVR → Beat Saber/VMC)
```

## トラッカー一覧

| Index | 部位   | 位置の元データ      | Yaw の元データ       |
|-------|--------|---------------------|----------------------|
| 0     | 腰     | Hip中点             | 肩方向               |
| 1     | 左足   | LeftAnkle           | 左膝→左足首          |
| 2     | 右足   | RightAnkle          | 右膝→右足首          |
| 3     | 胸     | Spine03（肩中点FB） | 肩方向               |
| 4     | 左膝   | LeftKnee            | 左腰→左膝            |
| 5     | 右膝   | RightKnee           | 右腰→右膝            |

- Spine03: SpinePoseキーポイント。検出失敗時は肩中点にフォールバック(FB)。
- 全トラッカーの回転はyawのみ（pitch, rollは常に0）。

## 出力形式

OSCアドレス: `/VMT/Room/Unity`

```
args: [index: i32, enable: i32, timeoffset: f32, x: f32, y: f32, z: f32, qx: f32, qy: f32, qz: f32, qw: f32]
```

- `enable = 1`（Tracker）
- 回転はY軸回転のみのクォータニオン: `[0, sin(yaw/2), 0, cos(yaw/2)]`

## 座標系

三角測量の出力はカメラ座標系（メートル単位）の3D座標。キャリブレーション時の基準位置を原点としてVR空間にマッピングする。

| 軸 | VR空間での意味 |
|----|----------------|
| x  | 左右           |
| y  | 上下           |
| z  | 前後（奥行き） |

## キャリブレーション

コンソール入力 `c + Enter` またはhip安定検出（3秒）で自動発動。5秒のカウントダウン後に現在のポーズを基準として記録する。

| 保存項目 | 用途 |
|----------|------|
| hip_x, hip_y, hip_z | 位置基準点 |
| yaw_shoulder | 腰・胸の回転基準 |
| yaw_left_foot, yaw_right_foot | 各足の回転基準 |
| yaw_left_knee, yaw_right_knee | 各膝の回転基準 |

キャリブレーション後、フィルタ・補間器はリセットされる。

## Yaw計算

### 腰・胸: 肩方向

```
dx = left_shoulder.x - right_shoulder.x  (mirror_x時は逆)
dy = right_shoulder.y - left_shoulder.y
yaw = atan2(dy, dx) - cal_yaw_shoulder
```

### 足: 膝→足首方向

```
dx = -(ankle.x - knee.x)  (mirror_x時は逆)
dy = ankle.y - knee.y
yaw = atan2(dx, dy) - cal_yaw_foot
```

### 膝: 腰→膝方向

```
dx = -(knee.x - hip.x)  (mirror_x時は逆)
dy = knee.y - hip.y
yaw = atan2(dx, dy) - cal_yaw_knee
```

## 平滑化

### One Euro Filter

ジッタ除去。速い動きには追従、静止時は安定。位置と回転それぞれに適用。

- クォータニオンのshortest path処理（dot積負なら符号反転、フィルタ後正規化）

### 補間モード（inference_server.toml `interpolation_mode`）

- **extrapolate**（デフォルト）: 速度ベース線形外挿。回転は外挿しない
- **lerp**: 線形補間。回転はslerp
- **none**: 最後の値をホールド

## 設定（inference_server.toml）

| パラメータ | 型 | デフォルト | 説明 |
|-----------|------|-----------|------|
| listen_addr | String | 0.0.0.0:9000 | TCPリッスンアドレス |
| vmt_addr | String | 127.0.0.1:39570 | VMT送信先 |
| model | String | spinepose_medium | ポーズモデル |
| detector | String | keypoint | 人物検出方式（keypoint / yolo） |
| interpolation_mode | String | extrapolate | 補間方式 |
| mirror_x | bool | false | 左右反転 |
| offset_y | f32 | 1.0 | Y座標のベースオフセット |
| foot_y_offset | f32 | 0.0 | 足トラッカーの高さ微調整 |

### [filter] (One Euro Filter)

| パラメータ          | 型   | デフォルト | 説明 |
|---------------------|------|-----------|------|
| position_min_cutoff | f32  | 1.5       | 位置フィルターのカットオフ周波数（低い=安定） |
| position_beta       | f32  | 1.5       | 位置フィルターの速度追従係数（高い=追従） |
| rotation_min_cutoff | f32  | 1.0       | 回転フィルターのカットオフ周波数 |
| rotation_beta       | f32  | 0.01      | 回転フィルターの速度追従係数 |
| lower_body_position_min_cutoff | f32 | (未設定) | 下半身用の位置カットオフ（上書き） |
| lower_body_position_beta | f32 | (未設定) | 下半身用の速度追従係数（上書き） |

## 姿勢推定モデル

### SpinePose (37キーポイント)

- 入力: NCHW `[1, 3, 256, 192]`、ImageNet正規化
- 出力: SimCC — `simcc_x[1, 37, 384]`, `simcc_y[1, 37, 512]`
- キーポイント: COCO17 + Halpe拡張 + Spine (01-05) + Latissimus + Clavicle + Neck (02-03)
- 胸トラッカーにSpine03を使用

### MoveNet (17キーポイント)

- 入力: NHWC `[1, 192, 192, 3]`、0-255 int32
- 出力: `[1, 1, 17, 3]` (x, y, confidence)
