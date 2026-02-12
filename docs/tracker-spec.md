# BodyTracker 詳細仕様

## 概要

単眼カメラからSpinePose（37キーポイント）で人体姿勢を推定し、6つの仮想トラッカーの位置(x, y, z)と回転(yaw)をOSC経由でVMTに送信する。

```
Mac (カメラ + SpinePose推論)
  ↓ OSC/UDP :39570
Windows (VMT → SteamVR → Beat Saber/VMC)
```

## トラッカー一覧

| Index | 部位   | 位置の元データ      | Yaw の元データ       | スケール   |
|-------|--------|---------------------|----------------------|------------|
| 0     | 腰     | Hip中点             | 肩方向               | body_scale |
| 1     | 左足   | LeftAnkle           | 左膝→左足首          | leg_scale  |
| 2     | 右足   | RightAnkle          | 右膝→右足首          | leg_scale  |
| 3     | 胸     | Spine03（肩中点FB） | 肩方向               | body_scale |
| 4     | 左膝   | LeftKnee            | 左腰→左膝            | leg_scale  |
| 5     | 右膝   | RightKnee           | 右腰→右膝            | leg_scale  |

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

| 軸 | VR空間での意味 | 画像座標との対応          |
|----|----------------|--------------------------|
| x  | 左右           | 画像x（左が正）          |
| y  | 上下           | 画像y（上が正）+ offset_y |
| z  | 前後（奥行き） | 胴体高さ比から推定        |

## 座標変換（convert_position）

入力: 画像上のキーポイント座標 (x, y) — 正規化済み [0, 1]

```
ref_x, ref_y = キャリブレーション時のHip中点（未キャリブレーション時は0.5, 0.5）

fov = fov_scale(y)  ... FOV補正係数

pos_x = ((ref_x - hip_x) * scale_x + (hip_x - x) * part_scale * fov) * body_ratio
pos_y = offset_y + ((ref_y - hip_y) * scale_y + (hip_y - y) * part_scale * fov) * body_ratio
pos_z = estimate_depth(pose)
```

### 式の構成要素

**グローバル移動項**: `(ref_x - hip_x) * scale_x`, `(ref_y - hip_y) * scale_y`
- キャリブレーション基準点からの腰の移動量。全トラッカー共通。

**パーツオフセット項**: `(hip_x - x) * part_scale * fov`
- 腰から各部位までの距離。`part_scale`はbody_scaleまたはleg_scale。
- FOV補正 `fov` で画像端の距離圧縮を補正。

**body_ratio**: 奥行き変化による見かけの体型変化の補正。
- `cal_torso_height / current_torso_height`
- 近づく → 体が大きく映る → ratio < 1 → 全変位を縮小して体型維持。

## キャリブレーション

`[C]` キーまたはコンソール `c + Enter` で5秒後に実行。以下を保存:

| 保存項目 | 用途 |
|----------|------|
| hip_x, hip_y | 位置基準点 |
| torso_height | 深度推定・body_ratio基準 |
| yaw_shoulder | 腰・胸の回転基準 |
| yaw_left_foot, yaw_right_foot | 各足の回転基準 |
| yaw_left_knee, yaw_right_knee | 各膝の回転基準 |
| left/right_ankle_offset | 足フォールバック用オフセット |
| left/right_knee_offset | 膝フォールバック用オフセット |

キャリブレーション後、フィルター・補間器はリセットされる。

## 奥行き推定（estimate_depth）

胴体の縦の長さ（肩中点→腰中点の垂直距離）の変化率から奥行きを推定。

```
torso_height = |avg(shoulder_y) - avg(hip_y)|
pos_z = (1.0 - cal_torso_height / current_torso_height) * depth_scale
```

- 近づく → torso_height増 → z > 0（前方）
- 離れる → torso_height減 → z < 0（後方）
- キャリブレーション距離 → z ≈ 0

胴体高さを選んだ理由: 体の回転（yaw）に対して不変。肩幅は回転で変化するため不適。

## 体型補正（body_ratio）

奥行きが変化すると画像上の体サイズが変わり、パーツ間距離が伸縮する。これを打ち消す。

```
body_ratio = cal_torso_height / current_torso_height
```

グローバル移動項とパーツオフセット項の両方に掛ける。

## FOV補正（fov_scale）

カメラが肩の高さにある場合、画像中心から離れた部位（膝、足首）はカメラからの斜め距離が増え、1ピクセルあたりの物理距離が大きくなる。

```
half_fov = camera_fov_v_rad / 2
theta = atan((y - 0.5) * 2 * tan(half_fov))
fov_scale = 1 / cos(theta)
```

- 画像中心（y = 0.5）: 補正 = 1.0
- 画像端: 補正 > 1.0（距離を拡大）
- `fov_v = 0` で無効（補正 = 1.0 固定）

パーツオフセット項にのみ適用。

## フォールバック（キーポイント見失い時）

足首・膝がカメラの画角から外れた場合、キャリブレーション時の腰からの相対オフセットで位置を推定。

```
足首見失い時:
  ax = hip_x + cal.ankle_offset_x
  ay = hip_y + cal.ankle_offset_y
  yaw = 0（キャリブレーション基準=まっすぐ）
```

前提条件:
- キャリブレーション済みであること
- キャリブレーション時にそのキーポイントが検出されていたこと
- 腰キーポイント（LeftHip, RightHip）が見えていること

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

## 非同期推論アーキテクチャ

SpinePose推論（約33-42ms/フレーム）はメインループをブロックしないよう別スレッドで実行。

```
[推論スレッド]
  frame_rx.recv() → detect/crop → preprocess → inference → result_tx.send()

[メインスレッド (90Hz)]
  try_send(frame)   ... 推論スレッドがビジーならスキップ
  try_recv(result)   ... 新しいPoseが来たら:
    → BodyTracker.compute → filter → 送信
  結果なし            ... 補間:
    → extrapolate/lerp/none → filter → 送信
```

チャンネル:
- `frame_tx/frame_rx`: `sync_channel(1)` — 推論スレッドが空いた時だけ最新フレームを渡す
- `result_tx/result_rx`: `channel()` — 推論結果をメインスレッドへ

## 平滑化パイプライン

```
BodyTracker.compute() → Extrapolator.update() / Lerper.update()
                       → PoseFilter.apply()  (One Euro Filter)
                       → VmtClient.send()
```

- **Extrapolator**: 直前2フレームから速度ベースで位置を外挿
- **Lerper**: 直前2フレーム間を線形補間
- **PoseFilter (One Euro Filter)**: ジッタ除去。速い動きには追従、静止時は安定

## 設定（config.toml）

### [tracker]

| パラメータ  | 型    | デフォルト | 説明 |
|-------------|-------|-----------|------|
| scale_x     | f32   | 1.0       | 水平方向のグローバル移動スケール |
| scale_y     | f32   | 1.0       | 垂直方向のグローバル移動スケール |
| body_scale  | f32   | 1.0       | 腰・胸のパーツオフセットスケール |
| leg_scale   | f32   | 1.0       | 足・膝のパーツオフセットスケール |
| depth_scale | f32   | 1.0       | 奥行き推定のスケール |
| mirror_x    | bool  | false     | 左右反転 |
| offset_y    | f32   | 1.0       | Y座標のベースオフセット（VR空間での高さ基準） |

### [camera]

| パラメータ | 型   | デフォルト | 説明 |
|-----------|------|-----------|------|
| index     | i32  | 0         | カメラデバイスID |
| width     | u32  | 640       | カメラ解像度（幅） |
| height    | u32  | 480       | カメラ解像度（高さ） |
| fov_v     | f32  | 0.0       | 垂直画角（度）。0で補正無効 |

### [app]

| パラメータ  | 型     | デフォルト        | 説明 |
|-------------|--------|-------------------|------|
| target_fps  | u32    | 30                | メインループのFPS上限 |
| model       | String | spinepose_medium  | ポーズモデル（movenet / spinepose_small / spinepose_medium） |
| detector    | String | keypoint          | 人物検出方式（keypoint / yolo / none） |

### [filter] (One Euro Filter)

| パラメータ          | 型   | デフォルト | 説明 |
|---------------------|------|-----------|------|
| position_min_cutoff | f32  | 1.5       | 位置フィルターのカットオフ周波数（低い=安定） |
| position_beta       | f32  | 0.01      | 位置フィルターの速度追従係数（高い=追従） |
| rotation_min_cutoff | f32  | 1.0       | 回転フィルターのカットオフ周波数 |
| rotation_beta       | f32  | 0.01      | 回転フィルターの速度追従係数 |

### [interpolation]

| パラメータ | 型     | デフォルト   | 説明 |
|-----------|--------|-------------|------|
| mode      | String | extrapolate | 補間方式（extrapolate / lerp / none） |

## 姿勢推定モデル

### SpinePose (37キーポイント)

- 入力: NCHW `[1, 3, 256, 192]`、ImageNet正規化
- 出力: SimCC — `simcc_x[1, 37, 384]`, `simcc_y[1, 37, 512]`
- キーポイント: COCO17 + Halpe拡張 + Spine (01-05) + Latissimus + Clavicle + Neck (02-03)
- 胸トラッカーにSpine03を使用

### MoveNet (17キーポイント)

- 入力: NHWC `[1, 192, 192, 3]`、0-255 int32
- 出力: `[1, 1, 17, 3]` (x, y, confidence)

## 既知の制約

### 単眼2Dポーズ推定の限界

- **回転**: yawのみ。pitch, rollは取得不可
- **奥行き**: 胴体高さ比による推定のみ（精度限定的）
- **オクルージョン**: 体の一部が隠れると追跡不可。フォールバック（固定オフセット）で部分的に対処
- **足の回転感度**: 膝→足首方向のyawは内股/がに股の微妙な変化を捉えにくい
- **つま先立ち**: 足首y座標の変化は捉えるが、足のpitch（傾き）は出力しない
