# 機能要件

## FR-01: トラッカー生成・送信

### FR-01-1: 6トラッカー構成

6つの仮想トラッカーを生成し、VMTに送信する。

| index | 部位 | ソースキーポイント | スケール |
|-------|------|-------------------|---------|
| 0 | 腰 (hip) | LeftHip/RightHip中点 | body_scale |
| 1 | 左足 (L_foot) | LeftAnkle（フォールバック: LeftKnee） | leg_scale |
| 2 | 右足 (R_foot) | RightAnkle（フォールバック: RightKnee） | leg_scale |
| 3 | 胸 (chest) | 肩-腰内挿(0.35)。片肩欠落時はhip_xをフォールバック | body_scale |
| 4 | 左膝 (L_knee) | LeftKnee | leg_scale |
| 5 | 右膝 (R_knee) | RightKnee | leg_scale |

### FR-01-2: 位置算出

- 三角測量の3D座標をそのまま使用（メートル単位）
- ミラーリング: `mirror_x`設定でX軸反転可能
- 足Y補正: `foot_y_offset`で足トラッカーの高さを微調整

### FR-01-3: 回転算出

- Yaw（Y軸回転）: 肩の傾き(`atan2(dy,dx)`)、膝・足は腰→膝/膝→足首ベクトル
- Pitch（X軸回転）: 複眼モード時のみ。膝は`atan2(-dz, dy)`で奥行き方向の曲がりを反映
- 出力: クォータニオン [qx, qy, qz, qw]

### FR-01-4: OSC送信

- アドレス: `/VMT/Room/Unity`
- 送信先: config.toml `vmt.addr`（デフォルト: 127.0.0.1:39570）
- プロトコル: UDP
- パラメータ: index(Int), enable(Int=1), timeoffset(Float=0.0), x/y/z(Float), qx/qy/qz/qw(Float)

## FR-02: 姿勢推定

### FR-02-1: ONNX推論

4モデル対応。config.toml `app.model`で選択。

| モデル | 入力サイズ | キーポイント | 3D | 現在の選択 |
|--------|-----------|-------------|-----|-----------|
| movenet_lightning | 192x192 NHWC | 17 (COCO) | x | 旧モデル |
| spinepose_small | 192x256 NCHW | 37 | o | 高速版 |
| spinepose_medium | 192x256 NCHW | 37 | o | **デフォルト** |
| rtmw3d-x | 288x384 NCHW | 133→37マッピング | o | 高精度版 |

### FR-02-2: 前処理

- MoveNet: BGR→RGB、192x192リサイズ、値域0-255
- SpinePose/RTMW3D: BGR→RGB、ImageNet正規化（mean=[123.675,116.28,103.53], std=[58.395,57.12,57.375]）、レターボックスリサイズ

### FR-02-3: 人物検出・クロップ

- 検出方式: `app.detector`で選択（"keypoint" or "yolo"）
- keypointモード: 前フレームのキーポイントからバウンディングボックスを予測
- クロップ後に推論、結果を元の座標系にunletterbox

### FR-02-4: 37キーポイント

SpinePoseモデルの出力。COCO-17 + 足指・踵 + 脊椎5点 + 補助点。

```
0-4:   Nose, LeftEye, RightEye, LeftEar, RightEar
5-10:  LeftShoulder, RightShoulder, LeftElbow, RightElbow, LeftWrist, RightWrist
11-16: LeftHip, RightHip, LeftKnee, RightKnee, LeftAnkle, RightAnkle
17-25: Head, Neck, Hip, LeftBigToe, RightBigToe, LeftSmallToe, RightSmallToe, LeftHeel, RightHeel
26-36: Spine01-05, LeftLatissimus, RightLatissimus, LeftClavicle, RightClavicle, Neck02, Neck03
```

各キーポイント: x(0-1正規化), y(0-1正規化), z(メートル、3D時), confidence(0-1)

## FR-03: マルチカメラ三角測量

### FR-03-1: DLT三角測量

- 複数カメラの2D観測からDLT法で3D座標を復元
- 射影行列 P = K[R|t]（Kは内部パラメータ、R/tは外部パラメータ）
- A行列をSVDで解き、最小固有値の固有ベクトルが3D座標

### FR-03-2: 再投影エラーチェック

- 3D→2D再投影と観測との最大誤差を検証
- 閾値: MAX_REPROJ_ERROR = 120px
- 実測精度: 95-101px（z=1.4m, fx=1273時に3D誤差~13cm、フィルタ後6-7cm）

### FR-03-3: Hip基準ペア固定

- 全キーポイントで同じカメラペアを使用（z値一貫性確保）
- 手順: Hipの最良ペアを特定 → 全キーポイントにそのペアを適用
- フォールバック: 基準ペア失敗時はキーポイント毎に最良ペアを選択

### FR-03-4: 歪み補正

- Newton-Raphson法（最大30イテレーション）でundistort
- 歪みモデル: radial(k1,k2,k3) + tangential(p1,p2)
- 異常な歪み係数時は[0,0,0,0,0]で上書き

### FR-03-5: 境界チェック

- 三角測量結果: ±10m以内、z>0
- 超過時はそのキーポイントを棄却

## FR-04: カメラキャプチャ

### FR-04-1: カメラ取り込み

- バックエンド: OpenCV VideoCapture (AVFoundation)
- ThreadedCamera: 別スレッドでフレーム連続取得、Arc<Mutex>で最新フレーム提供
- 設定: FPS=60、バッファサイズ=1（最新フレームのみ）
- カメラ検出: index 0から順にプローブ、最初の失敗で打ち切り

### FR-04-2: マルチカメラ均等化

- in_flightフラグで各カメラ最大1フレームのみチャンネルに保持
- sync_channel buffer = num_cameras

## FR-05: キャリブレーション

### FR-05-1: ChArUcoボードキャリブレーション（`cargo run --bin calibrate`）

- ChArUcoボード: DICT_4X4_50辞書、5x4マス、マス辺長0.04m、マーカー辺長0.03m
- 内部パラメータ: 最低3フレーム×6コーナー以上で`cv::calibrateCamera`実行
- 外部パラメータ: 全カメラ同時検出時に`cv::solvePnP`で推定、カメラ0を基準に相対姿勢計算
- 出力: calibration.json（JSON形式）

### FR-05-2: ランタイムキャリブレーション

- 手動: コンソール入力`c`またはSIGUSR1シグナルで5秒後に実行
- 自動: hipが3秒間安定（標準偏差<0.3m）で自動発動
- キャリブレーション内容: hip位置を基準点としてオフセット設定

### FR-05-3: キャリブレーション不要化

- scale変更後のキャリブレーション再実行を不要にした（修正済み）

## FR-06: 品質チェック・外れ値除去

### FR-06-1: 信頼度チェック

- CONFIDENCE_THRESHOLD = 0.3
- 信頼度が閾値未満のキーポイントは無効化

### FR-06-2: 速度外れ値除去

- max_displacement = 0.4 * scale_factor
- 前フレームからの移動距離が閾値を超えたら拒否

### FR-06-3: 解剖学的制約

- hip-四肢距離: max_limb_dist = 1.5 * scale_factor
- 足・膝はhipより下（limb_y < hip_y）
- 左右同一位置検出: 距離<0.05で両方無効化（アーティファクト除去）

### FR-06-4: ステールデータ処理

- 0.3秒: 送信停止、フィルタリセット（速度参照は維持）
- 3.0秒: 速度参照もクリア（再検出許可）

## FR-07: 平滑化・補間

### FR-07-1: One Euro Filter

- 位置と回転それぞれに適用
- パラメータ: position_min_cutoff=1.5, position_beta=0.3, rotation_min_cutoff=1.0, rotation_beta=0.01
- クォータニオンのshortest path処理（dot積負なら符号反転、フィルタ後正規化）

### FR-07-2: 補間モード（config.toml `interpolation.mode`で選択）

- extrapolate（デフォルト）: 速度ベース線形外挿。回転は外挿しない
- lerp: 線形補間。回転はslerp（球面線形補間）
- none: 最後の値をホールド

## FR-08: デバッグ・運用支援

### FR-08-1: デバッグビュー

- config.toml `debug.view = true`で有効化
- minifbウィンドウで複数カメラ映像を横並び表示
- キーポイントを緑色で描画（confidence >= 0.2）

### FR-08-2: ログ出力

- 全コンソールログをファイルに保存: `logs/tracker_YYYYMMDD_HHMMSS.log`

### FR-08-3: macOSスリープ防止

- `caffeinate -i -w`で起動中のスリープを禁止

### FR-08-4: カメラ診断ツール（`cargo run --bin camera_probe`）

- index 0-4をプローブし、解像度・FPS・フォーマット・バックエンド情報を表示
- 最初のフレームを`probe_cam{index}.png`に保存

### FR-08-5: ChArUcoボード生成（`cargo run --bin generate_board`）

- config.toml設定に基づくChArUcoボード画像を生成
- ルーラー・説明テキスト付き

### FR-08-6: 推論ベンチマーク（`cargo run --bin bench_inference`）

- ONNXモデルの推論速度を計測
