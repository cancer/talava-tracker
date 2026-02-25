# 開発経緯・要求変遷

開発期間: 2026-02-06 ~。GitHub Issue/PRで管理。

## タイムライン

### 2026-02-06: Phase 1-2（初期化・推論）

**コミット**: `31281ed` init, `6a0bf3a` Init cc, `bc25f6f` Impl pose inference

**要求**: 「先に出口を作り、次に人体を流し込む」
- OSC送信経路の確立（VMTにトラッカーが見える状態）
- MoveNet LightningによるONNX推論の単体動作

### 2026-02-07: Phase 3-5（トラッカー実装）

**コミット**:
- `a60eb0f` Add VMT tracker sender with hip tracking
- `6ef388c` Add calibration for hip tracker
- `aa8af38` Fix yaw calculation for mirrored camera
- `e1e4c79` Add EMA smoothing for tracker poses
- `6503e0a` Add hip, feet, and chest trackers via BodyTracker
- `8f7d8c8` Add Phase 6 (FPS improvement) to requirements

**要求**:
- 腰トラッカー1個がそれっぽく動く
- キャリブレーションで座標系を合わせる
- 静止時にプルプルしない（EMA導入）
- IKが成立する最低構成（腰+左足+右足+胸）

### 2026-02-08: Phase 6（FPS改善・フィルタ改善）

**コミット**:
- `67abdb6` Enable CoreML execution provider
- `6ca4a38` Bulk copy preprocess (42ms->0.2ms), disable CoreML
- `ab6ade0` Set camera buffer size to 1
- `5e2889e` Move camera capture to separate thread
- `d198899` Use spin wait for precise FPS control (53->60 FPS)
- `7dcc7aa` Add frame interpolation for smooth 90fps VMT output
- `b9f2e34` Add console-based calibration
- `96338e4` Replace EMA with One Euro Filter
- `a51f3ac` Separate body_scale from movement scale

**要求**:
- 「1fpsでは使えない。10fps以上に」
- 前処理のボトルネック解消（42ms→0.2ms）
- カメラ別スレッド化
- フレーム補間で90FPS VMT出力
- EMA → One Euro Filter（静止安定 + 動き出し追従の両立）
- body_scaleとmovement_scaleの分離（体型歪み防止）

**結果**: CoreMLはハングするため断念。前処理高速化とスレッド化で大幅改善。

### 2026-02-09 ~ 02-12: モデル切替・拡張

**コミット**:
- `929ac32` Replace MoveNet (17kp) with SpinePose (37kp)
- `59047ac` Add person crop pipeline, optimize SpinePose preprocessing
- `0baaace` Move inference to separate thread
- `e0b92f2` Add leg_scale parameter
- `da19bec` Add monocular depth estimation (torso height ratio)
- `1c2d1b9` Add knee trackers, body ratio compensation, FOV correction
- `6d051cf` Add detailed tracker specification document

**要求**:
- 脊椎キーポイントが必要（胸トラッカー精度向上）→ MoveNet→SpinePose切替
- 膝トラッカー追加（index 4, 5）
- FOV補正で画像端のパースペクティブ歪みを対処
- 単眼深度推定（胴体高さ比）
- 推論の非同期化

### 2026-02-13: 微調整

**コミット**:
- `3017189` Fix perspective-induced vertical drift
- `d8cb178` Tune config: adjust leg_scale, depth_scale, offset_y, fov_v

**要求**: パースペクティブによる垂直ドリフトの修正。各種パラメータチューニング。

### 2026-02-15: マルチカメラ三角測量（大規模変更）

**コミット**:
- `da44548` Add RTMW3D model support
- `0e54a4c` Add Bevy ECS multi-camera triangulation architecture
- `fb49d7d` Add ChArUco camera calibration and multi-camera triangulation
- `ae5689c` Fix multi-camera triangulation with lens distortion correction
- `a297d7e` Fix chest tracker: single-shoulder detection, hip_x fallback
- `e96af44` Add log file output
- `c612f86` Fix z-axis inversion: camera coords to VR coords
- `a29569c` Prevent macOS sleep while tracker is running

**要求**:
- 「単眼の奥行き推定に限界。複数カメラで3D座標を直接取得したい」
- DLT三角測量の実装
- ChArUcoキャリブレーション（`cargo run --bin calibrate`）
- Bevy ECSアーキテクチャへのリライト
- 歪み補正（undistort_point）
- z軸反転の修正（前に歩くとアバターが後ろに動く問題）
- ログファイル出力
- macOSスリープ防止

**影響**: アーキテクチャの大幅変更。旧コードの大部分を書き換え。

### 2026-02-15 ~ 02-16: 品質改善

**コミット**:
- `e2a56e3` Fix foot tracker jump: use ankle-only for position, knee as fallback

**要求**:
- 足トラッカーのジャンプ抑制
- ankleベースにkneeフォールバック

### 2026-02-23 ~ 02-25: camera_server + inference_server 分離構成

**コミット（PR #47）**:
- `src/protocol.rs` TCP通信プロトコル（bincode + LengthDelimitedCodec）
- `src/bin/camera_server.rs` Mac側バイナリ
- `src/bin/inference_server.rs` Win側バイナリ

**要求**:
- 「Mac側の推論負荷をなくし、Win側でGPU推論を可能にしたい」
- カメラ入力（Mac）と推論（Win）を別マシンに分離
- camera_server: カメラキャプチャ → JPEG圧縮 → TCP送信
- inference_server: TCP受信 → JPEG展開 → ONNX推論(DirectML) → 三角測量 → トラッカー → VMT送信
- Bevyなし（パイプラインが線形のためECSのメリットが薄い）

**影響**: アーキテクチャの大幅変更。inference_serverはBevy非依存の素の関数チェーン + スレッド構成。

## ユーザーの要求の本質的な変遷

```
Phase 1-2: 「動くものを作れ」
   ↓
Phase 3-5: 「トラッカーを増やせ」（腰→腰+足+胸）
   ↓
Phase 6:   「速くしろ」（1fps→60fps+補間90fps）
   ↓
フィルタ:   「揺れるな、でも遅れるな」
   ↓
モデル:    「精度を上げろ」（17kp→37kp、深度推定、膝追加）
   ↓
三角測量:  「奥行きを正確にしろ」（単眼→複眼三角測量）
   ↓
分離構成:  「GPUで推論しろ」（Mac+Win分離、camera_server+inference_server）
   ↓
現在:     「安定させろ」（z値ドリフト、キーポイント欠落、消失）
```

## 運用上の重要な教訓

1. **前処理がボトルネック**: 推論よりも画像前処理（42ms）が遅かった。バルクコピーで0.2msに改善。
2. **CoreMLはハングする**: macOSのCoreMLは利用可能だが一部モデルでハング。無効化が安全。
3. **One Euro Filterの.clone() vs .take()**: 毎フレーム`.clone()`でデータ供給→75Hz, alpha≈0.11（強い平滑化、安定）。`.take()`だと25Hz, alpha≈0.28（不安定）。
4. **キーポイント毎にカメラペアを変えるとz軸ノイズ**: Hip基準ペア固定で解決。
5. **分離構成でシンプル化**: パイプラインが線形のためBevy ECS不要。素の関数チェーン+スレッドで十分。
