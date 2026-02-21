# マルチカメラ三角測量 引き継ぎ資料

## 絶対ルール
- **撮影対象はユーザー本人。カメラの前に立ってもらわないとポーズ検出できない**
- **手順: コード修正・ビルド等を全部終わらせる → 準備完了後に「カメラの前に立ってください」と報告 → ユーザーが立ってからcargo run**
- **ユーザーは暇ではない。準備が終わる前に呼ぶな。報告なしに起動するな**
- **被写体が不要なcargo run(ビルド確認、cargo test等)では呼ぶな。呼ぶのはポーズ検出テスト時のみ**
- **動作確認が済むまでコミットするな**

## 現在の状態（2026-02-21）

### 最優先課題: レンズ歪み補正の有効化

全カメラの歪み補正が無効化されている（`src/bin/tracker_bevy.rs` L201: `let dc: [f64; 5] = [0.0; 5]`）。

**問題**: 2D観測は歪んだ画像上の座標だが、射影行列P=K[R|t]は歪みなしの座標系で定義されている。この座標系不一致がリプロジェクションエラーの主因。

**診断結果**（2026-02-21 実機ログ `tracker_20260221_160027.log`）:
- c0+c1: hip reproj error = 145-177px（閾値120pxをわずかに超過）→ ref_pair=NONE
- c0+c2: 1700-3300px（cam2の歪みが極端）
- c1+c2: 1600-3300px
- ref_pair=NONEが大量発生 → hipが三角測量できず → キャリブレーション失敗・no poseの主因

**無効化された経緯**: cam2のdist_coeffs（k2=11.5, k3=-49.9）が極端すぎてundistort_point()のNewton-Raphson法が収束しない → 全カメラ巻き添えで無効化された。

**解決方針**: 全カメラの歪み補正を有効にする。cam2の極端な係数に対しては、逆歪みモデル（観測を補正）ではなく順歪みモデル（投影側に歪みを適用）を使えば反復法不要で任意の係数に対応可能。

### z-jump棄却の永久ループ

`triangulate_poses()`のhip z-jumpチェック（MAX_HIP_Z_JUMP=0.3m）で棄却されると、PREV_HIP_Zが更新されない → 以降のフレームも全て棄却される永久ループに陥る。

実測: prev_hip_z=1.536が固定され、z=4-14mのフレームが延々と棄却される。

### 未コミットの変更（main上）

1. 速度チェック基準: last_poses→last_raw_poses（フィルタ前位置と比較）
2. max_displacement: 0.4→0.15
3. 下半身フィルタ分離: lower_body_position_min_cutoff=1.0, beta=0.5
4. キャリブレーション時のlast_raw_poses/reject_countリセット
5. TriangulationDiag: 三角測量の棄却理由をログファイルに出力
6. config.toml: leg_scale, offset_y, foot_y_offset変更

### カメラ構成（3台）

| カメラ | 解像度 | fx | dist_coeffs | 備考 |
|--------|--------|-----|-------------|------|
| cam0 (index=0) | 1760x1328 | 1567 | k1=0.22, k2=-1.34, k3=5.07 | 基準カメラ（rvec/tvec=0） |
| cam2 (index=2) | 1920x1440 | 1765 | k1=-0.80, k2=11.5, k3=-49.9 | 歪み係数が極端 |
| cam3 (index=3) | 1920x1080 | 1521 | k1=0.26, k2=-1.95, k3=1.76 | 上方見下ろし角度、最も広角 |

- **cam2・cam3は角度が異なるからこそ三角測量に不可欠。除外は選択肢にない**
- calibration.jsonのreprojection_error: cam0=0.20px, cam2=0.13px, cam3=0.38px（キャリブレーション自体は良好）

### 完了した修正

1. **歪み補正実装** (`src/triangulation.rs`): CameraParamsにundistort_point()実装（Newton-Raphson法）
2. **再投影エラーチェック**: MAX_REPROJ_ERROR=120px（max方式）
3. **Hip基準ペア固定**: 全キーポイントで同じカメラペアを使いz軸一貫性を確保
4. **per-keypointバウンダリチェック**: ±10m/z>0
5. **hip z-jumpチェック**: MAX_HIP_Z_JUMP=0.3mでフレーム棄却
6. **TriangulationDiag**: 棄却理由（ref_pair, reproj error, z-jump）のログ出力

### ファイル構成

| ファイル | 内容 |
|---------|------|
| `src/triangulation.rs` | DLT三角測量、歪み補正、再投影チェック、TriangulationDiag |
| `src/bin/tracker_bevy.rs` | Bevy ECSメインループ、品質チェック、VMT送信 |
| `src/tracker/body.rs` | キーポイント→トラッカー位置変換、キャリブレーション |
| `src/tracker/one_euro.rs` | One Euroフィルタ（上半身/下半身分離対応） |
| `src/config.rs` | 設定ファイル読み込み（lower_body_position_min_cutoff追加済み） |
| `calibration.json` | カメラ内部/外部パラメータ、歪み係数 |
| `config.toml` | アプリ設定（カメラ、フィルタ、VMT） |

### デバッグの進め方

#### TriangulationDiagのログ出力を確認する

`triangulate_system`が問題フレーム（ref_pair=NONE, z-jump棄却, kps=0, boundary棄却）を自動的にログファイルに出力する。

出力例:
```
tri: cams=3 ref=NONE hip_errs=[c0+c1=155px,c0+c2=2671px,c1+c2=2532px] kps=16/37
tri: cams=2 ref=c0+c1(113px) kps=22/37 boundary_rej=16
tri: cams=2 ref=c0+c1(44px) Z_JUMP(1.536->4.289) kps=0/37
```

読み方:
- `ref=NONE`: hipの基準カメラペアが見つからない（全ペアがMAX_REPROJ_ERROR超過）
- `hip_errs=[...]`: 各カメラペアのhipリプロジェクションエラー
- `ref=c0+c1(113px)`: c0+c1ペアが選ばれた（エラー113px）
- `Z_JUMP(prev->cur)`: hip z値のジャンプで全キーポイント棄却
- `boundary_rej=N`: 境界チェック（±10m, z>0）で除去されたキーポイント数
- `kps=N/37`: 有効キーポイント数/全キーポイント数
