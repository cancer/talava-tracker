# ワールド座標系の導入 (#59)

## 背景: なぜ変更が必要だったか

三角測量後の座標はcam0のカメラ座標系（x=右, y=下, z=奥）で表現される。カメラは被写体に対して斜め・俯角で配置されているため、カメラ座標系のy軸は鉛直（重力方向）と一致しない。

旧実装では2段階の近似回転でワールド座標に変換していた:

```
[三角測量結果] → rotate_pose_xz(yaw) → rotate_pose_yz(tilt, filtered_hip_z) → [近似ワールド座標]
```

この近似には3つの問題があった:

1. **x→yリーク**: カメラがyaw回転している場合、横移動(x)がtilt補正経由でy（高さ）に漏れる
2. **z→yリーク**: per-keypoint zノイズがsin(tilt)経由でyに漏れる
3. **filtered_hip_zハック**: 上記2の対策として、tilt補正のy計算にhipのフィルタ済みzを代用。これが1の原因でもあった

根本原因は、2つの独立した軸回転を順番に適用する近似が、完全な3D回転と等価ではないこと。

## 変更内容

### 旧パイプライン

```
三角測量(カメラ座標) → rotate_pose_xz(yaw) → rotate_pose_yz(tilt, filtered_hip_z)
  → estimate_depth() / keypoint_depth() で z 算出
  → convert_position(x, y, hip_x, hip_y, pos_z) で相対座標化
  → TrackerPose → VMT
```

### 新パイプライン

```
三角測量(カメラ座標) → to_world_coords(pose, M) で一括3D回転
  → convert_position(x, y, z) で直接差分
  → TrackerPose → VMT
```

### 回転行列 M の導出

カメラの俯角(tilt)と水平回転(yaw)を一括で補正する行列:

```
M = R_x(-tilt) * R_y(yaw)
```

ここで:
- `R_y(yaw)`: Y軸周りの回転（カメラの水平向き補正）
- `R_x(-tilt)`: X軸周りの逆回転（カメラの俯角補正）

展開すると:

```
     | cy      0     sy     |
M =  | -sy*st  ct    cy*st  |
     | -sy*ct  -st   cy*ct  |

(sy = sin(yaw), cy = cos(yaw), st = sin(tilt), ct = cos(tilt))
```

全キーポイントに対して `[wx, wy, wz] = M * [cx, cy, cz]` を適用する。

#### 重要: なぜ R^T ではなく M なのか

issue #59 の提案では `R^T`（R の転置）を使う記述があったが、実際には旧コードの `rotate_pose_xz` + `rotate_pose_yz` が計算していたのは `R_x(-tilt) * R_y(yaw)` であり、これは `R^T` とは異なる行列だった。

初回実装で R^T を使ったところトラッカーが完全に壊れた。基底ベクトル (1,0,0), (0,1,0), (0,0,1) に対する変換結果を旧コードと比較して不一致を発見し、正しい行列 M を導出した。

```
point (1,0,0): old=(0.9211,-0.1151,-0.3720) R^T=(0.9211,0.0000,0.3894)  ← 不一致
point (1,0,0): old=(0.9211,-0.1151,-0.3720) M  =(0.9211,-0.1151,-0.3720) ← 一致
```

### 削除したもの

| 要素 | 理由 |
|------|------|
| `rotate_pose_xz()` | `to_world_coords()` に統合 |
| `rotate_pose_yz()` | `to_world_coords()` に統合 |
| `estimate_depth()` | ワールド座標の z が直接 depth を表す |
| `keypoint_depth()` | 同上 |
| `hip_z_filter` (ScalarFilter) | filtered_hip_z ハックが不要に |
| `FilterConfig.hip_z_min_cutoff` | 上記に伴い不要 |
| `FilterConfig.hip_z_beta` | 上記に伴い不要 |

### 変更したもの

| 要素 | 旧 | 新 |
|------|-----|-----|
| `BodyCalibration` | `tilt_angle` + `camera_yaw` を個別保持 | `world_rotation: [f32; 9]` に統合（ログ用に yaw/tilt も保持） |
| ankle/knee offset | `Option<(f32, f32)>` (x, y) | `Option<(f32, f32, f32)>` (x, y, z) |
| `convert_position` | 5引数、hip_x/hip_y 経由の間接計算 | 3引数、`ref - kp` の直接差分 |
| `compute()` の `hip_center` | `Option<(f32, f32)>` | `Option<(f32, f32, f32)>` (z追加) |
| chest の z | hip z を流用 | shoulder-hip 内挿 (0.35) |
| `BodyTracker::new()` | 4引数 (FilterConfig含む) | 3引数 (FilterConfig不要) |

### 追加したテスト

| テスト名 | 検証内容 |
|---------|---------|
| `test_build_world_rotation_identity` | yaw=0, tilt=0 で単位行列 |
| `test_build_world_rotation_yaw_only` | yaw=90° で x→-z の回転 |
| `test_build_world_rotation_tilt_only` | tilt=45° で z→(y,z) の回転 |
| `test_to_world_coords_preserves_confidence` | confidence 値が変換で失われない |

### 更新したテスト

uncalibrated の基準点が `(0.5, 0.5)` → `(0, 0, 0)` に変更されたため、以下3テストの期待値を更新:
- `test_body_compute_uncalibrated_positions`
- `test_body_compute_offset_y`
- `test_body_compute_foot_y_offset`

既存の軸分離テスト（`test_x_movement_with_noise_does_not_leak_to_y` 等）は変更なしで合格。

## 数学的な補足

### convert_position の簡素化

旧コードの `convert_position(x, y, hip_x, hip_y, pos_z)`:
```
global_x = ref_x - hip_x
pos_x = global_x + (hip_x - x) = ref_x - x
```
hip_x/hip_y は式中で打ち消し合い、実質 `ref - kp` だった。新コードはこれを明示的に3引数で表現。

### filtered_hip_z ハックが不要になった理由

旧コード `rotate_pose_yz` では:
```
kp.y = y * cos(tilt) + filtered_hip_z * sin(tilt)  // y計算にhip z（フィルタ済み）
kp.z = -y * sin(tilt) + kp.z * cos(tilt)           // z計算にper-kp z
```

y 計算で per-keypoint z を使うと、各キーポイントの z ノイズが `sin(tilt)` 倍されて y にリークしていた。hip の z をフィルタして代用することでノイズを抑えていたが、これが hip の z 変動（横移動との相関）もリークさせていた。

新コードでは完全な3D回転を適用するため、y と z の計算に同じ per-keypoint 座標を一貫して使用する。ノイズ対策はワールド座標変換後の OneEuro フィルタ（PoseFilter）に委ねる。
