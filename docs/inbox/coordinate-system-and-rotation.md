# 座標系と回転の定義

## 三角測量後の座標系（カメラ座標系）

三角測量はcam0のカメラキャリブレーション座標系（OpenCV準拠）で3D座標を出力する。

| 軸 | 方向 |
|----|------|
| X軸 | カメラから見て右 |
| Y軸 | カメラから見て下 |
| Z軸 | カメラの奥行き（前方） |

## ワールド座標系への変換

`to_world_coords(pose, M)` でカメラ座標系からワールド座標系に変換する。
回転行列 M = R_x(-tilt) * R_y(yaw) を全キーポイントに適用:

```
[wx, wy, wz] = M * [cx, cy, cz]
```

変換後のワールド座標系:

| 軸 | 方向 | 補足 |
|----|------|------|
| X軸 | 体の左右 | yaw 補正により hip line に揃う |
| Y軸 | 鉛直下向き | tilt 補正により重力方向に揃う |
| Z軸 | 前後 | 水平面内の前方 |

詳細は `world-coordinate-transform.md` を参照。

## 回転軸と動作の対応

| 軸周りの回転 | 名称 | 動作 | 計算方法（肩ベース） |
|-------------|------|------|---------------------|
| X軸 | pitch | 前傾/後傾 | 脊椎ベクトルのY-Z平面角度 |
| Y軸 | yaw | 左右旋回 | 左右肩のz座標差: `atan2(dz, dx)` |
| Z軸 | roll | 左右傾き | 左右肩のy座標差: `atan2(dy, dx)` |

## camera_yaw / tilt_angle の推定

キャリブレーション時に被写体のポーズから推定する:

```
camera_yaw = atan2(lh.z - rh.z, lh.x - rh.x)    // 左右hipのX-Z平面角度
tilt_angle = atan2(lower_z - hip_z, lower_y - hip_y)  // hip→足のY-Z平面傾き
```

これらから `build_world_rotation(yaw, tilt)` で回転行列 M を構築し、`BodyCalibration.world_rotation` に保存する。

## VMT送信時の座標系

`convert_position(x, y, z)` でワールド座標をキャリブレーション基準の相対座標に変換:
```
pos_x = ref_x - x   (mirror_x の場合は符号反転)
pos_y = offset_y + (ref_y - y)
pos_z = ref_z - z
```

出力 `[pos_x, pos_y, pos_z]` がそのままVMTに送信される。
VMTはUnity座標系（左手系、Y-up）: X=右, Y=上, Z=前。

## クォータニオン変換

| 関数 | 回転軸 | クォータニオン [qx, qy, qz, qw] |
|------|--------|-------------------------------|
| `yaw_to_quaternion(θ)` | Y軸 | `[0, sin(θ/2), 0, cos(θ/2)]` |
| `yaw_pitch_to_quaternion(yaw, pitch)` | Y+X軸 | 合成（膝で使用） |
| `yaw_roll_to_quaternion(yaw, roll)` | Y+Z軸 | 合成（胸・腰で使用） |

## compute_shoulder_yaw の修正経緯

旧実装は `atan2(dy, dx)` を使用していた。
これはXY平面（左右-上下）での肩ラインの傾きであり、実質的にroll（Z軸回転）を計算していた。
しかし `yaw_to_quaternion()` でY軸回転に変換していたため、正しい軸に反映されなかった。

修正: `atan2(dz, dx)` に変更し、X-Z平面での肩ラインの角度（Y軸回転 = yaw）を正しく計算。
