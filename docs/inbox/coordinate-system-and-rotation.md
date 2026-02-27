# 座標系と回転の定義

## 三角測量後の座標系

三角測量はカメラキャリブレーション座標系（OpenCV準拠）で3D座標を出力する。
`rotate_pose_xz`（camera_yaw補正）でhip lineをX軸に揃えた後の座標系:

| 軸 | 方向 | 補足 |
|----|------|------|
| X軸 | 体の左右 | hip lineが揃う方向 |
| Y軸 | 上下 | |
| Z軸 | 前後（正中線） | カメラの奥行き方向から回転補正済み |

## 回転軸と動作の対応

| 軸周りの回転 | 名称 | 動作 | 計算方法（肩ベース） |
|-------------|------|------|---------------------|
| X軸 | pitch | 前傾/後傾 | 脊椎ベクトルのY-Z平面角度 |
| Y軸 | yaw | 左右旋回 | 左右肩のz座標差: `atan2(dz, dx)` |
| Z軸 | roll | 左右傾き | 左右肩のy座標差: `atan2(dy, dx)` |

## camera_yaw補正の仕組み

キャリブレーション時に左右hipのX-Z平面での角度を計測:

```
camera_yaw = atan2(lh.z - rh.z, lh.x - rh.x)
```

`rotate_pose_xz(pose, camera_yaw)` でhip lineをX軸に揃える。
これにより、カメラが被写体を斜めから撮影している場合でも座標系が正規化される。

## tilt補正の仕組み

hip→ankle/kneeのY-Z平面での傾き:

```
tilt_angle = atan2(lower_z - hip_z, lower_y - hip_y)
```

`rotate_pose_yz(pose, tilt_angle)` でY-Z平面の傾きを補正。

## VMT送信時の座標系

`convert_position()` の出力 `[pos_x, pos_y, pos_z]` がそのままVMTに送信される。
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
