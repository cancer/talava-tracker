# Models

このディレクトリにはONNXモデルファイルを配置します。
モデルファイル（*.onnx）はgit管理外です。

## MoveNet Lightning

### ダウンロード方法

#### 方法1: PINTO Model Zoo からダウンロード（推奨）

1. [PINTO Model Zoo - MoveNet](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/115_MoveNet) にアクセス
2. リポジトリをクローン:
   ```bash
   git clone https://github.com/PINTO0309/PINTO_model_zoo.git
   ```
3. モデルファイルをコピー:
   ```bash
   cp PINTO_model_zoo/115_MoveNet/saved_model_lightning_192x192/movenet_singlepose_lightning_192x192_p6_nopost_float32.onnx models/movenet_lightning.onnx
   ```

#### 方法2: TFLite から変換

1. TensorFlow Hub からTFLiteモデルをダウンロード:
   ```bash
   curl -L -o movenet.tflite "https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/int8/4?lite-format=tflite"
   ```

2. tf2onnx で変換:
   ```bash
   pip install tf2onnx
   python -m tf2onnx.convert --tflite movenet.tflite --output models/movenet_lightning.onnx --opset 13
   ```

### 配置

モデルを以下のパスに配置:
```
models/movenet_lightning.onnx
```

### モデル仕様

- 入力名: `input`
- 入力形状: `[1, 192, 192, 3]` (int32, RGB, 0-255)
- 出力名: `output_0`
- 出力形状: `[1, 1, 17, 3]` (float32, y, x, confidence)
- キーポイント数: 17

### 注意

モデルの入出力名はバージョンによって異なる場合があります。
実行時にエラーが出る場合は、以下のコマンドでモデルの入出力名を確認してください:

```python
import onnx
model = onnx.load('models/movenet_lightning.onnx')
print([i.name for i in model.graph.input])
print([o.name for o in model.graph.output])
```
