# VMC トラッカー割当手順

## 前提条件

- Windows PC に VMT (Virtual Motion Tracker) がインストール済み
- VMC (Virtual Motion Capture) がインストール済み
- SteamVR が起動済み
- VMT が起動済み

## トラッカー対応表

| Index | 部位 | VMTトラッカー名 |
|-------|------|-----------------|
| 0     | 腰   | VMT_0           |
| 1     | 左足 | VMT_1           |
| 2     | 右足 | VMT_2           |
| 3     | 胸   | VMT_3           |
| 4     | 左膝 | VMT_4           |
| 5     | 右膝 | VMT_5           |

## 手順

### 1. inference_server を起動

Windows 側で実行:

```bash
inference_server.exe
```

設定は `inference_server.toml` で管理。`vmt_addr` にVMTの受信アドレスを設定（デフォルト: 127.0.0.1:39570）。

### 2. camera_server を起動

Mac 側で実行:

```bash
cargo run --release --bin camera_server
```

設定は `camera_server.toml` で管理。`server_addr` にWindows側のIPとポートを設定。

### 3. VMT で受信確認

VMT のウィンドウでトラッカーが認識されていることを確認。

### 4. SteamVR でトラッカー確認

SteamVR のデバイス一覧に VMT_0 〜 VMT_5 が表示されることを確認。

### 5. VMC でトラッカー割当

1. VMC を起動
2. 設定 → トラッカー割当
3. 腰 → VMT_0、左足 → VMT_1、右足 → VMT_2、胸 → VMT_3 を選択
4. 適用

## トラブルシューティング

### トラッカーが表示されない

- VMT が起動しているか確認
- ファイアウォールで UDP 39570 が許可されているか確認
- inference_server のログでVMT送信が行われているか確認

### camera_server が接続できない

- inference_server が先に起動しているか確認
- ファイアウォールで TCP 9000 が許可されているか確認
- `camera_server.toml` の `server_addr` が正しいか確認

### 位置がずれる

- ポーズキャリブレーションが必要（コンソールで `c + Enter`、または自動発動を待つ）
- `inference_server.toml` の `offset_y` で高さを調整
