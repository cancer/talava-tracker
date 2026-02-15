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

※ 現在は腰（VMT_0）のみ実装済み

## 手順

### 1. talava-tracker を起動

Mac 側で実行:

```bash
VMT_ADDR=<WindowsのIP>:39570 cargo run --release --bin tracker_sender
```

### 2. VMT で受信確認

VMT のウィンドウでトラッカーが認識されていることを確認。

### 3. SteamVR でトラッカー確認

SteamVR のデバイス一覧に VMT_0 が表示されることを確認。

### 4. VMC でトラッカー割当

1. VMC を起動
2. 設定 → トラッカー割当
3. 腰 → VMT_0 を選択
4. 適用

## トラブルシューティング

### トラッカーが表示されない

- VMT が起動しているか確認
- ファイアウォールで UDP 39570 が許可されているか確認
- talava-tracker の `Sent:` が 0 でないか確認

### 位置がずれる

- キャリブレーションが必要（フェーズ3で実装予定）
- スケール調整が必要な場合あり
