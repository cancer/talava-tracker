# 制約条件

## CON-01: ハードウェア制約

### CON-01-1: 実行環境

- **Mac側（camera_server）**: macOS（AVFoundationバックエンド依存）。カメラキャプチャ + TCP送信
- **Win側（inference_server）**: Windows。ONNX推論（DirectML GPU）+ 三角測量 + VMT送信
- **通信**: Mac→Win TCP接続（ポート9000）、Win→VMT OSC/UDP（ポート39570）
- **VR接続**: Quest Link（Air Link）経由。Steam Linkは遅すぎてNG

### CON-01-2: カメラ

- **内蔵カメラ**: Mac内蔵カメラ（index 0）
- **外部カメラ**: iPhone連携（Camo Camera等経由）。iPhone 7, 8, 13を使用
- **解像度**: 1760x1328 / 1920x1440 / 1920x1080（カメラによる）
- **カメラFPS**: 30FPS（カメラ性能に依存。設定上は60だが実効30）
- **物理配置**:
  - カメラのアイラインが肩の高さ（下半身になるほど距離が縮まる）
  - 1台では人体フルボディが映らない（上半身がフレーム外になりがち）
  - カメラの物理配置を正確にできないため、ChArUcoキャリブレーションで自動推定
- **iPhone連携の制約**:
  - 望遠モードに勝手に切り替わる問題
  - 複数iPhoneの同時認識にCamo Camera等の外部アプリが必要

### CON-01-3: ネットワーク

- **Mac→Windows**: 同一LAN内でTCP通信（ポート9000、フレーム転送）
- **Windows内**: localhost UDP通信（ポート39570、VMT送信）
- **帯域**: Gigabit有線LAN前提。JPEG Q80 × 3カメラ × 30fps ≈ 190Mbps
- **遅延**: LAN内のため無視できるレベル

## CON-02: ソフトウェア制約

### CON-02-1: 言語・ランタイム

- **Rust**: 全コンポーネントをRustで実装（「全部Rustで閉じる」方針）
- **ONNX Runtime**: 推論エンジン。CoreML対応だが効果限定的
- **OpenCV**: C++バインディング（opencv crate）。ビルド時にOpenCVライブラリが必要

### CON-02-2: 推論モデル

- ONNXフォーマットのみ対応
- モデルファイルはgit管理外（models/ディレクトリ）
- SimCC出力形式（SpinePose）とヒートマップ出力形式（MoveNet）の両方に対応

### CON-02-3: TCPプロトコル

- tokio + tokio-util (LengthDelimitedCodec) + bincode + serde
- Mac→Win: フル画像JPEG送信（人物検出はWin側で実行）
- フレーム同期はMac側で実施（全カメラのフレームが揃ってから送信）

## CON-03: 運用制約

### CON-03-1: 用途

- **撮影用途**: Beat Saber撮影が目的。操作用途ではない
- **編集前提**: 致命的破綻がなければ許容。完璧なリアルタイムトラッキングは不要
- **最終表示先**: VMC（バーチャルモーションキャプチャー）のアバター。Beat Saber内のCustomAvatarsではない

### CON-03-2: キャリブレーション

- セッション開始時にキャリブレーション必須（立ち位置は毎回変わる）
- ChArUcoボードを使用（チェスボードは未所持）
- 全カメラから同時にボードが見える位置が必要（外部パラメータ推定時）

### CON-03-3: 被写体

- 撮影対象はユーザー本人
- カメラの前に立っていないとポーズ検出できない
- 人物部分がフレーム外の場合はhipのみ or no pose（単眼推定の限界）

## CON-04: 技術的制約

### CON-04-1: 推論速度のボトルネック

- SpinePose medium: ~40-50ms/フレーム
- マルチカメラ時: 各カメラ順次推論のため、カメラ数×推論時間
- 補間（extrapolate/lerp）でVMT送信レートを確保

### CON-04-2: 三角測量の精度限界

- 2カメラ間の基線長・角度に依存
- 歪み係数が大きい場合（k1=-1.29, k3=-33.4等）、undistort_pointの収束が不確実
- 再投影エラー閾値（120px）のバランス: 厳しくすると検出率低下、緩くすると精度低下
