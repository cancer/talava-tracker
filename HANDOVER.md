# マルチカメラ三角測量 引き継ぎ資料

## 絶対ルール
- **撮影対象はユーザー本人。カメラの前に立ってもらわないとポーズ検出できない**
- **手順: コード修正・ビルド等を全部終わらせる → 準備完了後に「カメラの前に立ってください」と報告 → ユーザーが立ってからcargo run**
- **ユーザーは暇ではない。準備が終わる前に呼ぶな。報告なしに起動するな**
- **被写体が不要なcargo run(ビルド確認、cargo test等)では呼ぶな。呼ぶのはポーズ検出テスト時のみ**

## 現在の状態（2026-02-15）

### 完了した修正
1. **歪み補正追加** (`src/triangulation.rs`): CameraParamsにundistort_point()実装。calibration.jsonの歪み係数(k1=-1.29等)を反映
2. **再投影エラーチェック** (`src/triangulation.rs`): 三角測量結果をカメラに再投影し、50px以上の誤差で棄却
3. **画像サイズ修正** (`src/triangulation.rs`): image_width/image_heightフィールド追加。P[0,2]からの推定を廃止
4. **per-keypointバウンダリチェック** (`src/bin/tracker_bevy.rs`): 個別キーポイントの±10m/z>0チェック
5. **2段階ステールタイムアウト** (`src/bin/tracker_bevy.rs`): Phase1(0.3s)送信停止、Phase2(3.0s)速度参照クリア
6. **limb_ok None→false** (`src/bin/tracker_bevy.rs`): hip基準なしの四肢を拒否
7. **自動キャリブレーション** (`src/bin/tracker_bevy.rs`): hip 3秒検出で自動発動

### 未解決の重大問題

#### 1. z値ドリフト（最優先）
最新の実行結果(bc947e5)で確認:
- キャリブレーション前: hip z=1.25（正常）→ z=6.33（異常ジャンプ）
- 自動キャリブレーションがz=6.5の異常値で発動してしまった
- キャリブレーション後: hip z=-0.08（正常）→ z=-7.11（ドリフト）
- **原因**: 三角測量の出力自体が不安定。z値が1.3と6.5の間で二峰性を示す

対策案:
- 自動キャリブレーションに値の妥当性チェックを追加（z値の分散が小さいことを確認）
- 再投影エラー閾値(50px)を下げて品質の低い三角測量を排除
- 三角測量結果に時間的な一貫性チェックを追加

#### 2. 右側キーポイント欠落
R_foot, R_kneeがほぼ出現しない。原因候補:
- カメラ2からの右側キーポイントの検出信頼度が低い
- 片方のカメラでしか見えないため三角測量の2カメラ要件を満たさない
- カメラ設置角度の問題（上半身がフレーム外になる問題と関連）

#### 3. L_footのフリーズ
L_foot(-0.37, -0.57, -6.39)が長時間同一座標。ステールタイムアウトが正しく機能していない可能性

#### 4. "no pose"の頻度が高い
ポーズ検出できるフレームが半分以下。考えられる原因:
- 再投影エラーで正当なキーポイントも棄却されている
- 両カメラで同時に検出される確率が低い（フレーム同期の問題）

### デバッグの進め方

#### 再投影エラーの実測値を確認する
`src/triangulation.rs`のtriangulate_poses()内、再投影チェックのところに一時的なeprintlnを追加:
```rust
// この位置（max_reproj_err計算後、if文の前）に追加
eprintln!("reproj[{}]: err={:.1}px pos=({:.2},{:.2},{:.2})",
    kp_idx, max_reproj_err, x, y, z);
```
これで歪み補正後の実際のエラー値がわかる。50pxの閾値が適切か判断できる。

#### undistort_point()の収束確認
calibration.jsonの歪み係数が非常に大きい(k1=-1.29, k3=-33.4)。反復法(10回)で収束しているか確認:
```rust
// undistort_point()のforループ内に追加
if i == 9 { eprintln!("undistort iter10: ({:.4},{:.4})", x, y); }
```

### ファイル構成
| ファイル | 内容 |
|---------|------|
| `src/triangulation.rs` | DLT三角測量、歪み補正、再投影チェック |
| `src/bin/tracker_bevy.rs` | Bevy ECSメインループ、品質チェック、VMT送信 |
| `src/tracker/body.rs` | キーポイント→トラッカー位置変換、キャリブレーション |
| `calibration.json` | カメラ内部/外部パラメータ、歪み係数 |
| `config.toml` | アプリ設定（カメラ、フィルタ、VMT） |

### calibration.jsonの歪み係数
- Camera 0: k1=-1.29, k2=7.25, p1=-0.14, p2=0.03, k3=-33.4
- Camera 2: k1=-2.73, k2=15.3, p1=0.10, p2=0.008, k3=-27.1

これらは非常に大きな値。広角レンズの可能性が高い。undistort_point()の反復法が正しく収束するか要確認。

### 過去の実行結果の場所
- `/private/tmp/claude-501/-Users-cancer-repos-talava-tracker/tasks/bc947e5.output` - 歪み補正あり、キャリブレーション成功→zドリフト
- `/private/tmp/claude-501/-Users-cancer-repos-talava-tracker/tasks/b0280aa.output` - 歪み補正あり、キャリブレーション前のデータ（hip安定）
- `/private/tmp/claude-501/-Users-cancer-repos-talava-tracker/tasks/bada63f.output` - キャリブレーション失敗（hip不検出タイミング）
