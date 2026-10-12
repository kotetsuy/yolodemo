# 実機検証記録

測定日: 2026-10-12（Asia/Tokyo）。入力448×448、batch=1。

その後の切り替え不具合修正とUSBカメラでの5回連続検証は [switch-fix/README.md](switch-fix/README.md) を参照。性能表は修正前の正式測定記録。

高解像度・複数カメラ対応の追加検証は [cameras/README.md](cameras/README.md) を参照。

## 正式測定: 30秒×3回

動画デコード、検出描画、MJPEG配信を動かし、MJPEGを受信しながら測定。選択中のworkerだけが推論を反復した。

| 演算器 | 1回目 | 2回目 | 3回目 | 完了数/経過秒の合算 | 目標 |
|---|---:|---:|---:|---:|---|
| GPU | 390.18 | 388.53 | 388.04 | 388.92 FPS | >300: 達成 |
| NPU | 82.35 | 82.73 | 82.85 | 82.64 FPS | >60: 達成 |

各回の完了数・実測秒・入力FPS・モデルハッシュ・環境・計測範囲は [benchmark-formal.json](benchmark-formal.json) に保存。

## Chrome表示中の連続動作: 各5分

Chromeの通常ウィンドウで映像を表示し、追加のMJPEG受信も行った。各演算器について300秒測定した。

| 演算器 | 完了数 | 実測秒 | 推論FPS | 入力FPSの平均 | worker RSS: 初期→末尾 / 最大 |
|---|---:|---:|---:|---:|---|
| GPU | 114807 | 300.028 | 382.65 | 30.60 | 2367.0→2367.0 / 2367.0 MiB |
| NPU | 24267 | 300.012 | 80.89 | 30.68 | 579.4→579.4 / 579.4 MiB |

- GPU測定: サーバーRSS 316.8→327.2 MiB、最大441.2 MiB。確認した動画10本、観測した検出結果の最大受信経過183.5 ms。
- NPU測定: サーバーRSS 327.5→328.2 MiB、最大440.0 MiB。確認した動画10本、観測した検出結果の最大受信経過78.8 ms。

完了カウンターのリセット、推論heartbeatの1秒超停止、演算器変更、workerエラーを測定スクリプトで監視。詳細とメモリの時系列は [benchmark-stress.json](benchmark-stress.json) に保存。

## 数値検証

- GPU: 実画像3枚、白黒2入力、乱数3入力で通常forward/グラフ再生が一致。CPUの安定Top-kとも一致。FP16のスコア値を変えず、同値の順序を安定化。
- NPU: 同じ448量子化モデルのCPU実行を参照。表示閾値0.25、検出数・クラス一致、IoU≥0.9、スコア差≤0.05を維持。8入力で合格、実画像の最小IoUは0.977042、最大スコア差は0.046207。12回の交互入力で生出力・検出の再現性を確認。
- 元の全NPU構成の最大スコア差0.059795は不合格。許容値を広げず、既存QDQパラメーターを保持して分類ヘッドと直前のSiLUをCPUへ分離した。
- NPUの計測は外部INT16量子化、NPUバックボーン・枠、CPU分類ヘッド、Top-kを含む。GPUはHIPグラフと同期、新規入力の転送・サンプル出力コピーを含む。
- GPU/NPU間の全数値一致やmAPを保証する検証ではない。

NPUモデルSHA-256: `6c1201a9959108ce40d522f450d1b48a3a703716e9cb8dac5d56be38cadb4c89`。GPU重みSHA-256: `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`。

## NPUハードウェアの確認

XRTのAIE context PIDがNPU worker PIDと一致。Active状態でハードウェア完了カウンターが3730→5847へ増加し、errors=0。記録: [1回目](xrt-npu-first.json)、[2回目](xrt-npu-second.json)、[同時のworker状態](status-npu-hardware.json)。VitisAI EPの登録だけでNPU動作と判定していない。

## Chromeと機能

- 実ChromeでGPU→NPU→GPU→NPU→GPUと連続選択を確認。worker PIDが変わらず、最後の指定が反映され、JavaScriptエラーなし。
- ページ上部に映像、検出枠、左上に演算器と推論FPS。動画は1280×720、接続後のUSBカメラは640×480。元の縦横比を維持。
- 切り替え時間はモデル初回使用時を含めて 14.3〜141.6 ms。ポーリングと画像受信を含むブラウザ観測値も記録。
- カメラ不在時に動画へ移行。10本の実動画の巡回は連続測定のsource記録で確認。
- CPUテスト7件: 前処理・座標復元、量子化の丸め/飽和、実完了数からのFPS、IPCの不変スナップショット、動画の相対パス/ループ、不正プレイリスト、古いepochの結果破棄。

- MJPEG受信を20秒止めたクライアントがあっても推論と描画は継続。サーバーRSS増加は360 KiB。記録: [slow-client.json](slow-client.json)。

- NPU workerを終了すると既存GPU workerへ自動移行し、epochを更新して新しい検出映像を表示。記録: [worker-error-fallback.json](worker-error-fallback.json)。
- SIGTERM終了後に所有するサーバー・GPU/NPU worker・共有メモリが残らないことを確認し、同じポートで再起動。記録: [shutdown-check.json](shutdown-check.json)、[restart-check.json](restart-check.json)。

Chrome記録: [結果](browser-check/result.json)、[GPU画面](browser-check/gpu.png)、[NPU画面](browser-check/npu.png)。

## 残る検証範囲

正式な性能測定時はカメラ不在。接続後のUSBカメラ取得・Chrome表示・GPU/NPU切り替えは追加確認済み。物理カメラ切断と448用モデルのCOCO mAP測定は未実施。

モデル検証で異常を検出したworkerは開始せず、検証結果を保存する。NPUのCPU並列数4固定だけでは交互入力の不一致が再発したため、NPU→GPUの順次初期化へ変更した。修正後の起動・切り替えを5回連続で確認。異常時は `npu-alternating-mismatch.npz` に比較データを保存する。
