# 検証ファイル

- `VALIDATION.md`: 実機で確認した内容の要約。
- `benchmark-formal.json`: デモ稼働中の30秒×3回、GPU/NPU両方。
- `benchmark-stress.json`: Chrome表示中の各5分、メモリ・入力動画・検出結果の受信経過を含む。
- `browser-check/`: 実Chromeでの操作結果とGPU/NPU画面。
- `environment.json`: OS・Chrome・GPU/NPUランタイム・モデルハッシュ。
- `xrt-npu-first.json`, `xrt-npu-second.json`: NPU workerのハードウェア完了カウンター。
- `slow-client.json`: MJPEG受信を20秒止めたクライアントの影響。
- `demo-*/`: 起動ごとのworkerログ・環境・モデル検証・切り替え・終了状態。
- `model-validation/`: モデル準備時のNPU検証（採用モデルの選択に使用）。

開発中の不合格モデル・性能調査の記録も残しています。採用構成と最終判断は `VALIDATION.md`、実際に起動するモデルは `models/npu/selection.json` を参照してください。
