# HANDOFF: YOLO26n GPU/NPUリアルタイムデモの実装

> 公開版参照の最終照合（2026-10-12）: [yolofast_pub README](https://github.com/kotetsuy/yolofast_pub/blob/main/README.md) の「yolodemo用の追加venv」は本デモのGPU・ONNX出力・ローカル量子化環境の配置と一致する。公開版単体の `.venv` とデモ用環境は別用途。重み・元校正レシピ・画像・NPU SDKは別途必要。独立CPUエクスポート環境による生成物は実機検証後に採用し、過去の測定と混同しない。現在の実行条件は [READMEJ.md](READMEJ.md) を参照。


更新日: 2026-10-12（Asia/Tokyo）  
実装担当: gpt-6.1-sol  
作業ディレクトリ: `~/yolodemo`  
参照プロジェクト: `~/yolofast`（以下 `~/yolofast`）

この文書の要件に基づき、ローカルデモを実装した。起動・構成は [README.md](README.md)、実機での検証記録は [results/VALIDATION.md](results/VALIDATION.md) を参照する。以下の環境確認・構成案は実装前の引き継ぎ時点の記録を含む。第1節はユーザーの確定要件とする。

## 1. ユーザー確認済みの要件

| 項目 | 要件 |
|---|---|
| 目的 | YOLO26nの高速な物体検出を、実際の映像と検出結果で見せる |
| 表示先 | 同じマシンのChromeブラウザ |
| 画面配置 | ページ上部に映像を大きく表示する |
| 入力 | カメラを優先し、利用できなければ `./videos/playlist.m3u8` の動画を順番にループ再生する |
| 推論解像度 | **GPU・NPUとも448×448**。映像表示の解像度とは別の設定 |
| モデル・精度 | YOLO26nを使用し、量子化方式は `~/yolofast` を引き継ぐ。GPUはFP16、NPUは最新採用構成のA16W8・重みper-channel |
| 演算器の選択 | GPU／NPUの**2位置スライダー**で実行中に切り替える |
| 推論ループ | 入力フレームレートに制限されないよう、最新フレームを繰り返し推論する |
| FPS表示 | 実際に完了した推論回数/秒を「推論FPS」として映像左上に大きく重ね、約0.3秒ごとに更新する |
| 速度目標 | GPU **300 FPS超**、NPU **60 FPS超**。448構成のデモを動かした状態で実測する |
| 再利用 | `~/yolofast` の重み、推論実装、モデル生成手順、既存環境を活用する |
| NPU数値基準 | スコア差の許容値 **0.05を維持**し、超過する場合はモデルの改善を続ける（追加確認済み） |

最新フレームの反復推論をFPSに含めることはユーザー了承済み。入力FPS・表示FPSとは区別する。スライダーは負荷配分用ではない。「ページ上部」の指定に、Chromeウィンドウの最前面固定は含まれない。

## 2. 確認済みの環境と参照資料

2026-10-12のファイル確認では、Ubuntu 26.04.1、`/usr/bin/google-chrome`、`/usr/bin/ffprobe`、`/dev/kfd`、`/dev/dri/renderD128`、`/dev/accel/accel0` が存在した。`/dev/video*` は見つからなかった。デバイスの存在確認のみで、今回の引き継ぎ整備では推論を実行していない。実装時にアクセス権・ランタイム・実演算を再確認する。

既存レポート上のハードウェアはRyzen AI Max+ 395、Radeon 8060S（gfx1151）、XDNA 2 NPU。

| 用途 | 既存のパス・参照箇所 |
|---|---|
| GPU環境 | `~/yolofast/.venv-rocm10-py314/bin/python`。保存記録はPython 3.14.3、PyTorch 2.13.0+rocm10.0.0、Ultralytics 8.4.158 |
| GPU推論 | [graph_replay.py](../yolofast/benchmarks/graph_replay.py) の `worker()`、[inference.py](../yolofast/benchmarks/inference.py) |
| GPU重み | `~/yolofast/models/yolo26n.pt` |
| ONNX書き出し環境 | `~/yolofast/.venv-npu-export/bin/python` |
| 量子化環境 | `./.venv-quant/bin/python`。既存採用モデルはAMD Quark 0.11.2で生成 |
| NPU環境 | `source "$HOME/ryzenai_1_8/setup_ryzenai_env.sh"` 後の `python`。保存記録はPython 3.12.13、ORT VitisAI 1.27.0、Ryzen AI SDK 1.8 |
| NPUの最新採用手順 | [distribute/README.md](../yolofast/distribute/README.md)、[REPORT.md](../yolofast/REPORT.md) 第30節 |
| NPU推論 | [npu_split.py](../yolofast/benchmarks/npu_split.py) の `Pipeline` とセッション生成処理 |
| 前処理・入力量子化 | [npu_common.py](../yolofast/benchmarks/npu_common.py) の `letterbox`、`preprocess`、`quantize_input`、`Executor` |
| NPU後処理 | [npu_postprocess.py](../yolofast/benchmarks/npu_postprocess.py) の `decode`（stable Top-k参照実装） |
| 座標の復元 | [evaluate_npu_coco.py](../yolofast/benchmarks/evaluate_npu_coco.py) の `restore_box`。戻り値はCOCO形式xywhなので描画用xyxyへの違いに注意 |
| 動画 | [playlist.m3u8](videos/playlist.m3u8)、[SOURCES.md](videos/SOURCES.md) |

参照先には過去の640入力・旧モデル・旧速度目標が残っている。このデモの要件は第1節を優先し、NPUの量子化手順は最新のper-channel採用モデルを参照する。

## 3. 448×448用モデルの準備

**既存の配布NPUモデルは640×640固定であり、448用としてそのまま読み込めない。** 画像だけ448に縮小したり、ONNX入力のshape表記だけ変更したりして対応済みとしない。以下の生成工程を `yolodemo` 内に実装する。

### GPU

- 既存 `yolo26n.pt` を読み込み、batch=1、FP16、448×448で使用する。
- `graph_replay.py` と同様に、融合前に `model.end2end = True` を設定し、eval、Conv/BN融合、FP16化、ウォームアップ、HIPグラフキャプチャを行う。ROCmでもAPI名は `torch.cuda.CUDAGraph`。
- 固定入力バッファに新しいフレームのテンソルをコピーしてグラフを再生する。キャプチャした戻り値全体の寿命を維持する。
- 448で通常forwardとグラフ再生の出力を検証してから採用する。過去の640での成功を448の検証に代用しない。

### NPU

1. 同じPT重みから、448×448・batch=1・FP32・NMS不要ヘッド・opset 17のONNXを生成する。参照は [export_npu.py](../yolofast/benchmarks/export_npu.py)。
2. **量子化前に**boxes/scores出力を分離する。参照は [prepare_npu_split.py](../yolofast/benchmarks/prepare_npu_split.py)。抽出ノードの存在・演算・実際のshapeを検証する。
3. 分離したFP32モデルを、既存の `A16W8_CONFIG`、`per_channel=True`、MinMax、CLE有効で量子化する。参照は [quantize_npu.py](../yolofast/benchmarks/quantize_npu.py)。
4. 校正には既存採用時の60枚と同じファイル・順序を使う。[既存量子化メタデータ](../yolofast/distribute/models/npu/accuracy95/perchannel/yolo26n-640-a16w8.json) の `calibration` 一覧とハッシュを参照する。場所は `~/yolotest/calib2`。校正前処理も448にそろえる。
5. 入力のQuantizeLinearだけを外し、外部INT16入力版を作成する。参照は [prepare_npu_int16.py](../yolofast/benchmarks/prepare_npu_int16.py)。変換前後のCPU出力一致を確認する。
6. ONNXと同名のJSONに、448用のscale・zero point、モデルのハッシュ、量子化設定、校正画像、検証結果を保存する。**量子化方式を引き継いでも数値パラメーターは再生成するため、640用scaleの転記はしない。**
7. Ryzen AI環境でコンパイル・実行を確認する。初期設定は `opt_level=2`、同期 `session.run`、stable Top-k、ディスクキャッシュ。CPUへの全体フォールバックをNPU動作として扱わない。

生成先の例は `models/npu/yolo26n-448-split-a16w8-perchannel-int16.onnx` と同名JSON。元FP32、分離FP32、量子化FP32入力版も検証用に残す。生成済みファイルの設定・ハッシュが一致する場合は再利用する。

既存コードの `640`・`8400` は、入力、校正、検証画像、scratchバッファ、shape検査、座標復元まで一貫して448対応にする。単純な全文置換ではなく、`imgsz` とモデル入出力メタデータから決定する。448ではstride 8/16/32に対して候補数は `56² + 28² + 14² = 4116` が想定されるが、生成したモデルで確認する。

| 入出力 | 448構成の想定 |
|---|---|
| GPU入力 | FP16 `[1,3,448,448]` |
| GPU最終検出 | `[1,300,6]`、xyxy・score・class ID。実際の戻り値のコンテナも確認 |
| NPU入力 | `images`、INT16 `[1,3,448,448]` |
| NPU生出力 | FLOAT boxes `[1,4,4116]`、FLOAT scores `[1,80,4116]` |
| NPU最終検出 | CPU stable Top-kで最大300件、xyxy・score・class ID |

前処理は縦横比を保持した中央letterbox、余白114、BGR→RGB、NCHW、0～1正規化。元画像サイズ、丸め後のresize幅・高さ、余白を保持し、描画時の座標復元に使う。既定の表示閾値は0.25、80クラスを保持する。GPU/NPUとも元のNMS不要ヘッドの意味を維持する。

## 4. アプリの構成案

PythonのローカルWebサーバーとHTML/CSS/JavaScriptで構成する。サーバーは既定で `127.0.0.1:8000`、ブラウザは1画面を想定する。FastAPI/uvicornなどWeb用依存は `yolodemo/.venv` に用意する。

GPUとNPUはPython・ライブラリ環境が異なるため、**別プロセスのworker**として起動する。NPU用setupスクリプトはNPU子プロセス内だけでsourceする。GPU/NPUの既存環境やSDKにWeb用パッケージを混ぜない。追加依存が必要ならデモ専用環境に分離する。

```text
カメラ／動画読込 → 最新フレームの保持 → 選択中workerの連続推論
                                      ↓
                         元フレームと検出結果の組
                                      ↓
                        描画・JPEG化 → Chrome
workerの完了カウンター → 約0.3秒ごとの統計 → 左上のFPS
```

- 入力取得、推論、描画/配信を独立させる。ブラウザの要求やJPEG化を待って推論する構造を避ける。
- GPU/NPUモデルは事前ロード・ウォームアップし、選択していないworkerは待機させる。同時に負荷をかけた値を単独GPU/NPUのFPSとして表示しない。
- フレーム受け渡しは共有メモリなどを用い、最新値だけを保持する。キューは有界とし、古い未処理フレームを捨てられるようにする。
- `frame_id` は動画切り替えやループ後も単調増加させる。コピーまたはバッファ所有権の管理で、推論中の画像上書きを防ぐ。
- 同じフレームを反復中は前処理済み入力を再利用し、新フレームが来れば次の安全な反復境界で差し替える。
- 描画へ渡す検出出力は、次の推論が再利用するGPU/CPUバッファから必要な分をコピーする。対応する元フレーム・`frame_id`・演算器とまとめて扱う。
- 配信の初期案は検出枠付きMJPEG（最大30 FPS）と、約0.3秒周期の状態JSON取得。配信頻度は推論回数と独立させ、遅いクライアントには古い配信フレームを破棄する。
- 終了時にカメラ、子プロセス、共有メモリ、HTTP接続を解放する。

## 5. 入力の動作

1. 起動時にカメラを開き、有効フレームを取得できれば使用する。`--camera` で指定できるようにし、既定は利用可能なデバイスの先頭とする。読込が無期限に待たない設計にする。
2. カメラ不在・オープン失敗・継続的な読込失敗時はプレイリストへ移行し、画面に入力名と理由を表示する。再接続後のカメラ復帰は初版ではアプリ再起動でよい。
3. `--source playlist` でカメラの有無によらず動画入力を選べるようにする。
4. `playlist.m3u8` はUTF-8のローカルMP4一覧で、HLS配信ではない。空行・`#`行を再生対象から除外し、ファイルパスをプレイリストの親ディレクトリ基準で解決する。
5. 記載順に再生し、最後まで再生したら先頭へ戻る。動画は時刻/PTSまたは実際の動画FPSに合わせて通常速度で進め、推論速度に合わせて早送りしない。
6. 壊れた動画・存在しないファイルは理由を記録して次へ進む。全件再生不能なら入力エラー状態にし、空回りを止める。
7. 入力断・ファイル切り替え中はその状態を表示する。入力を失った際に古い静止画で推論し続けて正常なライブ入力に見せない。

現プレイリストの10ファイルは存在確認済み。動画は2560×1440のMP4。表示は元映像の縦横比を保ち、推論用448×448のletterbox画像をそのまま引き伸ばして表示しない。[SOURCES.md](videos/SOURCES.md) と `#CREDIT` を保持し、画面下部の「動画の出典」からクレジットを見られるようにする。

## 6. 画面と切り替えの動作

- ページ上部に映像、その下にGPU/NPUの2位置スライダーを置く。検出枠、クラス名、confidenceを映像に重ねる。
- 映像左上に実行中の `GPU` または `NPU` と「推論FPS」を大きく表示する。小さな補足として「最新フレームを反復推論」、入力FPS、入力ソース名を表示する。
- 初期選択はGPU。GPUが利用できずNPUが利用可能ならNPUで開始し、実際の選択状態と理由を表示する。両方利用不可ならエラー表示にする。
- workerごとに `loading / ready / active / error` を管理する。初回コンパイルやウォームアップ中は準備状況を表示し、未準備の演算器は選択できないようにする。
- 切り替え要求を受けたら旧workerの実行中の1回を完了させて待機させ、新workerへ最新フレームを渡す。モデル再ロードやWebページ再読み込みを通常の切り替え処理にしない。
- 切り替えに通番 `backend_epoch` を付け、旧epochの検出結果・統計を捨てる。準備後の切り替えは1秒以内を実装上の目安とし、実測を残す。
- 「切り替え中」を表示し、切り替え後の完了を確認してから表示演算器とFPSを更新する。FPS集計窓をリセットし、最初の約0.3秒は `—` とする。
- 連続操作では最後の選択を有効にし、workerを重複起動しない。失敗時は理由を表示し、利用可能な元の演算器へ戻す。

## 7. FPSの計算と表示

```text
推論FPS = 集計窓内に実際に完了した推論回数 / 同じ窓の実経過秒
```

- 時刻は単調増加時計を使う。直近約0.3秒の差分から計算し、UIも約0.3秒ごとに更新する。単発の処理時間の逆数や過去の保存値をFPSとして表示しない。
- GPUはグラフ再生の**デバイス完了を同期/イベントで確認してから**1回と数える。単に投入した回数は数えない。
- NPUは同期 `session.run` とstable Top-kが完了してから1回と数える。参照実装に合わせ、外部INT16量子化も各反復に含める。
- 連続推論中の入力差し替え・同期・表示向け出力コピーなどの時間も実経過時間に含める。推論ループ内の費用を後から差し引いて表示を上げない。
- 読込・コンパイル・ウォームアップ・演算器切り替え中は定常FPSに含めず、状態として表示する。定常中の処理停止を検知したら、過去の高い値を表示し続けない。
- 入力FPSは新しいフレームの取得数で計算し、反復分を加算しない。表示FPSを示す場合はブラウザでの描画数を計測し、サーバー送信数とは区別する。
- GPUの完了位置はGPU上の最終検出、NPUはCPU Top-k完了であり、デバイスごとの測定範囲をREADMEと結果JSONに記録する。厳密に同じ処理範囲でのデバイス性能比較とは説明しない。

300/60 FPSは反復推論の目標であり、カメラ取得やChromeの画面更新に同じFPSを要求するものではない。達成判定にはデモ稼働中の30秒×3回の `sum(完了回数) / sum(実経過秒)` を使い、瞬間最高値で判定しない。未達なら測定値と律速・未解決事項を報告する。

## 8. 既存性能記録の扱い

| 構成 | 保存済み実測 | 測定範囲・出典 |
|---|---:|---|
| GPU、YOLO26n、640、FP16、HIPグラフ | 320.4086 FPS | GPU常駐合成入力→forward完了、30秒×3回。[summary.json](../yolofast/results/graph-20261007-171158/summary.json) |
| NPU、YOLO26n、640、A16W8 per-channel、INT16入力 | 68.3919 FPS | 正規化FP32入力→外部量子化→ORT→CPU stable Top-k、300秒。[result.json](../yolofast/distribute/results/npu-accuracy95-20261011/perchannel/sustained/result.json) |

これらは**640構成の参考値**。448用モデルの性能、精度、ブラウザ表示を含むデモ動作は未検証。NPUの640モデルではCOCO val2017全5,000枚でFP32基準比98.5196%のmAP50–95が記録されているが、448の精度として引用しない。今回、448のmAP数値目標や全COCO再評価は指定されていない。第10節の正しさ検証を行い、定量精度評価が必要になれば評価範囲を確認する。

参照資産のSHA-256（今回のファイル確認で一致）:

- PT: `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`
- 640配布NPUモデル: `663dbb1792126a3de2e7ea1a27a4db11839eb88a89a6f72507390d38d7d20c9b`、パスは `~/yolofast/distribute/yolo26n-640-split-v3-a16w8-perchannel-int16.onnx`

448生成物には別のハッシュを記録する。既存プロジェクトのモデル・結果・仮想環境を上書きせず、デモ用のコード・モデル・キャッシュ・ログを `yolodemo` 側に保存する。

## 9. 実装成果物と進め方

以下は**これから作成する構成案**で、現時点でファイルやCLIが存在するという意味ではない。責務を保てるならファイル分割は調整してよい。

```text
yolodemo/
  HANDOFF.md
  README.md                  # セットアップ、モデル生成、起動、測定、制限事項
  requirements.txt           # Web/入力/描画用の依存
  app/
    server.py                # ローカルHTTP、状態取得、演算器切り替え
    capture.py               # カメラ・プレイリスト・再生時刻
    preprocess.py            # 448 letterbox、座標復元
    coordinator.py           # worker管理、IPC、切り替え、配信
    workers/gpu.py
    workers/npu.py
    static/                  # HTML/CSS/JS
  scripts/
    prepare_models.sh        # 448用ONNX生成・量子化・検証
    run_demo.sh              # サーバー/worker起動、Chrome起動
    benchmark_demo.py        # デモ稼働中の完了数・経過時間を保存
  models/npu/                # 448用モデルとメタデータ
  tests/                     # 下記のCPUで確認できる仕様
  videos/                    # 既存動画とクレジット
  results/                   # 実機検証・性能の記録
```

1. 既存環境・重み・校正画像・デバイスを確認し、448用モデル生成と各workerの単体検証から始める。
2. 最新フレームを反復する推論ループと完了カウンターを実装する。
3. プレイリスト/カメラ、描画/配信、画面、動的切り替えを接続する。
4. 第10節を確認し、結果と再現コマンドをREADMEに記録する。

起動は `bash scripts/run_demo.sh` の1コマンドを目標にする。サーバー準備完了後にChromeでローカルURLを開く。起動オプションには `--source auto|playlist`、`--camera`、`--playlist`、`--backend gpu|npu`、`--port`、`--no-browser` を用意する。入力解像度を変更する機能を追加する場合は、その解像度のNPUモデルとの対応を検証する。

## 10. 完成条件・検証

### モデルと推論の正しさ

- GPU/NPUとも実際の入力shapeが `[1,3,448,448]`、batch=1であることを確認する。
- GPUは複数の異なる実画像・合成入力で通常forwardとグラフ再生を比較し、有限値・検出出力の一致・入力変更の反映を確認する。不一致は入力と出力を保存して調べる。
- NPUはFP32出力分離前後の生出力一致、INT16入力変換前後のCPU出力一致、同じ448量子化モデルのCPU/NPU比較を行う。既存のスモーク基準（表示閾値0.25、検出数・クラス一致、IoU≥0.9、スコア差≤0.05）を初期基準に使う。
- GPUの同値比較とNPUの量子化誤差検査を区別する。通すために許容誤差を広げない。入力を交互に変更して古い結果やバッファ破損がないことも確認する。
- GPU FP16とNPU A16W8の全数値一致は要求しない。両方で枠・クラス・座標が妥当であることを確認し、スモーク成功をmAP保証とは記載しない。

### デモの機能

- Chromeのページ上部に映像と検出枠、左上に実演算器と推論FPSが出る。
- カメラなしで10本のプレイリストが順に再生され、終端から先頭へ戻る。フレームIDと座標が混ざらない。
- カメラがある環境ではカメラ優先と切断時の動画移行を確認する。実カメラがない場合はその項目を未検証と記録し、模擬入力による確認範囲を明記する。
- GPU→NPU→GPUを複数回切り替えて、ページ再読み込み・worker再起動なしで映像と統計が更新される。連続操作やworkerエラーで古い結果が表示されない。
- 入力・描画が30 FPS程度でも推論回数が制限されない。ブラウザの受信が遅くてもキューとメモリが増え続けない。
- FPSは約0.3秒で更新され、準備・切り替え・入力断・推論停止が正しく表示される。
- 終了後に子プロセス、カメラ、共有メモリが残らず、再起動できる。

### 性能と記録

- デモを稼働させ、選択中の演算器だけに負荷をかけて各30秒×3回を測定する。GPU>300、NPU>60の達成状況をそれぞれ報告する。
- 完了数、経過秒、推論FPS、入力FPS、測定範囲、実入力サイズ、モデルハッシュ、環境、エラー、切り替え時間を `results/` に保存する。
- 各演算器で5分程度の連続動作を確認し、停止・遅延の蓄積・メモリの増え続けを調べる。性能目標が未達なら機能動作とは分けて報告する。
- CPUでの自動テストは、プレイリストの相対パス/ループ、448前処理と座標復元、完了数に基づくFPS、切り替え時の古いepoch破棄など、実際の不具合を防ぐ箇所に絞る。
- GPU/NPUにアクセスできない場合も実装とCPUで可能な確認を進め、実機での再現コマンドと未検証事項を残す。実行していない検証を完了扱いにしない。

## 量子化環境のローカル化（2026-10-12）

量子化用Pythonは本プロジェクトの `.venv-quant` へ移行済み。`./create_models.sh --setup-quant` で `.python/` 内のPython 3.12.13と `requirements-quant.txt` の固定版から再作成する。校正・検証画像の `~/yolotest` 参照は維持している。

## ライセンス変更（2026-10-12）

アプリ全体の配布ライセンスをAGPL-3.0-onlyへ変更した。LLaVA-NPU由来のApache-2.0表記はNOTICEとlicenses/Apache-2.0.txtに保持する。現在の条件はREADMEJ.mdのライセンス節を参照。
