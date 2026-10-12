# YOLO26n GPU/NPU デモ

[English](README.md) | [日本語](READMEJ.md)

Radeon GPUとRyzen AI NPUを切り替えて物体検出を行うローカルデモです。Chromeで検出映像を表示し、2位置スライダーで演算器を切り替え、複数のUSBカメラから1台を選択できます。両方の推論入力は **448×448、batch=1** で、表示映像の縦横比を保ちます。

カメラには **1920×1080・30FPS** を要求し、推論FPSと入力FPSを分けて表示します。モデルはローカルで生成し、リポジトリには含めません。

## 関連プロジェクト

| プロジェクト | このデモとの関係 |
|---|---|
| [LLaVA-NPU](https://github.com/kotetsuy/LLaVA-NPU) | USBカメラの検出、形式設定、カメラ選択の参考実装。 |
| [yolofast_pub](https://github.com/kotetsuy/yolofast_pub) | このデモの元となるyolofastのONNX量子化・Ryzen AI NPU評価ツールの公開版。 |

[yolofast_pubの最新README](https://github.com/kotetsuy/yolofast_pub/blob/main/README.md) には、公開版単体のONNX環境に加え、**yolodemo用の追加環境**の手順が記載されています。このデモでは追加環境の節に従ってGPU・ONNX出力環境を用意し、このリポジトリで `./create_models.sh --setup-quant` を実行します。公開版単体の `~/yolofast/.venv` は別用途で、このデモは参照しません。

記載の配置は現在のスクリプトと一致します。GPU・ONNX出力は `~/yolofast` 内、量子化はこのリポジトリの `.venv-quant`、NPU設定は `~/ryzenai_1_8/setup_ryzenai_env.sh` です。`YOLOFAST_DIR` が変更するのはyolofastのルートだけです。重み、元の60画像の校正レシピ、`~/yolotest` 内の画像、NPU SDKは別途用意する必要があります。公開版の画像マニフェストはデモのレシピの代用にはなりません。新たに記載された独立CPUエクスポート環境の生成物は、モデル生成・実機検証に合格してから採用してください。過去の測定記録はこの新環境で測定したものではありません。

LLaVA-NPUは参考プロジェクトで、起動しておく必要はありません。

## クローン

```bash
git clone https://github.com/kotetsuy/yolodemo.git ~/yolodemo
cd ~/yolodemo
```

モデル生成・起動の前に、下記の外部の前提環境を用意してください。配置先が既に存在する場合は内容を確認し、別のチェックアウト先を選んでください。

## 現在のデモの前提環境

下表は公開版READMEの「yolodemo用の追加venv」と一致する構成です。公開版単体のONNX環境とは用途が異なります。検証したハードウェアはAMD Ryzen AI Max+ 395、Radeon 8060S（gfx1151）、XDNA2 NPUです。現在のスクリプトは、次の環境・ファイルが既に用意されていることを前提にしています。GPU/NPUランタイムをWebのvenvへ混在させず、各workerを別プロセスで起動します。

| 用途 | 環境 |
|---|---|
| GPU | `~/yolofast/.venv-rocm10-py314`、PyTorch ROCm、Radeon 8060S |
| ONNX export | `~/yolofast/.venv-npu-export` |
| 量子化 | `./.venv-quant`、AMD Quark |
| NPU | `~/ryzenai_1_8/setup_ryzenai_env.sh`、VitisAI EP、`/dev/accel/accel0` |
| 重み | `~/yolofast/models/yolo26n.pt` |
| 校正レシピ | `~/yolofast/distribute/models/npu/accuracy95/perchannel/yolo26n-640-a16w8.json` と記載された60画像 |
| Web | `uv`、Python 3.12、`requirements.txt`、Chrome |
| ネイティブビルド | C++コンパイラ一式（Ubuntuの `build-essential` など）。Ninjaは `.venv-quant` に導入 |

`YOLOFAST_DIR`でyolofastの位置を変更できます。他のマシンでは互換性のあるランタイム、重み、校正画像と所定のディレクトリ構成を先に用意してください。モデル生成スクリプトは前提を確認しますが、GPU/NPU環境を構築するものではありません。VitisAI SDK環境を別環境のCPU用 `onnxruntime` で置き換えないでください。既存の640用モデルを上書きせず、448用モデルを `models/npu/` に作成します。

## モデルをローカルで作成する

モデルの準備は1本のスクリプトから実行します。

```bash
./stop_all.sh
./create_models.sh --setup-quant
./create_models.sh --check
./create_models.sh
./start_all.sh
```

`--check` はPython環境、重み、校正・検証画像、NPUデバイスとランタイムを確認し、パッケージのインストールやモデル生成は行いません。引数なしでは必要に応じてローカル量子化環境とWeb環境を作成・依存をインストールし、YOLO26nを448×448でONNX出力、枠・スコア分離、A16W8・per-channel・MinMax・CLE量子化、INT16入力化、NPU本体とCPU分類ヘッドの生成まで実行します。NPUの厳密な検証に合格した場合だけ採用し、その後GPUも検証します。内部で `scripts/` のPython補助コードを呼ぶため、段階ごとの手動実行は不要です。

前提は上表の環境、`~/yolofast/models/yolo26n.pt`、元の量子化レシピとハッシュが一致する60枚の校正画像、`~/yolotest/calib_images/bus.jpg` と `~/yolotest/calib2/` 内の2枚以上のJPGです。**重み・データセットのダウンロードやGPU/NPUランタイムのインストールは行いません。** 各ライセンスに従って事前に用意してください。yolofastは `YOLOFAST_DIR` で変更でき、yolotestとRyzen AI設定は上表の配置を使います。レシピ内の画像パスも実行環境で参照可能である必要があります。

`models/` 以下のONNX、CPUヘッド、来歴JSON、`selection.json` はすべてローカル生成物としてGit対象外です。検証記録は `results/model-validation/` に保存します。既存の基本モデルは来歴ハッシュが一致する場合に再利用し、採用候補は実機で再検証します。変更・不完全なファイルでは失敗するため、該当生成物を退避してから再生成してください。NPU検証失敗は無視せず、自動再試行もしません。スコア差の基準0.05を維持します。

`./create_models.sh --help` で使い方を表示します。従来の `scripts/prepare_models.sh` はこのスクリプトへ転送します。起動時も `selection.json` がなければ呼び出します。新しいチェックアウトでは上記の前提を用意し、起動前に `./create_models.sh` を実行してください。

### ローカル量子化環境

`./create_models.sh --setup-quant` は、このリポジトリ内の `.python/` にPython 3.12.13を導入し、`.venv-quant` を新規作成します。`requirements-quant.txt` に記録した版（AMD Quark 0.11.2、CPU版PyTorch 2.6.0など）をインストールします。初回のダウンロードにはネットワーク接続が必要です。通常のモデル生成コマンドでも、この準備を自動実行します。`.python/` はvenvと一緒に保持し、他プロジェクトのvenvはコピーしないでください。

量子化では `~/yolotest/.venv-quant` を使用しなくなりました。校正・検証画像は引き続き `~/yolotest` を参照し、エクスポート・GPU・NPU環境も記載の場所を使います。`.venv-quant/` と `.python/` はGit対象外で、再作成用の `requirements-quant.txt` はGit対象です。旧環境は元のプロジェクト用に保持しています。Quarkは初回import時にCPU用のカスタム演算をコンパイルする場合があるため、C++コンパイラ一式も必要です。

## 起動

```bash
cd ~/yolodemo
./start_all.sh
```

前述の前提環境とモデルを準備した後に実行すると、Chromeで `http://127.0.0.1:8000` が開き、サーバーはバックグラウンドで動作します。初回はWeb用venvと`create_models.sh` で不足している448用モデルを作成し、起動時は両workerのモデル検証・ウォームアップを実行します。NPUの初回コンパイルには時間がかかります。

起動時はNPUの検証を先に完了し、その後GPUを初期化します。両方の準備後にカメラ・動画の取得と推論を開始します。このAPUでGPU/NPUの並行初期化中に発生したNPU中間出力の不一致を避けるためです。準備中は切り替えスライダーが無効になり、両方の準備が完了すると操作できます。

```bash
./stop_all.sh
```

停止すると、このディレクトリのサーバー・推論worker・共有メモリを解放します。起動済みの場合は二重起動せず、停止済みの場合は何もせず終了します。既存の `scripts/run_demo.sh` から起動したデモも停止できます。Chromeは開いたままです。ログは `.cache/run/demo.log` に追記します。

前景で実行してCtrl+Cで終了したい場合は `bash scripts/run_demo.sh` を使えます。

```bash
# カメラを使わず、動画からNPUで開始
./start_all.sh --source playlist --backend npu
# ブラウザを自動で開かず起動
./start_all.sh --no-browser
# 指定カメラを優先する。開けない・切断時は動画に移行
./start_all.sh --camera /dev/video0
```

既定はカメラ優先です。利用可能なカメラがなければ、設定したローカルの動画プレイリストに移行します。動画はGitに含まれないため、この移行を使う場合は事前にプレイリストを用意してください。

## カメラ入力

画面の「映像入力」で、**自動・接続中の各カメラ・動画プレイリスト**を選べます。複数接続時も映像は1台分を表示し、手動選択したカメラを優先します。「自動」に戻すと設定の優先順位を使います。

[LLaVA-NPU](https://github.com/kotetsuy/LLaVA-NPU) のカメラ処理を参考に、[camera.yaml](camera.yaml) に安定したUSB by-id名・VID:PIDによる優先順位を設定しました。既定は **1920×1080・30FPSを要求し、MJPG → YUYVの順に試行**します。未対応の場合はカメラが受け付けた解像度・形式を使用し、実際の寸法・設定FPSを画面に表示します。設定30FPSと実測の入力FPSは別の値です。推論サイズは448×448です。

カメラ一覧を1秒ごとに更新し、映像取得に対応するデバイスだけを表示します。自動選択では優先カメラの接続が1秒安定した後に切り替えます。別カメラの最初の映像が届くまでは現在の映像を継続します。選択カメラが切断された場合は利用可能な別カメラ、なければ動画へ移行し、再接続を検出します。

`camera.yaml` の `preferred`、`fallback: any/none`、`preempt`、`preempt_settle_sec`、`format` を編集して変更できます。`fallback: none` は自動選択の対象を優先リストのカメラに限定します。手動選択はリスト外でも可能です。

```bash
# 解像度・設定FPSを起動引数で上書き（起動中なら先に停止）
./start_all.sh --camera-width 2560 --camera-height 1440 --camera-fps 30
# 別のカメラ設定を使用
./start_all.sh --camera-config /path/to/camera.yaml
# 配信負荷を抑えたい場合
./start_all.sh --display-width 1280
```

## ローカル動画入力

開発環境では2560×1440の短い動画10本をプレイリストで再生しました。`videos/` 全体がGit対象外のため、新しいチェックアウトには動画、`playlist.m3u8`、`SOURCES.md` は含まれません。

独自の動画を使う場合は、1行に1ファイルのパスを記載したM3Uプレイリストを作成してください。相対パスはプレイリストのディレクトリを基準に解釈し、`#` で始まる行はコメントとして扱います。

```bash
mkdir -p videos
# 起動前に videos/example.mp4 を配置する
printf '%s\n' '#EXTM3U' 'example.mp4' > videos/playlist.m3u8
./start_all.sh --source playlist --playlist videos/playlist.m3u8
```

既定は `videos/playlist.m3u8` です。別のプレイリストは `--playlist /path/to/playlist.m3u8` で指定できます。配信幅は既定1920で、`--display-width` で変更できます。動画の出典・ライセンス情報はローカルの `videos/SOURCES.md` に保存してください。画面の「動画の出典」はこのファイルを参照します。開発時の動画を引き続き使う場合も、元の出典・ライセンス表記を保持してください。

## FPSの意味

大きく表示する値は **実際に完了した推論回数 ÷ 実測経過秒** です。最新フレームを繰り返し推論するので、入力が30FPSでも推論FPSは300を超えられます。新しい映像フレーム数を表す値ではありません。約0.3秒ごとに更新します。

GPUは同期を含み、NPUはINT16量子化・NPU実行・CPU分類ヘッド・Top-kを含みます。動画デコード、letterbox、配信は別スレッドで並行して動作し、ライブ測定ではこれらの負荷もかかります。入力FPSを別表示し、入力断・準備・切り替え中は推論FPSを空欄にします。

切り替えでは旧workerの処理停止を確認し、epochを更新して古い結果を破棄します。workerを再起動せず切り替えます。フレームと検出をIDで対応させ、保持フレーム数と配信画像を制限しています。

## 推論方式と数値検証

GPUはFP16のHIPグラフを使い、GPU同期完了後にカウンターを増やします。同スコアのTop-kをインデックス順で安定化し、スコア値は変更しません。起動時に8入力で通常forward、グラフ再生、CPUの安定Top-kとの一致を確認します。

NPUはyolofastの **A16W8・per-channel・MinMax・CLE** と60枚の校正画像を引き継ぎます。448で再校正したQDQパラメーターを保持し、バックボーンと枠の推論をNPU、分類ヘッドと直前のSiLUをCPUで実行します。全NPU構成ではスコア差が基準を超えたため、この構成で改善しました。CPUヘッドとINT16入力量子化、Top-kを含む実際のループ速度を表示します。

比較先は元の448量子化モデルのCPU実行です。**閾値0.25、検出数・クラス一致、IoU≥0.9、スコア差≤0.05を維持**しています。採用モデルは実画像3枚・合成5入力で合格し、最大スコア差は0.046207です。交互入力の再現性も起動時に確認します。この検証はmAP測定ではありません。

モデル・CPUヘッド・参照モデルのSHA-256を確認します。採用モデルは `models/npu/selection.json` に記録し、NPUコンパイルキャッシュはモデルハッシュ別に `.cache/npu-448/` へ保存します。

## 検証・性能測定

```bash
uv pip install --python .venv/bin/python pytest
.venv/bin/python -m pytest -q
# デモを別ターミナルで起動してから実行
.venv/bin/python scripts/benchmark_demo.py --output results/benchmark-formal.json
.venv/bin/python scripts/benchmark_demo.py --seconds 300 --repeats 1 --output results/benchmark-stress.json
# Chrome自動検証（検証用依存）
uv pip install --python .venv/bin/python playwright
.venv/bin/python scripts/check_browser.py
```

性能測定は選択した演算器の実カウンターを使い、MJPEGも受信します。正式測定と連続動作の結果は `results/VALIDATION.md` にまとめます。各起動の環境、数値検証、切り替え履歴、終了状態は `results/demo-日時-ID/` に保存します。

高解像度カメラと入力選択の検証は [results/cameras/README.md](results/cameras/README.md) を参照してください。実USBカメラ1台で1920×1080の取得・Chrome表示・入力切り替えを確認しました。複数カメラの優先順位・切断・再接続は模擬デバイスで検証し、実カメラ2台と物理切断は未検証です。

## ソース構成

- `app/cameras.py`, `camera.yaml`: USBカメラ一覧、優先順位、V4L2の形式設定。
- `app/capture.py`, `app/preprocess.py`: 入力取得、実時間再生、448前処理と元映像への座標復元。
- `app/ipc.py`, `app/coordinator.py`: 異なるPython環境のIPC、epoch、worker停止確認、描画と配信。
- `app/workers/`: GPU/NPUの検証、同期推論、実完了カウンター。
- `app/server.py`, `app/static/`: localhostのHTTPサーバー、Chrome画面。
- `create_models.sh`, `scripts/prepare_models.py`, `scripts/refine_npu.py`: 同じ量子化方式での448モデル生成とCPUヘッド分離。
- `scripts/benchmark_demo.py`, `scripts/check_browser.py`: 実測と実Chrome検証。

## HTTP API

| エンドポイント | 用途・リクエスト本文 |
|---|---|
| `GET /api/status` | 演算器、worker、入力、FPSの状態。 |
| `GET /api/cameras` | カメラ一覧と現在の入力状態。 |
| `POST /api/backend` | `{"backend":"gpu"}` または `{"backend":"npu"}`。 |
| `POST /api/camera` | `{"camera":"auto"}`、`{"camera":"playlist"}`、またはカメラ一覧のIDを使った `{"camera":"<camera-id>"}`。 |
| `GET /frame.jpg` | 最新の検出画像。 |
| `GET /stream.mjpg` | 連続MJPEG映像。 |
| `GET /source.zip` | アプリケーションのソースZIP。 |
| `GET /license` | AGPL-3.0のライセンス本文。 |

## Git対象外のファイル

生成モデルとそのメタデータ、実行キャッシュ、生の検証結果、ログ、スクリーンショットはローカルに保持し、`.gitignore` で除外しています。これらにはローカルパスやカメラ映像が含まれる場合があります。`results/` 配下の文書はGit対象ですが、生データへのリンクはローカル実行時のファイルを参照します。モデル生成物は `create_models.sh` で再生成できます。

## ライセンス

本アプリケーションの独自コード、スクリプト、設定、文書を含む全体を、**GNU Affero General Public License version 3のみ（AGPL-3.0-only）** で配布します。[LICENSE](LICENSE) と [NOTICE](NOTICE) を参照してください。無保証で提供します。

LLaVA-NPUから取り入れた部分のApache-2.0表記は、[NOTICE](NOTICE) と [licenses/Apache-2.0.txt](licenses/Apache-2.0.txt) に保持しています。第三者ライブラリ、重み、データセット、SDKにはそれぞれの条件が適用されます。Ultralytics YOLO26のコード・派生モデルには、元の [AGPL-3.0／Enterprise条件](https://www.ultralytics.com/license) が適用され、モデルはGitに含めません。

画面にはライセンスと、実行中のアプリケーションのソースZIPへのリンクを設けています。改変・運用する場合もソース入手方法を維持し、改変版に必要な追加の対応ソースを提供してください。ZIPにはアプリのコード、スクリプト、テスト、設定、文書、ライセンス表記を含め、ローカルモデル、画像、実行ログ、認証情報、実行環境は含めません。モデルやSDKを同梱する配布物ではありません。
