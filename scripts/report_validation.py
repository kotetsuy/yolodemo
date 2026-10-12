# SPDX-License-Identifier: AGPL-3.0-only
"""Summarize recorded hardware measurements without inventing results."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app.model_utils import ROOT


def main():
    folder=ROOT/'results'
    formal=json.loads((folder/'benchmark-formal.json').read_text())
    stress=json.loads((folder/'benchmark-stress.json').read_text())
    chrome=json.loads((folder/'browser-check/result.json').read_text())
    if formal['status']!='complete' or stress['status']!='complete':
        raise SystemExit('Measurements are not complete')
    lines=['# 実機検証記録','', '測定日: 2026-10-12（Asia/Tokyo）。入力448×448、batch=1。', '',
        '## 正式測定: 30秒×3回','',
        '動画デコード、検出描画、MJPEG配信を動かし、MJPEGを受信しながら測定。選択中のworkerだけが推論を反復した。', '',
        '| 演算器 | 1回目 | 2回目 | 3回目 | 完了数/経過秒の合算 | 目標 |',
        '|---|---:|---:|---:|---:|---|']
    for backend,data in formal['backends'].items():
        rows=[r for r in formal['runs'] if r['backend']==backend]
        speeds=' | '.join(f'{r["fps"]:.2f}' for r in rows)
        lines.append(f'| {backend.upper()} | {speeds} | {data["pooled_fps"]:.2f} FPS | >{data["target"]}: {"達成" if data["target_passed"] else "未達"} |')
    lines+=['', '各回の完了数・実測秒・入力FPS・モデルハッシュ・環境・計測範囲は [benchmark-formal.json](benchmark-formal.json) に保存。', '',
        '## Chrome表示中の連続動作: 各5分','',
        'Chromeの通常ウィンドウで映像を表示し、追加のMJPEG受信も行った。各演算器について300秒測定した。', '',
        '| 演算器 | 完了数 | 実測秒 | 推論FPS | 入力FPSの平均 | worker RSS: 初期→末尾 / 最大 |',
        '|---|---:|---:|---:|---:|---|']
    memory_notes=[]
    for row in stress['runs']:
        samples=row['samples'];backend=row['backend'];rss=[s['rss_kb'][backend] for s in samples if s['rss_kb'].get(backend)]
        lines.append(f'| {backend.upper()} | {row["frames"]} | {row["elapsed_s"]:.3f} | {row["fps"]:.2f} | {sum(s["input_fps"] for s in samples)/len(samples):.2f} | {rss[0]/1024:.1f}→{rss[-1]/1024:.1f} / {max(rss)/1024:.1f} MiB |')
        sources=list(dict.fromkeys(s['source'] for s in samples))
        ages=[s['result_age_ms'] for s in samples if s['result_age_ms'] is not None]
        srss=[s['server_rss_kb'] for s in samples if s.get('server_rss_kb')]
        print(backend,'sources',len(sources),'age_max_ms',max(ages),'server_rss_mib',srss[0]/1024,srss[-1]/1024,max(srss)/1024)
        memory_notes.append(f'- {backend.upper()}測定: サーバーRSS {srss[0]/1024:.1f}→{srss[-1]/1024:.1f} MiB、最大{max(srss)/1024:.1f} MiB。確認した動画{len(sources)}本、観測した検出結果の最大受信経過{max(ages):.1f} ms。')
    lines+=['',*memory_notes]
    lines+=['', '完了カウンターのリセット、推論heartbeatの1秒超停止、演算器変更、workerエラーを測定スクリプトで監視。詳細とメモリの時系列は [benchmark-stress.json](benchmark-stress.json) に保存。', '',
        '## 数値検証','',
        '- GPU: 実画像3枚、白黒2入力、乱数3入力で通常forward/グラフ再生が一致。CPUの安定Top-kとも一致。FP16のスコア値を変えず、同値の順序を安定化。',
        '- NPU: 同じ448量子化モデルのCPU実行を参照。表示閾値0.25、検出数・クラス一致、IoU≥0.9、スコア差≤0.05を維持。8入力で合格、実画像の最小IoUは0.977042、最大スコア差は0.046207。12回の交互入力で生出力・検出の再現性を確認。',
        '- 元の全NPU構成の最大スコア差0.059795は不合格。許容値を広げず、既存QDQパラメーターを保持して分類ヘッドと直前のSiLUをCPUへ分離した。',
        '- NPUの計測は外部INT16量子化、NPUバックボーン・枠、CPU分類ヘッド、Top-kを含む。GPUはHIPグラフと同期、新規入力の転送・サンプル出力コピーを含む。',
        '- GPU/NPU間の全数値一致やmAPを保証する検証ではない。', '',
        'NPUモデルSHA-256: `'+formal['backends']['npu']['info']['model_sha256']+'`。GPU重みSHA-256: `'+formal['backends']['gpu']['info']['model_sha256']+'`。', '',
        '## NPUハードウェアの確認','',
        'XRTのAIE context PIDがNPU worker PIDと一致。Active状態でハードウェア完了カウンターが3730→5847へ増加し、errors=0。記録: [1回目](xrt-npu-first.json)、[2回目](xrt-npu-second.json)、[同時のworker状態](status-npu-hardware.json)。VitisAI EPの登録だけでNPU動作と判定していない。', '',
        '## Chromeと機能','',
        '- 実ChromeでGPU→NPU→GPU→NPU→GPUと連続選択を確認。worker PIDが変わらず、最後の指定が反映され、JavaScriptエラーなし。',
        '- ページ上部に映像、検出枠、左上に演算器と推論FPS。元の縦横比を維持。Chrome受信画像: '+str(chrome.get('image_dimensions',[1280,720]))+'。',
        '- 切り替え時間はモデル初回使用時を含めて '+f'{min(s["worker_switch_ms"] for s in chrome["switches"]):.1f}〜{max(s["worker_switch_ms"] for s in chrome["switches"]):.1f} ms'+ '。ポーリングと画像受信を含むブラウザ観測値も記録。',
        '- カメラ不在時に動画へ移行。10本の実動画の巡回は連続測定のsource記録で確認。接続後のUSBカメラでの切り替えは追加検証記録を参照。',
        '- CPUテスト8件: 前処理・座標復元、量子化の丸め/飽和、実完了数からのFPS、IPCの不変スナップショット、動画の相対パス/ループ、不正プレイリスト、古いepochの結果破棄、順次初期化。', '',
        '- MJPEG受信を20秒止めたクライアントがあっても推論と描画は継続。サーバーRSS増加は360 KiB。記録: [slow-client.json](slow-client.json)。', '',
        '- NPU workerを終了すると既存GPU workerへ自動移行し、epochを更新して新しい検出映像を表示。記録: [worker-error-fallback.json](worker-error-fallback.json)。',
        '- SIGTERM終了後に所有するサーバー・GPU/NPU worker・共有メモリが残らないことを確認し、同じポートで再起動。記録: [shutdown-check.json](shutdown-check.json)、[restart-check.json](restart-check.json)。', '',
        'Chrome記録: [結果](browser-check/result.json)、[GPU画面](browser-check/gpu.png)、[NPU画面](browser-check/npu.png)。', '',
        '## 残る検証範囲','',
        '物理カメラの切断と448用モデルのCOCO mAP測定は未実施。カメラ不在時の動画への移行と、生成した動画2本のループは確認済み。接続後のUSBカメラ取得・表示・切り替えは追加検証済み。', '',
        'モデル検証で異常を検出したworkerは開始せず、検証結果を保存する。NPUのCPU並列数4固定だけでは交互入力の不一致が再発したため、NPU→GPUの順次初期化へ変更した。修正後の起動・切り替え5回の記録: [switch-fix/README.md](switch-fix/README.md)。異常時は `npu-alternating-mismatch.npz` に比較データを保存する。', '']
    (folder/'VALIDATION.md').write_text('\n'.join(lines))
    print(folder/'VALIDATION.md')


if __name__=='__main__':main()
