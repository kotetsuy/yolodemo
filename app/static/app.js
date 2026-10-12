'use strict';
const el = id => document.getElementById(id);
const slider = el('backend-slider');
const cameraSelect = el('camera-select');
let posting = false, cameraPosting = false, streamKey = '', inventoryKey = '', cameraError = '';
const states = {loading:'準備中',ready:'待機中',active:'実行中',error:'利用不可'};
slider.addEventListener('input', async () => {
  const backend = slider.value === '0' ? 'gpu' : 'npu';
  posting = true;
  el('switch-status').textContent = `${backend.toUpperCase()}へ切り替え中`;
  try {
    const response = await fetch('/api/backend', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({backend})});
    const status = await response.json();
    if (!response.ok) throw new Error(status.error);
  } catch (error) { el('error').textContent = error.message; }
  finally { posting = false; }
});
cameraSelect.addEventListener('change', async () => {
  cameraPosting = true;
  cameraError = '';
  el('camera-status').textContent = '入力を切り替えています';
  try {
    const response = await fetch('/api/camera', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({camera:cameraSelect.value})});
    const result = await response.json();
    if (!response.ok) throw new Error(result.error);
  } catch (error) { cameraError = error.message; }
  finally { cameraPosting = false; }
});
function updateCameras(input) {
  const devices = input.cameras || [];
  const selection = input.camera_selection || 'auto';
  const key = JSON.stringify([devices,selection]);
  if (key !== inventoryKey && !cameraPosting) {
    inventoryKey = key;
    cameraSelect.replaceChildren(new Option('自動（優先カメラ）','auto'),new Option('動画プレイリスト','playlist'));
    for (const device of devices) cameraSelect.add(new Option(`${device.label} (${device.path})`,device.id));
    if (!['auto','playlist',...devices.map(d=>d.id)].includes(selection)) cameraSelect.add(new Option('指定カメラ（再接続待ち）',selection));
  }
  if (!cameraPosting) cameraSelect.value = selection;
  const camera = input.camera;
  const dimensions = input.width && input.height ? `${input.width} × ${input.height}` : '';
  const format = camera ? `${camera.fourcc} · 設定 ${camera.fps.toFixed(1)} FPS` : '動画';
  el('camera-status').textContent = cameraError || (input.pending_camera ? '別のカメラを準備中 · 現在の映像を継続' : camera ? `${dimensions} · ${format}` : `${dimensions} · ${devices.length}台のカメラを検出`);
}
async function refresh() {
  try {
    const response = await fetch('/api/status', {cache:'no-store'});
    if (!response.ok) throw new Error('サーバーから状態を取得できません');
    const s = await response.json();
    cameraSelect.disabled = false;
    updateCameras(s.input);
    for (const name of ['gpu','npu']) {
      const w = s.workers[name];
      el(`${name}-state`).textContent = states[w.state] || w.state;
      el(`${name}-dot`).className = ['ready','active'].includes(w.state) ? 'ready' : w.state === 'error' ? 'error' : '';
    }
    slider.disabled = !Object.values(s.workers).every(w => ['ready','active'].includes(w.state));
    if (!posting) slider.value = s.desired === 'gpu' ? '0' : '1';
    el('backend').textContent = s.switching ? '切り替え中' : s.backend ? s.backend.toUpperCase() : '準備中';
    el('fps').textContent = s.fps === null ? '—' : s.fps.toFixed(1);
    el('source').textContent = (s.input.source || '入力を準備中') + (s.input.width ? ` · ${s.input.width}×${s.input.height}` : '');
    el('input-fps').textContent = `入力 ${s.input.fps.toFixed(1)} FPS`;
    const usable = s.result_frame_id !== null && s.result_age_ms < 1000 && !s.switching && s.input.state === 'running';
    el('placeholder').style.display = usable ? 'none' : 'flex';
    el('message').textContent = s.switching ? '演算器を切り替えています' : s.input.state === 'error' ? '映像入力を確認してください' : s.backend ? '推論結果を待っています' : 'モデルを準備しています';
    const errors = Object.entries(s.workers).filter(([,w])=>w.error).map(([k,w])=>`${k.toUpperCase()}: ${w.error}`);
    el('detail').textContent = errors.join(' / ') || s.input.reason || '初回のNPUコンパイルには時間がかかる場合があります';
    el('error').textContent = [...errors,cameraError,s.input.state === 'error' ? s.input.reason : ''].filter(Boolean).join(' / ');
    el('switch-status').textContent = errors.length ? '演算器にエラーがあります（下の詳細を確認）' : slider.disabled ? 'GPU / NPUを準備中' : s.switching ? `${s.desired.toUpperCase()}へ切り替え中` : s.switch_ms !== null ? `切り替え ${s.switch_ms.toFixed(0)} ms` : 'GPU / NPUを準備中';
    const key = `${s.server_pid}:${s.epoch}:${s.input.generation || 0}`;
    if (usable && streamKey !== key) {
      streamKey = key;
      el('video').src = `/stream.mjpg?epoch=${encodeURIComponent(key)}`;
    }
  } catch (error) {
    el('fps').textContent = '—';
    el('placeholder').style.display = 'flex';
    el('message').textContent = 'サーバーへの接続を確認してください';
    el('detail').textContent = error.message;
    slider.disabled = true;
    cameraSelect.disabled = true;
  } finally { setTimeout(refresh,300); }
}
refresh();
