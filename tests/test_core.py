# SPDX-License-Identifier: AGPL-3.0-only
import json
import threading
import time
from pathlib import Path
import cv2
import numpy as np
import pytest
from app.capture import Capture, playlist_entries
from app.ipc import InputBus
from app.metrics import FpsMeter
from app.coordinator import Coordinator
from app.preprocess import preprocess, restore, quantize


def test_letterbox_rgb_and_inverse_with_rounded_resize():
    image = np.full((601,999,3), [10,20,30], np.uint8)
    x, g = preprocess(image)
    assert x.shape == (1,3,448,448) and x.dtype == np.float32
    np.testing.assert_allclose(x[0,:,224,224], np.array([30,20,10])/255)
    assert x[0,0,0,0] == np.float32(114)/np.float32(255)
    rows = [[g['left'], g['top'], g['left']+g['resized_width'], g['top']+g['resized_height'], .9, 0],
            [0,0,1,1,float('nan'),0], [30,30,20,20,.9,0], [0,0,448,448,.9,80]]
    result = restore(rows,g)
    assert len(result)==1
    np.testing.assert_allclose(result[0,:4],[0,0,999,601],atol=.001)


def test_quantization_half_even_and_saturation():
    x=np.array([-.5,.5,1.5,2.5,100000,-100000],np.float32)
    q=quantize(x,1,0,np.empty_like(x),np.empty(x.shape,np.int16))
    np.testing.assert_array_equal(q,[0,0,2,2,32767,-32768])


def test_fps_uses_completions_actual_elapsed_and_resets():
    m=FpsMeter()
    assert m.update(45,.15) is None
    assert m.update(120,.4)==300
    assert m.update(240,.8)==300
    assert m.update(15,.05) is None
    assert m.update(90,.3)==300


def test_shared_input_is_immutable_and_epoch_survives_frame_publish(tmp_path):
    owner=InputBus(tmp_path/'bus',create=True)
    reader=InputBus(tmp_path/'bus')
    try:
        x=np.ones((1,3,448,448),np.float32)
        owner.select('gpu',7)
        owner.publish(12,x)
        control, snapshot=reader.read()
        owner.publish(13,np.zeros_like(x))
        assert snapshot.min()==1
        assert control['epoch']==7 and control['frame_id']==12
        control,new=reader.read(12)
        assert new.max()==0 and control['frame_id']==13 and control['epoch']==7
        owner.select('npu',8)
        control,new=reader.read(13)
        assert new is None and control['selected']==2 and control['epoch']==8
        owner.invalidate()
        assert reader.read(-1)[0]['valid'] is False
    finally:
        reader.close();owner.close()


def test_playlist_relative_paths_loop_and_monotonic_frames(tmp_path):
    for name,color in [('a.avi',30),('b.avi',200)]:
        writer=cv2.VideoWriter(str(tmp_path/name),cv2.VideoWriter_fourcc(*'MJPG'),20,(32,24))
        assert writer.isOpened()
        for _ in range(2):writer.write(np.full((24,32,3),color,np.uint8))
        writer.release()
    p=tmp_path/'list.m3u8';p.write_text('#EXTM3U\n#CREDIT: example\na.avi\nb.avi\n')
    assert playlist_entries(p)==[tmp_path/'a.avi',tmp_path/'b.avi']
    seen=[];done=threading.Event()
    def frame(i,image):
        seen.append((i,int(image.mean())))
        if len(seen)>=10:done.set()
    capture=Capture(p,'playlist',None,frame,lambda _:None)
    capture.thread.start()
    try:assert done.wait(3)
    finally:capture.close()
    assert not capture.thread.is_alive()
    assert [r[0] for r in seen]==list(range(1,len(seen)+1))
    # MJPEG is lossy; compare which clip was played, not pixel identity.
    assert [r[1]>100 for r in seen[:8]]==[False,False,True,True,False,False,True,True]


def test_unreadable_playlist_stops_without_spin(tmp_path):
    p=tmp_path/'list.m3u8';p.write_text('missing.mp4\n')
    capture=Capture(p,'playlist',None,lambda *_:None,lambda _:None)
    capture.thread.start();capture.thread.join(2)
    assert not capture.thread.is_alive()
    assert capture.status()['state']=='error'
    capture.close()


def test_old_worker_results_do_not_replace_current_frame_or_counters():
    coordinator=Coordinator.__new__(Coordinator)
    coordinator.lock=threading.RLock()
    coordinator.epoch=9
    coordinator.active='npu'
    coordinator.switching=False
    coordinator.workers={name:dict(paused=True,completed=0,elapsed=0.) for name in ('gpu','npu')}
    coordinator.frames={42:None}
    coordinator.meter=FpsMeter()
    coordinator.latest=None
    def result(backend,epoch,count):
        coordinator.event(backend,dict(kind='result',backend=backend,epoch=epoch,
            frame_id=42,detections=[],completed=count,elapsed=.4))
    result('npu',9,30)
    current=coordinator.latest
    result('gpu',8,900)
    result('npu',8,900)
    assert coordinator.latest is current
    assert coordinator.workers['npu']['completed']==30
    assert coordinator.workers['gpu']['completed']==0
    assert coordinator.meter.fps==75


def test_initialization_waits_for_each_worker_before_starting_capture():
    coordinator=Coordinator.__new__(Coordinator)
    coordinator.lock=threading.RLock()
    coordinator.stop=threading.Event()
    coordinator.workers={name:dict(state='loading') for name in ('npu','gpu')}
    started=[]
    npu_started,gpu_started,capture_started=threading.Event(),threading.Event(),threading.Event()
    def start_worker(name):
        started.append(name)
        (npu_started if name=='npu' else gpu_started).set()
    coordinator.start_worker=start_worker
    class CaptureThread:
        def start(self):capture_started.set()
    class FakeCapture:
        thread=CaptureThread()
    coordinator.capture=FakeCapture()
    thread=threading.Thread(target=coordinator.initialize)
    thread.start()
    try:
        assert npu_started.wait(1)
        assert not gpu_started.is_set() and not capture_started.is_set()
        with coordinator.lock:coordinator.workers['npu']['state']='ready'
        assert gpu_started.wait(1)
        assert not capture_started.is_set()
        with coordinator.lock:coordinator.workers['gpu']['state']='ready'
        assert capture_started.wait(1)
        assert started==['npu','gpu']
    finally:
        coordinator.stop.set();thread.join(1)


def test_camera_change_during_jpeg_encode_discards_old_source(monkeypatch):
    coordinator=Coordinator.__new__(Coordinator)
    coordinator.lock=threading.RLock()
    coordinator.changed=threading.Condition(coordinator.lock)
    coordinator.stop=threading.Event()
    coordinator.epoch=1;coordinator.active='gpu';coordinator.switching=False
    image=np.zeros((24,32,3),np.uint8)
    _,geometry=preprocess(image)
    old=dict(epoch=1,backend='gpu',frame_id=1,received=time.monotonic(),detections=[])
    coordinator.frames={1:(image,geometry)};coordinator.latest=old
    coordinator.names={};coordinator.jpeg=None;coordinator.jpeg_sequence=0
    encoding=threading.Event();release=threading.Event()
    def encode(*args):
        encoding.set()
        assert release.wait(2)
        return True,np.array([1,2,3],np.uint8)
    monkeypatch.setattr(cv2,'imencode',encode)
    thread=threading.Thread(target=coordinator.render_loop);thread.start()
    try:
        assert encoding.wait(1)
        with coordinator.lock:
            # Input changed while rendering; backend and epoch remain the same.
            coordinator.frames={2:(image,geometry)}
            coordinator.latest=dict(old,frame_id=2)
            coordinator.stop.set()
        release.set();thread.join(1)
        assert coordinator.jpeg is None and coordinator.jpeg_sequence==0
    finally:
        coordinator.stop.set();release.set();thread.join(1)
