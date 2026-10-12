# SPDX-License-Identifier: AGPL-3.0-only
import threading
import time
import cv2
import numpy as np
import pytest
from app import capture as capture_module
from app.cameras import CameraDevice, candidates, rank, matches, load_config, configure_format, capture_capable
from app.capture import Capture

A=CameraDevice('camera-a','/dev/video2','Camera A','usb-camera-A-video-index0','aaaa:0001')
B=CameraDevice('camera-b','/dev/video4','Camera B',None,'bbbb:0002')
CONFIG=dict(preferred=[dict(by_id='usb-camera-A*',name='Preferred A')],fallback='any',
            preempt=True,preempt_settle_sec=0.,format=dict(width=1920,height=1080,fps=30,fourcc_priority=['MJPG','YUYV']))


def wait_for(predicate, timeout=3):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        if predicate():return
        time.sleep(.01)
    raise AssertionError('Capture did not reach expected state')


def test_priority_stable_identity_manual_override_and_fallback():
    assert rank(A,CONFIG)==0 and rank(B,CONFIG)==1
    assert matches(A,{'by_id':'usb-camera-A*'})
    assert matches(B,{'vid_pid':'BBBB:0002'})
    assert matches(A,'/dev/v4l/by-id/usb-camera-A-video-index0')
    assert matches(A,'2')
    assert candidates([B,A],CONFIG)==[A,B]
    assert candidates([B,A],CONFIG,'camera-b')==[B,A]
    restricted=dict(CONFIG,fallback='none')
    assert candidates([B],restricted)==[]
    assert candidates([B],restricted,'camera-b')==[B]


def test_capture_capability_excludes_metadata_node(monkeypatch):
    import struct
    from app import cameras
    monkeypatch.setattr(cameras.os,'open',lambda *_:7)
    monkeypatch.setattr(cameras.os,'close',lambda _:None)
    def ioctl(_fd,_request,buf,_mutate):
        struct.pack_into('=II',buf,84,0x80000000,0x00800000)  # META_CAPTURE
    monkeypatch.setattr(cameras.fcntl,'ioctl',ioctl)
    assert not capture_capable('/dev/video1')
    def video_ioctl(_fd,_request,buf,_mutate):
        struct.pack_into('=II',buf,84,0x80000000,0x04000001)
    monkeypatch.setattr(cameras.fcntl,'ioctl',video_ioctl)
    assert capture_capable('/dev/video0')


def test_config_and_negotiated_resolution_are_distinct(tmp_path):
    class Cap:
        def __init__(self):self.calls=[]
        def set(self,key,value):self.calls.append((key,value));return True
        def get(self,key):return {cv2.CAP_PROP_FOURCC:cv2.VideoWriter_fourcc(*'MJPG'),cv2.CAP_PROP_FRAME_WIDTH:1280,cv2.CAP_PROP_FRAME_HEIGHT:720,cv2.CAP_PROP_FPS:20}[key]
    cap=Cap();fmt=configure_format(cap,'MJPG',CONFIG['format'])
    assert cap.calls[0]==(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))
    assert (cv2.CAP_PROP_FRAME_WIDTH,1920) in cap.calls
    assert fmt==dict(fourcc='MJPG',width=1280,height=720,fps=20)
    p=tmp_path/'camera.yaml';p.write_text('camera:\n  format:\n    width: 1920\n    height: 1080\n')
    assert load_config(p,width=2560,height=1440)['format']['width']==2560
    p.write_text('camera:\n  format:\n    width: 0\n')
    with pytest.raises(ValueError):load_config(p)


def test_hotplug_manual_switch_disconnect_and_reconnect(monkeypatch):
    available=[B];created=[];seen=[]
    class Session:
        def __init__(self,device,requested):
            self.device=device;self.closed=False;self.last=0.;created.append(self)
        def poll(self):
            if time.monotonic()-self.last<.02:return None
            self.last=time.monotonic()
            frame=np.full((48,64,3),30 if self.device==A else 200,np.uint8)
            return frame,dict(width=64,height=48,fourcc='MJPG',fps=30.),self.last,None
        def close(self):self.closed=True
    class Player:
        name='fallback.mp4'
        def __init__(self,_):pass
        def close(self):pass
        def read(self):return np.full((48,64,3),90,np.uint8)
    monkeypatch.setattr(capture_module,'enumerate_cameras',lambda:available.copy())
    monkeypatch.setattr(capture_module,'CameraSession',Session)
    monkeypatch.setattr(capture_module,'PlaylistPlayer',Player)
    c=Capture('unused','auto',None,lambda i,f:seen.append((i,int(f.mean()))),lambda _:None,dict(CONFIG))
    c.thread.start()
    try:
        wait_for(lambda:c.status()['camera'] and c.status()['camera']['id']=='camera-b')
        available[:]=[A,B]
        wait_for(lambda:c.status()['camera'] and c.status()['camera']['id']=='camera-a')
        c.request_camera('camera-b')
        wait_for(lambda:c.status()['camera'] and c.status()['camera']['id']=='camera-b')
        c.request_camera('playlist')
        wait_for(lambda:c.status()['source']=='fallback.mp4' and c.status()['camera'] is None)
        c.request_camera('auto')
        wait_for(lambda:c.status()['camera'] and c.status()['camera']['id']=='camera-a')
        available[:]=[B]
        wait_for(lambda:c.status()['camera'] and c.status()['camera']['id']=='camera-b')
        available[:]=[]
        wait_for(lambda:c.status()['source']=='fallback.mp4' and c.status()['camera'] is None)
        available[:]=[A]
        wait_for(lambda:c.status()['camera'] and c.status()['camera']['id']=='camera-a')
        with pytest.raises(ValueError):c.request_camera('missing-camera')
    finally:c.close()
    assert not c.thread.is_alive() and all(s.closed for s in created)
    assert [i for i,_ in seen]==list(range(1,len(seen)+1))
    assert {color for _,color in seen}=={30,90,200}
