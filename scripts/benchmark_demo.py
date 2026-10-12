# SPDX-License-Identifier: AGPL-3.0-only
"""Measure the live demo counters and optionally drain the browser MJPEG load."""
import argparse
import json
from pathlib import Path
import sys
import threading
import time
import urllib.request
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app.model_utils import ROOT,write_json


def status(url):
    return json.load(urllib.request.urlopen(url+'/api/status',timeout=5))


def select(url,backend,timeout=120):
    start=time.monotonic()
    while time.monotonic()-start<timeout:
        s=status(url)
        w=s['workers'][backend]
        if w['state']=='error':raise RuntimeError(w['error'])
        if w['state'] in ('ready','active'):break
        time.sleep(.3)
    else:raise RuntimeError('Backend loading timeout')
    req=urllib.request.Request(url+'/api/backend',data=json.dumps(dict(backend=backend)).encode(),headers={'Content-Type':'application/json'})
    json.load(urllib.request.urlopen(req,timeout=5))
    while time.monotonic()-start<timeout:
        s=status(url)
        if s['backend']==backend and not s['switching'] and s['fps'] is not None:return s
        time.sleep(.1)
    raise RuntimeError('Backend switch timeout')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url',default='http://127.0.0.1:8000')
    p.add_argument('--backends',nargs='+',choices=['gpu','npu'],default=['gpu','npu'])
    p.add_argument('--seconds',type=float,default=30)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--no-stream',action='store_true')
    p.add_argument('--output',type=Path)
    a=p.parse_args()
    if a.seconds<=0 or a.repeats<1:p.error('Positive seconds and repeats required')
    out=a.output or ROOT/'results'/('benchmark-'+time.strftime('%Y%m%d-%H%M%S')+'.json')
    stop=threading.Event();stats={'stream_bytes':0}
    def stream():
        try:
            with urllib.request.urlopen(a.url+'/stream.mjpg',timeout=5) as r:
                while not stop.is_set():
                    data=r.read(65536)
                    if not data:break
                    stats['stream_bytes']+=len(data)
        except Exception as exc:stats['stream_error']=str(exc)
    thread=threading.Thread(target=stream,daemon=True)
    if not a.no_stream:thread.start()
    report=dict(status='started',seconds=a.seconds,repeats=a.repeats,stream=not a.no_stream,runs=[],backends={})
    write_json(out,report)
    try:
        for backend in a.backends:
            select(a.url,backend)
            runs=[]
            for trial in range(1,a.repeats+1):
                first=status(a.url);first_w=first['workers'][backend]
                begin=time.monotonic();samples=[]
                while time.monotonic()-begin<a.seconds:
                    time.sleep(min(.5,max(.001,a.seconds-(time.monotonic()-begin))))
                    last=status(a.url);w=last['workers'][backend]
                    if last['epoch']!=first['epoch'] or last['backend']!=backend or w['state']=='error':
                        raise RuntimeError('Backend changed/failed during measurement')
                    if w['completed']<first_w['completed'] or w['elapsed']<first_w['elapsed']:
                        raise RuntimeError('Counter reset/input stopped during measurement')
                    if time.monotonic()-w['heartbeat']>1:
                        raise RuntimeError('Inference stalled')
                    try:
                        rss_line=next(v for v in Path(f'/proc/{last["server_pid"]}/status').read_text().splitlines() if v.startswith('VmRSS:'))
                        server_rss=int(rss_line.split()[1])
                    except (OSError,StopIteration):server_rss=None
                    samples.append(dict(fps=last['fps'],input_fps=last['input']['fps'],source=last['input']['source'],
                        total_frames=last['input']['total_frames'],result_age_ms=last['result_age_ms'],
                        server_rss_kb=server_rss,
                        rss_kb={k:v.get('rss_kb') for k,v in last['workers'].items()}))
                frames=w['completed']-first_w['completed'];elapsed=w['elapsed']-first_w['elapsed']
                row=dict(backend=backend,trial=trial,frames=frames,elapsed_s=elapsed,fps=frames/elapsed,
                    epoch=first['epoch'],size=448,boundary=w['info']['boundary'],samples=samples)
                runs.append(row);report['runs'].append(row)
                report['backends'][backend]=dict(info=w['info'],pooled_fps=sum(x['frames'] for x in runs)/sum(x['elapsed_s'] for x in runs),
                    target=300 if backend=='gpu' else 60)
                print(f'{backend} trial {trial}: {row["fps"]:.2f} FPS',flush=True)
                write_json(out,report)
        report['status']='complete'
        for data in report['backends'].values():data['target_passed']=data['pooled_fps']>data['target']
    except Exception as exc:
        report.update(status='failed',error=str(exc));raise
    finally:
        stop.set()
        if thread.is_alive():thread.join(timeout=6)
        report.update(stats)
        write_json(out,report)
    print(out)


if __name__=='__main__':main()
