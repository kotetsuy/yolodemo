# SPDX-License-Identifier: AGPL-3.0-only
"""Exercise the real Chrome UI, hardware switches, and MJPEG image."""
import json
from pathlib import Path
import sys
import time
import urllib.request
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app.model_utils import ROOT,write_json
from playwright.sync_api import sync_playwright

URL='http://127.0.0.1:8000'


def state():
    return json.load(urllib.request.urlopen(URL+'/api/status',timeout=5))


if __name__=='__main__':
    folder=ROOT/'results/browser-check'
    folder.mkdir(exist_ok=True)
    errors=[];switches=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True,args=['--disable-dev-shm-usage'])
        page=browser.new_page(viewport={'width':1440,'height':1100})
        page.on('pageerror',lambda e:errors.append(str(e)))
        page.goto(URL,wait_until='domcontentloaded')
        slider=page.locator('#backend-slider')
        page.wait_for_function("!document.getElementById('backend-slider').disabled",timeout=120000)
        slider.focus();slider.press('Home')
        page.wait_for_function('document.getElementById("backend").textContent==="GPU" && document.getElementById("fps").textContent!=="—"',timeout=15000)
        pids={name:w['pid'] for name,w in state()['workers'].items()}
        for backend in ('npu','gpu','npu','gpu'):
            start=time.monotonic()
            slider.focus();slider.press('End' if backend=='npu' else 'Home')
            page.wait_for_function('(backend)=>document.getElementById("backend").textContent===backend && document.getElementById("fps").textContent!=="—"',arg=backend.upper(),timeout=15000)
            page.wait_for_function('document.getElementById("video").naturalWidth>0 && document.getElementById("placeholder").style.display==="none"')
            s=state()
            assert s['backend']==backend and not s['switching'] and s['fps']>0
            assert {name:w['pid'] for name,w in s['workers'].items()}==pids
            switches.append(dict(backend=backend,observed_ms=(time.monotonic()-start)*1000,worker_switch_ms=s['switch_ms'],epoch=s['epoch'],fps=s['fps']))
            page.screenshot(path=str(folder/f'{backend}.png'),full_page=True)
        # Requests can arrive while the previous transition is still in flight.
        slider.focus()
        for key in ('End','Home','End'):slider.press(key)
        page.wait_for_function('document.getElementById("backend").textContent==="NPU" && document.getElementById("fps").textContent!=="—"',timeout=15000)
        page.wait_for_function('document.getElementById("video").naturalWidth>0 && document.getElementById("video").naturalHeight>0 && document.getElementById("placeholder").style.display==="none"',timeout=15000)
        assert state()['desired']=='npu'
        dimensions=page.locator('#video').evaluate('(e)=>[e.naturalWidth,e.naturalHeight]')
        assert 0<dimensions[0]<=state()['display_width'] and dimensions[1]>0
        assert page.locator('a[href="/credits"]').count()==1
        assert not errors,errors
        write_json(folder/'result.json',dict(status='complete',errors=errors,switches=switches,pids=pids,rapid_last_selection='npu',image_dimensions=dimensions,final=state()))
        browser.close()
    print(folder/'result.json')
