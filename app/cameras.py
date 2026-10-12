# SPDX-License-Identifier: AGPL-3.0-only
"""Stable udev camera identities and V4L2 format negotiation.

Adapted to this app from ~/LLaVA-NPU/src/capture/device_manager.py and
format_negotiator.py. Capability queries do not start/re-open live streams.
"""
from dataclasses import asdict, dataclass
import fcntl
import fnmatch
import os
from pathlib import Path
import struct
import cv2
import pyudev
import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[1]/'camera.yaml'


@dataclass(frozen=True)
class CameraDevice:
    id: str
    path: str
    name: str
    by_id: str | None = None
    vid_pid: str | None = None


def capture_capable(path):
    fd = None
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
        buf = bytearray(104)  # struct v4l2_capability, linux/videodev2.h
        fcntl.ioctl(fd, 0x80685600, buf, True)  # VIDIOC_QUERYCAP
        caps, device_caps = struct.unpack_from('=II', buf, 84)
        effective = device_caps if caps & 0x80000000 else caps
        return bool(effective & (0x1 | 0x1000))
    except OSError:
        return False
    finally:
        if fd is not None:
            os.close(fd)


def enumerate_cameras(context=None):
    out = []
    for dev in (context or pyudev.Context()).list_devices(subsystem='video4linux'):
        path = dev.device_node
        if not path or not capture_capable(path):
            continue
        links = sorted(Path(v).name for v in dev.device_links if v.startswith('/dev/v4l/by-id/'))
        by_id = links[0] if links else None
        vid, pid = dev.properties.get('ID_VENDOR_ID'), dev.properties.get('ID_MODEL_ID')
        usb = dev.find_parent('usb', 'usb_device')
        if usb is not None:
            try:
                vid = vid or usb.attributes.asstring('idVendor')
                pid = pid or usb.attributes.asstring('idProduct')
            except (KeyError, OSError):
                pass
        try:
            name = dev.attributes.asstring('name')
        except (KeyError, OSError):
            name = dev.properties.get('ID_MODEL', 'USB Camera')
        try:
            index = dev.attributes.asstring('index')
        except (KeyError, OSError):
            index = dev.sys_name
        identity = by_id or (f'{dev.properties.get("ID_PATH")}-index{index}' if dev.properties.get('ID_PATH') else path)
        out.append(CameraDevice(identity, path, name, by_id,
                                f'{vid}:{pid}'.lower() if vid and pid else None))
    return sorted(out, key=lambda d: (d.id, d.path))


def matches(device, value):
    if isinstance(value, dict):
        if value.get('by_id'):
            return bool(device.by_id and fnmatch.fnmatchcase(device.by_id, value['by_id']))
        return bool(value.get('vid_pid') and device.vid_pid == value['vid_pid'].lower())
    if str(value).isdecimal():
        value = f'/dev/video{value}'
    return (value in (device.id, device.path, device.vid_pid) or
            bool(device.by_id and (fnmatch.fnmatchcase(device.by_id, str(value)) or
                                  value == str(Path('/dev/v4l/by-id')/device.by_id))))


def rank(device, config):
    return next((i for i, p in enumerate(config['preferred']) if matches(device, p)), len(config['preferred']))


def label(device, config):
    i = rank(device, config)
    return config['preferred'][i].get('name', device.name) if i < len(config['preferred']) else device.name


def camera_inventory(devices, config):
    return [dict(asdict(d), label=label(d, config), priority=rank(d, config)) for d in devices]


def candidates(devices, config, selection='auto'):
    ordered = sorted(devices, key=lambda d: (rank(d, config), d.id))
    if config['fallback'] == 'none':
        ordered = [d for d in ordered if rank(d, config) < len(config['preferred'])]
    explicit = [d for d in devices if selection not in ('auto', 'playlist') and matches(d, selection)]
    return explicit + [d for d in ordered if d not in explicit]


def load_config(path=DEFAULT_CONFIG, width=None, height=None, fps=None):
    data = yaml.safe_load(Path(path).expanduser().read_text()) or {}
    cfg = dict(data.get('camera', data))
    cfg.setdefault('preferred', [])
    cfg.setdefault('fallback', 'any')
    cfg.setdefault('preempt', True)
    cfg.setdefault('preempt_settle_sec', 1.)
    fmt = dict(cfg.get('format', {}))
    fmt.update(width=width if width is not None else fmt.get('width', 1920),
               height=height if height is not None else fmt.get('height', 1080),
               fps=fps if fps is not None else fmt.get('fps', 30))
    fmt.setdefault('fourcc_priority', ['MJPG', 'YUYV'])
    if (not isinstance(cfg['preferred'], list) or
            any(not isinstance(p, dict) or not (p.get('by_id') or p.get('vid_pid')) for p in cfg['preferred'])):
        raise ValueError('camera.preferred requires by_id or vid_pid for each entry')
    if cfg['fallback'] not in ('any', 'none') or not isinstance(cfg['preempt'], bool):
        raise ValueError('Invalid camera fallback/preempt')
    if not 0 <= float(cfg['preempt_settle_sec']) <= 30:
        raise ValueError('Invalid camera preempt_settle_sec')
    if not (64 <= int(fmt['width']) <= 7680 and 64 <= int(fmt['height']) <= 4320 and 1 <= float(fmt['fps']) <= 120):
        raise ValueError('Invalid requested camera resolution/FPS')
    if (not isinstance(fmt['fourcc_priority'], list) or not fmt['fourcc_priority'] or
            any(not isinstance(v, str) or len(v) != 4 for v in fmt['fourcc_priority'])):
        raise ValueError('fourcc_priority must contain FourCC strings')
    cfg['format'] = fmt
    return cfg


def configure_format(cap, fourcc, requested):
    # MJPG comes first to reduce USB bandwidth for high-resolution UVC input.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(requested['width']))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(requested['height']))
    cap.set(cv2.CAP_PROP_FPS, float(requested['fps']))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    code = int(cap.get(cv2.CAP_PROP_FOURCC))
    return dict(fourcc=''.join(chr((code >> (8*i)) & 0xff) for i in range(4)),
                width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), fps=cap.get(cv2.CAP_PROP_FPS))
