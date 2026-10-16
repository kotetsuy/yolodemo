# SPDX-License-Identifier: AGPL-3.0-only
"""Load and verify the local, version-specific NPU output synchronization shim."""
import ctypes
import fcntl
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import sysconfig
import tempfile
import threading

from app.model_utils import ROOT, sha256

SUPPORTED = {
    'runtime': '172385afec6bfef49ab251f571cadfba8e72a803bf8f9b6078577db302f65027',
    'provider': 'cf5206d4810e503e017dfac1fb0cfe89789ff9c3eae57d6db1682a040d28a09b',
    'xrt': 'acbc609939c35e9edc0aed3709a0b5bdc6d0b9e5f53e9e6d2e0a94100ada2e97',
}
XRT = Path('/opt/xilinx/xrt')
_lock = threading.Lock()
_instance = None


def verify_libraries(libraries):
    hashes = {key: sha256(path) for key, path in libraries.items()}
    if hashes != SUPPORTED:
        raise RuntimeError('NPU output sync: unvalidated runtime libraries; '
                           'validate this SDK/XRT version before updating the compatibility shim')
    return hashes


def build_library(hashes):
    source = Path(__file__).with_suffix('.cpp')
    header = XRT/'include/xrt/xrt_bo.h'
    key = hashlib.sha256((sha256(source)+sha256(header)+str(sorted(hashes.items()))).encode()).hexdigest()
    folder = ROOT/'.cache/native/npu-output-sync'/key
    folder.mkdir(parents=True, exist_ok=True)
    target = folder/'libyolodemo_npu_output_sync.so'
    with (folder/'build.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if not target.exists():
            fd, tmp = tempfile.mkstemp(suffix='.so', dir=folder)
            os.close(fd)
            try:
                result = subprocess.run([
                    'g++', '-std=c++17', '-O2', '-fPIC', '-shared', str(source),
                    '-I'+str(XRT/'include'), '-L'+str(XRT/'lib'),
                    '-Wl,-rpath,'+str(XRT/'lib'), '-lxrt_coreutil', '-ldl', '-o', tmp,
                ], capture_output=True, text=True)
                if result.returncode:
                    raise RuntimeError('NPU output sync: native build failed: '+result.stderr[-3000:])
                os.replace(tmp, target)
            finally:
                Path(tmp).unlink(missing_ok=True)
    return target


def runtime_libraries():
    folder = Path(sysconfig.get_paths()['purelib'])/'voe/lib'
    return dict(runtime=folder/'libvitis-ai-runtime.so',
                provider=folder/'libonnxruntime_vitisai_ep.so',
                xrt=XRT/'lib/libxrt_coreutil.so')


def bootstrap_output_sync():
    """Re-exec a dedicated NPU entry point once, before opening IPC or devices."""
    hashes = verify_libraries(runtime_libraries())
    path = str(build_library(hashes))
    marker = os.environ.get('YOLODEMO_NPU_OUTPUT_SYNC')
    preloads = os.environ.get('LD_PRELOAD', '').replace(':', ' ').split()
    if marker == path and path in preloads:
        return
    if marker:
        raise RuntimeError('NPU output sync: stale preload configuration; restart the NPU process')
    env = os.environ.copy()
    env['YOLODEMO_NPU_OUTPUT_SYNC'] = path
    env['LD_PRELOAD'] = ':'.join([path, *[p for p in preloads if p != path]])
    os.execve(sys.executable, [sys.executable, *sys.orig_argv[1:]], env)


class OutputSync:
    def __init__(self):
        self.libraries = runtime_libraries()
        self.hashes = verify_libraries(self.libraries)
        self.path = build_library(self.hashes)
        if os.environ.get('YOLODEMO_NPU_OUTPUT_SYNC') != str(self.path):
            raise RuntimeError('NPU output sync: call bootstrap_output_sync() in the NPU entry point')
        # Refuse to load the interposer late: CPU-op registration and symbol
        # binding must happen in their normal order during process startup.
        self.native = ctypes.CDLL(str(self.path), mode=os.RTLD_NOLOAD | os.RTLD_NOW)
        self.native.yolodemo_npu_sync_stats.argtypes = [ctypes.POINTER(ctypes.c_uint64)]
        self.native.yolodemo_npu_sync_stats.restype = None
        self.native.yolodemo_npu_sync_original_library.argtypes = [ctypes.c_int]
        self.native.yolodemo_npu_sync_original_library.restype = ctypes.c_char_p

    def verify_loaded(self):
        for index, key in enumerate(('provider', 'provider', 'xrt')):
            path = self.native.yolodemo_npu_sync_original_library(index)
            if not path or sha256(os.fsdecode(path)) != self.hashes[key]:
                raise RuntimeError(f'NPU output sync: unexpected loaded runtime library for {index}: {path!r}')

    def stats(self):
        values = (ctypes.c_uint64 * 4)()
        self.native.yolodemo_npu_sync_stats(values)
        return dict(zip(('runner_calls', 'npu_completions', 'output_syncs', 'live_buffers'), values))

    def run(self, session, inputs):
        before = self.stats()
        raw = session.run(None, inputs)
        after = self.stats()
        completed = after['npu_completions']-before['npu_completions']
        if (not completed or after['output_syncs']-before['output_syncs'] != completed
                or after['runner_calls'] == before['runner_calls'] or after['live_buffers']):
            raise RuntimeError('NPU output sync: synchronization hook did not cover this inference')
        return raw

    def info(self):
        return dict(method='XRT FROM_DEVICE after NPU completion, before CPU depadding',
                    library=str(self.path), library_sha256=sha256(self.path),
                    runtime_sha256=self.hashes, **self.stats())


def ensure_output_sync():
    global _instance
    with _lock:
        if _instance is None:
            _instance = OutputSync()
        return _instance
