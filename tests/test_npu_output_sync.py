# SPDX-License-Identifier: AGPL-3.0-only
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.workers import npu_output_sync as sync


def test_unknown_runtime_is_rejected_before_loading_native_code(tmp_path):
    library = tmp_path/'runtime.so'
    library.write_bytes(b'another SDK build')
    with pytest.raises(RuntimeError, match='unvalidated runtime'):
        sync.verify_libraries(dict(runtime=library, provider=library, xrt=library))


def test_bootstrap_reexec_preserves_arguments_and_only_changes_child_environment(monkeypatch):
    monkeypatch.setattr(sync, 'verify_libraries', lambda _: {})
    monkeypatch.setattr(sync, 'build_library', lambda _: Path('/tmp/npu-sync.so'))
    monkeypatch.setattr(sync.sys, 'orig_argv', ['python', '-m', 'app.workers.run', '--backend', 'npu'])
    monkeypatch.delenv('YOLODEMO_NPU_OUTPUT_SYNC', raising=False)
    monkeypatch.setenv('LD_PRELOAD', '/tmp/existing.so')
    class Reexecuted(Exception):
        pass
    def execute(executable, argv, env):
        assert executable == sync.sys.executable
        assert argv == [executable, '-m', 'app.workers.run', '--backend', 'npu']
        assert env['LD_PRELOAD'] == '/tmp/npu-sync.so:/tmp/existing.so'
        assert env['YOLODEMO_NPU_OUTPUT_SYNC'] == '/tmp/npu-sync.so'
        raise Reexecuted
    monkeypatch.setattr(sync.os, 'execve', execute)
    with pytest.raises(Reexecuted):
        sync.bootstrap_output_sync()
    assert os.environ['LD_PRELOAD'] == '/tmp/existing.so'
    assert 'YOLODEMO_NPU_OUTPUT_SYNC' not in os.environ
    monkeypatch.setenv('YOLODEMO_NPU_OUTPUT_SYNC', '/tmp/old.so')
    with pytest.raises(RuntimeError, match='stale preload'):
        sync.bootstrap_output_sync()


@pytest.mark.parametrize('after', [
    (0, 0, 0, 0),  # Missing symbol interposition.
    (1, 1, 0, 0),  # Completed device execution without synchronization.
    (1, 1, 1, 1),  # BO references retained after the runner invocation.
])
def test_unsynchronized_or_retained_outputs_are_never_returned(after):
    guard = sync.OutputSync.__new__(sync.OutputSync)
    snapshots = iter([(0, 0, 0, 0), after])
    guard.stats = lambda: dict(zip(('runner_calls', 'npu_completions', 'output_syncs', 'live_buffers'), next(snapshots)))
    with pytest.raises(RuntimeError, match='did not cover'):
        guard.run(SimpleNamespace(run=lambda *_: ['unchecked']), {'images': object()})


def test_synchronized_output_is_returned():
    guard = sync.OutputSync.__new__(sync.OutputSync)
    snapshots = iter([(5, 5, 5, 0), (6, 6, 6, 0)])
    guard.stats = lambda: dict(zip(('runner_calls', 'npu_completions', 'output_syncs', 'live_buffers'), next(snapshots)))
    assert guard.run(SimpleNamespace(run=lambda *_: ['checked']), {}) == ['checked']
