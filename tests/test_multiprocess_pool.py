import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
import pytest
from utilz.helpers import multiprocess_multiarg


def ordered_result(value):  #AI
    time.sleep(.03 * (4-value))
    return value


def maybe_error(value):  #AI
    if value == 1:
        raise ValueError('task exploded')
    return value


def task_exit(value):  #AI
    if value == 1:
        raise SystemExit('task exit')
    return value


def killed_result(value, marker):  #AI
    if value == 0:
        Path(marker).touch()
        return 42
    if value == 1:
        while not Path(marker).exists():
            time.sleep(.01)
        time.sleep(.5)
        os.kill(os.getpid(), signal.SIGKILL)
    time.sleep(30)
    return value


def numpy_result(value):  #AI
    result = np.arange(2**20, dtype=np.uint8) + value
    return result


def tensor_result(value):  #AI
    import torch

    result = {'nested': [torch.arange(8) + value]}
    return result


def oom_score(_):  #AI
    value = int(Path('/proc/self/oom_score_adj').read_text())
    return value


@pytest.mark.parametrize('case', ['order', 'exception', 'systemexit', 'killed', 'numpy', 'tensor', 'oom'])
def test_process_pool_contract(case, tmp_path):  #AI
    proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), case, str(tmp_path)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        stdout, stderr = proc.communicate(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.communicate()
        pytest.fail(f'{case}: multiprocessing did not finish within20s')
    assert proc.returncode == 0, stdout + stderr
    if case == 'exception':
        assert 'Task 1 failed' in stdout and 'ValueError' in stdout and 'task exploded' in stdout
    if case == 'killed':
        assert 'BrokenProcessPool' in stdout and '2/3 tasks lost' in stdout


if __name__ == '__main__':
    case, scratch = sys.argv[1:]
    kwargs = dict(num_processes=2, progress_bar=False)
    if case == 'order':
        result = multiprocess_multiarg(ordered_result, [(i,) for i in range(5)], **kwargs)
        assert result == list(range(5)), result
    elif case == 'exception':
        result = multiprocess_multiarg(maybe_error, [(i,) for i in range(3)], **kwargs)
        assert result == [0, None, 2], result
    elif case == 'systemexit':
        result = multiprocess_multiarg(task_exit, [(i,) for i in range(3)], **kwargs)
        assert result == [0, None, 2], result
    elif case == 'killed':
        result = multiprocess_multiarg(killed_result, [(i, str(Path(scratch)/'done')) for i in range(3)], **kwargs)
        assert result == [42, None, None], result
    elif case == 'numpy':
        result = multiprocess_multiarg(numpy_result, [(i,) for i in range(2)], **kwargs)
        for i, value in enumerate(result):
            np.testing.assert_array_equal(value, numpy_result(i))
    elif case == 'tensor':
        import torch

        result = multiprocess_multiarg(tensor_result, [(i,) for i in range(2)], **kwargs)
        for i, value in enumerate(result):
            assert torch.equal(value['nested'][0], torch.arange(8)+i)
    elif case == 'oom':
        before = int(Path('/proc/self/oom_score_adj').read_text())
        result = multiprocess_multiarg(oom_score, [(i,) for i in range(2)], **kwargs)
        assert result == [1000, 1000], result
        assert int(Path('/proc/self/oom_score_adj').read_text()) == before
