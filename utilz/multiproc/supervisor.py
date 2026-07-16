"""Adaptive multiprocessing supervisor with RAM-pressure kill/requeue."""
from __future__ import annotations

import json
import multiprocessing as mp
import time
import traceback
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

from tqdm.auto import tqdm

from utilz.multiproc.mem import mem_frac


class MultiprocMemAbort(RuntimeError):
    """Raised after two consecutive floor (1-worker) memory kills."""


@dataclass
class _ActiveJob:
    index: int
    args: tuple
    process: mp.Process
    queue: Any
    launched_at: float = field(default_factory=time.time)


def _worker_entry(func, args, result_queue, job_index, initializer, initargs):  #AI
    if initializer is not None:
        initializer(*initargs)
    try:
        result = func(*args)
        result_queue.put(("ok", job_index, result))
    except BaseException as exc:
        result_queue.put(
            ("err", job_index, {"type": type(exc).__name__, "msg": str(exc), "tb": traceback.format_exc()})
        )


def _default_log_dir() -> Path:  #AI
    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    path = Path("/tmp/utilz_multiproc_logs") / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def _append_log(log_path: Path, event: str, **fields):  #AI
    row = {"ts": time.time(), "event": event, "mem_frac": round(mem_frac(), 4), **fields}
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, default=str) + "\n")


def _args_summary(args: tuple) -> str:  #AI
    text = repr(args)
    if len(text) > 200:
        return text[:200] + "..."
    return text


def run_adaptive(  #AI
    func: Callable,
    arguments,
    num_processes: int = 8,
    mem_threshold: float = 0.85,
    log_dir: Optional[Path] = None,
    initializer=None,
    initargs=(),
    progress_bar: bool = True,
) -> List[Any]:
    """
    Run func(*args) for each args in arguments with adaptive concurrency.

    On mem_frac > mem_threshold: kill newest workers, requeue their jobs, lower target.
    At 1 worker still over threshold: kill, log fail, skip; 2 consecutive → abort.
    """
    arguments = list(arguments)
    n = len(arguments)
    if n == 0:
        return []

    log_dir = Path(log_dir) if log_dir is not None else _default_log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "log.jsonl"

    ctx = mp.get_context("spawn")
    pending: Deque[Tuple[int, tuple]] = deque(
        (i, tuple(arguments[i])) for i in range(n)
    )
    results: List[Any] = [None] * n
    failed: List[Dict[str, Any]] = []
    active: Dict[int, _ActiveJob] = {}  # keyed by job index
    target = max(1, min(num_processes, n))
    consecutive_floor_fails = 0
    done_count = 0

    pbar = tqdm(total=n) if progress_bar else None

    def _reap() -> None:
        nonlocal done_count, consecutive_floor_fails
        finished = []
        for idx, job in list(active.items()):
            if job.process.is_alive():
                continue
            job.process.join(timeout=1)
            finished.append(idx)
            status = None
            payload = None
            try:
                if not job.queue.empty():
                    status, _, payload = job.queue.get_nowait()
            except Exception:
                status = "err"
                payload = {"type": "QueueEmpty", "msg": "no result after exit", "tb": ""}
            if status == "ok":
                results[idx] = payload
                consecutive_floor_fails = 0
                _append_log(
                    log_path,
                    "done",
                    job_index=idx,
                    active=len(active) - 1,
                    target=target,
                )
            else:
                results[idx] = None
                failed.append({"job_index": idx, "args": _args_summary(job.args), "error": payload})
                _append_log(
                    log_path,
                    "fail_skip",
                    job_index=idx,
                    reason="worker_error",
                    error=payload,
                    active=len(active) - 1,
                    target=target,
                )
            done_count += 1
            if pbar is not None:
                pbar.update(1)
        for idx in finished:
            active.pop(idx, None)

    def _kill_job(idx: int, reason: str) -> None:
        nonlocal done_count
        job = active.pop(idx)
        if job.process.is_alive():
            job.process.terminate()
            job.process.join(timeout=5)
            if job.process.is_alive():
                job.process.kill()
                job.process.join(timeout=2)
        try:
            while not job.queue.empty():
                job.queue.get_nowait()
        except Exception:
            pass

        if reason == "fail_skip":
            results[idx] = None
            failed.append({"job_index": idx, "args": _args_summary(job.args), "error": reason})
            done_count += 1
            if pbar is not None:
                pbar.update(1)
            _append_log(
                log_path,
                "fail_skip",
                job_index=idx,
                reason="mem_floor",
                active=len(active),
                target=target,
            )
        else:
            pending.appendleft((idx, job.args))
            _append_log(
                log_path,
                "kill_requeue",
                job_index=idx,
                reason=reason,
                active=len(active),
                target=target,
            )

    def _shrink_for_mem() -> None:
        nonlocal target, consecutive_floor_fails
        while mem_frac() > mem_threshold and active:
            if len(active) == 1:
                idx = next(iter(active))
                _kill_job(idx, "fail_skip")
                consecutive_floor_fails += 1
                if consecutive_floor_fails >= 2:
                    _append_log(
                        log_path,
                        "abort",
                        failed=failed,
                        consecutive_floor_fails=consecutive_floor_fails,
                    )
                    if pbar is not None:
                        pbar.close()
                    raise MultiprocMemAbort(
                        f"mem_threshold={mem_threshold} floor fails x2; "
                        f"failed={failed}; log={log_path.resolve()}"
                    )
                target = 1
                return
            # kill newest
            newest_idx = max(active, key=lambda i: active[i].launched_at)
            _kill_job(newest_idx, "mem_pressure")
            target = max(1, len(active))

    def _launch_one() -> None:
        if not pending:
            return
        if len(active) >= target:
            return
        if mem_frac() > mem_threshold and active:
            return
        idx, args = pending.popleft()
        q = ctx.Queue()
        proc = ctx.Process(
            target=_worker_entry,
            args=(func, args, q, idx, initializer, initargs),
        )
        proc.start()
        active[idx] = _ActiveJob(index=idx, args=args, process=proc, queue=q)
        _append_log(
            log_path,
            "launch",
            job_index=idx,
            args=_args_summary(args),
            active=len(active),
            target=target,
        )

    _append_log(log_path, "start", n_jobs=n, num_processes=num_processes, mem_threshold=mem_threshold)

    try:
        while pending or active:
            _reap()
            if mem_frac() > mem_threshold:
                _shrink_for_mem()
            while pending and len(active) < target and mem_frac() <= mem_threshold:
                _launch_one()
            if pending and not active and mem_frac() > mem_threshold:
                # system already hot with no workers — wait briefly then try one
                time.sleep(0.5)
                if mem_frac() > mem_threshold:
                    # force one launch; floor path will fail_skip if still hot
                    target = 1
                    _launch_one()
            if active or pending:
                time.sleep(0.1)
    finally:
        for idx in list(active):
            job = active[idx]
            if job.process.is_alive():
                job.process.terminate()
                job.process.join(timeout=2)
        if pbar is not None:
            pbar.close()

    _append_log(log_path, "complete", failed=failed, log=str(log_path.resolve()))
    print(f"multiproc log={log_path.resolve()}")
    return results
