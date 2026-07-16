#!/usr/bin/env python3
"""Ad-hoc trial: mem-aware multiproc with dummy high-RAM jobs. Delete after."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from utilz.helpers import multiprocess_multiarg
from utilz.multiproc.mem import mem_frac
from utilz.multiproc.supervisor import MultiprocMemAbort


def hog(job_id: int, gib: float, hold_s: float) -> str:
    n = int(gib * (1024**3) / 8)
    arr = np.ones(n, dtype=np.float64)
    arr[0] = job_id
    time.sleep(hold_s)
    return f"ok-{job_id}-{arr[0]}"


def main():
    log_dir = Path("/s/agent_rw/tmp/multiproc_mem_trial")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "log.jsonl"
    if log_path.exists():
        log_path.unlink()

    print(f"start mem_frac={mem_frac():.3f}")
    # 8 jobs × ~20 GiB would OOM hard; with threshold 0.5 supervisor must shrink
    args = [[i, 18.0, 2.0] for i in range(8)]
    try:
        results = multiprocess_multiarg(
            hog,
            args,
            num_processes=6,
            mem_threshold=0.5,
            log_dir=log_dir,
            progress_bar=True,
        )
        print("results", results)
    except MultiprocMemAbort as exc:
        print("ABORT", exc)

    events = [json.loads(line) for line in log_path.read_text().splitlines() if line]
    kinds = [e["event"] for e in events]
    print("events", kinds)
    print(f"log={log_path.resolve()}")
    assert "kill_requeue" in kinds or "fail_skip" in kinds, kinds
    print("trial ok")


if __name__ == "__main__":
    main()
