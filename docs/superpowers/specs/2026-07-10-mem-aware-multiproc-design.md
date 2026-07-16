# Mem-aware multiprocess supervisor — design

Date: 2026-07-10

## Goal

Replace fixed-size `multiprocess_multiarg` with an adaptive process supervisor that shrinks concurrency under RAM pressure, never drops the job list, and fails loud only after two consecutive floor (1-worker) failures.

## Decisions

| Item | Choice |
|------|--------|
| Approach | Adaptive Process supervisor (manual spawn/join) |
| Over threshold | Kill newest excess workers, requeue their args, lower `target` |
| Floor (1 worker) still over | Kill task, log failed, skip, next |
| Abort | 2 consecutive floor-fails → raise; print abs `log.jsonl` path + all failed |
| API | Replace `multiprocess_multiarg` in place |
| Default threshold | `0.85` (trial uses `0.5`) |
| Log file | `{log_dir}/log.jsonl` |
| Merge log_dir | `{output_folder}/logs` |
| Deps | No new packages; RAM via `/proc/meminfo` (Linux) |

## Layout

```
utilz/utilz/multiproc/
  __init__.py      # re-exports
  mem.py           # mem_frac()
  supervisor.py    # run_adaptive(...)
```

`utilz/helpers.py`: `multiprocess_multiarg` thin-wraps supervisor; keep existing kwargs; add `mem_threshold=0.85`, `log_dir=None`.

`label_analysis`: pass `log_dir` from merge output folder (`…/logs`).

## Runtime

1. Pending deque of `(index, args)`; results list sized to `len(arguments)`.
2. `target = min(num_processes, n_jobs)`; spawn via `multiprocessing.get_context("spawn")`.
3. Before launch: if `mem_frac > mem_threshold` and `active > 0` → kill newest until under pressure or `active==1`; requeue; set `target = max(1, active)`.
4. On worker exit: store result or exception; if success at floor, reset `consecutive_floor_fails`.
5. Floor kill path: append fail event; `consecutive_floor_fails += 1`; if `>= 2` abort.
6. Preserve input order in results; failed slots `None`.

## Log events (JSONL)

`launch`, `kill_requeue`, `fail_skip`, `done`, `abort` — each with `ts`, `mem_frac`, `active`, `target`, job index/args summary.

## Trial

Ad hoc script under `/s/agent_rw/tmp/`: dummy jobs allocate large numpy arrays; `mem_threshold=0.5`; assert shrink + remaining jobs complete (or controlled fail_skip). Delete after.

## Merge validation

Run `fix-touching-mp` on `/s/fran_storage/predictions/lits/LITS-ROOST/litq` → `litq_fixed_mc`, logs in `litq_fixed_mc/logs/log.jsonl`.
