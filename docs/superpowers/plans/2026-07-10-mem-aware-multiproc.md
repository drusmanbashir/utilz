# Mem-aware multiprocess supervisor — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Adaptive mem-aware replacement for `multiprocess_multiarg` that kills/requeues under RAM pressure and still drains the full job list.

**Architecture:** New `utilz.multiproc` package (`mem.py` + `supervisor.py`); `helpers.multiprocess_multiarg` becomes a thin wrapper. Merge path passes `log_dir=output_folder/logs`.

**Tech Stack:** Python stdlib `multiprocessing` (spawn), `/proc/meminfo`, JSONL logging. No new deps.

**Spec:** `docs/superpowers/specs/2026-07-10-mem-aware-multiproc-design.md`

---

### Task 1: `mem_frac`

**Files:**
- Create: `utilz/multiproc/mem.py`
- Create: `utilz/multiproc/__init__.py`

- [ ] **Step 1:** Implement `mem_frac() -> float` reading `MemTotal` / `MemAvailable` from `/proc/meminfo` → `1 - available/total`.
- [ ] **Step 2:** Smoke: `python -c "from utilz.multiproc.mem import mem_frac; print(mem_frac())"` in `dl` env.

### Task 2: Supervisor

**Files:**
- Create: `utilz/multiproc/supervisor.py`

- [ ] **Step 1:** Implement `run_adaptive(func, arguments, num_processes, mem_threshold, log_dir, initializer=None, initargs=(), progress_bar=True)`.
- [ ] **Step 2:** Kill newest + requeue on over-threshold; floor fail_skip; abort after 2 consecutive floor fails; write `{log_dir}/log.jsonl`.
- [ ] **Step 3:** Export from `__init__.py`.

### Task 3: Wire helpers

**Files:**
- Modify: `utilz/helpers.py` (`multiprocess_multiarg`)

- [ ] **Step 1:** Add `mem_threshold=0.85`, `log_dir=None`; default `log_dir` → `/tmp/utilz_multiproc_logs/<run_id>`.
- [ ] **Step 2:** Multiprocess path calls `run_adaptive`; keep debug/single-process path unchanged.
- [ ] **Step 3:** Preserve `io`/`algebra` initializer behaviour.

### Task 4: Dummy RAM trial

**Files:**
- Ad hoc: `/s/agent_rw/tmp/trial_multiproc_mem.py` (delete after)

- [ ] **Step 1:** Jobs allocate ~8–15 GB numpy; `num_processes=6`, `mem_threshold=0.5`.
- [ ] **Step 2:** Run; confirm `kill_requeue` in log and jobs complete or controlled fail_skip.
- [ ] **Step 3:** Delete trial script after success.

### Task 5: Merge wire + litq run

**Files:**
- Modify: `label_analysis/merge.py` (`mergetouchinglabelfiles_multiprocessor` / wrapper)
- Modify: `label_analysis/run/merge.py` if needed for log_dir

- [ ] **Step 1:** Pass `log_dir` under default/fixed output folder `…/logs`.
- [ ] **Step 2:** Run `fix-touching-mp` on `/s/fran_storage/predictions/lits/LITS-ROOST/litq`.
- [ ] **Step 3:** Confirm outputs in `litq_fixed_mc` and `litq_fixed_mc/logs/log.jsonl`.
