#!/usr/bin/env python3
"""Exercise ImageBBoxViewer with synthetic data and programmatic slice scroll."""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
import traceback

import numpy as np
import torch

from utilz.imageviewers import ImageBBoxViewer


def make_volume(shape=(96, 128, 128), seed=0):
    rng = np.random.default_rng(seed)
    vol = rng.normal(0.0, 1.0, size=shape).astype(np.float32)
    vol += np.linspace(0, 1, shape[0], dtype=np.float32)[:, None, None]
    return vol


def make_boxes(shape, n_boxes=3, seed=0):
    rng = np.random.default_rng(seed)
    zmax, ymax, xmax = shape[0] - 1, shape[1] - 1, shape[2] - 1
    boxes = []
    for _ in range(n_boxes):
        x0 = int(rng.integers(0, max(1, xmax - 20)))
        y0 = int(rng.integers(0, max(1, ymax - 20)))
        z0 = int(rng.integers(0, max(1, zmax - 10)))
        x1 = min(xmax, x0 + int(rng.integers(8, 30)))
        y1 = min(ymax, y0 + int(rng.integers(8, 30)))
        z1 = min(zmax, z0 + int(rng.integers(4, 12)))
        boxes.append([x0, y0, z0, x1, y1, z1])
    return np.asarray(boxes, dtype=np.float64)


def make_monai_volume(shape=(96, 128, 128), seed=0):
    if importlib.util.find_spec("monai") is None:
        return None
    from monai.data import MetaTensor

    vol = make_volume(shape=shape, seed=seed)
    chw = vol[np.newaxis, ...]
    meta = {
        "filename_or_obj": "synthetic_case",
        "spatial_shape": np.array(shape, dtype=np.int64),
        "spacing": np.array([1.25, 0.7, 0.7], dtype=np.float64),
        "affine": np.eye(4, dtype=np.float64),
        "original_channel_dim": 0,
    }
    return MetaTensor(torch.as_tensor(chw), meta=meta)


def scroll_viewer(viewer, timeout_s=2.0, label="", full_sweep=False):
    n_slices = viewer.image.shape[viewer.slice_axis]
    if full_sweep:
        indices = list(range(n_slices))
    else:
        indices = [0, 1, n_slices // 2, n_slices - 1, n_slices // 3, 2 * n_slices // 3]
        indices = sorted(set(max(0, min(i, n_slices - 1)) for i in indices))
        indices += list(range(0, min(n_slices, 16)))

    seen = []
    slow = []
    for idx in indices:
        t0 = time.perf_counter()
        viewer.slider.set_val(idx)
        viewer.fig.canvas.draw_idle()
        viewer.fig.canvas.flush_events()
        elapsed = time.perf_counter() - t0
        actual = int(round(viewer.slider.val))
        seen.append((idx, actual, elapsed))
        if elapsed > 0.5:
            slow.append((idx, elapsed))
        if elapsed > timeout_s:
            raise TimeoutError(
                f"{label} slice {idx} took {elapsed:.2f}s (timeout {timeout_s}s); "
                f"slider={actual}"
            )
        if actual != idx:
            raise AssertionError(f"{label} requested slice {idx}, slider at {actual}")

    if slow:
        print(f"  slow slices (>500ms): {slow[:6]}", flush=True)
    return seen


def run_case(name, image, bbox, timeout_s):
    print(f"\n=== {name} ===", flush=True)
    t0 = time.perf_counter()
    viewer = ImageBBoxViewer(image, bbox)
    init_s = time.perf_counter() - t0
    print(f"init {init_s:.2f}s shape={viewer.image.shape} axis={viewer.slice_axis}", flush=True)
    start_val = int(round(viewer.slider.val))
    if start_val != 0:
        raise AssertionError(f"{name} opened at slice {start_val}, expected 0")

    seen = scroll_viewer(
        viewer,
        timeout_s=timeout_s,
        label=name,
        full_sweep=args.full_sweep,
    )
    for idx, actual, elapsed in seen[:8]:
        print(f"  slice {idx:3d} -> {actual:3d} in {elapsed*1000:6.1f}ms", flush=True)
    if len(seen) > 8:
        print(f"  ... {len(seen) - 8} more slices ok", flush=True)

    import matplotlib.pyplot as plt

    plt.close(viewer.fig)
    print(f"{name} PASS", flush=True)


def main(args):
    cases = []

    vol = make_volume(shape=tuple(args.shape), seed=args.seed)
    boxes = make_boxes(vol.shape, n_boxes=args.n_boxes, seed=args.seed)
    cases.append(("numpy", vol, boxes))

    vol_t = torch.from_numpy(vol.copy())
    cases.append(("torch", vol_t, torch.from_numpy(boxes.copy())))

    vol5 = vol[None, None]
    boxes1 = boxes[:1]
    cases.append(("5d_squeezed", vol5, boxes1))

    bad_boxes = np.array(
        [
            [-50, -50, -50, 200, 200, 200],
            [10, 10, 10, 40, 40, 40],
        ],
        dtype=np.float64,
    )
    cases.append(("wide_bbox_coords", vol, bad_boxes))

    monai_vol = make_monai_volume(shape=tuple(args.shape), seed=args.seed)
    if monai_vol is not None:
        cases.append(("monai_metatensor", monai_vol, torch.from_numpy(boxes.copy())))

    large = make_volume(shape=(128, 256, 256), seed=args.seed + 1)
    cases.append(("large_volume", large, make_boxes(large.shape, n_boxes=5, seed=args.seed + 1)))

    failures = []
    for name, image, bbox in cases:
        try:
            run_case(name, image, bbox, timeout_s=args.timeout)
        except Exception as exc:
            failures.append((name, exc))
            print(f"{name} FAIL: {exc}", flush=True)
            traceback.print_exc()

    if failures:
        print("\nFAILED:", ", ".join(n for n, _ in failures), flush=True)
        return 1

    print("\nALL CASES PASS", flush=True)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", type=int, nargs=3, default=[48, 96, 96])
    parser.add_argument("--n-boxes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument("--full-sweep", action="store_true")
    args = parser.parse_known_args()[0]
    sys.exit(main(args))
