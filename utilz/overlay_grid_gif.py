#!/usr/bin/env python3
"""Create animated NIfTI image+label overlay grids as GIFs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import matplotlib
import numpy as np
import SimpleITK as sitk


@dataclass(frozen=True)
class CaseVolume:
    case_id: str
    image: np.ndarray  # z, y, x
    label: np.ndarray  # z, y, x


def _list_case_dirs(dataset_root: Path) -> List[Path]:
    return sorted([p for p in dataset_root.iterdir() if p.is_dir()])


def _pick_label_file(case_dir: Path, preferred_label_name: str) -> Path:
    preferred = case_dir / preferred_label_name
    if preferred.exists():
        return preferred
    candidates = sorted(case_dir.glob("annotation*.nii.gz"))
    if not candidates:
        raise FileNotFoundError(f"No annotation*.nii.gz found in {case_dir}")
    return candidates[0]


def _read_zyx(path: Path) -> np.ndarray:
    img = sitk.ReadImage(str(path))
    return sitk.GetArrayFromImage(img)


def _normalize_slice(x: np.ndarray) -> np.ndarray:
    lo = np.percentile(x, 1.0)
    hi = np.percentile(x, 99.0)
    if hi <= lo:
        lo = float(x.min())
        hi = float(x.max())
        if hi <= lo:
            return np.zeros_like(x, dtype=np.float32)
    y = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return y.astype(np.float32)


def _stencil_edges(mask2d: np.ndarray) -> np.ndarray:
    m = mask2d > 0
    if not np.any(m):
        return np.zeros_like(m, dtype=bool)
    edge = np.zeros_like(m, dtype=bool)
    edge[1:, :] |= m[1:, :] != m[:-1, :]
    edge[:-1, :] |= m[:-1, :] != m[1:, :]
    edge[:, 1:] |= m[:, 1:] != m[:, :-1]
    edge[:, :-1] |= m[:, :-1] != m[:, 1:]
    edge &= m
    return edge


def _overlay_stencil_rgb(
    image2d: np.ndarray,
    mask2d: np.ndarray,
    color: Tuple[float, float, float] = (1.0, 0.1, 0.1),
) -> np.ndarray:
    base = _normalize_slice(image2d)
    rgb = np.stack([base, base, base], axis=-1)
    edge = _stencil_edges(mask2d)
    rgb[edge, 0] = color[0]
    rgb[edge, 1] = color[1]
    rgb[edge, 2] = color[2]
    return rgb


def _load_cases(
    dataset_root: Path,
    preferred_label_name: str,
    max_cases: int,
) -> List[CaseVolume]:
    cases: List[CaseVolume] = []
    for case_dir in _list_case_dirs(dataset_root):
        img_path = case_dir / "image.nii.gz"
        if not img_path.exists():
            continue
        label_path = _pick_label_file(case_dir, preferred_label_name)
        image = _read_zyx(img_path)
        label = _read_zyx(label_path)
        if image.shape != label.shape:
            raise ValueError(
                f"Shape mismatch in {case_dir.name}: image={image.shape}, label={label.shape}"
            )
        cases.append(CaseVolume(case_id=case_dir.name, image=image, label=label))
        if len(cases) >= max_cases:
            break
    if not cases:
        raise RuntimeError(f"No usable cases found under {dataset_root}")
    return cases


def create_nifti_overlay_grid_gif(
    dataset_root: Path | str,
    output_gif: Path | str,
    grid_shape: Tuple[int, int] = (6, 6),
    preferred_label_name: str = "annotation_staple.nii.gz",
    num_frames: int = 90,
    fps: int = 10,
    stride: int = 1,
) -> Path:
    """
    Create an animated GIF grid where each panel scrolls its own z-slices.

    Args:
        dataset_root: Folder containing case subfolders with image/annotation NIfTI files.
        output_gif: Destination GIF path.
        grid_shape: Grid rows, cols.
        preferred_label_name: Annotation file preferred per case.
        num_frames: Total animation frames.
        fps: GIF frame rate.
        stride: Slice step per frame.
    """
    # Avoid forcing Agg at module-import time; only use it for headless GIF rendering.
    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        matplotlib.use("Agg", force=True)

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    dataset_root = Path(dataset_root)
    output_gif = Path(output_gif)
    rows, cols = grid_shape
    max_cases = rows * cols
    cases = _load_cases(dataset_root, preferred_label_name, max_cases=max_cases)

    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows), constrained_layout=True)
    ax_list = np.atleast_1d(axes).ravel()

    artists = []
    for i, ax in enumerate(ax_list):
        ax.set_axis_off()
        if i < len(cases):
            c = cases[i]
            z = 0
            frame = _overlay_stencil_rgb(c.image[z], c.label[z])
            im = ax.imshow(frame, interpolation="nearest")
            ax.set_title(c.case_id, fontsize=7)
            artists.append(im)
        else:
            artists.append(None)

    fig.suptitle(
        f"Overlay Stencil Grid {rows}x{cols} | Cases: {len(cases)} | Label: {preferred_label_name}",
        fontsize=12,
    )

    def update(frame_idx: int):
        for i, im in enumerate(artists):
            if im is None:
                continue
            c = cases[i]
            depth = int(c.image.shape[0])
            if depth == 0:
                continue
            speed = 1 + (i % 5)
            z = (frame_idx * stride * speed + i) % depth
            out = _overlay_stencil_rgb(c.image[z], c.label[z])
            im.set_data(out)
        return [im for im in artists if im is not None]

    anim = FuncAnimation(fig, update, frames=num_frames, interval=int(1000 / max(fps, 1)), blit=False)
    output_gif.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(output_gif), writer=PillowWriter(fps=fps))
    plt.close(fig)
    return output_gif


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Create a 6x6 animated NIfTI overlay stencil grid GIF.")
    p.add_argument("--dataset-root", type=Path, required=True, help="Path with case subfolders.")
    p.add_argument("--output-gif", type=Path, required=True, help="Output GIF path.")
    p.add_argument(
        "--preferred-label-name",
        type=str,
        default="annotation_staple.nii.gz",
        help="Preferred annotation filename inside each case folder.",
    )
    p.add_argument("--rows", type=int, default=6, help="Grid rows.")
    p.add_argument("--cols", type=int, default=6, help="Grid cols.")
    p.add_argument("--num-frames", type=int, default=90, help="Total GIF frames.")
    p.add_argument("--fps", type=int, default=10, help="GIF frame rate.")
    p.add_argument("--stride", type=int, default=1, help="Slice step multiplier per frame.")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    out = create_nifti_overlay_grid_gif(
        dataset_root=args.dataset_root,
        output_gif=args.output_gif,
        grid_shape=(args.rows, args.cols),
        preferred_label_name=args.preferred_label_name,
        num_frames=args.num_frames,
        fps=args.fps,
        stride=args.stride,
    )
    print(out)


if __name__ == "__main__":
    main()
