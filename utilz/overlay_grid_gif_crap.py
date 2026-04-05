#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

from utilz.grid_helper import (
    _Spinner,
    _iter_progress,
    _load_cases,
    _orientation_axis,
    _panel_frame,
)


def create_nifti_overlay_grid_gif(
    dataset_root: Path | str,
    output_gif: Path | str | None = None,
    grid_shape: Tuple[int, int] = (4, 4),
    preferred_label_name: str = "annotation_staple.nii.gz",
    num_frames: int = 30,
    fps: int = 5,
    stride: int = 3,
    slice_axis: int | None = None,
    window: str = "auto",
    rotate_cw_degrees: int = 0,
    panel_px: int = 500,
    case_ids: Sequence[str] | None = None,
    orientations: Tuple[str, str] = ("axial", "coronal"),
) -> Path:

    dataset_root = Path(dataset_root)

    if output_gif is None:
        output_gif = dataset_root / "dataset_stats" / "snapshot.gif"

    output_gif = Path(output_gif)

    rows, cols = grid_shape
    max_cases = rows * cols

    cases = _load_cases(
        dataset_root,
        preferred_label_name=preferred_label_name,
        max_cases=max_cases,
        slice_axis=slice_axis,
        case_ids=case_ids,
    )

    orientation_axes = [_orientation_axis(name) for name in orientations]

    def _sample_indices(depth: int) -> np.ndarray:
        if depth <= 1:
            return np.zeros((max(1, num_frames),), dtype=np.int32)

        sample_count = max(1, num_frames)

        if stride <= 1:
            idx = np.linspace(0, depth - 1, num=sample_count, endpoint=True).round().astype(np.int32)
        else:
            expanded_count = ((sample_count - 1) * stride) + 1
            expanded = np.linspace(0, depth - 1, num=expanded_count, endpoint=True).round().astype(np.int32)
            idx = expanded[::stride]

            if idx.shape[0] > sample_count:
                idx = idx[:sample_count]
            elif idx.shape[0] < sample_count:
                pad = np.full((sample_count - idx.shape[0],), idx[-1], dtype=np.int32)
                idx = np.concatenate([idx, pad], axis=0)

        idx[0] = 0
        idx[-1] = depth - 1
        return idx

    case_frame_indices: List[Dict[int, np.ndarray]] = []
    for c in cases:
        case_frame_indices.append(
            {axis: _sample_indices(int(c.image.shape[axis])) for axis in orientation_axes}
        )

    panel_px = max(48, int(panel_px))
    title_h = max(20, panel_px // 6)
    line_h = max(9, panel_px // 14)

    panel_h = panel_px
    panel_w = panel_px

    split_widths = [panel_w // len(orientations) for _ in orientations]
    split_widths[-1] = panel_w - sum(split_widths[:-1])

    frame_h = title_h + (rows * panel_h)
    frame_w = cols * panel_w

    output_gif.parent.mkdir(parents=True, exist_ok=True)

    # preallocate
    canvas = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
    frames = []
    font = cv2.FONT_HERSHEY_SIMPLEX

    with _Spinner("Encoding GIF"):
        for frame_idx in _iter_progress(range(num_frames), desc="Rendering GIF frames"):

            canvas.fill(0)

            # title
            cv2.putText(
                canvas,
                f"Overlay Stencil Grid {rows}x{cols} | Cases: {len(cases)}",
                (6, 14),
                font,
                0.4,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            for i, case in enumerate(cases):

                panel_rgb, labels_present, panel_info = _panel_frame(
                    case,
                    frame_idx=frame_idx,
                    case_frame_indices=case_frame_indices[i],
                    orientation_axes=orientation_axes,
                    orientation_names=orientations,
                    split_widths=split_widths,
                    window=window,
                    rotate_cw_degrees=rotate_cw_degrees,
                    panel_hw=(panel_h, panel_w),
                )

                row = i // cols
                col = i % cols

                y0 = title_h + (row * panel_h)
                x0 = col * panel_w

                canvas[y0 : y0 + panel_h, x0 : x0 + panel_w] = panel_rgb

                # text (no second loop)
                cv2.putText(canvas, case.case_id, (x0 + 4, y0 + 14),
                            font, 0.35, (255,255,255), 1, cv2.LINE_AA)

                cv2.putText(canvas, f"Img {i + 1} | {panel_info}",
                            (x0 + 4, y0 + panel_h - 6),
                            font, 0.35, (255,255,255), 1, cv2.LINE_AA)

                max_visible = 10
                for line_idx, label_value in enumerate(labels_present[:max_visible]):
                    y = y0 + 28 + line_idx * line_h
                    cv2.putText(canvas, str(label_value), (x0 + 4, y),
                                font, 0.3, (255,255,255), 1, cv2.LINE_AA)

                if len(labels_present) > max_visible:
                    y = y0 + 28 + max_visible * line_h
                    cv2.putText(canvas, "...", (x0 + 4, y),
                                font, 0.3, (255,255,255), 1, cv2.LINE_AA)

            frames.append(Image.fromarray(canvas.copy()))

    duration = int(1000 / max(fps, 1))

    frames[0].save(
        output_gif,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
        optimize=True,
    )

    return output_gif


def main(args) -> None:
    dataset_root = args.dataset_root if args.dataset_root is not None else args.datafolder

    if dataset_root is None:
        raise SystemExit("dataset_root is required")

    out = create_nifti_overlay_grid_gif(
        dataset_root=dataset_root,
        output_gif=args.output_gif,
        grid_shape=(args.rows, args.cols),
        preferred_label_name=args.preferred_label_name,
        num_frames=args.num_frames,
        fps=args.fps,
        stride=args.stride,
        slice_axis=args.slice_axis,
        window=args.window,
        rotate_cw_degrees=args.rotate_cw,
        panel_px=args.panel_px,
        case_ids=args.case_ids,
        orientations=tuple(args.orientations),
    )

    print(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a 6x6 animated image/label overlay stencil grid GIF.")

    parser.add_argument("datafolder", nargs="?", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--output-gif", type=Path, default=None)
    parser.add_argument("--preferred-label-name", type=str, default="annotation_staple.nii.gz")
    parser.add_argument("--rows", type=int, default=6)
    parser.add_argument("--cols", type=int, default=6)
    parser.add_argument("--num-frames", type=int, default=90)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--panel-px", type=int, default=120)
    parser.add_argument("--case-ids", nargs="*", default=None)
    parser.add_argument("--orientations", nargs=2, default=("axial", "coronal"),
                        choices=["axial", "coronal", "sag", "sagittal"])
    parser.add_argument("--window", type=str, default="auto",
                        choices=["auto", "lung", "abdomen", "bone"])
    parser.add_argument("--rotate-cw", type=int, default=0, choices=[0, 90, 180, 270])
    parser.add_argument("--slice-axis", type=int, default=None, choices=[0, 1, 2])

    main(parser.parse_args())
