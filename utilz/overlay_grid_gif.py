#!/usr/bin/env python3
"""Create animated image+label overlay grids as GIFs from NIfTI or Torch tensors."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import itertools
from pathlib import Path
import sys
import threading
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np
import SimpleITK as sitk
from PIL import Image, ImageDraw
from tqdm import tqdm

from utilz.stringz import cleanup_fname


@dataclass(frozen=True)
class CaseVolume:
    case_id: str
    image: np.ndarray
    label: np.ndarray


SUPPORTED_EXTENSIONS = (".pt", ".nii", ".nii.gz")
ORIENTATION_TO_AXIS = {"axial": 0, "coronal": 1, "sag": 2, "sagittal": 2}
WINDOW_PRESETS: Dict[str, Tuple[float, float]] = {
    "lung": (-600.0, 1500.0),
    "abdomen": (40.0, 400.0),
    "bone": (300.0, 1500.0),
}
LABEL_COLORS_U8 = np.array(
    [
        (255, 26, 26),
        (26, 204, 26),
        (26, 102, 255),
        (255, 178, 26),
        (204, 26, 204),
        (26, 204, 204),
        (255, 255, 26),
        (230, 128, 230),
    ],
    dtype=np.uint8,
)


class _Spinner:
    def __init__(self, message: str) -> None:
        self.message = message
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self) -> None:
        for ch in itertools.cycle("|/-\\"):
            if self._stop.is_set():
                break
            sys.stderr.write(f"\r{self.message} {ch}")
            sys.stderr.flush()
            time.sleep(0.1)
        sys.stderr.write(f"\r{self.message} done\n")
        sys.stderr.flush()

    def __enter__(self) -> "_Spinner":
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)


def _iter_progress(items: Sequence, desc: str, unit: str):
    pbar = tqdm(items, total=len(items), desc=desc, unit=unit)
    for item in pbar:
        if isinstance(item, tuple) and len(item) > 0:
            pbar.set_postfix_str(str(item[0]), refresh=False)
        else:
            pbar.set_postfix_str(str(item), refresh=False)
        yield item


def _supported_files(folder: Path) -> List[Path]:
    files: List[Path] = []
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.name.endswith(SUPPORTED_EXTENSIONS):
            files.append(path)
    return files


def _select_items(
    items: Sequence,
    key_fn,
    case_ids: Sequence[str] | None,
    max_items: int,
) -> List:
    selected: List = []
    if not case_ids:
        for item in items:
            selected.append(item)
            if len(selected) >= max_items:
                break
        return selected

    wanted: List[str] = []
    for case_id in case_ids:
        wanted.append(cleanup_fname(case_id))
    if len(wanted) > max_items:
        raise ValueError(f"Received {len(wanted)} case_ids for only {max_items} panels")

    keyed: List[Tuple[str, object]] = []
    for item in items:
        keyed.append((cleanup_fname(key_fn(item)), item))

    lookup: Dict[str, object] = {}
    for key, item in keyed:
        lookup[key] = item

    missing: List[str] = []
    for key in wanted:
        if key not in lookup:
            missing.append(key)
    if missing:
        raise ValueError(f"Requested case_ids were not found: {missing}")

    seen: set[str] = set()
    for key in wanted:
        selected.append(lookup[key])
        seen.add(key)
    if len(selected) >= max_items:
        return selected[:max_items]

    for key, item in keyed:
        if key in seen:
            continue
        selected.append(item)
        if len(selected) >= max_items:
            break
    return selected


def _pick_case_files(case_dir: Path, preferred_label_name: str) -> Tuple[Path, Path]:
    image_path = None
    for name in ("image.nii.gz", "image.nii", "image.pt", "img.nii.gz", "img.nii", "img.pt"):
        candidate = case_dir / name
        if candidate.exists():
            image_path = candidate
            break

    candidates = _supported_files(case_dir)
    if image_path is None:
        non_label_candidates: List[Path] = []
        for path in candidates:
            name = path.name.lower()
            if "annotation" in name or "label" in name or "mask" in name or "lm" in name or "seg" in name:
                continue
            non_label_candidates.append(path)
        if len(non_label_candidates) == 1:
            image_path = non_label_candidates[0]
        elif len(non_label_candidates) == 0:
            raise FileNotFoundError(f"No supported image file found in {case_dir}")
        else:
            names = [path.name for path in non_label_candidates]
            raise FileNotFoundError(f"Multiple possible image files found in {case_dir}: {names}")

    label_path = case_dir / preferred_label_name
    if not label_path.exists():
        preferred_stem = cleanup_fname(preferred_label_name)
        exact_name_matches: List[Path] = []
        label_candidates: List[Path] = []
        annotation_candidates: List[Path] = []

        for path in candidates:
            cleaned = cleanup_fname(path.name)
            lowered = path.name.lower()
            if cleaned == preferred_stem:
                exact_name_matches.append(path)
            if "annotation" in lowered or "label" in lowered or "mask" in lowered or "lm" in lowered or "seg" in lowered:
                label_candidates.append(path)
            if "annotation" in lowered:
                annotation_candidates.append(path)

        if len(exact_name_matches) == 1:
            label_path = exact_name_matches[0]
        elif len(label_candidates) == 1:
            label_path = label_candidates[0]
        elif len(annotation_candidates) == 1:
            label_path = annotation_candidates[0]
        elif len(label_candidates) == 0:
            raise FileNotFoundError(f"No supported label file found in {case_dir}")
        else:
            names = [path.name for path in label_candidates]
            raise FileNotFoundError(f"Multiple possible label files found in {case_dir}: {names}")

    return image_path, label_path


def _to_numpy(data: object) -> np.ndarray:
    if isinstance(data, np.ndarray):
        return data
    import torch

    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    if hasattr(data, "detach") and hasattr(data, "cpu") and hasattr(data, "numpy"):
        return data.detach().cpu().numpy()
    return np.asarray(data)


def _read_volume(path: Path, slice_axis: int | None, is_label: bool) -> np.ndarray:
    if path.name.endswith((".nii", ".nii.gz")):
        image = sitk.ReadImage(str(path))
        array = sitk.GetArrayFromImage(image)
        if is_label:
            return array.astype(np.int16, copy=False)
        return array.astype(np.float32, copy=False)

    if path.suffix == ".pt":
        import torch
        from det3d.monai.data.meta_tensor import MetaTensor
        from torch.serialization import add_safe_globals

        add_safe_globals([MetaTensor])
        data = torch.load(path, map_location="cpu", weights_only=False)
        volume = _to_numpy(data)
        if volume.ndim == 4:
            if volume.shape[0] <= 32:
                if is_label and volume.shape[0] > 1:
                    volume = np.argmax(volume, axis=0)
                else:
                    volume = volume[0]
            elif volume.shape[-1] <= 32:
                if is_label and volume.shape[-1] > 1:
                    volume = np.argmax(volume, axis=-1)
                else:
                    volume = volume[..., 0]
        if volume.ndim != 3:
            raise ValueError(f"Expected 3D volume, got shape={volume.shape}")
        axis = slice_axis
        if axis is None:
            axis = int(np.argmin(volume.shape))
        volume = np.moveaxis(volume, axis, 0)
        if is_label:
            return volume.astype(np.int16, copy=False)
        return volume.astype(np.float32, copy=False)

    raise ValueError(f"Unsupported file type: {path}")


def _window_normalize_slice(image2d: np.ndarray, window: str | None) -> np.ndarray:
    if not window or window == "auto":
        lo = np.percentile(image2d, 1.0)
        hi = np.percentile(image2d, 99.0)
    else:
        level, width = WINDOW_PRESETS[window]
        lo = level - (width / 2.0)
        hi = level + (width / 2.0)
    if hi <= lo:
        lo = float(image2d.min())
        hi = float(image2d.max())
        if hi <= lo:
            return np.zeros_like(image2d, dtype=np.float32)
    norm = np.clip((image2d - lo) / (hi - lo), 0.0, 1.0)
    return norm.astype(np.float32)


def _apply_pair_transform(arr2d: np.ndarray, pair_index: int, rotate_cw_degrees: int) -> np.ndarray:
    transformed = arr2d
    if pair_index == 0:
        transformed = np.rot90(transformed, k=1)
    elif pair_index == 1:
        transformed = np.flipud(transformed)
    k_ccw = ((rotate_cw_degrees // 90) * -1) % 4
    if k_ccw == 0:
        return transformed
    return np.rot90(transformed, k=k_ccw)


def _resize_2d(arr2d: np.ndarray, target_hw: Tuple[int, int], nearest: bool) -> np.ndarray:
    target_h, target_w = target_hw
    if arr2d.shape == (target_h, target_w):
        return arr2d
    resample = Image.Resampling.NEAREST
    if not nearest:
        resample = Image.Resampling.BILINEAR
    resized = Image.fromarray(arr2d).resize((target_w, target_h), resample=resample)
    return np.asarray(resized)


def _outline_mask(mask2d: np.ndarray, line_thickness: int) -> np.ndarray:
    present = mask2d > 0
    if not np.any(present):
        return np.zeros_like(present, dtype=bool)

    edge = np.zeros_like(present, dtype=bool)
    edge[1:, :] |= mask2d[1:, :] != mask2d[:-1, :]
    edge[:-1, :] |= mask2d[:-1, :] != mask2d[1:, :]
    edge[:, 1:] |= mask2d[:, 1:] != mask2d[:, :-1]
    edge[:, :-1] |= mask2d[:, :-1] != mask2d[:, 1:]
    outline = edge & present

    grow_steps = max(0, int(line_thickness) - 1)
    for _ in range(grow_steps):
        grown = outline.copy()
        grown[1:, :] |= outline[:-1, :]
        grown[:-1, :] |= outline[1:, :]
        grown[:, 1:] |= outline[:, :-1]
        grown[:, :-1] |= outline[:, 1:]
        outline = grown & present
    return outline


def _render_overlay(
    image2d: np.ndarray,
    mask2d: np.ndarray,
    window: str | None,
    pair_index: int,
    rotate_cw_degrees: int,
    target_hw: Tuple[int, int],
    line_thickness: int,
) -> Tuple[np.ndarray, List[int]]:
    image2d = _apply_pair_transform(image2d, pair_index, rotate_cw_degrees)
    mask2d = _apply_pair_transform(mask2d, pair_index, rotate_cw_degrees)

    image2d = _resize_2d(image2d, target_hw=target_hw, nearest=False)
    mask2d = _resize_2d(mask2d, target_hw=target_hw, nearest=True)

    base = _window_normalize_slice(image2d, window)
    base = np.clip(np.round(base * 255.0), 0, 255).astype(np.uint8)
    rgb = np.repeat(base[..., None], 3, axis=2)

    maski = np.rint(mask2d).astype(np.int32)
    labels_present: List[int] = []
    seen: set[int] = set()
    for value in np.unique(maski):
        label_value = int(round(float(value)))
        if label_value <= 0 or label_value in seen:
            continue
        labels_present.append(label_value)
        seen.add(label_value)
    outline = _outline_mask(maski, line_thickness=line_thickness)
    if np.any(outline):
        color_idx = (maski - 1) % len(LABEL_COLORS_U8)
        rgb[outline] = LABEL_COLORS_U8[color_idx[outline]]
    return rgb, labels_present


def _draw_label_legend(
    draw: ImageDraw.ImageDraw,
    x0: int,
    y0: int,
    labels_present: Sequence[int],
    line_h: int,
    max_visible: int,
) -> None:
    swatch = max(6, line_h - 2)
    for row_index, label_value in enumerate(labels_present[:max_visible]):
        y = y0 + (row_index * line_h)
        color = tuple(int(v) for v in LABEL_COLORS_U8[(label_value - 1) % len(LABEL_COLORS_U8)])
        draw.rectangle((x0, y + 1, x0 + swatch, y + swatch + 1), fill=color, outline=(255, 255, 255))
        draw.text((x0 + swatch + 4, y), f"Label {label_value}", fill=(255, 255, 255))
    if len(labels_present) > max_visible:
        draw.text((x0, y0 + (max_visible * line_h)), "...", fill=(255, 255, 255))


def _sample_indices(depth: int, num_frames: int, stride: int) -> np.ndarray:
    if depth <= 1:
        return np.zeros((max(1, num_frames),), dtype=np.int32)

    sample_count = max(1, num_frames)
    if stride <= 1:
        idx = np.linspace(0, depth - 1, num=sample_count, endpoint=True)
        idx = np.rint(idx).astype(np.int32)
    else:
        expanded_count = ((sample_count - 1) * stride) + 1
        expanded = np.linspace(0, depth - 1, num=expanded_count, endpoint=True)
        expanded = np.rint(expanded).astype(np.int32)
        idx = expanded[::stride]
        if idx.shape[0] > sample_count:
            idx = idx[:sample_count]
        elif idx.shape[0] < sample_count:
            pad = np.full((sample_count - idx.shape[0],), idx[-1], dtype=np.int32)
            idx = np.concatenate([idx, pad], axis=0)

    idx[0] = 0
    idx[-1] = depth - 1
    return idx


def _load_cases_from_images_lms(
    dataset_root: Path,
    max_cases: int,
    slice_axis: int | None,
    case_ids: Sequence[str] | None,
) -> List[CaseVolume]:
    images_dir = dataset_root / "images"
    lms_dir = dataset_root / "lms"
    if not images_dir.is_dir() or not lms_dir.is_dir():
        raise RuntimeError(f"Expected images/ and lms/ under {dataset_root}")

    label_by_id: Dict[str, Path] = {}
    for label_path in _supported_files(lms_dir):
        label_by_id[cleanup_fname(label_path.name)] = label_path

    pairs: List[Tuple[str, Path, Path]] = []
    for image_path in _supported_files(images_dir):
        case_id = cleanup_fname(image_path.name)
        if case_id in label_by_id:
            pairs.append((case_id, image_path, label_by_id[case_id]))

    if not pairs:
        raise RuntimeError(f"No matched image/label pairs found in {images_dir} and {lms_dir}")

    pairs = _select_items(pairs, key_fn=lambda pair: pair[0], case_ids=case_ids, max_items=max_cases)

    cases: List[CaseVolume] = []
    for case_id, image_path, label_path in _iter_progress(pairs, desc="Loading image/lm pairs", unit="pair"):
        image = _read_volume(image_path, slice_axis=slice_axis, is_label=False)
        label = _read_volume(label_path, slice_axis=slice_axis, is_label=True)
        if image.shape != label.shape:
            raise ValueError(f"Shape mismatch in {case_id}: image={image.shape}, label={label.shape}")
        cases.append(CaseVolume(case_id=case_id, image=image, label=label))
    return cases


def _load_cases_from_case_dirs(
    dataset_root: Path,
    preferred_label_name: str,
    max_cases: int,
    slice_axis: int | None,
    case_ids: Sequence[str] | None,
) -> List[CaseVolume]:
    case_dirs: List[Path] = []
    for path in sorted(dataset_root.iterdir()):
        if path.is_dir():
            case_dirs.append(path)

    case_dirs = _select_items(case_dirs, key_fn=lambda path: path.name, case_ids=case_ids, max_items=max_cases)

    cases: List[CaseVolume] = []
    for case_dir in _iter_progress(case_dirs, desc="Loading case folders", unit="case"):
        image_path, label_path = _pick_case_files(case_dir, preferred_label_name)
        image = _read_volume(image_path, slice_axis=slice_axis, is_label=False)
        label = _read_volume(label_path, slice_axis=slice_axis, is_label=True)
        if image.shape != label.shape:
            raise ValueError(f"Shape mismatch in {case_dir.name}: image={image.shape}, label={label.shape}")
        cases.append(CaseVolume(case_id=case_dir.name, image=image, label=label))

    if not cases:
        raise RuntimeError(f"No usable cases found under {dataset_root}")
    return cases


def _load_cases(
    dataset_root: Path,
    preferred_label_name: str,
    max_cases: int,
    slice_axis: int | None,
    case_ids: Sequence[str] | None,
) -> List[CaseVolume]:
    images_dir = dataset_root / "images"
    lms_dir = dataset_root / "lms"
    if images_dir.is_dir() and lms_dir.is_dir():
        return _load_cases_from_images_lms(
            dataset_root=dataset_root,
            max_cases=max_cases,
            slice_axis=slice_axis,
            case_ids=case_ids,
        )
    return _load_cases_from_case_dirs(
        dataset_root=dataset_root,
        preferred_label_name=preferred_label_name,
        max_cases=max_cases,
        slice_axis=slice_axis,
        case_ids=case_ids,
    )


def _render_panel(
    case: CaseVolume,
    frame_idx: int,
    frame_indices: Dict[int, np.ndarray],
    orientations: Sequence[str],
    split_widths: Sequence[int],
    window: str,
    rotate_cw_degrees: int,
    panel_hw: Tuple[int, int],
    line_thickness: int,
) -> Tuple[np.ndarray, List[int], str]:
    panel_h, _ = panel_hw
    parts: List[np.ndarray] = []
    labels_seen: set[int] = set()
    info_parts: List[str] = []

    for pair_index, orientation_name in enumerate(orientations):
        axis = ORIENTATION_TO_AXIS[orientation_name.lower()]
        slice_idx = int(frame_indices[axis][frame_idx])

        image2d = np.take(case.image, indices=slice_idx, axis=axis)
        mask2d = np.take(case.label, indices=slice_idx, axis=axis)

        rendered, labels_present = _render_overlay(
            image2d=image2d,
            mask2d=mask2d,
            window=window,
            pair_index=pair_index,
            rotate_cw_degrees=rotate_cw_degrees,
            target_hw=(panel_h, split_widths[pair_index]),
            line_thickness=line_thickness,
        )
        parts.append(rendered)

        for label_value in labels_present:
            labels_seen.add(label_value)

        axis_len = case.image.shape[axis]
        info_parts.append(f"{orientation_name[0].upper()} {slice_idx + 1}/{axis_len}")

    return np.concatenate(parts, axis=1), sorted(labels_seen), " | ".join(info_parts)


def _render_grid_frames(
    cases: Sequence[CaseVolume],
    case_frame_indices: Sequence[Dict[int, np.ndarray]],
    orientations: Sequence[str],
    window: str,
    rotate_cw_degrees: int,
    rows: int,
    cols: int,
    num_frames: int,
    panel_px: int,
    line_thickness: int,
) -> List["Image.Image"]:
    title_h = max(20, panel_px // 6)
    line_h = max(9, panel_px // 14)
    panel_h = panel_px
    panel_w = panel_px

    split_widths: List[int] = []
    for _ in orientations:
        split_widths.append(panel_w // len(orientations))
    split_widths[-1] = panel_w - sum(split_widths[:-1])

    frames: List[Image.Image] = []
    for frame_idx in _iter_progress(range(num_frames), desc="Rendering GIF frames", unit="frame"):
        canvas = np.zeros((title_h + (rows * panel_h), cols * panel_w, 3), dtype=np.uint8)
        frame_text: List[Tuple[int, List[int], str, str]] = []

        for case_index, case in enumerate(cases):
            panel_rgb, labels_present, panel_info = _render_panel(
                case=case,
                frame_idx=frame_idx,
                frame_indices=case_frame_indices[case_index],
                orientations=orientations,
                split_widths=split_widths,
                window=window,
                rotate_cw_degrees=rotate_cw_degrees,
                panel_hw=(panel_h, panel_w),
                line_thickness=line_thickness,
            )
            row = case_index // cols
            col = case_index % cols
            y0 = title_h + (row * panel_h)
            x0 = col * panel_w
            canvas[y0 : y0 + panel_h, x0 : x0 + panel_w, :] = panel_rgb
            frame_text.append((case_index, labels_present, panel_info, case.case_id))

        frame_im = Image.fromarray(canvas, mode="RGB")
        draw = ImageDraw.Draw(frame_im)
        draw.text((6, 4), f"Overlay Stencil Grid {rows}x{cols} | Cases: {len(cases)}", fill=(255, 255, 255))

        for case_index, labels_present, panel_info, case_id in frame_text:
            row = case_index // cols
            col = case_index % cols
            y0 = title_h + (row * panel_h)
            x0 = col * panel_w

            draw.text((x0 + 4, y0 + 3), case_id, fill=(255, 255, 255))
            draw.text((x0 + 4, y0 + panel_h - (2 * line_h)), f"Img {case_index + 1} | {panel_info}", fill=(255, 255, 255))

            max_visible = max(1, (panel_h - 24 - (3 * line_h)) // line_h)
            _draw_label_legend(
                draw=draw,
                x0=x0 + 4,
                y0=y0 + 16,
                labels_present=labels_present,
                line_h=line_h,
                max_visible=max_visible,
            )

        frames.append(frame_im)

    return frames


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
    line_thickness: int = 1,
    gif_colors: int = 96,
    case_ids: Sequence[str] | None = None,
    orientations: Tuple[str, str] = ("axial", "coronal"),
) -> Path:
    dataset_root = Path(dataset_root)
    if output_gif is None:
        output_gif = dataset_root / "dataset_stats" / "snapshot.gif"
    output_gif = Path(output_gif)

    rows, cols = grid_shape
    max_cases = rows * cols
    panel_px = max(48, int(panel_px))
    line_thickness = max(1, int(line_thickness))
    _ = gif_colors

    cases = _load_cases(
        dataset_root=dataset_root,
        preferred_label_name=preferred_label_name,
        max_cases=max_cases,
        slice_axis=slice_axis,
        case_ids=case_ids,
    )
    case_frame_indices: List[Dict[int, np.ndarray]] = []
    for case in cases:
        per_case: Dict[int, np.ndarray] = {}
        for orientation_name in orientations:
            axis = ORIENTATION_TO_AXIS[orientation_name.lower()]
            per_case[axis] = _sample_indices(
                depth=int(case.image.shape[axis]),
                num_frames=num_frames,
                stride=stride,
            )
        case_frame_indices.append(per_case)

    output_gif.parent.mkdir(parents=True, exist_ok=True)
    frames = _render_grid_frames(
        cases=cases,
        case_frame_indices=case_frame_indices,
        orientations=orientations,
        window=window,
        rotate_cw_degrees=rotate_cw_degrees,
        rows=rows,
        cols=cols,
        num_frames=num_frames,
        panel_px=panel_px,
        line_thickness=line_thickness,
    )

    with _Spinner("Encoding GIF"):
        frame_duration_ms = int(round(1000 / max(fps, 1)))
        frames[0].save(
            output_gif,
            save_all=True,
            append_images=frames[1:],
            duration=frame_duration_ms,
            loop=0,
            disposal=2,
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
        gif_colors=args.gif_colors,
        case_ids=args.case_ids,
        orientations=tuple(args.orientations),
    )
    print(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a 6x6 animated image/label overlay stencil grid GIF.")
    parser.add_argument(
        "datafolder",
        nargs="?",
        type=Path,
        help="Dataset folder. Same as --dataset-root.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        help="Path with case subfolders or with images/ and lms/ subfolders. Supports .nii.gz, .nii, and .pt.",
    )
    parser.add_argument("--output-gif", type=Path, default=None, help="Output GIF path.")
    parser.add_argument(
        "--preferred-label-name",
        type=str,
        default="annotation_staple.nii.gz",
        help="Preferred annotation filename inside each case folder. The same basename is matched across .nii.gz, .nii, and .pt.",
    )
    parser.add_argument("--rows", type=int, default=6, help="Grid rows.")
    parser.add_argument("--cols", type=int, default=6, help="Grid cols.")
    parser.add_argument("--num-frames", type=int, default=90, help="Total GIF frames.")
    parser.add_argument("--fps", type=int, default=10, help="GIF frame rate.")
    parser.add_argument("--stride", type=int, default=1, help="Slice step multiplier per frame.")
    parser.add_argument("--panel-px", type=int, default=120, help="Target pixels per panel side in output GIF.")
    parser.add_argument("--gif-colors", type=int, default=96, help="Palette size for GIF optimization (2-256).")
    parser.add_argument(
        "--case-ids",
        nargs="*",
        default=None,
        help="Optional case IDs to prioritize first when filling grid panels.",
    )
    parser.add_argument(
        "--orientations",
        nargs=2,
        default=("axial", "coronal"),
        choices=["axial", "coronal", "sag", "sagittal"],
        help="Two orientations rendered side by side inside each grid panel.",
    )
    parser.add_argument(
        "--window",
        type=str,
        default="auto",
        choices=["auto", "lung", "abdomen", "bone"],
        help="Intensity window preset.",
    )
    parser.add_argument(
        "--rotate-cw",
        type=int,
        default=0,
        choices=[0, 90, 180, 270],
        help="Rotate each rendered frame clockwise.",
    )
    parser.add_argument(
        "--slice-axis",
        type=int,
        default=None,
        choices=[0, 1, 2],
        help="Depth axis for tensor inputs. Default infers the smallest axis.",
    )
    main(parser.parse_args())
