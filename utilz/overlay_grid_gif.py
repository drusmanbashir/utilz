#!/usr/bin/env python3
"""Create animated image+label overlay grids as GIFs from NIfTI or Torch tensors."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import itertools
import os
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
    image: np.ndarray  # z, y, x
    label: np.ndarray  # z, y, x


SUPPORTED_EXTENSIONS = (".pt", ".nii", ".nii.gz")
ORIENTATION_TO_AXIS = {"axial": 0, "coronal": 1, "sag": 2, "sagittal": 2}
WINDOW_PRESETS: Dict[str, Tuple[float, float]] = {
    "lung": (-600.0, 1500.0),
    "abdomen": (40.0, 400.0),
    "bone": (300.0, 1500.0),
}
LABEL_COLORS: Tuple[Tuple[float, float, float], ...] = (
    (1.0, 0.1, 0.1),   # red
    (0.1, 0.8, 0.1),   # green
    (0.1, 0.4, 1.0),   # blue
    (1.0, 0.7, 0.1),   # orange
    (0.8, 0.1, 0.8),   # magenta
    (0.1, 0.8, 0.8),   # cyan
    (1.0, 1.0, 0.1),   # yellow
    (0.9, 0.5, 0.9),   # pink
)
LABEL_COLORS_U8 = np.array([(255, 26, 26), (26, 204, 26), (26, 102, 255), (255, 178, 26), (204, 26, 204), (26, 204, 204), (255, 255, 26), (230, 128, 230)], dtype=np.uint8)


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


def _iter_progress(items: Sequence, desc: str):
    pbar = tqdm(items, total=len(items), desc=desc, unit="pair")
    for item in pbar:
        case_name = str(item[0]) if isinstance(item, tuple) and item else "case"
        pbar.set_postfix_str(case_name, refresh=False)
        yield item


def _list_case_dirs(dataset_root: Path) -> List[Path]:
    return sorted([p for p in dataset_root.iterdir() if p.is_dir()])


def _pick_first_existing(case_dir: Path, names: Sequence[str]) -> Path | None:
    for name in names:
        candidate = case_dir / name
        if candidate.exists():
            return candidate
    return None


def _pick_case_image_file(case_dir: Path) -> Path:
    preferred = _pick_first_existing(
        case_dir,
        (
            "image.nii.gz",
            "image.nii",
            "image.pt",
            "img.nii.gz",
            "img.nii",
            "img.pt",
        ),
    )
    if preferred is not None:
        return preferred

    candidates = _list_supported_files(case_dir)
    non_label_candidates = [
        p for p in candidates if not any(tok in p.name.lower() for tok in ("annotation", "label", "mask", "lm", "seg"))
    ]
    if len(non_label_candidates) == 1:
        return non_label_candidates[0]
    if not non_label_candidates:
        raise FileNotFoundError(f"No supported image file found in {case_dir}")
    raise FileNotFoundError(
        f"Multiple possible image files found in {case_dir}: {[p.name for p in non_label_candidates]}"
    )


def _pick_label_file(case_dir: Path, preferred_label_name: str) -> Path:
    preferred = case_dir / preferred_label_name
    if preferred.exists():
        return preferred

    preferred_stem = cleanup_fname(preferred_label_name)
    candidates = _list_supported_files(case_dir)
    for candidate in candidates:
        if cleanup_fname(candidate.name) == preferred_stem:
            return candidate

    label_candidates = [
        p for p in candidates if any(tok in p.name.lower() for tok in ("annotation", "label", "mask", "lm", "seg"))
    ]
    if len(label_candidates) == 1:
        return label_candidates[0]
    if not label_candidates:
        raise FileNotFoundError(f"No supported label file found in {case_dir}")
    annotation_candidates = [p for p in label_candidates if "annotation" in p.name.lower()]
    if len(annotation_candidates) == 1:
        return annotation_candidates[0]
    raise FileNotFoundError(
        f"Multiple possible label files found in {case_dir}: {[p.name for p in label_candidates]}"
    )


def _base_stem(path: Path) -> str:
    return cleanup_fname(path.name)


def _list_supported_files(folder: Path) -> List[Path]:
    return sorted([p for p in folder.iterdir() if p.is_file() and p.name.endswith(SUPPORTED_EXTENSIONS)])


def _normalize_case_id(case_id: str) -> str:
    return cleanup_fname(case_id)


def _prioritize_items(
    items: Sequence,
    key_fn,
    case_ids: Sequence[str] | None,
    max_items: int,
) -> List:
    if not case_ids:
        return list(items[:max_items])

    normalized_requested = [_normalize_case_id(case_id) for case_id in case_ids]
    if len(normalized_requested) > max_items:
        raise ValueError(f"Received {len(normalized_requested)} case_ids for only {max_items} panels")

    keyed_items = [(_normalize_case_id(key_fn(item)), item) for item in items]
    keyed_lookup = {case_id: item for case_id, item in keyed_items}
    missing = [case_id for case_id in normalized_requested if case_id not in keyed_lookup]
    if missing:
        raise ValueError(f"Requested case_ids were not found: {missing}")

    ordered: List = []
    seen: set[str] = set()
    for case_id in normalized_requested:
        ordered.append(keyed_lookup[case_id])
        seen.add(case_id)
    for case_id, item in keyed_items:
        if case_id in seen:
            continue
        ordered.append(item)
        if len(ordered) >= max_items:
            break
    return ordered[:max_items]


def _to_numpy(x: object) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    import torch
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if hasattr(x, "detach") and hasattr(x, "cpu") and hasattr(x, "numpy"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _normalize_to_3d(arr: np.ndarray, *, is_label: bool = False) -> np.ndarray:
    x = np.asarray(arr)
    if x.ndim == 3:
        return x
    if x.ndim == 4:
        # Channel-first [C, D, H, W]
        if x.shape[0] <= 32:
            if is_label and x.shape[0] > 1:
                return np.argmax(x, axis=0)
            return x[0]
        # Channel-last [D, H, W, C]
        if x.shape[-1] <= 32:
            if is_label and x.shape[-1] > 1:
                return np.argmax(x, axis=-1)
            return x[..., 0]
    raise ValueError(f"Expected 3D volume, got shape={x.shape}")


def _infer_slice_axis(vol3d: np.ndarray, slice_axis: int | None) -> int:
    if slice_axis is not None:
        return int(slice_axis)
    return int(np.argmin(vol3d.shape))


def _read_zyx(path: Path, slice_axis: int | None, *, is_label: bool = False) -> np.ndarray:
    if path.name.endswith((".nii", ".nii.gz")):
        img = sitk.ReadImage(str(path))
        arr = sitk.GetArrayFromImage(img)
        return arr.astype(np.int16 if is_label else np.float32, copy=False)
    if path.suffix == ".pt":
        import torch
        from monai.data.meta_tensor import MetaTensor
        from torch.serialization import add_safe_globals
        add_safe_globals([MetaTensor])
        data = torch.load(path, map_location="cpu", weights_only=False)
        vol3d = _normalize_to_3d(_to_numpy(data), is_label=is_label)
        axis = _infer_slice_axis(vol3d, slice_axis=slice_axis)
        out = np.moveaxis(vol3d, axis, 0)
        return out.astype(np.int16 if is_label else np.float32, copy=False)
    raise ValueError(f"Unsupported file type: {path}")


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


def _window_normalize_slice(x: np.ndarray, window: str | None) -> np.ndarray:
    if not window or window == "auto":
        return _normalize_slice(x)
    level, width = WINDOW_PRESETS[window]
    lo = level - (width / 2.0)
    hi = level + (width / 2.0)
    if hi <= lo:
        return _normalize_slice(x)
    y = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return y.astype(np.float32)


def _rotate_clockwise_2d(arr2d: np.ndarray, rotate_cw_degrees: int) -> np.ndarray:
    k_ccw = ((rotate_cw_degrees // 90) * -1) % 4
    if k_ccw == 0:
        return arr2d
    return np.rot90(arr2d, k=k_ccw)


def _apply_pair_transform(arr2d: np.ndarray, pair_index: int, rotate_cw_degrees: int) -> np.ndarray:
    # Pair-specific transforms requested by the user:
    # panel 1 -> 90 degrees anticlockwise, panel 2 -> vertical flip.
    if pair_index == 0:
        arr2d = np.rot90(arr2d, k=1)
    elif pair_index == 1:
        arr2d = np.flipud(arr2d)
    return _rotate_clockwise_2d(arr2d, rotate_cw_degrees=rotate_cw_degrees)


def _resize_2d(arr2d: np.ndarray, target_hw: Tuple[int, int], *, nearest: bool) -> np.ndarray:
    target_h, target_w = target_hw
    if arr2d.shape == (target_h, target_w):
        return arr2d
    resample = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
    out = Image.fromarray(arr2d).resize((target_w, target_h), resample=resample)
    return np.asarray(out)


def _stencil_edges(mask2d: np.ndarray) -> np.ndarray:
    m = mask2d > 0
    if not np.any(m):
        return np.zeros_like(m, dtype=bool)
    edge = np.zeros_like(m, dtype=bool)
    edge[1:, :] |= mask2d[1:, :] != mask2d[:-1, :]
    edge[:-1, :] |= mask2d[:-1, :] != mask2d[1:, :]
    edge[:, 1:] |= mask2d[:, 1:] != mask2d[:, :-1]
    edge[:, :-1] |= mask2d[:, :-1] != mask2d[:, 1:]
    edge &= m
    edge = _thicken_mask(edge, iterations=1)
    return edge


def _thicken_mask(mask2d: np.ndarray, iterations: int = 1) -> np.ndarray:
    out = mask2d.astype(bool, copy=True)
    for _ in range(max(0, iterations)):
        grown = out.copy()
        grown[1:, :] |= out[:-1, :]
        grown[:-1, :] |= out[1:, :]
        grown[:, 1:] |= out[:, :-1]
        grown[:, :-1] |= out[:, 1:]
        grown[1:, 1:] |= out[:-1, :-1]
        grown[1:, :-1] |= out[:-1, 1:]
        grown[:-1, 1:] |= out[1:, :-1]
        grown[:-1, :-1] |= out[1:, 1:]
        out = grown
    return out


def _label_color(label_value: int) -> Tuple[float, float, float]:
    return LABEL_COLORS[(int(label_value) - 1) % len(LABEL_COLORS)]


def _labels_present(mask2d: np.ndarray) -> List[int]:
    vals = np.unique(mask2d)
    labels: List[int] = []
    seen: set[int] = set()
    for v in vals:
        fv = float(v)
        if fv <= 0:
            continue
        iv = int(round(fv))
        if iv <= 0 or iv in seen:
            continue
        seen.add(iv)
        labels.append(iv)
    return labels


def _overlay_stencil_rgb(
    image2d: np.ndarray,
    mask2d: np.ndarray,
    window: str | None = "auto",
    pair_index: int = 0,
    rotate_cw_degrees: int = 0,
    target_hw: Tuple[int, int] | None = None,
) -> Tuple[np.ndarray, List[int]]:
    image2d = _apply_pair_transform(image2d, pair_index=pair_index, rotate_cw_degrees=rotate_cw_degrees)
    mask2d = _apply_pair_transform(mask2d, pair_index=pair_index, rotate_cw_degrees=rotate_cw_degrees)
    if target_hw is not None:
        image2d = _resize_2d(image2d, target_hw=target_hw, nearest=False)
        mask2d = _resize_2d(mask2d, target_hw=target_hw, nearest=True)
    base = np.clip(np.round(_window_normalize_slice(image2d, window=window) * 255.0), 0, 255).astype(np.uint8)
    rgb = np.repeat(base[..., None], 3, axis=2)
    maski = np.rint(mask2d).astype(np.int32)
    labels_present = _labels_present(maski)
    edge = _stencil_edges(maski)
    if np.any(edge):
        color_idx = (maski - 1) % len(LABEL_COLORS_U8)
        rgb[edge] = LABEL_COLORS_U8[color_idx[edge]]
    return rgb, labels_present


def _orientation_axis(name: str) -> int:
    return ORIENTATION_TO_AXIS[name.lower()]


def _extract_slice(volume_zyx: np.ndarray, axis: int, index: int) -> np.ndarray:
    return np.take(volume_zyx, indices=index, axis=axis)


def _panel_frame(
    case: CaseVolume,
    frame_idx: int,
    case_frame_indices: Dict[int, np.ndarray],
    orientation_axes: Sequence[int],
    orientation_names: Sequence[str],
    split_widths: Sequence[int],
    window: str,
    rotate_cw_degrees: int,
    panel_hw: Tuple[int, int],
) -> Tuple[np.ndarray, List[int], str]:
    panel_parts: List[np.ndarray] = []
    labels_seen: set[int] = set()
    info_parts: List[str] = []
    panel_h, panel_w = panel_hw
    for pair_index, axis in enumerate(orientation_axes):
        indices = case_frame_indices[axis]
        slice_idx = int(indices[frame_idx])
        image2d = _extract_slice(case.image, axis=axis, index=slice_idx)
        mask2d = _extract_slice(case.label, axis=axis, index=slice_idx)
        rendered, labels_present = _overlay_stencil_rgb(
            image2d,
            mask2d,
            window=window,
            pair_index=pair_index,
            rotate_cw_degrees=rotate_cw_degrees,
            target_hw=(panel_h, split_widths[pair_index]),
        )
        panel_parts.append(rendered)
        labels_seen.update(labels_present)
        info_parts.append(f"{orientation_names[pair_index][0].upper()} {slice_idx + 1}/{case.image.shape[axis]}")
    return np.concatenate(panel_parts, axis=1), sorted(labels_seen), " | ".join(info_parts)


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
) -> List["Image.Image"]:
    title_h = max(20, panel_px // 6)
    line_h = max(9, panel_px // 14)
    panel_h = panel_px
    panel_w = panel_px
    frames: List[Image.Image] = []
    worker_count = min(len(cases), max(1, os.cpu_count() or 1))
    orientation_axes = [_orientation_axis(name) for name in orientations]
    split_widths = [panel_w // len(orientations) for _ in orientations]
    split_widths[-1] = panel_w - sum(split_widths[:-1])

    def _render_case_panel(i: int) -> Tuple[int, np.ndarray, List[int], str, str]:
        case = cases[i]
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
        return i, panel_rgb, labels_present, panel_info, case.case_id

    with ThreadPoolExecutor(max_workers=worker_count) as ex:
        for frame_idx in _iter_progress(range(num_frames), desc="Rendering GIF frames"):
            canvas = np.zeros((title_h + (rows * panel_h), cols * panel_w, 3), dtype=np.uint8)
            panel_outputs = list(ex.map(_render_case_panel, range(len(cases))))
            panel_text: List[Tuple[int, List[int], str, str]] = []
            for i, panel_rgb_u8, labels_present, panel_info, case_id in panel_outputs:
                row = i // cols
                col = i % cols
                y0 = title_h + (row * panel_h)
                x0 = col * panel_w
                canvas[y0 : y0 + panel_h, x0 : x0 + panel_w, :] = panel_rgb_u8
                panel_text.append((i, labels_present, panel_info, case_id))
            frame_im = Image.fromarray(canvas, mode="RGB")
            draw = ImageDraw.Draw(frame_im)
            draw.text((6, 4), f"Overlay Stencil Grid {rows}x{cols} | Cases: {len(cases)}", fill=(255, 255, 255))
            for i, labels_present, panel_info, case_id in panel_text:
                row = i // cols
                col = i % cols
                y0 = title_h + (row * panel_h)
                x0 = col * panel_w
                draw.text((x0 + 4, y0 + 3), case_id, fill=(255, 255, 255))
                draw.text((x0 + 4, y0 + panel_h - (2 * line_h)), f"Img {i + 1} | {panel_info}", fill=(255, 255, 255))
                max_visible = 10
                for line_idx, label_value in enumerate(labels_present[:max_visible]):
                    color = tuple(int(round(c * 255.0)) for c in _label_color(label_value))
                    draw.text((x0 + 4, y0 + 16 + (line_idx * line_h)), f"{label_value}", fill=color)
                if len(labels_present) > max_visible:
                    draw.text((x0 + 4, y0 + 16 + (max_visible * line_h)), "...", fill=(255, 255, 255))
            frames.append(frame_im)
    return frames


def _load_cases_from_case_dirs(
    dataset_root: Path,
    preferred_label_name: str,
    max_cases: int,
    slice_axis: int | None,
    case_ids: Sequence[str] | None,
) -> List[CaseVolume]:
    cases: List[CaseVolume] = []
    case_dirs = _list_case_dirs(dataset_root)
    case_dirs = _prioritize_items(case_dirs, key_fn=lambda path: path.name, case_ids=case_ids, max_items=max_cases)
    for case_dir in _iter_progress(case_dirs, desc="Loading case folders"):
        img_path = _pick_case_image_file(case_dir)
        label_path = _pick_label_file(case_dir, preferred_label_name)
        image = _read_zyx(img_path, slice_axis=slice_axis, is_label=False)
        label = _read_zyx(label_path, slice_axis=slice_axis, is_label=True)
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


def _match_flat_pairs(images_dir: Path, lms_dir: Path) -> List[Tuple[str, Path, Path]]:
    image_files = _list_supported_files(images_dir)
    label_files = _list_supported_files(lms_dir)
    label_by_id: Dict[str, Path] = {_base_stem(p): p for p in label_files}
    pairs: List[Tuple[str, Path, Path]] = []
    for image_path in image_files:
        case_id = _base_stem(image_path)
        label_path = label_by_id.get(case_id)
        if label_path is None:
            continue
        pairs.append((case_id, image_path, label_path))
    return pairs


def _load_cases_from_images_lms(
    dataset_root: Path,
    max_cases: int,
    slice_axis: int | None,
    case_ids: Sequence[str] | None,
) -> List[CaseVolume]:
    images_dir = dataset_root / "images"
    lms_dir = dataset_root / "lms"
    if not images_dir.exists() or not lms_dir.exists():
        raise RuntimeError(f"Expected images/ and lms/ under {dataset_root}")

    pairs = _match_flat_pairs(images_dir, lms_dir)
    if not pairs:
        raise RuntimeError(f"No matched image/label pairs found in {images_dir} and {lms_dir}")

    cases: List[CaseVolume] = []
    selected_pairs = _prioritize_items(pairs, key_fn=lambda pair: pair[0], case_ids=case_ids, max_items=max_cases)
    for case_id, image_path, label_path in _iter_progress(selected_pairs, desc="Loading image/lm pairs"):
        image = _read_zyx(image_path, slice_axis=slice_axis, is_label=False)
        label = _read_zyx(label_path, slice_axis=slice_axis, is_label=True)
        if image.shape != label.shape:
            raise ValueError(
                f"Shape mismatch in {case_id}: image={image.shape}, label={label.shape}"
            )
        cases.append(CaseVolume(case_id=case_id, image=image, label=label))
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
            dataset_root,
            max_cases=max_cases,
            slice_axis=slice_axis,
            case_ids=case_ids,
        )
    return _load_cases_from_case_dirs(
        dataset_root,
        preferred_label_name=preferred_label_name,
        max_cases=max_cases,
        slice_axis=slice_axis,
        case_ids=case_ids,
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
    gif_colors: int = 96,
    case_ids: Sequence[str] | None = None,
    orientations: Tuple[str, str] = ("axial", "coronal"),
) -> Path:
    """
    Create an animated GIF grid where each panel scrolls its own slices.

    Args:
        dataset_root: Either
            1) case subfolders with image/annotation NIfTI files, or
            2) flat images/ and lms/ subfolders with matched basenames.
        output_gif: Destination GIF path. Default is <dataset_root>/dataset_stats/snapshot.gif.
        grid_shape: Grid rows, cols.
        preferred_label_name: Annotation file preferred per case.
        num_frames: Total animation frames.
        fps: GIF frame rate.
        stride: Slice step per frame.
        slice_axis: For tensor inputs, which axis is depth (0/1/2). None -> infer smallest axis.
        window: Window preset for image intensity. One of auto, lung, abdomen, bone.
        rotate_cw_degrees: Rotate each rendered slice clockwise by 0/90/180/270 degrees.
        panel_px: Target pixels per panel side for the rendered grid.
        gif_colors: Unused.
        case_ids: Optional case-id list to prioritize when filling panels.
        orientations: Two orientations rendered side by side in each panel.
    """
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
        case_frame_indices.append({axis: _sample_indices(int(c.image.shape[axis])) for axis in orientation_axes})

    panel_px = max(48, int(panel_px))
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
