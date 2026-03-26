#!/usr/bin/env python3
"""Create animated image+label overlay grids as GIFs from NIfTI or Torch tensors."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import itertools
import os
from pathlib import Path
import sys
import threading
import time
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import SimpleITK as sitk

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


def _optimize_gif_palette(path: Path, colors: int) -> None:
    try:
        from PIL import Image, ImageSequence
    except Exception:
        return
    color_count = max(2, min(256, int(colors)))
    with Image.open(path) as im:
        loop = int(im.info.get("loop", 0))
        frames = []
        durations = []
        for fr in ImageSequence.Iterator(im):
            durations.append(int(fr.info.get("duration", 100)))
            q = fr.convert("P", palette=Image.ADAPTIVE, colors=color_count, dither=Image.Dither.NONE)
            frames.append(q)
        if not frames:
            return
        frames[0].save(
            path,
            save_all=True,
            append_images=frames[1:],
            optimize=True,
            loop=loop,
            duration=durations,
            disposal=2,
        )


def _iter_progress(items: Sequence, desc: str):
    try:
        from tqdm import tqdm

        pbar = tqdm(items, total=len(items), desc=desc, unit="pair")
        for item in pbar:
            if isinstance(item, tuple) and item:
                case_name = str(item[0])
                pbar.set_postfix_str(case_name, refresh=False)
            yield item
        return
    except Exception:
        pass

    total = len(items)
    for idx, item in enumerate(items, start=1):
        case_name = str(item[0]) if isinstance(item, tuple) and item else "case"
        print(f"[{idx}/{total}] processing {case_name}")
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
    # Lazy torch handling so NIfTI-only usage does not require torch runtime.
    try:
        import torch
    except Exception:
        torch = None
    if torch is not None and isinstance(x, torch.Tensor):
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
        axis = int(slice_axis)
        if axis < 0 or axis > 2:
            raise ValueError(f"slice_axis must be in [0,1,2], got {slice_axis}")
        return axis
    # For anisotropic medical volumes, depth is commonly the smallest spatial axis.
    return int(np.argmin(vol3d.shape))


def _read_zyx(path: Path, slice_axis: int | None, *, is_label: bool = False) -> np.ndarray:
    if path.name.endswith((".nii", ".nii.gz")):
        img = sitk.ReadImage(str(path))
        return sitk.GetArrayFromImage(img)
    if path.suffix == ".pt":
        try:
            import torch
            from monai.data.meta_tensor import MetaTensor
            from torch.serialization import add_safe_globals
        except Exception as e:
            raise RuntimeError(f"Loading .pt requires torch and MONAI: {e}") from e
        add_safe_globals([MetaTensor])
        data = torch.load(path, map_location="cpu", weights_only=False)
        vol3d = _normalize_to_3d(_to_numpy(data), is_label=is_label)
        axis = _infer_slice_axis(vol3d, slice_axis=slice_axis)
        return np.moveaxis(vol3d, axis, 0)
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
    if window not in WINDOW_PRESETS:
        raise ValueError(f"Unsupported window preset: {window}")
    level, width = WINDOW_PRESETS[window]
    lo = level - (width / 2.0)
    hi = level + (width / 2.0)
    if hi <= lo:
        return _normalize_slice(x)
    y = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return y.astype(np.float32)


def _rotate_clockwise_2d(arr2d: np.ndarray, rotate_cw_degrees: int) -> np.ndarray:
    if rotate_cw_degrees not in {0, 90, 180, 270}:
        raise ValueError(f"rotate_cw_degrees must be one of 0,90,180,270. Got {rotate_cw_degrees}")
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


def _pad_center_2d(arr2d: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_hw
    h, w = arr2d.shape
    if h > target_h or w > target_w:
        # Safety fallback; should not happen when target is max over all cases.
        y0 = max(0, (h - target_h) // 2)
        x0 = max(0, (w - target_w) // 2)
        return arr2d[y0 : y0 + target_h, x0 : x0 + target_w]
    out = np.zeros((target_h, target_w), dtype=arr2d.dtype)
    y0 = (target_h - h) // 2
    x0 = (target_w - w) // 2
    out[y0 : y0 + h, x0 : x0 + w] = arr2d
    return out


def _pad_center_rgb(arr3d: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_hw
    h, w, c = arr3d.shape
    if h > target_h or w > target_w:
        y0 = max(0, (h - target_h) // 2)
        x0 = max(0, (w - target_w) // 2)
        return arr3d[y0 : y0 + target_h, x0 : x0 + target_w, :]
    out = np.zeros((target_h, target_w, c), dtype=arr3d.dtype)
    y0 = (target_h - h) // 2
    x0 = (target_w - w) // 2
    out[y0 : y0 + h, x0 : x0 + w, :] = arr3d
    return out


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
        image2d = _pad_center_2d(image2d, target_hw=target_hw)
        mask2d = _pad_center_2d(mask2d, target_hw=target_hw)
    base = _window_normalize_slice(image2d, window=window)
    rgb = np.stack([base, base, base], axis=-1)
    labels_present = _labels_present(mask2d)
    for label_value in labels_present:
        edge = _stencil_edges(np.isclose(mask2d, label_value, atol=0.5))
        color = _label_color(label_value)
        rgb[edge, 0] = color[0]
        rgb[edge, 1] = color[1]
        rgb[edge, 2] = color[2]
    return rgb, labels_present


def _orientation_axis(name: str) -> int:
    try:
        return ORIENTATION_TO_AXIS[name.lower()]
    except KeyError as e:
        raise ValueError(f"Unsupported orientation: {name}") from e


def _extract_slice(volume_zyx: np.ndarray, axis: int, index: int) -> np.ndarray:
    return np.take(volume_zyx, indices=index, axis=axis)


def _transformed_hw(
    hw: Tuple[int, int],
    *,
    pair_index: int,
    rotate_cw_degrees: int,
) -> Tuple[int, int]:
    hh, ww = hw
    if pair_index == 0:
        hh, ww = ww, hh
    if rotate_cw_degrees in {90, 270}:
        hh, ww = ww, hh
    return hh, ww


def _orientation_target_hw(
    cases: Sequence[CaseVolume],
    axis: int,
    pair_index: int,
    rotate_cw_degrees: int,
) -> Tuple[int, int]:
    heights: List[int] = []
    widths: List[int] = []
    for case in cases:
        sample = _extract_slice(case.image, axis=axis, index=case.image.shape[axis] // 2)
        hh, ww = _transformed_hw(sample.shape, pair_index=pair_index, rotate_cw_degrees=rotate_cw_degrees)
        heights.append(hh)
        widths.append(ww)
    return max(heights), max(widths)


def _panel_frame(
    case: CaseVolume,
    frame_idx: int,
    case_frame_indices: Dict[int, np.ndarray],
    orientations: Sequence[str],
    target_hw_by_axis: Dict[int, Tuple[int, int]],
    window: str,
    rotate_cw_degrees: int,
) -> Tuple[np.ndarray, List[int], str]:
    panel_parts: List[np.ndarray] = []
    labels_seen: set[int] = set()
    info_parts: List[str] = []
    for pair_index, orientation in enumerate(orientations):
        axis = _orientation_axis(orientation)
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
            target_hw=target_hw_by_axis[axis, pair_index],
        )
        panel_parts.append(rendered)
        labels_seen.update(labels_present)
        info_parts.append(f"{orientation[0].upper()} {slice_idx + 1}/{case.image.shape[axis]}")
    panel_height = max(part.shape[0] for part in panel_parts)
    panel_parts = [_pad_center_rgb(part, target_hw=(panel_height, part.shape[1])) for part in panel_parts]
    return np.concatenate(panel_parts, axis=1), sorted(labels_seen), " | ".join(info_parts)


def _rgb_float_to_uint8(rgb: np.ndarray) -> np.ndarray:
    return np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)


def _fit_rgb_to_panel(rgb: np.ndarray, panel_hw: Tuple[int, int]) -> np.ndarray:
    try:
        from PIL import Image
    except Exception as e:
        raise RuntimeError(f"Pillow is required for GIF rendering: {e}") from e

    target_h, target_w = panel_hw
    src_h, src_w = rgb.shape[:2]
    if src_h == target_h and src_w == target_w:
        return rgb

    scale = min(target_h / max(src_h, 1), target_w / max(src_w, 1))
    new_h = max(1, int(round(src_h * scale)))
    new_w = max(1, int(round(src_w * scale)))
    resized = np.asarray(
        Image.fromarray(rgb, mode="RGB").resize((new_w, new_h), resample=Image.Resampling.NEAREST)
    )
    out = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    y0 = (target_h - new_h) // 2
    x0 = (target_w - new_w) // 2
    out[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return out


def _render_grid_frames(
    cases: Sequence[CaseVolume],
    case_frame_indices: Sequence[Dict[int, np.ndarray]],
    orientations: Sequence[str],
    target_hw_by_axis: Dict[Tuple[int, int], Tuple[int, int]],
    window: str,
    rotate_cw_degrees: int,
    rows: int,
    cols: int,
    num_frames: int,
    panel_px: int,
) -> List["Image.Image"]:
    try:
        from PIL import Image, ImageDraw
    except Exception as e:
        raise RuntimeError(f"Pillow is required for GIF rendering: {e}") from e

    title_h = max(20, panel_px // 6)
    line_h = max(9, panel_px // 14)
    panel_h = panel_px
    panel_w = panel_px
    frames: List[Image.Image] = []
    for frame_idx in range(num_frames):
        canvas = np.zeros((title_h + (rows * panel_h), cols * panel_w, 3), dtype=np.uint8)
        frame_im = Image.fromarray(canvas, mode="RGB")
        draw = ImageDraw.Draw(frame_im)
        draw.text((6, 4), f"Overlay Stencil Grid {rows}x{cols} | Cases: {len(cases)}", fill=(255, 255, 255))
        for i, case in enumerate(cases):
            panel_rgb, labels_present, panel_info = _panel_frame(
                case,
                frame_idx=frame_idx,
                case_frame_indices=case_frame_indices[i],
                orientations=orientations,
                target_hw_by_axis=target_hw_by_axis,
                window=window,
                rotate_cw_degrees=rotate_cw_degrees,
            )
            panel_rgb_u8 = _fit_rgb_to_panel(_rgb_float_to_uint8(panel_rgb), panel_hw=(panel_h, panel_w))
            row = i // cols
            col = i % cols
            y0 = title_h + (row * panel_h)
            x0 = col * panel_w
            frame_im.paste(Image.fromarray(panel_rgb_u8, mode="RGB"), (x0, y0))

            draw.text((x0 + 4, y0 + 3), case.case_id, fill=(255, 255, 255))
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
        try:
            img_path = _pick_case_image_file(case_dir)
        except FileNotFoundError:
            continue
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
    output_gif: Path | str,
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
        output_gif: Destination GIF path.
        grid_shape: Grid rows, cols.
        preferred_label_name: Annotation file preferred per case.
        num_frames: Total animation frames.
        fps: GIF frame rate.
        stride: Slice step per frame.
        slice_axis: For tensor inputs, which axis is depth (0/1/2). None -> infer smallest axis.
        window: Window preset for image intensity. One of auto, lung, abdomen, bone.
        rotate_cw_degrees: Rotate each rendered slice clockwise by 0/90/180/270 degrees.
        panel_px: Target pixels per panel side for the rendered grid.
        gif_colors: Palette size used for post-encoding GIF optimization.
        case_ids: Optional case-id list to prioritize when filling panels.
        orientations: Two orientations rendered side by side in each panel.
    """
    try:
        from PIL import Image
    except Exception as e:
        raise RuntimeError(f"Pillow is required for GIF rendering: {e}") from e

    dataset_root = Path(dataset_root)
    output_gif = Path(output_gif)
    rows, cols = grid_shape
    max_cases = rows * cols
    if len(orientations) != 2:
        raise ValueError(f"orientations must contain exactly 2 entries, got {orientations}")
    cases = _load_cases(
        dataset_root,
        preferred_label_name=preferred_label_name,
        max_cases=max_cases,
        slice_axis=slice_axis,
        case_ids=case_ids,
    )
    orientation_axes = [_orientation_axis(name) for name in orientations]
    target_hw_by_axis = {
        (axis, pair_index): _orientation_target_hw(
            cases,
            axis=axis,
            pair_index=pair_index,
            rotate_cw_degrees=rotate_cw_degrees,
        )
        for pair_index, axis in enumerate(orientation_axes)
    }

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
    with _Spinner("Postprocessing (encoding/optimizing GIF)"):
        frames = _render_grid_frames(
            cases=cases,
            case_frame_indices=case_frame_indices,
            orientations=orientations,
            target_hw_by_axis=target_hw_by_axis,
            window=window,
            rotate_cw_degrees=rotate_cw_degrees,
            rows=rows,
            cols=cols,
            num_frames=num_frames,
            panel_px=panel_px,
        )
        if not frames:
            raise RuntimeError("No GIF frames were rendered")
        frame_duration_ms = int(round(1000 / max(fps, 1)))
        frames[0].save(
            output_gif,
            save_all=True,
            append_images=frames[1:],
            duration=frame_duration_ms,
            loop=0,
            disposal=2,
        )
        _optimize_gif_palette(output_gif, colors=gif_colors)
    return output_gif


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Create a 6x6 animated image/label overlay stencil grid GIF.")
    p.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Path with case subfolders or with images/ and lms/ subfolders. Supports .nii.gz, .nii, and .pt.",
    )
    p.add_argument("--output-gif", type=Path, required=True, help="Output GIF path.")
    p.add_argument(
        "--preferred-label-name",
        type=str,
        default="annotation_staple.nii.gz",
        help="Preferred annotation filename inside each case folder. The same basename is matched across .nii.gz, .nii, and .pt.",
    )
    p.add_argument("--rows", type=int, default=6, help="Grid rows.")
    p.add_argument("--cols", type=int, default=6, help="Grid cols.")
    p.add_argument("--num-frames", type=int, default=90, help="Total GIF frames.")
    p.add_argument("--fps", type=int, default=10, help="GIF frame rate.")
    p.add_argument("--stride", type=int, default=1, help="Slice step multiplier per frame.")
    p.add_argument("--panel-px", type=int, default=120, help="Target pixels per panel side in output GIF.")
    p.add_argument("--gif-colors", type=int, default=96, help="Palette size for GIF optimization (2-256).")
    p.add_argument(
        "--case-ids",
        nargs="*",
        default=None,
        help="Optional case IDs to prioritize first when filling grid panels.",
    )
    p.add_argument(
        "--orientations",
        nargs=2,
        default=("axial", "coronal"),
        choices=["axial", "coronal", "sag", "sagittal"],
        help="Two orientations rendered side by side inside each grid panel.",
    )
    p.add_argument(
        "--window",
        type=str,
        default="auto",
        choices=["auto", "lung", "abdomen", "bone"],
        help="Intensity window preset.",
    )
    p.add_argument(
        "--rotate-cw",
        type=int,
        default=0,
        choices=[0, 90, 180, 270],
        help="Rotate each rendered frame clockwise.",
    )
    p.add_argument(
        "--slice-axis",
        type=int,
        default=None,
        choices=[0, 1, 2],
        help="Depth axis for tensor inputs. Default infers the smallest axis.",
    )
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
    main()
