#!/usr/bin/env python3
"""Create case GIFs windowed to abdominal settings, cropped to overlays, with multi-source contour overlays."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageSequence
import SimpleITK as sitk

try:
    from utilz.stringz import info_from_filename
except Exception:
    from stringz import info_from_filename


ColorRGB = Tuple[int, int, int]
DEFAULT_COLORS: List[ColorRGB] = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 255, 255),
    (255, 255, 0),
    (255, 0, 255),
    (255, 165, 0),
]


def _list_nifti_files(folder: Path) -> List[Path]:
    return sorted([p for p in folder.iterdir() if p.is_file() and p.name.endswith((".nii", ".nii.gz"))])


def _resample_like(moving: sitk.Image, reference: sitk.Image, is_label: bool = True) -> sitk.Image:
    if (
        moving.GetSize() == reference.GetSize()
        and moving.GetSpacing() == reference.GetSpacing()
        and moving.GetOrigin() == reference.GetOrigin()
        and moving.GetDirection() == reference.GetDirection()
    ):
        return moving
    r = sitk.ResampleImageFilter()
    r.SetReferenceImage(reference)
    r.SetTransform(sitk.Transform())
    r.SetDefaultPixelValue(0)
    r.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)
    return r.Execute(moving)


def _bbox_from_mask_xyz(mask_zyx: np.ndarray) -> Tuple[List[int], List[int]]:
    coords = np.where(mask_zyx)
    if len(coords[0]) == 0:
        raise ValueError("Empty mask")
    zmin, ymin, xmin = [int(c.min()) for c in coords]
    zmax, ymax, xmax = [int(c.max()) for c in coords]
    mins_xyz = [xmin, ymin, zmin]
    maxs_xyz = [xmax, ymax, zmax]
    return mins_xyz, maxs_xyz


def _axis_aligned_crop_for_label(
    label_img: sitk.Image,
    target_label: int,
    margin_cm: float,
) -> tuple[list[int], list[int]]:
    mask = sitk.Equal(label_img, int(target_label))
    mask_np = sitk.GetArrayFromImage(mask) > 0
    mins, maxs = _bbox_from_mask_xyz(mask_np)

    size = list(label_img.GetSize())
    spacing = label_img.GetSpacing()
    margin_mm = float(margin_cm) * 10.0
    margin_vox = [int(round(margin_mm / float(spacing[a]))) for a in range(3)]

    idx: List[int] = []
    crop_size: List[int] = []
    for a in range(3):
        start = max(0, mins[a] - margin_vox[a])
        end = min(size[a], maxs[a] + 1 + margin_vox[a])
        idx.append(int(start))
        crop_size.append(max(1, int(end - start)))
    return idx, crop_size


def _axis_aligned_crop_for_overlay_union(
    reference_img: sitk.Image,
    overlays: Sequence[Tuple[sitk.Image, Sequence[int]]],
    margin_cm: float,
    fallback_label_img: sitk.Image | None = None,
    fallback_label: int | None = None,
) -> tuple[list[int], list[int]]:
    size = list(reference_img.GetSize())
    spacing = reference_img.GetSpacing()
    margin_mm = float(margin_cm) * 10.0
    margin_vox = [int(round(margin_mm / float(spacing[a]))) for a in range(3)]

    mins = [size[0], size[1], size[2]]
    maxs = [-1, -1, -1]

    for ov_img, labels in overlays:
        ov_np = sitk.GetArrayFromImage(ov_img)  # z,y,x
        m = np.isin(ov_np, [int(x) for x in labels])
        if not np.any(m):
            continue
        bb_mins, bb_maxs = _bbox_from_mask_xyz(m)
        for a in range(3):
            mins[a] = min(mins[a], bb_mins[a])
            maxs[a] = max(maxs[a], bb_maxs[a])

    has_union = maxs[0] >= 0
    if not has_union:
        if fallback_label_img is None or fallback_label is None:
            raise ValueError("No overlay labels found and no fallback crop label available")
        return _axis_aligned_crop_for_label(fallback_label_img, target_label=int(fallback_label), margin_cm=margin_cm)

    idx: List[int] = []
    crop_size: List[int] = []
    for a in range(3):
        start = max(0, mins[a] - margin_vox[a])
        end = min(size[a], maxs[a] + 1 + margin_vox[a])
        idx.append(int(start))
        crop_size.append(max(1, int(end - start)))
    return idx, crop_size


def _window_to_uint8(img: sitk.Image, level: float, width: float) -> sitk.Image:
    lo = float(level) - float(width) / 2.0
    hi = float(level) + float(width) / 2.0
    out = sitk.IntensityWindowing(img, windowMinimum=lo, windowMaximum=hi, outputMinimum=0.0, outputMaximum=255.0)
    return sitk.Cast(out, sitk.sitkUInt8)


def _sample_slice_indices(depth: int, num_images: int) -> np.ndarray:
    if depth <= 0:
        return np.array([], dtype=int)
    n = min(depth, max(1, int(num_images)))
    if n == depth:
        return np.arange(depth, dtype=int)
    idx = np.linspace(0, depth - 1, n)
    return np.unique(np.round(idx).astype(int))


def _contour_from_mask_np(mask_zyx: np.ndarray) -> np.ndarray:
    if not np.any(mask_zyx):
        return np.zeros_like(mask_zyx, dtype=bool)
    mask_img = sitk.GetImageFromArray(mask_zyx.astype(np.uint8, copy=False))
    contour_img = sitk.LabelContour(mask_img, fullyConnected=False)
    contour_np = sitk.GetArrayFromImage(contour_img) > 0
    return contour_np


def _normalize_overlay_sources(overlay_sources: Sequence[Dict[str, Sequence[int] | int]]) -> List[Tuple[Path, List[int]]]:
    out: List[Tuple[Path, List[int]]] = []
    for source_dict in overlay_sources:
        for folder, labels in source_dict.items():
            if isinstance(labels, int):
                labels_list = [labels]
            else:
                labels_list = [int(x) for x in labels]
            out.append((Path(folder), labels_list))
    return out


def _parse_overlay_source_arg(raw: str) -> Dict[str, List[int]]:
    if ":" not in raw:
        raise ValueError(f"Invalid --overlay-source '{raw}'. Expected folder:label[,label...]")
    folder, labels_raw = raw.split(":", 1)
    labels = [int(x.strip()) for x in labels_raw.split(",") if x.strip() != ""]
    if not labels:
        raise ValueError(f"Invalid --overlay-source '{raw}': no labels provided")
    return {folder: labels}


def _upscale_rgb(frame: np.ndarray, scale: int) -> np.ndarray:
    if scale <= 1:
        return frame
    img = Image.fromarray(frame)
    out = img.resize((img.width * scale, img.height * scale), resample=Image.Resampling.NEAREST)
    return np.asarray(out, dtype=np.uint8)


def _add_case_id_footer(frame: np.ndarray, case_id: str, footer_px: int = 24) -> np.ndarray:
    if footer_px <= 0:
        return frame
    h, w = frame.shape[:2]
    canvas = np.zeros((h + footer_px, w, 3), dtype=np.uint8)
    canvas[:h, :, :] = frame
    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    text = str(case_id)
    tx = 6
    ty = h + max(1, (footer_px - 10) // 2)
    draw.text((tx, ty), text, fill=(255, 255, 255), font=font)
    return np.asarray(img, dtype=np.uint8)


def _read_gif_frames(path: Path) -> List[np.ndarray]:
    with Image.open(path) as im:
        return [np.asarray(frm.convert("RGB"), dtype=np.uint8) for frm in ImageSequence.Iterator(im)]


def create_paginated_gif_grid(
    gifs_dir: Path | str,
    output_dir: Path | str,
    rows: int = 6,
    cols: int = 6,
    fps: int = 1,
) -> List[Path]:
    gifs_dir = Path(gifs_dir)
    output_dir = Path(output_dir)
    gif_files = sorted(gifs_dir.glob("*.gif"))
    if not gif_files:
        return []

    per_page = rows * cols
    n_pages = math.ceil(len(gif_files) / per_page)
    output_dir.mkdir(parents=True, exist_ok=True)

    out_paths: List[Path] = []
    for page in range(n_pages):
        page_files = gif_files[page * per_page : (page + 1) * per_page]
        gif_frames = [_read_gif_frames(p) for p in page_files]

        h = max(frames[0].shape[0] for frames in gif_frames)
        w = max(frames[0].shape[1] for frames in gif_frames)
        t = max(len(frames) for frames in gif_frames)

        out_frames: List[np.ndarray] = []
        for ti in range(t):
            canvas = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
            for i, frames in enumerate(gif_frames):
                r = i // cols
                c = i % cols
                frm = frames[ti % len(frames)]
                fh, fw = frm.shape[:2]
                y0 = r * h + (h - fh) // 2
                x0 = c * w + (w - fw) // 2
                canvas[y0 : y0 + fh, x0 : x0 + fw] = frm
            out_frames.append(canvas)

        out_path = output_dir / f"grid_page_{page + 1:02d}.gif"
        imageio.mimsave(str(out_path), out_frames, format="GIF", duration=1.0 / max(1, int(fps)), loop=0)
        out_paths.append(out_path)
        print(f"[OK] GRID page {page + 1}/{n_pages} -> {out_path}")

    return out_paths


def create_case_crop_overlay_gif(
    image_path: Path | str,
    crop_labelmap_path: Path | str,
    output_gif: Path | str,
    overlay_case_paths_and_labels: Sequence[Tuple[Path | str, Sequence[int]]],
    crop_label: int = 12,
    margin_cm: float = 10.0,
    abdominal_window_level: float = 40.0,
    abdominal_window_width: float = 400.0,
    num_images: int = 24,
    fps: int = 2,
    scale: int = 1,
) -> Path:
    image_path = Path(image_path)
    crop_labelmap_path = Path(crop_labelmap_path)
    output_gif = Path(output_gif)
    case_info = info_from_filename(image_path.name, full_caseid=True)
    case_id = case_info.get("case_id", image_path.stem.replace(".nii", ""))

    img = sitk.ReadImage(str(image_path))

    crop_map = sitk.ReadImage(str(crop_labelmap_path))
    crop_map = _resample_like(crop_map, img, is_label=True)

    overlays_full: List[Tuple[sitk.Image, List[int], ColorRGB]] = []
    overlays_for_bbox: List[Tuple[sitk.Image, List[int]]] = []
    for i, (overlay_case_path, labels) in enumerate(overlay_case_paths_and_labels):
        ov = sitk.ReadImage(str(overlay_case_path))
        ov = _resample_like(ov, img, is_label=True)
        labels_i = [int(x) for x in labels]
        overlays_for_bbox.append((ov, labels_i))
        overlays_full.append((ov, labels_i, DEFAULT_COLORS[i % len(DEFAULT_COLORS)]))

    crop_idx, crop_size = _axis_aligned_crop_for_overlay_union(
        reference_img=img,
        overlays=overlays_for_bbox,
        margin_cm=margin_cm,
        fallback_label_img=crop_map,
        fallback_label=crop_label,
    )

    img_crop = sitk.RegionOfInterest(img, size=crop_size, index=crop_idx)
    img_u8 = _window_to_uint8(img_crop, level=abdominal_window_level, width=abdominal_window_width)

    overlay_contours_and_colors: List[Tuple[np.ndarray, ColorRGB]] = []
    for ov, labels, color in overlays_full:
        ov_crop = sitk.RegionOfInterest(ov, size=crop_size, index=crop_idx)
        ov_np = sitk.GetArrayFromImage(ov_crop)
        mask_np = np.isin(ov_np, labels)
        contour_np = _contour_from_mask_np(mask_np)
        overlay_contours_and_colors.append((contour_np, color))

    gray3d = sitk.GetArrayFromImage(img_u8)
    depth = int(gray3d.shape[0])
    z_indices = _sample_slice_indices(depth=depth, num_images=num_images)

    frames: List[np.ndarray] = []
    for z in z_indices:
        gray2d = gray3d[int(z)]
        rgb = np.stack([gray2d, gray2d, gray2d], axis=-1).astype(np.uint8, copy=False)

        for contour_np, color in overlay_contours_and_colors:
            edge = contour_np[int(z)]
            if np.any(edge):
                rgb[edge, 0] = color[0]
                rgb[edge, 1] = color[1]
                rgb[edge, 2] = color[2]

        frame = _upscale_rgb(rgb, scale=scale)
        footer_px = max(18, 12 * max(1, int(scale)))
        frame = _add_case_id_footer(frame, case_id=case_id, footer_px=footer_px)
        frames.append(frame)

    output_gif.parent.mkdir(parents=True, exist_ok=True)
    duration_sec = 1.0 / max(1, int(fps))
    imageio.mimsave(str(output_gif), frames, format="GIF", duration=duration_sec, loop=0)
    return output_gif


def create_folder_crop_overlay_gifs(
    images_dir: Path | str,
    crop_labelmaps_dir: Path | str,
    output_dir: Path | str,
    overlay_sources: Sequence[Dict[str, Sequence[int] | int]],
    crop_label: int = 12,
    margin_cm: float = 10.0,
    abdominal_window_level: float = 40.0,
    abdominal_window_width: float = 400.0,
    num_images: int = 24,
    fps: int = 1,
    scale: int = 1,
    limit: int | None = None,
) -> List[Path]:
    images_dir = Path(images_dir)
    crop_labelmaps_dir = Path(crop_labelmaps_dir)
    output_dir = Path(output_dir)

    overlays = _normalize_overlay_sources(overlay_sources)
    outputs: List[Path] = []
    candidates = _list_nifti_files(images_dir)
    done = 0

    for image_path in candidates:
        crop_labelmap_path = crop_labelmaps_dir / image_path.name
        if not crop_labelmap_path.exists():
            print(f"[SKIP] {image_path.name}: crop labelmap missing: {crop_labelmap_path}")
            continue

        overlay_case_paths_and_labels: List[Tuple[Path, List[int]]] = []
        missing_overlay = False
        for folder, labels in overlays:
            ov_path = folder / image_path.name
            if not ov_path.exists():
                print(f"[SKIP] {image_path.name}: overlay source missing file: {ov_path}")
                missing_overlay = True
                break
            overlay_case_paths_and_labels.append((ov_path, labels))
        if missing_overlay:
            continue

        out_path = output_dir / f"{image_path.stem.replace('.nii', '')}.gif"
        try:
            out = create_case_crop_overlay_gif(
                image_path=image_path,
                crop_labelmap_path=crop_labelmap_path,
                output_gif=out_path,
                overlay_case_paths_and_labels=overlay_case_paths_and_labels,
                crop_label=crop_label,
                margin_cm=margin_cm,
                abdominal_window_level=abdominal_window_level,
                abdominal_window_width=abdominal_window_width,
                num_images=num_images,
                fps=fps,
                scale=scale,
            )
            outputs.append(out)
            done += 1
            print(f"[OK] {image_path.name} -> {out}")
        except Exception as exc:
            print(f"[SKIP] {image_path.name}: {exc}")

        if limit is not None and done >= limit:
            break

    return outputs


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Create GIFs from CT + segmentation: crop to overlay union, overlay contours from one or more labelmaps."
    )
    p.add_argument("--images-dir", type=Path, required=True)
    p.add_argument("--crop-labelmaps-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--overlay-source", action="append", required=True, help="Format: /path/to/folder:label[,label...]. Repeat for multiple sources.")
    p.add_argument("--crop-label", type=int, default=12)
    p.add_argument("--margin-cm", type=float, default=10.0, help="Margin in cm, applied on x/y/z around union bbox.")
    p.add_argument("--wl", type=float, default=40.0, help="Window level (abdominal default 40).")
    p.add_argument("--ww", type=float, default=400.0, help="Window width (abdominal default 400).")
    p.add_argument("--num-images", type=int, default=24, help="Number of slices/frames per GIF.")
    p.add_argument("--fps", type=int, default=1, help="GIF playback speed (lower = slower).")
    p.add_argument("--scale", type=int, default=1, help="Nearest-neighbor frame upscale factor.")
    p.add_argument("--limit", type=int, default=None)

    p.add_argument("--make-grid", action="store_true", help="Also create paginated animated 6x6 GIF grids.")
    p.add_argument("--grid-rows", type=int, default=6)
    p.add_argument("--grid-cols", type=int, default=6)
    p.add_argument("--grid-output-dir", type=Path, default=None, help="Defaults to <output-dir>/grid_pages")
    p.add_argument("--grid-fps", type=int, default=2)
    return p


def main() -> None:
    args = _build_parser().parse_args()
    overlay_sources = [_parse_overlay_source_arg(x) for x in args.overlay_source]

    outs = create_folder_crop_overlay_gifs(
        images_dir=args.images_dir,
        crop_labelmaps_dir=args.crop_labelmaps_dir,
        output_dir=args.output_dir,
        overlay_sources=overlay_sources,
        crop_label=args.crop_label,
        margin_cm=args.margin_cm,
        abdominal_window_level=args.wl,
        abdominal_window_width=args.ww,
        num_images=args.num_images,
        fps=args.fps,
        scale=args.scale,
        limit=args.limit,
    )
    print(f"Generated {len(outs)} GIFs")

    if args.make_grid:
        grid_out_dir = args.grid_output_dir or (Path(args.output_dir) / "grid_pages")
        grid_paths = create_paginated_gif_grid(
            gifs_dir=args.output_dir,
            output_dir=grid_out_dir,
            rows=args.grid_rows,
            cols=args.grid_cols,
            fps=args.grid_fps,
        )
        print(f"Generated {len(grid_paths)} grid GIF pages")


if __name__ == "__main__":
    main()
