#!/usr/bin/env python3
"""Interactive 8x8 bone lesion grid viewer with synchronized scrolling.

Features:
- Matches bone images and labels by case ID (e.g. bone_00523)
- Reads .nii/.nii.gz and zipped .nii.gz.zip files
- Resamples image+label pairs to a common XYZ shape
- Displays 8x8 grid (64 cases) in full-screen mode
- Mouse wheel scrolls all 64 cases through Z together
- Pagination for >64 cases via keyboard
- Optional JPG export for current page
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import re
import tempfile
import threading
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

CASE_RE = re.compile(r"(bone_\d{5})", re.IGNORECASE)
SUPPORTED_SUFFIXES = (".nii", ".nii.gz", ".nii.zip", ".nii.gz.zip")


@dataclass(frozen=True)
class CasePair:
    case_id: str
    image_path: Path
    label_path: Path


class ZipCache:
    def __init__(self, cache_root: Optional[Path] = None) -> None:
        self.cache_root = cache_root or Path(tempfile.gettempdir()) / "bone_grid_cache"
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._write_locks: Dict[Path, threading.Lock] = {}

    def _lock_for(self, out_path: Path) -> threading.Lock:
        with self._lock:
            lk = self._write_locks.get(out_path)
            if lk is None:
                lk = threading.Lock()
                self._write_locks[out_path] = lk
            return lk

    def materialize(self, path: Path) -> Path:
        if path.suffix != ".zip":
            return path

        with zipfile.ZipFile(path, "r") as zf:
            nii_members = [n for n in zf.namelist() if n.lower().endswith((".nii", ".nii.gz"))]
            if not nii_members:
                raise RuntimeError(f"No NIfTI file found inside zip: {path}")
            member = nii_members[0]
            out_path = self.cache_root / Path(member).name
            lock = self._lock_for(out_path)
            with lock:
                if not out_path.exists() or out_path.stat().st_size == 0:
                    with zf.open(member, "r") as src, out_path.open("wb") as dst:
                        dst.write(src.read())
            return out_path


class PreprocessedDatasetCache:
    def __init__(self, cache_root: Optional[Path] = None) -> None:
        self.cache_root = cache_root or Path(tempfile.gettempdir()) / "bone_grid_preprocessed_cache"
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self._known_cache_dirs: Dict[str, Path] = {}

    def dataset_key(
        self,
        labels_dir: Path,
        images_dir: Path,
        pairs: Sequence[CasePair],
        target_shape: Tuple[int, int, int],
    ) -> str:
        digest = hashlib.sha1()
        digest.update(str(labels_dir.resolve()).encode("utf-8"))
        digest.update(b"\n")
        digest.update(str(images_dir.resolve()).encode("utf-8"))
        digest.update(b"\n")
        digest.update(str(target_shape).encode("utf-8"))
        digest.update(b"\n")
        for pair in pairs:
            digest.update(pair.case_id.encode("utf-8"))
            digest.update(b"|")
            digest.update(str(pair.image_path).encode("utf-8"))
            digest.update(b"|")
            digest.update(str(pair.label_path).encode("utf-8"))
            digest.update(b"\n")
        return digest.hexdigest()

    def ensure_preprocessed(
        self,
        labels_dir: Path,
        images_dir: Path,
        pairs: Sequence[CasePair],
        target_shape: Tuple[int, int, int],
        zip_cache: ZipCache,
        workers: int,
        compressed: bool,
    ) -> Path:
        key = self.dataset_key(labels_dir, images_dir, pairs, target_shape)
        if key in self._known_cache_dirs:
            return self._known_cache_dirs[key]

        dataset_dir = self.cache_root / key
        manifest_path = dataset_dir / "manifest.json"
        if manifest_path.exists():
            self._known_cache_dirs[key] = dataset_dir
            return dataset_dir

        dataset_dir.mkdir(parents=True, exist_ok=True)
        print(f"Preprocessing {len(pairs)} pairs into cache: {dataset_dir}")
        print(f"Workers: {workers} | Cache compression: {'on' if compressed else 'off'}")

        items_by_idx: Dict[int, Dict[str, object]] = {}
        total = len(pairs)
        completed = 0

        def preprocess_one(i: int, pair: CasePair) -> Tuple[int, Dict[str, object]]:
            img_xyz, lbl_xyz = load_and_resample_pair(pair, zip_cache, target_shape)
            rel_npz = f"{i:06d}.npz"
            out_path = dataset_dir / rel_npz
            if compressed:
                np.savez_compressed(out_path, image=img_xyz, label=lbl_xyz)
            else:
                np.savez(out_path, image=img_xyz, label=lbl_xyz)
            return i, {
                "index": i,
                "case_id": pair.case_id,
                "image_path": str(pair.image_path),
                "label_path": str(pair.label_path),
                "cache_file": rel_npz,
            }

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(preprocess_one, i, pair) for i, pair in enumerate(pairs)]
            for fut in concurrent.futures.as_completed(futures):
                i, item = fut.result()
                items_by_idx[i] = item
                completed += 1
                if completed % 25 == 0 or completed == total:
                    print(f"  preprocessed {completed}/{total}")

        items = [items_by_idx[i] for i in range(total)]

        manifest = {
            "target_shape": list(target_shape),
            "count": len(items),
            "items": items,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2))
        self._known_cache_dirs[key] = dataset_dir
        return dataset_dir

    def load_pair(self, dataset_dir: Path, pair_index: int) -> Tuple[np.ndarray, np.ndarray]:
        npz_path = dataset_dir / f"{pair_index:06d}.npz"
        with np.load(npz_path) as data:
            return data["image"], data["label"]


def parse_args() -> argparse.Namespace:
    default_workers = max(1, min(16, os.cpu_count() or 1))
    parser = argparse.ArgumentParser(description="Bone 8x8 synchronized slice grid viewer")
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=Path("annotations/ULS23/novel_data/ULS23_Radboudumc_Bone/labels"),
        help="Directory with bone label files (.nii.gz or .nii.gz.zip)",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        required=True,
        help="Directory with matching bone image volumes",
    )
    parser.add_argument(
        "--target-shape",
        type=int,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=None,
        help="Target resample shape. Default: shape of first matched image",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Optional limit on number of matched cases",
    )
    parser.add_argument(
        "--export-page-jpg",
        type=Path,
        default=None,
        help="Optional output path to save current page as JPG after each redraw",
    )
    parser.add_argument(
        "--preprocess-workers",
        type=int,
        default=default_workers,
        help=f"Thread workers for preprocessing (default: {default_workers})",
    )
    parser.add_argument(
        "--cache-compressed",
        action="store_true",
        help="Write compressed .npz cache (smaller, slower). Default is uncompressed for speed.",
    )
    return parser.parse_args()


def is_volume_file(path: Path) -> bool:
    name = path.name.lower()
    return any(name.endswith(s) for s in SUPPORTED_SUFFIXES)


def case_id_from_name(path: Path) -> Optional[str]:
    m = CASE_RE.search(path.name)
    if not m:
        return None
    return m.group(1).lower()


def discover_labels(labels_dir: Path) -> Dict[str, List[Path]]:
    out: Dict[str, List[Path]] = {}
    for p in sorted(labels_dir.rglob("*")):
        if not p.is_file() or not is_volume_file(p):
            continue
        cid = case_id_from_name(p)
        if cid is None:
            continue
        out.setdefault(cid, []).append(p)
    return out


def discover_images(images_dir: Path) -> Dict[str, List[Path]]:
    out: Dict[str, List[Path]] = {}
    for p in sorted(images_dir.rglob("*")):
        if not p.is_file() or not is_volume_file(p):
            continue
        cid = case_id_from_name(p)
        if cid is None:
            continue
        out.setdefault(cid, []).append(p)
    return out


def choose_best_image(paths: Sequence[Path], case_id: str) -> Path:
    case_exact = f"{case_id}.nii.gz"
    case_exact2 = f"{case_id}.nii"

    def score(p: Path) -> Tuple[int, int, str]:
        n = p.name.lower()
        exact = 0 if n in (case_exact, case_exact2) else 1
        lesion_penalty = 1 if "lesion" in n else 0
        return (exact + lesion_penalty, len(n), n)

    return sorted(paths, key=score)[0]


def build_pairs(labels_by_case: Dict[str, List[Path]], images_by_case: Dict[str, List[Path]]) -> List[CasePair]:
    pairs: List[CasePair] = []
    for cid in sorted(labels_by_case.keys()):
        if cid not in images_by_case:
            continue
        img = choose_best_image(images_by_case[cid], cid)
        for lbl in labels_by_case[cid]:
            pairs.append(CasePair(case_id=cid, image_path=img, label_path=lbl))
    return sorted(
        pairs,
        key=lambda p: (
            p.image_path.name.lower(),
            p.case_id.lower(),
            p.label_path.stem.replace(".nii", "").lower(),
            p.label_path.name.lower(),
        ),
    )


def sitk_to_numpy_xyz(img: sitk.Image) -> np.ndarray:
    # SITK gives [z, y, x]; convert to [x, y, z]
    arr_zyx = sitk.GetArrayFromImage(img)
    return np.transpose(arr_zyx, (2, 1, 0))


def numpy_xyz_to_sitk(arr_xyz: np.ndarray) -> sitk.Image:
    arr_zyx = np.transpose(arr_xyz, (2, 1, 0))
    return sitk.GetImageFromArray(arr_zyx)


def resample_xyz(vol_xyz: np.ndarray, out_shape_xyz: Tuple[int, int, int], is_label: bool) -> np.ndarray:
    in_shape = vol_xyz.shape
    sitk_img = numpy_xyz_to_sitk(vol_xyz.astype(np.float32, copy=False))

    in_spacing = [1.0, 1.0, 1.0]
    out_spacing = [
        in_spacing[i] * (in_shape[i] / out_shape_xyz[i])
        for i in range(3)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetSize([int(out_shape_xyz[0]), int(out_shape_xyz[1]), int(out_shape_xyz[2])])
    resampler.SetOutputSpacing(out_spacing)
    resampler.SetOutputOrigin(sitk_img.GetOrigin())
    resampler.SetOutputDirection(sitk_img.GetDirection())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)

    out = resampler.Execute(sitk_img)
    out_xyz = sitk_to_numpy_xyz(out)

    if is_label:
        return (out_xyz > 0.5).astype(np.uint8)
    return out_xyz.astype(np.float32)


def normalize_slice(img2d: np.ndarray) -> np.ndarray:
    finite = np.isfinite(img2d)
    if not np.any(finite):
        return np.zeros_like(img2d, dtype=np.float32)

    vals = img2d[finite]
    lo, hi = np.percentile(vals, [1.0, 99.0])
    if hi <= lo:
        lo, hi = float(np.min(vals)), float(np.max(vals)) + 1e-5

    x = np.clip(img2d, lo, hi)
    x = (x - lo) / (hi - lo + 1e-8)
    return x.astype(np.float32)


def edge_from_mask(mask2d: np.ndarray) -> np.ndarray:
    m = mask2d.astype(bool)
    if m.size == 0:
        return m
    interior = m.copy()
    interior &= np.roll(m, 1, axis=0)
    interior &= np.roll(m, -1, axis=0)
    interior &= np.roll(m, 1, axis=1)
    interior &= np.roll(m, -1, axis=1)
    interior[0, :] = False
    interior[-1, :] = False
    interior[:, 0] = False
    interior[:, -1] = False
    return m & (~interior)


def overlay_rgb(img2d: np.ndarray, mask2d: np.ndarray) -> np.ndarray:
    base = normalize_slice(img2d)
    rgb = np.stack([base, base, base], axis=-1)
    edge = edge_from_mask(mask2d)
    rgb[edge, 0] = 1.0
    rgb[edge, 1] = 0.1
    rgb[edge, 2] = 0.1
    return rgb


def chunked(seq: Sequence[CasePair], n: int) -> List[List[CasePair]]:
    return [list(seq[i : i + n]) for i in range(0, len(seq), n)]


def load_and_resample_pair(pair: CasePair, zc: ZipCache, target_shape: Tuple[int, int, int]) -> Tuple[np.ndarray, np.ndarray]:
    img_path = zc.materialize(pair.image_path)
    lbl_path = zc.materialize(pair.label_path)

    img = sitk.ReadImage(str(img_path))
    lbl = sitk.ReadImage(str(lbl_path))

    img_xyz = sitk_to_numpy_xyz(img).astype(np.float32)
    lbl_xyz = sitk_to_numpy_xyz(lbl)
    lbl_xyz = (lbl_xyz > 0).astype(np.uint8)

    if img_xyz.shape != target_shape:
        img_xyz = resample_xyz(img_xyz, target_shape, is_label=False)
    if lbl_xyz.shape != target_shape:
        lbl_xyz = resample_xyz(lbl_xyz, target_shape, is_label=True)

    return img_xyz, lbl_xyz


def infer_target_shape(first_pair: CasePair, zc: ZipCache) -> Tuple[int, int, int]:
    img_path = zc.materialize(first_pair.image_path)
    img = sitk.ReadImage(str(img_path))
    xyz = sitk_to_numpy_xyz(img).shape
    return int(xyz[0]), int(xyz[1]), int(xyz[2])


class GridViewer:
    def __init__(
        self,
        all_pairs: List[CasePair],
        target_shape: Tuple[int, int, int],
        dataset_cache_dir: Path,
        preprocessed_cache: PreprocessedDatasetCache,
        export_page_jpg: Optional[Path] = None,
    ) -> None:
        self.all_pairs = all_pairs
        self.pages = chunked(all_pairs, 64)
        self.target_shape = target_shape
        self.dataset_cache_dir = dataset_cache_dir
        self.preprocessed_cache = preprocessed_cache
        self.export_page_jpg = export_page_jpg

        self.page_idx = 0
        self.z_idx = max(0, target_shape[2] // 2)

        self.fig = None
        self.axes = None
        self.artists = []

        self.curr_pairs: List[CasePair] = []
        self.curr_images: List[np.ndarray] = []
        self.curr_labels: List[np.ndarray] = []

    def load_page(self, idx: int) -> None:
        self.page_idx = int(np.clip(idx, 0, len(self.pages) - 1))
        self.curr_pairs = self.pages[self.page_idx]
        self.curr_images = []
        self.curr_labels = []

        print(f"Loading page {self.page_idx + 1}/{len(self.pages)} ({len(self.curr_pairs)} cases)...")
        start_idx = self.page_idx * 64
        for i, pair in enumerate(self.curr_pairs, start=1):
            try:
                pair_idx = start_idx + (i - 1)
                img, lbl = self.preprocessed_cache.load_pair(self.dataset_cache_dir, pair_idx)
                self.curr_images.append(img)
                self.curr_labels.append(lbl)
            except Exception as exc:
                print(f"[WARN] Skipping {pair.label_path.name}: {exc}")
        print(f"Loaded {len(self.curr_images)} cases on current page.")

        self.z_idx = int(np.clip(self.z_idx, 0, self.target_shape[2] - 1))

    def draw(self) -> None:
        if self.fig is None:
            self.fig, self.axes = plt.subplots(8, 8, figsize=(16, 9), constrained_layout=True)
            # Try fullscreen for monitor-sized layout.
            try:
                mgr = plt.get_current_fig_manager()
                if hasattr(mgr, "full_screen_toggle"):
                    mgr.full_screen_toggle()
            except Exception:
                pass

            self.fig.canvas.mpl_connect("scroll_event", self.on_scroll)
            self.fig.canvas.mpl_connect("key_press_event", self.on_key)

        ax_list = self.axes.ravel()
        for ax in ax_list:
            ax.set_axis_off()

        for i, ax in enumerate(ax_list):
            ax.clear()
            ax.set_axis_off()
            if i < len(self.curr_images):
                img3d = self.curr_images[i]
                lbl3d = self.curr_labels[i]
                z = int(np.clip(self.z_idx, 0, img3d.shape[2] - 1))
                rgb = overlay_rgb(img3d[:, :, z], lbl3d[:, :, z])
                ax.imshow(np.transpose(rgb, (1, 0, 2)), interpolation="nearest")
                label_name = self.curr_pairs[i].label_path.stem.replace(".nii", "")
                ax.set_title(label_name, fontsize=7)

        self.fig.suptitle(
            f"Bone Viewer | Page {self.page_idx + 1}/{len(self.pages)} | Z {self.z_idx + 1}/{self.target_shape[2]} "
            f"| Scroll: Z  Left/Right or N/P: page  Home/End: first/last  Q: quit",
            fontsize=10,
        )
        self.fig.canvas.draw_idle()

        if self.export_page_jpg is not None:
            out = self.export_page_jpg
            if out.is_dir() or out.suffix == "":
                out.mkdir(parents=True, exist_ok=True)
                out = out / f"bone_grid_page_{self.page_idx + 1:03d}_z_{self.z_idx + 1:03d}.jpg"
            else:
                out.parent.mkdir(parents=True, exist_ok=True)
            self.fig.savefig(out, dpi=160, format="jpg")

    def on_scroll(self, event) -> None:
        step = 1 if event.button == "up" else -1
        self.z_idx = int(np.clip(self.z_idx + step, 0, self.target_shape[2] - 1))
        self.draw()

    def on_key(self, event) -> None:
        key = (event.key or "").lower()
        if key in ("right", "n", "pagedown"):
            if self.page_idx < len(self.pages) - 1:
                self.load_page(self.page_idx + 1)
                self.draw()
        elif key in ("left", "p", "pageup"):
            if self.page_idx > 0:
                self.load_page(self.page_idx - 1)
                self.draw()
        elif key in ("home",):
            self.load_page(0)
            self.draw()
        elif key in ("end",):
            self.load_page(len(self.pages) - 1)
            self.draw()
        elif key in ("q", "escape"):
            plt.close(self.fig)

    def run(self) -> None:
        self.load_page(0)
        self.draw()
        plt.show()


def main() -> None:
    args = parse_args()
    labels_dir: Path = args.labels_dir
    images_dir: Path = args.images_dir

    if not labels_dir.exists():
        raise SystemExit(f"Labels directory does not exist: {labels_dir}")
    if not images_dir.exists():
        raise SystemExit(f"Images directory does not exist: {images_dir}")

    labels_by_case = discover_labels(labels_dir)
    images_by_case = discover_images(images_dir)
    pairs = build_pairs(labels_by_case, images_by_case)

    if args.max_cases is not None:
        pairs = pairs[: max(1, int(args.max_cases))]

    if not pairs:
        raise SystemExit(
            "No matched image/label pairs were found. "
            "Ensure image files contain case IDs like bone_00523 in their filenames."
        )

    zc = ZipCache()
    target_shape = tuple(args.target_shape) if args.target_shape is not None else infer_target_shape(pairs[0], zc)
    preprocessed_cache = PreprocessedDatasetCache()
    dataset_cache_dir = preprocessed_cache.ensure_preprocessed(
        labels_dir,
        images_dir,
        pairs,
        target_shape,
        zc,
        workers=max(1, int(args.preprocess_workers)),
        compressed=bool(args.cache_compressed),
    )

    print(f"Matched pairs: {len(pairs)}")
    print(f"Pages (64 per page): {math.ceil(len(pairs) / 64)}")
    print(f"Target shape XYZ: {target_shape}")
    print(f"Preprocessed cache: {dataset_cache_dir}")

    viewer = GridViewer(
        pairs,
        target_shape=target_shape,
        dataset_cache_dir=dataset_cache_dir,
        preprocessed_cache=preprocessed_cache,
        export_page_jpg=args.export_page_jpg,
    )
    viewer.run()


if __name__ == "__main__":
    main()
