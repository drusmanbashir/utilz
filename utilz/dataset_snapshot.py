from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

from utilz.fileio import sitk_filename_to_numpy
from utilz.helpers import create_df_from_folder, str_to_path
from utilz.image_utils import get_bbox_from_mask


def _slice_center(bounds: slice, axis_size: int) -> int:
    if bounds.stop <= bounds.start:
        return axis_size // 2
    return int((bounds.start + bounds.stop - 1) // 2)


def _bbox_center_slices(mask_zyx: np.ndarray) -> tuple[int, int]:
    if np.count_nonzero(mask_zyx) == 0:
        return mask_zyx.shape[0] // 2, mask_zyx.shape[1] // 2
    bbox, _ = get_bbox_from_mask(mask_zyx)
    axial_idx = _slice_center(bbox[0], mask_zyx.shape[0])
    coronal_idx = _slice_center(bbox[1], mask_zyx.shape[1])
    return axial_idx, coronal_idx


def _normalize_image(slice_2d: np.ndarray) -> np.ndarray:
    lo = np.percentile(slice_2d, 1.0)
    hi = np.percentile(slice_2d, 99.0)
    if hi <= lo:
        lo = float(slice_2d.min())
        hi = float(slice_2d.max())
        if hi <= lo:
            return np.zeros_like(slice_2d, dtype=np.float32)
    return np.clip((slice_2d - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _extract_views(image_zyx: np.ndarray, mask_zyx: np.ndarray) -> tuple[np.ndarray, ...]:
    axial_idx, coronal_idx = _bbox_center_slices(mask_zyx)
    axial_img = np.rot90(_normalize_image(image_zyx[axial_idx]))
    axial_lm = np.rot90(mask_zyx[axial_idx])
    coronal_img = np.rot90(_normalize_image(image_zyx[:, coronal_idx, :]))
    coronal_lm = np.rot90(mask_zyx[:, coronal_idx, :])
    return axial_img, axial_lm, coronal_img, coronal_lm


def _iter_rows(axes: np.ndarray) -> Iterable[np.ndarray]:
    if axes.ndim == 1:
        yield axes
        return
    for row in axes:
        yield row


@str_to_path(0)
def create_dataset_snapshot(
    dataset_root: Path,
    n: int = 3,
    output_path: Path | None = None,
    dpi: int = 150,
):
    """
    Build an n-row, 4-column snapshot grid for a dataset with images/ and lms/.

    Each row contains: axial image, axial labelmap, coronal image, coronal labelmap.
    Slice selection is driven by the central slice of the non-zero label bounding box.
    """
    df = create_df_from_folder(dataset_root)
    if len(df) == 0:
        raise ValueError(f"No image/label pairs found in {dataset_root}")

    pairs = df.head(n)
    fig, axes = plt.subplots(len(pairs), 4, figsize=(12, 3 * len(pairs)), dpi=dpi)
    axes_rows = list(_iter_rows(np.asarray(axes, dtype=object)))
    col_titles = ("Axial image", "Axial lm", "Coronal image", "Coronal lm")

    for row_idx, (_, pair) in enumerate(pairs.iterrows()):
        image = sitk_filename_to_numpy(pair["image"])
        mask = sitk_filename_to_numpy(pair["lm"])
        if image.shape != mask.shape:
            raise ValueError(
                f"Shape mismatch for {pair['image']} and {pair['lm']}: {image.shape} != {mask.shape}"
            )
        views = _extract_views(image, mask)
        for col_idx, (ax, view) in enumerate(zip(axes_rows[row_idx], views)):
            cmap = "gray" if col_idx in (0, 2) else "viridis"
            ax.imshow(view, cmap=cmap, interpolation="nearest")
            ax.axis("off")
            if row_idx == 0:
                ax.set_title(col_titles[col_idx])
        axes_rows[row_idx][0].set_ylabel(str(pair["case_id"]), rotation=90, fontsize=9)

    fig.tight_layout()
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create a simple dataset snapshot grid.")
    parser.add_argument("dataset_root", type=Path, help="Folder containing images/ and lms/")
    parser.add_argument("-n", type=int, default=3, help="Number of image/label pairs to show")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Optional output image path")
    args = parser.parse_args()
    create_dataset_snapshot(args.dataset_root, n=args.n, output_path=args.output)
    if args.output is None:
        plt.show()
