import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

from utilz.helpers import is_close, multiprocess_multiarg


def _vec3_delta(a, b):
    return [float(x - y) for x, y in zip(a, b)]


def _direction_delta(a, b):
    da = np.array(a, dtype=float).reshape(3, 3)
    db = np.array(b, dtype=float).reshape(3, 3)
    return (da - db).reshape(-1).tolist()


def _meta_row(case_id, img, lm, within_tol, aligned):
    img_spacing = img.GetSpacing()
    lm_spacing = lm.GetSpacing()
    img_origin = img.GetOrigin()
    lm_origin = lm.GetOrigin()
    img_direction = img.GetDirection()
    lm_direction = lm.GetDirection()
    spacing_diff = _vec3_delta(lm_spacing, img_spacing)
    origin_diff = _vec3_delta(lm_origin, img_origin)
    direction_diff = _direction_delta(lm_direction, img_direction)
    row = {
        "case_id": case_id,
        "within_tol": within_tol,
        "aligned": aligned,
        "img_spacing_x": img_spacing[0],
        "img_spacing_y": img_spacing[1],
        "img_spacing_z": img_spacing[2],
        "lm_spacing_x": lm_spacing[0],
        "lm_spacing_y": lm_spacing[1],
        "lm_spacing_z": lm_spacing[2],
        "spacing_delta_x": spacing_diff[0],
        "spacing_delta_y": spacing_diff[1],
        "spacing_delta_z": spacing_diff[2],
        "img_origin_x": img_origin[0],
        "img_origin_y": img_origin[1],
        "img_origin_z": img_origin[2],
        "lm_origin_x": lm_origin[0],
        "lm_origin_y": lm_origin[1],
        "lm_origin_z": lm_origin[2],
        "origin_delta_x": origin_diff[0],
        "origin_delta_y": origin_diff[1],
        "origin_delta_z": origin_diff[2],
    }
    for i in range(9):
        row[f"img_direction_{i}"] = img_direction[i]
        row[f"lm_direction_{i}"] = lm_direction[i]
        row[f"direction_delta_{i}"] = direction_diff[i]
    return row


def _align_case(img_fn, lm_fn, out_fn, tol):
    img = sitk.ReadImage(str(img_fn))
    lm = sitk.ReadImage(str(lm_fn))

    within_tol = (
        is_close(img.GetSpacing(), lm.GetSpacing(), tol)
        and is_close(img.GetOrigin(), lm.GetOrigin(), tol)
        and is_close(img.GetDirection(), lm.GetDirection(), tol)
    )

    out_fn.parent.mkdir(parents=True, exist_ok=True)
    if within_tol:
        shutil.copy2(lm_fn, out_fn)
        aligned = False
    else:
        out_lm = sitk.Image(lm)
        out_lm.SetSpacing(img.GetSpacing())
        out_lm.SetOrigin(img.GetOrigin())
        out_lm.SetDirection(img.GetDirection())
        sitk.WriteImage(out_lm, str(out_fn))
        aligned = True

    row = _meta_row(lm_fn.name, img, lm, within_tol, aligned)
    return row


def align_images_lms(
    dataset_root,
    tol=1e-5,
    images_subdir="images",
    lms_subdir="lms",
    out_subdir="lms_aligned",
    csv_name="lms_metadata_deltas.csv",
    num_processes=16,
):
    root = Path(dataset_root)
    images_dir = root / images_subdir
    lms_dir = root / lms_subdir
    out_dir = root / out_subdir
    csv_fn = root / csv_name

    image_names = {p.name for p in images_dir.glob("*.nii.gz")}
    lm_names = {p.name for p in lms_dir.glob("*.nii.gz")}
    common = sorted(image_names & lm_names)

    args = [
        (images_dir / name, lms_dir / name, out_dir / name, tol) for name in common
    ]
    rows = multiprocess_multiarg(
        _align_case, args, num_processes=num_processes, io=True
    )
    rows = sorted(rows, key=lambda row: row["case_id"])
    pd.DataFrame(rows).to_csv(csv_fn, index=False)

    n_within_tol = sum(row["within_tol"] for row in rows)
    n_aligned = sum(row["aligned"] for row in rows)
    summary = {
        "cases": len(common),
        "within_tol": n_within_tol,
        "aligned": n_aligned,
        "out_dir": out_dir,
        "csv_fn": csv_fn,
        "missing_images": sorted(lm_names - image_names),
        "missing_lms": sorted(image_names - lm_names),
    }
    return summary


def main(args):
    summary = align_images_lms(
        args.dataset_root,
        tol=args.tol,
        images_subdir=args.images_subdir,
        lms_subdir=args.lms_subdir,
        out_subdir=args.out_subdir,
        csv_name=args.csv_name,
        num_processes=args.num_processes,
    )
    print(f"cases: {summary['cases']}")
    print(f"within tol: {summary['within_tol']}")
    print(f"metadata aligned: {summary['aligned']}")
    print(f"aligned folder: {summary['out_dir']}")
    print(f"delta csv: {summary['csv_fn']}")
    if summary["missing_images"]:
        print(f"missing images for {len(summary['missing_images'])} lms")
    if summary["missing_lms"]:
        print(f"missing lms for {len(summary['missing_lms'])} images")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Align lms NIfTI metadata to images within tolerance and record CSV deltas."
    )
    parser.add_argument(
        "dataset_root",
        help="Dataset root containing images/, lms/, writes lms_aligned/ and csv",
    )
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument("--images-subdir", default="images")
    parser.add_argument("--lms-subdir", default="lms")
    parser.add_argument("--out-subdir", default="lms_aligned")
    parser.add_argument("--csv-name", default="lms_metadata_deltas.csv")
    parser.add_argument("--num-processes", type=int, default=16)
    main(parser.parse_args())
