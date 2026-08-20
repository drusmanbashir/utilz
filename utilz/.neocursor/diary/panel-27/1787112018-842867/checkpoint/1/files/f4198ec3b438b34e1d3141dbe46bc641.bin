from pathlib import Path

import pandas as pd

from utilz.fileio import is_img_file
from utilz.stringz import cleanup_fname

SKIP_NAMES = frozenset({"notes.csv", "notes.xlsx"})
IMAGES_DIR = "images"
LMS_DIR = "lms"


def _case_ids_from_folder(folder: Path) -> list[str]:  #AI
    """Collect sorted unique case IDs from image filenames in folder."""
    case_ids = sorted(
        {
            cleanup_fname(path.name)
            for path in folder.iterdir()
            if path.is_file()
            and not path.name.startswith(".")
            and path.name not in SKIP_NAMES
            and is_img_file(path)
        }
    )
    return case_ids


def _resolve_case_folder_and_output(folder: Path) -> tuple[Path, Path]:  #AI
    """Resolve source folder for case IDs and parent folder for notes.csv."""
    folder = folder.resolve()
    images_sub = folder / IMAGES_DIR
    lms_sub = folder / LMS_DIR
    if images_sub.is_dir() and lms_sub.is_dir():
        return images_sub, folder

    if folder.name in (IMAGES_DIR, LMS_DIR):
        parent = folder.parent
        sibling_images = parent / IMAGES_DIR
        sibling_lms = parent / LMS_DIR
        if sibling_images.is_dir() and sibling_lms.is_dir():
            return sibling_images, parent

    return folder, folder


def make_case_notes_csv(folder: Path) -> Path:  #AI
    """Build notes.csv with case_id from image filenames and empty notes column."""
    case_folder, output_folder = _resolve_case_folder_and_output(folder)
    case_ids = _case_ids_from_folder(case_folder)
    df = pd.DataFrame({"case_id": case_ids, "notes": ""})
    out_path = output_folder / "notes.csv"
    df.to_csv(out_path, index=False)
    return out_path
