# FUNCTIONS.md

Scope: only code under `run/`, `cli/`, or `tools/` belongs here.

## CLI
- `run/case_notes_csv_maker.py`: scan a folder of image filenames, write `notes.csv` with `case_id` and empty `notes`. If the folder has `images/` and `lms/` subfolders, reads case IDs from `images/` and writes `notes.csv` in the parent. If called on `images/` or `lms/` with the sibling present, same: IDs from `images/`, output in their parent.
- `run/delete_files.py`: delete one or more files from positional args or `--files-json`.

## Functions
- `main` (`run/case_notes_csv_maker.py`): parse `--folder`, call `utilz.case_notes_csv.make_case_notes_csv`, print output path.
- `make_case_notes_csv` (`utilz/case_notes_csv.py`): extract case IDs from image filenames in a folder and write `notes.csv`.
- `main` (`run/delete_files.py`): parse args, call `utilz.fileio.delete_files`, and print deleted paths as JSON.
