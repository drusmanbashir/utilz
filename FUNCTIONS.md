# FUNCTIONS.md

Scope: `run/`

## CLI
- `run/case_notes_csv_maker.py`: scan an image folder and write notes.csv with case_id and empty notes; accepts --folder.
- `run/delete_files.py`: delete files passed as positional arguments or --files-json; print deleted paths as JSON.

## Libraries (import, not run)
- `utilz/helpers.py`: multiprocess_multiarg runs ordered tasks with spawn ProcessPoolExecutor, IO/algebra thread limits, worker oom_score_adj=1000, and None plus printed errors for failed tasks; serial/debug execution stays in the caller.
- `utilz/case_notes_csv.py`: make_case_notes_csv extracts case IDs from image filenames and writes notes.csv.
