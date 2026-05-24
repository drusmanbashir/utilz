# FUNCTIONS.md

Scope: only code under `run/`, `cli/`, or `tools/` belongs here.

## CLI
- `run/delete_files.py`: delete one or more files from positional args or `--files-json`.

## Functions
- `main`: parse args, call `utilz.fileio.delete_files`, and print deleted paths as JSON.
