#!/usr/bin/env python3
import argparse
from pathlib import Path

from utilz.case_notes_csv import make_case_notes_csv


def main(args):
    out_path = make_case_notes_csv(Path(args.folder))
    print(out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build notes.csv with case_id and empty notes from folder filenames."
    )
    parser.add_argument(
        "--folder",
        required=True,
        help="Folder containing case volume files (e.g. prediction output folder)",
    )
    args = parser.parse_known_args()[0]
    main(args)
