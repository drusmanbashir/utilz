#!/usr/bin/env python3
import argparse
import json

from utilz.fileio import delete_files


def main():
    parser = argparse.ArgumentParser(description="Delete one or more files.")
    parser.add_argument("files", nargs="*", help="File paths to delete")
    parser.add_argument(
        "--files-json",
        dest="files_json",
        default=None,
        help='JSON list of file paths, e.g. \'["/tmp/a","/tmp/b"]\'',
    )
    args = parser.parse_args()
    files = args.files
    if args.files_json is not None:
        files = json.loads(args.files_json)
    deleted = delete_files(files)
    print(json.dumps({"deleted_files": deleted}, indent=2))


if __name__ == "__main__":
    main()
