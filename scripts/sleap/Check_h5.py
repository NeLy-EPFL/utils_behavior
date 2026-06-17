#!/usr/bin/env python3
"""Validate SLEAP analysis ``.h5`` files and regenerate any that are corrupt.

uv-native (no conda): regeneration goes through
:func:`utils_behavior.sleap.convert.slp_to_analysis_h5`.

Run with::

    uv run python scripts/sleap/Check_h5.py [--data-path DIR] [--pattern '*ball*']
"""

import argparse
from pathlib import Path

import h5py

from utils_behavior.sleap.convert import slp_to_analysis_h5

DEFAULT_DATA_PATH = Path("/mnt/upramdya_data/MD/F1_Tracks/Videos/")


def is_h5_file_valid(file_path):
    """Return True if the h5 file opens and is readable."""
    try:
        with h5py.File(file_path, "r") as f:
            _ = list(f.keys())
        return True
    except Exception as e:
        print(f"Invalid h5 file: {file_path}. Error: {e}")
        return False


def regenerate_invalid_h5_files(h5_files, slp_file):
    """Remove invalid h5 files and regenerate them from ``slp_file``."""
    for h5_file in h5_files:
        if not is_h5_file_valid(h5_file):
            Path(h5_file).unlink()
            print(f"Removed invalid h5 file: {h5_file}")
            slp_to_analysis_h5(slp_file, h5_file)
            print(f"Regenerated h5 file: {h5_file}")


def generate_h5_file(slp_file, h5_file=None):
    """Generate an analysis h5 file from an slp file."""
    out = slp_to_analysis_h5(slp_file, h5_file)
    print(f"Generated h5 file from slp file: {slp_file} -> {out}")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH, help="Root directory to search")
    parser.add_argument("--pattern", default="*ball*", help="Base glob for slp/h5 pairs to check (without extension)")
    args = parser.parse_args()

    data_path = args.data_path
    if not data_path.is_dir():
        raise SystemExit(f"Data path not found: {data_path}")

    video_folders = {folder.parent for folder in data_path.rglob("*.mp4")}
    print(f"Checking {len(video_folders)} video folder(s)...")

    for folder in video_folders:
        slp_files = list(folder.rglob(f"{args.pattern}.slp"))
        h5_files = list(folder.rglob(f"{args.pattern}.h5"))

        if not slp_files:
            print(f"No .slp file found in folder: {folder}")
            continue

        slp_file = slp_files[0]  # one tracking .slp per folder
        if h5_files:
            regenerate_invalid_h5_files(h5_files, slp_file)
        else:
            generate_h5_file(slp_file)


if __name__ == "__main__":
    main()
