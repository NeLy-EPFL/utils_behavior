#!/usr/bin/env python3
"""Ensure every SLEAP ``.slp`` in the F1_Tracks videos tree has an analysis ``.h5``.

uv-native (no conda): conversion goes through
:func:`utils_behavior.sleap.convert.slp_to_analysis_h5`, which runs
``sleap_io.save_analysis_h5`` in an isolated ``uv run --with sleap-io`` env.

Run with::

    uv run python scripts/sleap/convert_h5.py [--data-path DIR] [--overwrite]
"""

import argparse
from pathlib import Path

from utils_behavior.sleap.convert import slp_to_analysis_h5

DEFAULT_DATA_PATH = Path("/mnt/upramdya_data/MD/F1_Tracks/Videos/")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH, help="Root directory to search for .slp files")
    parser.add_argument("--overwrite", action="store_true", help="Reconvert even if the .h5 already exists")
    args = parser.parse_args()

    data_path = args.data_path
    if not data_path.is_dir():
        raise SystemExit(f"Data path not found: {data_path}")

    slp_files = sorted(data_path.rglob("*.slp"))
    print(f"Found {len(slp_files)} .slp file(s) under {data_path}")

    converted = skipped = 0
    for slp in slp_files:
        h5 = slp.with_suffix(".h5")
        if h5.exists() and not args.overwrite:
            skipped += 1
            continue
        slp_to_analysis_h5(slp, h5)
        converted += 1
        print(f"  [ok] {slp.name} -> {h5.name}")

    print(f"\nConverted {converted}, skipped {skipped} (already had .h5).")
    print("Total ball h5 files:", len(list(data_path.rglob("*ball*.h5"))))
    print("Total fly h5 files:", len(list(data_path.rglob("*fly*.h5"))))


if __name__ == "__main__":
    main()
