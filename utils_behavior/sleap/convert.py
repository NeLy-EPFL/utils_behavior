#!/usr/bin/env python3
"""Convert SLEAP ``.slp`` files to analysis ``.h5`` (uv-native, no conda).

The conversion runs ``sleap_io.save_analysis_h5`` inside an isolated
``uv run --with sleap-io`` environment. This is independent of whatever
interpreter/conda env launched the caller, which avoids the library conflicts
(e.g. a conda env's Qt6 libs) that crash the uv ``sleap-convert`` tool when its
subprocess inherits a polluted environment.

The produced ``.h5`` uses the SLEAP analysis schema (``tracks``, ``node_names``,
``edge_names``, ``point_scores``, ``instance_scores``, ``track_occupancy`` ...),
which is the same regardless of whether the ``.slp`` came from the legacy
TensorFlow backend or the new sleap-nn backend, so existing readers keep working.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

# sleap_io >=0.7 exposes ``save_analysis_h5`` (older ``Labels.export`` does not
# exist). Passed to ``python -c`` so paths arrive via argv with no quoting pain.
# A ``.slp`` with no labeled frames (the model tracked nothing) cannot become an
# analysis ``.h5`` — ``save_analysis_h5`` raises "No labeled frames in video".
# We detect that up front and exit with ``_EMPTY_EXIT`` so the caller can skip
# the video instead of aborting the whole batch.
_EMPTY_EXIT = 2
_EXPORT = (
    "import sys, sleap_io as sio; "
    "labels = sio.load_file(sys.argv[1]); "
    "sys.exit(2) if not labels.labeled_frames "
    "else sio.save_analysis_h5(labels, sys.argv[2])"
)


def clean_env() -> dict:
    """Environment for SLEAP subprocesses, scrubbed of conda library pollution.

    The uv-installed tools and the isolated uv env are self-contained; inheriting
    a conda env's ``LD_LIBRARY_PATH`` / ``PYTHONPATH`` can inject conflicting
    shared libraries or site-packages, so drop them for the children.
    """
    env = os.environ.copy()
    env.pop("LD_LIBRARY_PATH", None)
    env.pop("PYTHONPATH", None)
    return env


def slp_to_analysis_h5(slp_path, h5_path=None, check=True):
    """Convert one ``.slp`` to an analysis ``.h5``.

    Args:
        slp_path: Path to the ``.slp`` file.
        h5_path: Output path. Defaults to the ``.slp`` path with a ``.h5``
            suffix (e.g. ``foo_tracked.slp`` -> ``foo_tracked.h5``).
        check: Raise on a genuine non-zero exit (default True).

    Returns:
        The output ``.h5`` path, or ``None`` if the ``.slp`` had no labeled
        frames (nothing was tracked) and so no analysis ``.h5`` was written.
    """
    slp_path = Path(slp_path)
    h5_path = slp_path.with_suffix(".h5") if h5_path is None else Path(h5_path)
    result = subprocess.run(
        [
            "uv", "run", "--no-project", "--with", "sleap-io",
            "python", "-c", _EXPORT, str(slp_path), str(h5_path),
        ],
        check=False,
        env=clean_env(),
    )
    if result.returncode == _EMPTY_EXIT:
        return None
    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, result.args)
    return h5_path


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert SLEAP .slp files to analysis .h5 (uv-native)."
    )
    parser.add_argument("paths", nargs="+", help="One or more .slp files or directories to search")
    parser.add_argument("--recursive", action="store_true", help="Search directories recursively for *.slp")
    parser.add_argument("--pattern", default="*.slp", help="Glob used when a path is a directory")
    parser.add_argument("--overwrite", action="store_true", help="Reconvert even if the .h5 already exists")
    parser.add_argument("--suffix", default="", help="Optional suffix added before .h5 (e.g. '_processed')")
    args = parser.parse_args()

    slp_files: list[Path] = []
    for p in args.paths:
        path = Path(p)
        if path.is_dir():
            globber = path.rglob if args.recursive else path.glob
            slp_files.extend(sorted(globber(args.pattern)))
        else:
            slp_files.append(path)

    if not slp_files:
        print("No .slp files found.")
        return

    print(f"Converting {len(slp_files)} .slp file(s)...")
    for slp in slp_files:
        h5 = slp.with_name(f"{slp.stem}{args.suffix}.h5")
        if h5.exists() and not args.overwrite:
            print(f"  [skip] {h5.name} exists")
            continue
        slp_to_analysis_h5(slp, h5)
        print(f"  [ok]   {slp.name} -> {h5.name}")


if __name__ == "__main__":
    main()
