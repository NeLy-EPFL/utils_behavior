#!/usr/bin/env python3
"""Remove failed-attempt artifacts (DLC outputs, annotated videos) before tracking.

Dry-run by default: it prints exactly what it *would* delete and the space that
would be freed. Nothing is removed unless ``--apply`` is passed.

Default targets (the things explicitly flagged as junk):
  - ``*_tracked.mp4``         annotated/tracked videos from previous runs
  - ``*_tracked.pkl``         pickled outputs from previous runs
  - ``resultsDLC`` dirs       DLC per-clip outputs
  - ``*DLC*`` files           any stray DeepLabCut files
  - ``*.csv``                 stray CSV exports

Optional extras (only with the corresponding flag), because they are part of the
same failed pipeline but were not explicitly named:
  - ``--registered``          ``images_registered`` directories (large frame dumps)
  - ``--experiment-dict``     ``experiment_dict.npy`` files
  - ``--logs``                ``process_log.txt`` files

The raw arena ``*_80fps.mp4`` videos are never matched and never touched.

Usage:
    # See what would be removed (safe):
    python cleanup_failed_attempts.py /mnt/upramdya_data/SB/Optogenetics/Optobot/PG_20260521_Vglut_B6VF_1min
    # Actually delete:
    python cleanup_failed_attempts.py <dir> --apply
    # Include the big intermediate dirs too:
    python cleanup_failed_attempts.py <dir> --registered --experiment-dict --logs --apply
"""

import argparse
import fnmatch
import shutil
from pathlib import Path

# (label, kind, matcher) where kind is "file" or "dir".
FILE_PATTERNS = [
    ("tracked video", "*_tracked.mp4"),
    ("tracked pickle", "*_tracked.pkl"),
    ("DLC file", "*DLC*"),
    ("CSV", "*.csv"),
]
DIR_NAMES = ["resultsDLC"]

# Guard: never delete anything matching these, even if another pattern would.
PROTECTED = ["*_80fps.mp4"]


def _is_protected(path: Path) -> bool:
    return any(fnmatch.fnmatch(path.name, p) for p in PROTECTED)


def _dir_size(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            pass
    return total


def _human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def collect_targets(base: Path, include_registered, include_experiment_dict, include_logs):
    """Return a list of (path, label, size_bytes, is_dir) to remove."""
    targets = []
    seen = set()

    def add(path: Path, label: str, is_dir: bool):
        rp = path.resolve()
        if rp in seen or _is_protected(path):
            return
        seen.add(rp)
        size = _dir_size(path) if is_dir else path.stat().st_size
        targets.append((path, label, size, is_dir))

    # Directories first (so their files aren't also listed individually).
    dir_names = list(DIR_NAMES)
    if include_registered:
        dir_names.append("images_registered")
    removed_dirs = []
    for dname in dir_names:
        for d in base.rglob(dname):
            if d.is_dir():
                add(d, f"{dname}/ dir", True)
                removed_dirs.append(d.resolve())

    def inside_removed_dir(path: Path) -> bool:
        return any(str(path.resolve()).startswith(str(d) + "/") for d in removed_dirs)

    # Files.
    file_patterns = list(FILE_PATTERNS)
    if include_experiment_dict:
        file_patterns.append(("experiment dict", "experiment_dict.npy"))
    if include_logs:
        file_patterns.append(("process log", "process_log.txt"))

    for label, pattern in file_patterns:
        for f in base.rglob(pattern):
            if f.is_file() and not inside_removed_dir(f):
                add(f, label, False)

    return targets


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("paths", nargs="+", help="Directories (or files) to scan")
    parser.add_argument("--apply", action="store_true", help="Actually delete (default: dry run)")
    parser.add_argument("--registered", action="store_true", help="Also remove images_registered/ directories")
    parser.add_argument("--experiment-dict", action="store_true", help="Also remove experiment_dict.npy files")
    parser.add_argument("--logs", action="store_true", help="Also remove process_log.txt files")
    args = parser.parse_args()

    all_targets = []
    for p in args.paths:
        base = Path(p)
        if not base.exists():
            print(f"Skipping missing path: {base}")
            continue
        if base.is_file():
            if not _is_protected(base):
                all_targets.append((base, "file", base.stat().st_size, False))
            continue
        all_targets.extend(
            collect_targets(base, args.registered, args.experiment_dict, args.logs)
        )

    if not all_targets:
        print("Nothing to remove.")
        return

    total = sum(size for _, _, size, _ in all_targets)
    mode = "DELETING" if args.apply else "DRY RUN — would delete"
    print(f"{mode} {len(all_targets)} item(s), {_human(total)} total:\n")
    for path, label, size, is_dir in sorted(all_targets, key=lambda t: -t[2]):
        kind = "DIR " if is_dir else "FILE"
        print(f"  [{kind}] {_human(size):>9}  {label:<16} {path}")

    if not args.apply:
        print("\nNothing was deleted. Re-run with --apply to remove these.")
        return

    print()
    errors = 0
    for path, _, _, is_dir in all_targets:
        try:
            if is_dir:
                shutil.rmtree(path)
            else:
                path.unlink()
        except OSError as e:
            errors += 1
            print(f"  ERROR removing {path}: {e}")
    print(f"\nRemoved {len(all_targets) - errors}/{len(all_targets)} item(s), freed up to {_human(total)}.")


if __name__ == "__main__":
    main()
