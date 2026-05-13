"""Copy a nested confocal-stacks directory into a flat layout for Dataverse upload.

For every .tiff/.tif found under a top-level subdirectory the file is renamed
``<top_level_dir_name>.tiff`` and placed directly in the destination root.
Files that already sit at the source root (e.g. a .yaml manifest) are copied
as-is.  Directories listed in ``--exclude-dir`` and OS junk files (.DS_Store
etc.) are silently skipped.

Run with ``python -m utils_behavior.dataverse.prepare_flat_copy --help``.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

# Same ignore list as upload.py
IGNORED_FILENAMES: frozenset[str] = frozenset(
    {
        ".DS_Store",
        "Thumbs.db",
        "desktop.ini",
        ".Spotlight-V100",
        ".Trashes",
        ".fseventsd",
    }
)

TIFF_SUFFIXES: frozenset[str] = frozenset({".tiff", ".tif"})

_DATE_PREFIX_RE = re.compile(r"^\d{6,8}_")


def strip_date_prefix(name: str) -> str:
    """Remove a leading YYMMDD_ or YYYYMMDD_ date prefix if present."""
    return _DATE_PREFIX_RE.sub("", name)


def collect_source_files(
    source: Path,
    exclude_dirs: set[str],
) -> list[Path]:
    """Return all non-ignored files under *source*, excluding *exclude_dirs*."""
    result: list[Path] = []
    for p in sorted(source.rglob("*")):
        if not p.is_file():
            continue
        if p.name in IGNORED_FILENAMES:
            continue
        rel = p.relative_to(source)
        if any(part in exclude_dirs for part in rel.parts[:-1]):
            continue
        result.append(p)
    return result


def destination_name(source_root: Path, file: Path) -> str:
    """Derive the flat destination filename for *file*.

    - Files at the source root keep their original name.
    - .tiff/.tif files anywhere deeper get renamed to
      ``<top_level_dir>.tiff``.
    - Any other nested file keeps its original name (with a warning if a
      collision would occur).
    """
    rel = file.relative_to(source_root)
    parts = rel.parts

    if len(parts) == 1:
        # Already at root level (e.g. stack_infos.yaml)
        return parts[0]

    top_dir = parts[0]
    if file.suffix.lower() in TIFF_SUFFIXES:
        return f"{strip_date_prefix(top_dir)}.tiff"

    # Non-tiff nested file: keep original name
    return file.name


def build_plan(
    source: Path,
    dest: Path,
    exclude_dirs: set[str],
) -> list[tuple[Path, Path]]:
    """Return list of (src, dst) pairs."""
    files = collect_source_files(source, exclude_dirs)
    plan: list[tuple[Path, Path]] = []
    seen: dict[str, Path] = {}

    for f in files:
        name = destination_name(source, f)
        dst = dest / name

        if name in seen:
            print(
                f"  WARNING: collision — '{f}' and '{seen[name]}' both map to "
                f"'{name}'. Second file will be skipped.",
                file=sys.stderr,
            )
            continue
        seen[name] = f
        plan.append((f, dst))

    return plan


def print_plan(plan: list[tuple[Path, Path]], dest: Path) -> None:
    print(f"\nDestination : {dest}")
    print(f"Files to copy/rename : {len(plan)}\n")
    for src, dst in plan:
        if src.name == dst.name:
            print(f"  {src.name}")
        else:
            print(f"  {src.name}  →  {dst.name}")
    print()


def execute_plan(plan: list[tuple[Path, Path]], dest: Path) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    errors = 0
    for idx, (src, dst) in enumerate(plan, 1):
        prefix = f"[{idx}/{len(plan)}]"
        try:
            shutil.copy2(src, dst)
            print(f"{prefix} {src.name}  →  {dst.name}")
        except Exception as exc:
            print(f"{prefix} FAILED {src.name}: {exc}", file=sys.stderr)
            errors += 1
    return errors


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="utils_behavior.dataverse.prepare_flat_copy",
        description=(
            "Copy a nested confocal-stacks directory into a flat layout, "
            "renaming .tiff files after their top-level parent directory."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("source", type=Path, help="Source directory")
    p.add_argument(
        "destination", type=Path, help="Destination directory (will be created)"
    )
    p.add_argument(
        "--exclude-dir",
        action="append",
        default=[],
        metavar="DIR",
        dest="exclude_dirs",
        help="Skip subdirectories with this name (can be repeated)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be copied/renamed without touching any files",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.source.exists():
        print(f"Source not found: {args.source}", file=sys.stderr)
        return 2

    exclude_dirs: set[str] = set(args.exclude_dirs)

    plan = build_plan(args.source, args.destination, exclude_dirs)

    if not plan:
        print("No files matched.")
        return 0

    print_plan(plan, args.destination)

    if args.dry_run:
        print("Dry run — no files were copied.")
        return 0

    if args.destination.exists() and any(args.destination.iterdir()):
        try:
            ans = (
                input(
                    f"Destination '{args.destination}' already exists and is not empty. "
                    "Continue? [y/N] "
                )
                .strip()
                .lower()
            )
        except (EOFError, KeyboardInterrupt):
            print()
            return 1
        if ans not in ("y", "yes"):
            print("Aborted.")
            return 1

    errors = execute_plan(plan, args.destination)

    print(f"\nDone. {len(plan) - errors} copied, {errors} failed.")
    return 0 if errors == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
