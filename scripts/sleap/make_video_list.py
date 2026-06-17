#!/usr/bin/env python3
"""Generate a YAML list of videos to track for the Optobot Vglut experiment.

Walks the Optobot tree, selects experiment directories named
``PG_YYYYMMDD_Vglut_B6VF_1min`` whose date falls in a given range, and collects
the full-arena ``*_80fps.mp4`` videos (skipping the single-fly ``vid_*`` DLC clips
and any previously ``*_tracked*`` outputs).

The output YAML is consumed by ``SleapTracker`` (``--yaml_file``):

    videos:
      - /abs/path/to/PG_..._p6-0_80fps.mp4
      ...

Usage:
    python make_video_list.py \
        --base /mnt/upramdya_data/SB/Optogenetics/Optobot \
        --start 20260521 --end 20260605 \
        --output optobot_vglut_videos.yaml
"""

import argparse
import re
from pathlib import Path

import yaml

# PG_YYYYMMDD_Vglut_B6VF_1min  (tolerate optional extra token between Vglut/B6VF)
DIR_RE = re.compile(r"^PG_(\d{8})_Vglut_?B6VF_1min$", re.IGNORECASE)


def find_experiment_dirs(base: Path, start: int, end: int):
    """Yield (date_int, dir_path) for experiment dirs with date in [start, end]."""
    for child in sorted(base.iterdir()):
        if not child.is_dir():
            continue
        m = DIR_RE.match(child.name)
        if not m:
            continue
        date = int(m.group(1))
        if start <= date <= end:
            yield date, child


def collect_videos(exp_dir: Path, include="*_80fps.mp4", exclude=("*_tracked*", "vid_*")):
    """Find arena videos under an experiment dir, excluding clips/tracked outputs."""
    videos = []
    for video in sorted(exp_dir.rglob("*.mp4")):
        name = video.name
        if not _fnmatch_any(name, [include]):
            continue
        if _fnmatch_any(name, exclude):
            continue
        videos.append(video)
    return videos


def _fnmatch_any(name, patterns):
    from fnmatch import fnmatch

    return any(fnmatch(name, p) for p in patterns)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default="/mnt/upramdya_data/SB/Optogenetics/Optobot", help="Base directory to search")
    parser.add_argument("--start", type=int, default=20260521, help="Start date (YYYYMMDD, inclusive)")
    parser.add_argument("--end", type=int, default=20260605, help="End date (YYYYMMDD, inclusive)")
    parser.add_argument("--include", default="*_80fps.mp4", help="Glob for arena videos to include")
    parser.add_argument("--exclude", nargs="*", default=["*_tracked*", "vid_*"], help="Globs to exclude")
    parser.add_argument("--output", default="optobot_vglut_videos.yaml", help="Output YAML path")
    parser.add_argument("--list-dirs", action="store_true", help="Also include a directories: block")
    parser.add_argument(
        "--extra-dirs",
        nargs="*",
        default=[],
        help="Additional directories to scan wholesale (any name/date), e.g. the 'PG' dir. "
        "Paths may be absolute or relative to --base.",
    )
    args = parser.parse_args()

    base = Path(args.base)
    if not base.is_dir():
        raise SystemExit(f"Base directory not found: {base}")

    all_videos = []
    matched_dirs = []
    print(f"Searching {base} for PG_YYYYMMDD_Vglut_B6VF_1min in [{args.start}, {args.end}]\n")
    for date, exp_dir in find_experiment_dirs(base, args.start, args.end):
        videos = collect_videos(exp_dir, include=args.include, exclude=tuple(args.exclude))
        matched_dirs.append(str(exp_dir))
        print(f"  {exp_dir.name}: {len(videos)} video(s)")
        all_videos.extend(str(v) for v in videos)

    for extra in args.extra_dirs:
        extra_dir = Path(extra)
        if not extra_dir.is_absolute():
            extra_dir = base / extra
        if not extra_dir.is_dir():
            print(f"  [extra] skipping missing dir: {extra_dir}")
            continue
        videos = collect_videos(extra_dir, include=args.include, exclude=tuple(args.exclude))
        matched_dirs.append(str(extra_dir))
        print(f"  [extra] {extra_dir.name}: {len(videos)} video(s)")
        all_videos.extend(str(v) for v in videos)

    all_videos = sorted(set(all_videos))
    data = {"videos": all_videos}
    if args.list_dirs:
        data = {"directories": sorted(matched_dirs), "videos": all_videos}

    out_path = Path(args.output)
    with open(out_path, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)

    print(f"\nTotal: {len(all_videos)} videos across {len(matched_dirs)} directories.")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
