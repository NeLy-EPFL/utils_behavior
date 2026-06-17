#!/usr/bin/env python3
"""Denoise SLEAP analysis ``.h5`` files by removing parasite tracks.

A "parasite" is a spurious instance, usually detected on top of a real fly with a
low confidence score (< ~0.3), that gets assigned its own short-lived track. Real
flies, by contrast, have a track present for most of the video.

The cleanup keeps tracks that are (a) present for a large enough fraction of the
video **and** (b) have a high enough mean confidence score, capped at a maximum
number of identities (3 flies here). It then optionally drops any remaining track
that spatially overlaps a higher-confidence kept track (a parasite sitting on top
of a real fly). Everything is parameterised so the policy can be tuned per dataset.

The function operates purely on the analysis HDF5 schema (``tracks``,
``point_scores``, ``instance_scores``, ``track_names``, ``track_occupancy`` ...),
so it works for both legacy ``sleap-track`` and new ``sleap-nn`` exports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np


@dataclass
class TrackStats:
    """Per-track summary used to decide whether to keep a track."""

    index: int
    name: str
    occupancy_frac: float
    mean_score: float
    n_frames_present: int
    kept: bool = False
    reason: str = ""


@dataclass
class CleanResult:
    """Outcome of cleaning a single file."""

    input_path: Path
    output_path: Path
    n_tracks_in: int
    n_tracks_out: int
    kept_indices: list = field(default_factory=list)
    stats: list = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"{self.input_path.name}: {self.n_tracks_in} -> {self.n_tracks_out} tracks",
        ]
        for s in self.stats:
            flag = "KEEP" if s.kept else "drop"
            lines.append(
                f"  [{flag}] {s.name}: occupancy={s.occupancy_frac:.2%} "
                f"score={s.mean_score:.3f} ({s.reason})"
            )
        return "\n".join(lines)


def _present_mask(track_xy: np.ndarray) -> np.ndarray:
    """Boolean mask over frames where a track has at least one visible node.

    Args:
        track_xy: array of shape (2, n_nodes, n_frames) for one track.
    """
    # A frame is "present" if any node has a non-NaN coordinate.
    return ~np.all(np.isnan(track_xy), axis=(0, 1))


def _mean_score(
    idx: int,
    present: np.ndarray,
    instance_scores: np.ndarray | None,
    point_scores: np.ndarray | None,
) -> float:
    """Mean confidence over the frames a track is present.

    Prefers instance scores, falls back to mean point score, then to 1.0 if no
    score datasets are available (legacy files without scores -> never dropped on
    score grounds).
    """
    if not present.any():
        return 0.0
    if instance_scores is not None:
        vals = instance_scores[idx][present]
        vals = vals[~np.isnan(vals)]
        if vals.size:
            return float(np.mean(vals))
    if point_scores is not None:
        vals = point_scores[idx][:, present]
        vals = vals[~np.isnan(vals)]
        if vals.size:
            return float(np.mean(vals))
    return 1.0


def _centroids(tracks: np.ndarray) -> np.ndarray:
    """Per-track, per-frame centroid. Shape (n_tracks, 2, n_frames)."""
    # tracks: (n_tracks, 2, n_nodes, n_frames); average over nodes ignoring NaN.
    # Frames where a track is absent are all-NaN -> nanmean warns; suppress it.
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(tracks, axis=2)


def _overlaps(
    cen_a: np.ndarray,
    cen_b: np.ndarray,
    present_a: np.ndarray,
    present_b: np.ndarray,
    dist_thresh: float,
    min_overlap_frac: float,
) -> bool:
    """True if track A sits "on top of" track B for a large fraction of A's frames."""
    shared = present_a & present_b
    if not shared.any():
        return False
    d = np.sqrt(
        (cen_a[0, shared] - cen_b[0, shared]) ** 2
        + (cen_a[1, shared] - cen_b[1, shared]) ** 2
    )
    close = np.sum(d < dist_thresh)
    # Fraction relative to A's own presence (A is the suspected parasite).
    denom = max(int(present_a.sum()), 1)
    return (close / denom) >= min_overlap_frac


def select_tracks_to_keep(
    tracks: np.ndarray,
    track_names: list[str],
    instance_scores: np.ndarray | None = None,
    point_scores: np.ndarray | None = None,
    score_threshold: float = 0.3,
    min_occupancy: float = 0.5,
    max_tracks: int = 3,
    overlap_dist: float = 20.0,
    overlap_frac: float = 0.7,
    remove_overlapping: bool = True,
) -> list[TrackStats]:
    """Decide which tracks to keep.

    Policy:
      1. A track qualifies if it is present for at least ``min_occupancy`` of the
         video AND has a mean score >= ``score_threshold``.
      2. Qualifying tracks are ranked by (mean_score, occupancy) and capped to
         ``max_tracks``.
      3. Optionally, any kept track that overlaps (sits on top of) a
         higher-ranked kept track is dropped as a parasite.

    Returns a list of :class:`TrackStats`, one per input track, with ``kept`` set.
    """
    n_tracks, _, _, n_frames = tracks.shape
    stats: list[TrackStats] = []

    for i in range(n_tracks):
        present = _present_mask(tracks[i])
        n_present = int(present.sum())
        occ = n_present / n_frames if n_frames else 0.0
        score = _mean_score(i, present, instance_scores, point_scores)
        name = track_names[i] if i < len(track_names) else f"track_{i}"
        stats.append(
            TrackStats(
                index=i,
                name=name,
                occupancy_frac=occ,
                mean_score=score,
                n_frames_present=n_present,
            )
        )

    # Step 1 + 2: qualify and rank.
    qualifiers = [
        s
        for s in stats
        if s.occupancy_frac >= min_occupancy and s.mean_score >= score_threshold
    ]
    qualifiers.sort(key=lambda s: (s.mean_score, s.occupancy_frac), reverse=True)
    kept = qualifiers[:max_tracks]
    kept_set = {s.index for s in kept}

    # Step 3: remove parasites overlapping a higher-ranked kept track.
    if remove_overlapping and kept:
        cen = _centroids(tracks)
        present_masks = {s.index: _present_mask(tracks[s.index]) for s in stats}
        # kept is already ranked best-first; check lower-ranked against higher.
        survivors: list[TrackStats] = []
        for s in kept:
            is_parasite = False
            for better in survivors:
                if _overlaps(
                    cen[s.index],
                    cen[better.index],
                    present_masks[s.index],
                    present_masks[better.index],
                    overlap_dist,
                    overlap_frac,
                ):
                    is_parasite = True
                    s.reason = f"overlaps higher-score track '{better.name}'"
                    break
            if not is_parasite:
                survivors.append(s)
        kept = survivors
        kept_set = {s.index for s in kept}

    # Annotate reasons.
    for s in stats:
        if s.index in kept_set:
            s.kept = True
            if not s.reason:
                s.reason = "present + confident"
        elif not s.reason:
            if s.occupancy_frac < min_occupancy and s.mean_score < score_threshold:
                s.reason = "short + low-score parasite"
            elif s.occupancy_frac < min_occupancy:
                s.reason = f"occupancy < {min_occupancy:.0%}"
            elif s.mean_score < score_threshold:
                s.reason = f"score < {score_threshold}"
            else:
                s.reason = f"exceeds max_tracks={max_tracks}"

    return stats


# Datasets in the analysis HDF5 that are indexed by track and must be subset.
_TRACK_AXIS0_DATASETS = (
    "tracks",
    "track_names",
    "point_scores",
    "instance_scores",
    "tracking_scores",
)


def clean_h5(
    input_path,
    output_path=None,
    overwrite=False,
    inplace=False,
    score_threshold: float = 0.3,
    min_occupancy: float = 0.5,
    max_tracks: int = 3,
    overlap_dist: float = 20.0,
    overlap_frac: float = 0.7,
    remove_overlapping: bool = True,
    dry_run: bool = False,
) -> CleanResult:
    """Clean a single analysis ``.h5`` file, writing the kept tracks to a new file.

    Args:
        input_path: Path to the analysis ``.h5``.
        output_path: Destination. Defaults to ``<stem>_clean.h5`` next to input
            (ignored when ``inplace`` is True).
        overwrite: Allow overwriting an existing output file.
        inplace: Write back to ``input_path`` (atomically via a temp file).
        score_threshold, min_occupancy, max_tracks, overlap_dist, overlap_frac,
        remove_overlapping: see :func:`select_tracks_to_keep`.
        dry_run: Compute decisions and return them without writing any file.

    Returns:
        CleanResult with the decision stats and output path.
    """
    input_path = Path(input_path)
    if inplace:
        output_path = input_path
    elif output_path is None:
        output_path = input_path.with_name(f"{input_path.stem}_clean.h5")
    else:
        output_path = Path(output_path)

    with h5py.File(input_path, "r") as f:
        tracks = f["tracks"][:]
        track_names = (
            [n.decode("utf-8") if isinstance(n, bytes) else str(n) for n in f["track_names"][:]]
            if "track_names" in f
            else [f"track_{i}" for i in range(tracks.shape[0])]
        )
        instance_scores = f["instance_scores"][:] if "instance_scores" in f else None
        point_scores = f["point_scores"][:] if "point_scores" in f else None

    stats = select_tracks_to_keep(
        tracks,
        track_names,
        instance_scores=instance_scores,
        point_scores=point_scores,
        score_threshold=score_threshold,
        min_occupancy=min_occupancy,
        max_tracks=max_tracks,
        overlap_dist=overlap_dist,
        overlap_frac=overlap_frac,
        remove_overlapping=remove_overlapping,
    )
    kept_indices = [s.index for s in stats if s.kept]

    result = CleanResult(
        input_path=input_path,
        output_path=output_path,
        n_tracks_in=tracks.shape[0],
        n_tracks_out=len(kept_indices),
        kept_indices=kept_indices,
        stats=stats,
    )

    if dry_run:
        return result

    if not inplace and output_path.exists() and not overwrite:
        raise FileExistsError(
            f"{output_path} exists. Pass overwrite=True or choose another path."
        )

    write_target = (
        output_path.with_suffix(output_path.suffix + ".tmp") if inplace else output_path
    )
    n_tracks_in = tracks.shape[0]
    with h5py.File(input_path, "r") as fin, h5py.File(write_target, "w") as fout:
        for key in fin:
            obj = fin[key]
            is_dataset = isinstance(obj, h5py.Dataset)
            # Track-indexed datasets: subset along the track axis.
            if is_dataset and key in _TRACK_AXIS0_DATASETS and obj.ndim >= 1 and obj.shape[0] == n_tracks_in:
                fout.create_dataset(key, data=obj[()][kept_indices])
            elif is_dataset and key == "track_occupancy":
                # track_occupancy may be (n_frames, n_tracks) or (n_tracks, n_frames).
                data = obj[()]
                axis = next((a for a, size in enumerate(data.shape) if size == n_tracks_in), None)
                if axis is not None:
                    data = np.take(data, kept_indices, axis=axis)
                fout.create_dataset(key, data=data)
            else:
                # Everything else (scalars like video_path, strings, groups):
                # copy verbatim so dtypes/attrs are preserved.
                fin.copy(key, fout)
        # Preserve top-level attributes.
        for attr_key, attr_val in fin.attrs.items():
            fout.attrs[attr_key] = attr_val

    if inplace:
        write_target.replace(output_path)

    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Remove parasite/noise tracks from SLEAP analysis .h5 files."
    )
    parser.add_argument("paths", nargs="+", help="One or more .h5 files (or directories to search)")
    parser.add_argument("--recursive", action="store_true", help="Search directories recursively for *_tracked.h5")
    parser.add_argument("--pattern", default="*_tracked.h5", help="Glob used when a path is a directory")
    parser.add_argument("--inplace", action="store_true", help="Overwrite the input file with the cleaned version")
    parser.add_argument("--suffix", default="_clean", help="Output suffix when not --inplace (default: _clean)")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting existing output files")
    parser.add_argument("--score-threshold", type=float, default=0.3, help="Minimum mean confidence to keep a track")
    parser.add_argument("--min-occupancy", type=float, default=0.5, help="Minimum fraction of the video a kept track must be present")
    parser.add_argument("--max-tracks", type=int, default=3, help="Maximum number of tracks to keep")
    parser.add_argument("--overlap-dist", type=float, default=20.0, help="Centroid distance (px) for overlap detection")
    parser.add_argument("--overlap-frac", type=float, default=0.7, help="Fraction of shared frames within overlap-dist to call a parasite")
    parser.add_argument("--keep-overlapping", action="store_true", help="Do not drop tracks overlapping a higher-score track")
    parser.add_argument("--dry-run", action="store_true", help="Print decisions without writing files")
    args = parser.parse_args()

    files: list[Path] = []
    for p in args.paths:
        path = Path(p)
        if path.suffix.lower() in (".yaml", ".yml"):
            # A tracker video-list YAML: map each video to its *_tracked.h5.
            import yaml

            with open(path) as fh:
                data = yaml.safe_load(fh) or {}
            for v in data.get("videos", []):
                vp = Path(v)
                h5 = vp.with_name(f"{vp.stem}_tracked.h5")
                if h5.exists():
                    files.append(h5)
                else:
                    print(f"  [skip] no tracked h5 for {vp.name}")
        elif path.is_dir():
            globber = path.rglob if args.recursive else path.glob
            files.extend(sorted(globber(args.pattern)))
        else:
            files.append(path)

    if not files:
        print("No .h5 files found.")
        return

    print(f"Cleaning {len(files)} file(s){' (dry run)' if args.dry_run else ''}...\n")
    for f in files:
        out = None if args.inplace else f.with_name(f"{f.stem}{args.suffix}.h5")
        result = clean_h5(
            f,
            output_path=out,
            overwrite=args.overwrite,
            inplace=args.inplace,
            score_threshold=args.score_threshold,
            min_occupancy=args.min_occupancy,
            max_tracks=args.max_tracks,
            overlap_dist=args.overlap_dist,
            overlap_frac=args.overlap_frac,
            remove_overlapping=not args.keep_overlapping,
            dry_run=args.dry_run,
        )
        print(result.summary())
        if not args.dry_run:
            print(f"  -> wrote {result.output_path}")
        print()


if __name__ == "__main__":
    main()
