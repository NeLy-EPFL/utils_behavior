#!/usr/bin/env python3
"""Batch SLEAP inference wrapper (uv-native).

Supports two backends that coexist so older experiments keep working:

- ``sleap-nn`` (SLEAP 1.6 PyTorch backend). Models are directories containing
  ``best.ckpt`` + ``training_config.yaml``. Tracking runs via
  ``sleap-nn-track -i <video> -m <model> -t [--use_flow] [-n N]``.
- ``legacy`` (classic TensorFlow ``sleap-track``). Models contain
  ``best_model.h5`` + ``training_config.json``. Tracking uses
  ``sleap-track <video> -m <model> --tracking.tracker flow ...``.

All SLEAP CLIs (``sleap-nn-track``, ``sleap-track``, ``sleap-convert``) are the
uv-installed ``sleap`` tool on PATH — there is **no** conda/mamba dependency.
``.slp`` -> analysis ``.h5`` conversion runs ``sleap_io.save_analysis_h5`` inside
an isolated ``uv run --with sleap-io`` environment, which is immune to whatever
interpreter launched the tracker (a conda env's Qt libs used to crash
``sleap-convert`` when they leaked into the subprocess).

The backend is auto-detected from the model directory by default. Both paths
write ``*_tracked.slp`` + ``*_tracked.h5`` next to each video, so downstream code
(``Sleap_Tracks``) is unchanged.
"""

import argparse
import fnmatch
import subprocess
import warnings
from pathlib import Path

import yaml

from .convert import clean_env as _clean_env
from .convert import slp_to_analysis_h5
from .tracks import Sleap_Tracks


class SleapTracker:
    def __init__(
        self,
        model_path,
        data_folder=None,
        model_centered_instance_path=None,
        output_folder=None,
        sleap_nn_executable="sleap-nn-track",
        legacy_executable="sleap-track",
        sleap_convert_executable="sleap-convert",
        batch_size=16,
        max_instances=3,
        max_tracks=None,
        tracker="flow",
        backend="auto",
        use_flow=True,
        video_filter=None,
        exclude_filters=("*_tracked*", "vid_*fly*"),
        yaml_file=None,
        conda_env=None,  # deprecated, ignored — kept for backwards compatibility
    ):
        """Initialize the SleapTracker.

        Args:
            model_path (str or Path): Trained SLEAP model directory. For a
                top-down pipeline this is the centroid model.
            data_folder (str or Path, optional): Directory of videos to track.
                Ignored when ``yaml_file`` is provided.
            model_centered_instance_path (str or Path, optional): Second model
                (centered-instance) for top-down pipelines.
            output_folder (str or Path, optional): Kept for compatibility;
                outputs are written next to each video.
            sleap_nn_executable (str): sleap-nn track entry point on PATH
                (uv tool ``sleap`` exposes ``sleap-nn-track``).
            legacy_executable (str): legacy tracker entry point on PATH
                (``sleap-track``).
            sleap_convert_executable (str): legacy converter on PATH
                (``sleap-convert``).
            batch_size (int): Frames per inference batch.
            max_instances (int or None): Cap on instances detected per frame
                (sleap-nn ``-n``). Defaults to 3; identities are consolidated to
                <=3 tracks afterwards by clean_tracks.
            max_tracks (int or None): Only used with the ``local_queues``
                candidate method (incompatible with ``--use_flow``). Legacy
                backend uses it as the max number of identities.
            tracker (str): Legacy tracker name ("flow", "simple", ...).
            backend (str): "auto" (detect from model dir), "sleap-nn", "legacy".
            use_flow (bool): Use optical-flow candidates (sleap-nn backend).
            video_filter (str, optional): glob/substring inclusion filter
                (e.g. "*_80fps.mp4").
            exclude_filters (tuple): glob patterns to exclude (matched on name).
            yaml_file (str or Path, optional): YAML with ``directories:`` and/or
                ``videos:`` lists to process instead of ``data_folder``.
            conda_env: Deprecated and ignored (the tools are uv-native now).
        """
        if conda_env is not None:
            warnings.warn(
                "SleapTracker(conda_env=...) is deprecated and ignored; SLEAP "
                "tools are invoked via uv on PATH (no conda).",
                DeprecationWarning,
                stacklevel=2,
            )

        self.model_path = Path(model_path)
        self.model_centered_instance = (
            Path(model_centered_instance_path) if model_centered_instance_path else None
        )
        self.data_folder = Path(data_folder) if data_folder else None
        self.output_folder = Path(output_folder) if output_folder else self.data_folder
        self.sleap_nn_cmd = sleap_nn_executable.split()
        self.legacy_cmd = legacy_executable.split()
        self.sleap_convert_cmd = sleap_convert_executable.split()
        self.batch_size = batch_size
        self.max_instances = max_instances if max_instances else None
        self.max_tracks = max_tracks if max_tracks else None
        self.tracker = tracker
        self.use_flow = use_flow
        self.video_filter = video_filter
        self.exclude_filters = tuple(exclude_filters) if exclude_filters else ()
        self.yaml_file = Path(yaml_file) if yaml_file else None
        self.videos_to_process = []

        self.backend = (
            self._detect_backend(self.model_path) if backend == "auto" else backend
        )
        if self.backend not in ("sleap-nn", "legacy"):
            raise ValueError(
                f"Unknown backend '{self.backend}'. Use 'auto', 'sleap-nn', or 'legacy'."
            )

    @staticmethod
    def _detect_backend(model_path):
        """Detect the SLEAP backend from a model directory's contents.

        sleap-nn models ship ``best.ckpt`` / ``training_config.yaml``; legacy
        TensorFlow models ship ``best_model.h5`` / ``training_config.json``.
        """
        model_path = Path(model_path)
        if model_path.is_dir():
            names = {p.name for p in model_path.iterdir()}
            if "best.ckpt" in names or "training_config.yaml" in names:
                return "sleap-nn"
            if "best_model.h5" in names or "training_config.json" in names:
                return "legacy"
        if model_path.suffix in {".ckpt", ".yaml", ".yml"}:
            return "sleap-nn"
        if model_path.suffix in {".h5", ".json"}:
            return "legacy"
        print(
            f"Could not auto-detect backend for {model_path}; defaulting to 'sleap-nn'. "
            "Pass backend='legacy' to override."
        )
        return "sleap-nn"

    def load_directories_from_yaml(self):
        """Load directories (and explicit videos) from the YAML file, if any."""
        directories, videos = [], []
        if self.yaml_file:
            with open(self.yaml_file, "r") as file:
                data = yaml.safe_load(file) or {}
            directories = [Path(d) for d in data.get("directories", [])]
            videos = [Path(v) for v in data.get("videos", [])]
            print(
                f"Loaded {len(directories)} directories and {len(videos)} videos "
                f"from {self.yaml_file}."
            )
        return directories, videos

    def _matches_filters(self, video):
        """Return True if a video passes the inclusion filter and no exclusion."""
        name = video.name
        if self.video_filter and not (
            fnmatch.fnmatch(name, self.video_filter) or self.video_filter in name
        ):
            return False
        for pattern in self.exclude_filters:
            if fnmatch.fnmatch(name, pattern):
                return False
        return True

    def collect_videos(self, video_extension=".mp4"):
        """Collect videos to track from the YAML lists or the data folder."""
        directories, explicit_videos = (
            self.load_directories_from_yaml()
            if self.yaml_file
            else ([self.data_folder] if self.data_folder else [], [])
        )

        collected = []
        for folder in directories:
            if folder and folder.exists() and folder.is_dir():
                collected.extend(folder.rglob(f"*{video_extension}"))
            else:
                print(f"Skipping missing directory: {folder}")
        collected.extend(v for v in explicit_videos if v.exists())

        seen = set()
        self.videos_to_process = []
        for video in collected:
            resolved = video.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            if self._matches_filters(video):
                self.videos_to_process.append(video)

        print(f"Collected {len(self.videos_to_process)} videos for tracking.")
        return self.videos_to_process

    def filter_tracked_videos(self, render=False):
        """Drop videos that already have a ``.slp`` + ``.h5`` (and annotated video)."""
        videos_filtered = []
        skipped = []
        for video in self.videos_to_process:
            out_dir = video.parent
            slp_file = out_dir / f"{video.stem}_tracked.slp"
            h5_file = out_dir / f"{video.stem}_tracked.h5"
            empty_marker = out_dir / f"{video.stem}_tracked.empty"
            annotated_file = out_dir / f"{video.stem}_tracked_annotated.mp4"
            # An .empty marker means a prior run tracked nothing for this video;
            # re-tracking would just fail the same way, so treat it as done.
            already_processed = (slp_file.exists() and h5_file.exists()) or empty_marker.exists()
            if render:
                already_processed = already_processed and annotated_file.exists()
            if not already_processed:
                videos_filtered.append(video)
            else:
                skipped.append(video)

        self.skipped_videos = skipped
        self.videos_to_process = videos_filtered
        print(
            f"Filtered to {len(self.videos_to_process)} videos needing tracking "
            f"({len(skipped)} already tracked, skipped)."
        )
        return self.videos_to_process

    def _build_sleap_nn_cmd(self, video, slp_out):
        """Build the ``sleap-nn-track`` command for one video.

        For a top-down pipeline the centroid model is passed first, then the
        centered-instance model (each with its own ``-m``).
        """
        cmd = [
            *self.sleap_nn_cmd,
            "-i", str(video),
            "-m", str(self.model_path),
        ]
        if self.model_centered_instance:
            cmd.extend(["-m", str(self.model_centered_instance)])
        cmd.extend([
            "-o", str(slp_out),
            "-b", str(self.batch_size),
            "-t",  # enable tracking
        ])
        if self.max_instances:
            cmd.extend(["-n", str(self.max_instances)])
        if self.use_flow:
            cmd.append("--use_flow")
        elif self.max_tracks:
            # max_tracks only applies to the local_queues candidate method.
            cmd.extend(["--candidates_method", "local_queues",
                        "--max_tracks", str(self.max_tracks)])
        return cmd

    def _build_legacy_cmd(self, video, slp_out):
        """Build the legacy ``sleap-track`` command for one video."""
        cmd = [
            *self.legacy_cmd, str(video),
            "--model", str(self.model_path),
            "--output", str(slp_out),
            "--batch_size", str(self.batch_size),
            "--tracking.tracker", self.tracker,
            "--verbosity", "rich",
        ]
        if self.max_tracks:
            cmd.extend(["--tracking.max_tracking", "1", "--max_tracks", str(self.max_tracks)])
        if self.model_centered_instance:
            cmd.extend(["--model", str(self.model_centered_instance)])
        return cmd

    def _export_h5(self, slp_out, h5_out):
        """Convert a ``.slp`` to an analysis ``.h5`` (uv-native; both backends).

        Delegates to :func:`utils_behavior.sleap.convert.slp_to_analysis_h5`,
        which runs ``sleap_io.save_analysis_h5`` in an isolated
        ``uv run --with sleap-io`` env — robust to the launching interpreter.

        Returns the ``.h5`` path, or ``None`` when the ``.slp`` had no labeled
        frames (nothing was tracked) so no analysis ``.h5`` could be written.
        """
        return slp_to_analysis_h5(slp_out, h5_out)

    def process_videos(self):
        """Run inference + h5 export for every collected video.

        Per-video failures are recorded and the batch continues, so one bad
        video can't abort a long resumable run. Videos whose ``.slp`` has no
        labeled frames (nothing tracked) get a ``*_tracked.empty`` marker so
        they're skipped on resume instead of re-tracked indefinitely.
        """
        print(
            f"Processing {len(self.videos_to_process)} videos with the "
            f"'{self.backend}' backend..."
        )

        self.empty_videos = []
        self.failed_videos = []
        env = _clean_env()
        for video in self.videos_to_process:
            out_dir = video.parent
            slp_out = out_dir / f"{video.stem}_tracked.slp"
            h5_out = out_dir / f"{video.stem}_tracked.h5"

            if self.backend == "sleap-nn":
                track_cmd = self._build_sleap_nn_cmd(video, slp_out)
            else:
                track_cmd = self._build_legacy_cmd(video, slp_out)

            print(f"  Tracking: {video.name}")
            try:
                subprocess.run(track_cmd, check=True, env=env)
                h5 = self._export_h5(slp_out, h5_out)
                if h5 is None:
                    marker = out_dir / f"{video.stem}_tracked.empty"
                    marker.touch()
                    self.empty_videos.append(video)
                    print(
                        f"    [empty] no instances tracked; wrote {marker.name} "
                        "(delete it to retry this video)"
                    )
            except Exception as exc:
                self.failed_videos.append((video, exc))
                print(f"    [fail] {video.name}: {exc}")

        print("Video processing complete.")
        if self.empty_videos:
            print(f"  {len(self.empty_videos)} video(s) tracked nothing (marked .empty).")
        if self.failed_videos:
            print(f"  {len(self.failed_videos)} video(s) failed:")
            for video, exc in self.failed_videos:
                print(f"    - {video}: {exc}")

    def run(self, video_extension=".mp4", render=False, dry_run=False):
        """Collect, filter, and track videos.

        When ``dry_run`` is True, report what would be tracked vs skipped and
        return without running any inference (lets you verify resumability).
        """
        self.collect_videos(video_extension=video_extension)
        self.filter_tracked_videos(render=render)

        if dry_run:
            print("\n--- DRY RUN (no inference will be performed) ---")
            print(
                f"Backend: {self.backend} | "
                f"already tracked (skip): {len(getattr(self, 'skipped_videos', []))} | "
                f"to track: {len(self.videos_to_process)}"
            )
            if self.videos_to_process:
                print("\nWould track:")
                for video in self.videos_to_process:
                    print(f"  + {video}")
            else:
                print("\nNothing to track — all collected videos already have "
                      "*_tracked.slp + *_tracked.h5.")
            return

        if not self.videos_to_process:
            print("No new videos to track.")
            return

        self.process_videos()

        if render:
            from .tracks import generate_annotated_video

            skip = set(getattr(self, "empty_videos", [])) | {
                v for v, _ in getattr(self, "failed_videos", [])
            }
            for video in self.videos_to_process:
                if video in skip:
                    continue
                h5_path = video.parent / f"{video.stem}_tracked.h5"
                sleap_tracks = Sleap_Tracks(h5_path)
                if getattr(sleap_tracks, "video", None) is None:
                    print(f"Warning: no video path in {h5_path}, skipping annotation.")
                    continue
                generate_annotated_video([sleap_tracks], save=True)


def main():
    parser = argparse.ArgumentParser(description="Run SLEAP tracking on video files.")
    parser.add_argument("model_path", type=str, help="Path to the trained SLEAP model directory")
    parser.add_argument("--data_folder", type=str, help="Directory containing videos to track")
    parser.add_argument("--model_centered_instance_path", type=str, help="Centered-instance model for top-down pipelines")
    parser.add_argument("--output_folder", type=str, help="(Compatibility) outputs are written next to each video")
    parser.add_argument("--sleap_nn_executable", type=str, default="sleap-nn-track", help="sleap-nn track entry point on PATH")
    parser.add_argument("--legacy_executable", type=str, default="sleap-track", help="Legacy sleap-track entry point on PATH")
    parser.add_argument("--sleap_convert_executable", type=str, default="sleap-convert", help="Legacy sleap-convert entry point on PATH")
    parser.add_argument("--batch_size", type=int, default=16, help="Frames per inference batch")
    parser.add_argument("--max_instances", type=int, default=3, help="Max instances detected per frame (sleap-nn -n)")
    parser.add_argument("--max_tracks", type=int, default=None, help="Max track identities (local_queues / legacy only)")
    parser.add_argument("--tracker", type=str, default="flow", help="Legacy tracker name (legacy backend only)")
    parser.add_argument("--backend", type=str, default="auto", choices=["auto", "sleap-nn", "legacy"], help="SLEAP backend to use")
    parser.add_argument("--no_flow", action="store_true", help="Disable optical-flow candidates (sleap-nn backend)")
    parser.add_argument("--video_filter", type=str, help="Only track videos whose name matches this glob/substring (e.g. '*_80fps.mp4')")
    parser.add_argument("--yaml_file", type=str, help="YAML with directories/videos to process")
    parser.add_argument("--video_extension", type=str, default=".mp4", help="Video file extension to process")
    parser.add_argument("--render", action="store_true", help="Generate annotated video after tracking")
    parser.add_argument("--dry_run", action="store_true", help="Report which videos would be tracked/skipped, then exit without running inference")

    args = parser.parse_args()

    tracker = SleapTracker(
        model_path=args.model_path,
        data_folder=args.data_folder,
        model_centered_instance_path=args.model_centered_instance_path,
        output_folder=args.output_folder,
        sleap_nn_executable=args.sleap_nn_executable,
        legacy_executable=args.legacy_executable,
        sleap_convert_executable=args.sleap_convert_executable,
        batch_size=args.batch_size,
        max_instances=args.max_instances,
        max_tracks=args.max_tracks,
        tracker=args.tracker,
        backend=args.backend,
        use_flow=not args.no_flow,
        video_filter=args.video_filter,
        yaml_file=args.yaml_file,
    )

    tracker.run(video_extension=args.video_extension, render=args.render, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
