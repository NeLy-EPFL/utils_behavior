#!/usr/bin/env python3

import argparse
from pathlib import Path
from Sleap_utils import Sleap_Tracks, generate_annotated_video
import subprocess

class SleapTracker:
    def __init__(
        self,
        model_path,
        data_folder=None,
        model_centered_instance_path=None,
        output_folder=None,
        conda_env="sleap",
        batch_size=16,
        max_tracks=None,
        tracker="simple",
        video_filter=None,
        yaml_file=None,
    ):
        """
        Initialize the SleapTracker class.

        Args:
            model_path (str or pathlib.Path): Path to the trained SLEAP model for tracking. If multiple models are needed, this one should be the centroid model.
            data_folder (str or pathlib.Path, optional): Directory containing videos to track. This will be ignored if a YAML file is provided.
            model_centered_instance_path (str or pathlib.Path, optional): Path to the centered instance model for tracking.
            output_folder (str or pathlib.Path, optional): Directory to store tracked .slp files. Defaults to data_folder.
            conda_env (str, optional): Conda environment where SLEAP is installed. Defaults to 'sleap'.
            batch_size (int, optional): Number of frames to process at once in sleap-track command. Defaults to 16.
            max_tracks (int, optional): The maximum number of tracks to predict. Set to None for unlimited.
            tracker (str, optional): The type of tracker to use (e.g., 'simple', 'flow', etc.). Defaults to 'simple'.
            video_filter (str, optional): Optional filter for videos to process.
            yaml_file (str or pathlib.Path, optional): Path to a YAML file containing a list of directories to process instead of data_folder.

        Example usage:
            tracker = SleapTracker(model_path="path/to/model", data_folder="path/to/videos", batch_size=16)
            tracker.run()

        If you have a YAML file with folders:
            tracker = SleapTracker(model_path="path/to/model", yaml_file="path/to/config.yaml")
            tracker.run()

            example YAML file:
            directories:
              - path/to/videos1
              - path/to/videos2
        """
        self.model_path = Path(model_path)
        self.model_centered_instance = (
            Path(model_centered_instance_path) if model_centered_instance_path else None
        )
        self.data_folder = Path(data_folder) if data_folder else None
        self.output_folder = Path(output_folder) if output_folder else self.data_folder
        self.conda_env = conda_env
        self.batch_size = batch_size
        self.max_tracks = max_tracks if max_tracks else None
        self.tracker = tracker
        self.videos_to_process = []
        self.video_filter = video_filter if video_filter else None
        self.yaml_file = Path(yaml_file) if yaml_file else None

    def load_directories_from_yaml(self):
        """
        Load directories from a YAML file if provided.
        """
        if self.yaml_file:
            with open(self.yaml_file, "r") as file:
                directories = yaml.safe_load(file).get("directories", [])
                if directories:
                    print(f"Loaded {len(directories)} directories from YAML file.")
                    return [Path(d) for d in directories]
        return []

    def collect_videos(self, video_extension=".mp4"):
        """
        Collect all videos from the data folder(s) that need tracking.

        If a YAML file is provided, directories from the YAML file will be used instead of data_folder.
        """
        directories_to_process = (
            self.load_directories_from_yaml() if self.yaml_file else [self.data_folder]
        )

        for folder in directories_to_process:
            if folder.exists() and folder.is_dir():
                self.videos_to_process.extend(list(folder.rglob(f"*{video_extension}")))
        print(f"Collected {len(self.videos_to_process)} videos for tracking.")

    def filter_tracked_videos(self, render=False):
        """
        Filter out videos that have already been tracked (i.e., videos with corresponding .slp, .h5, or annotated video files).
        """
        videos_filtered = []
        for video in self.videos_to_process:
            video_name = video.stem
            output_folder = video.parent  # Check in the video's directory
            slp_file = output_folder / f"{video_name}_tracked.slp"
            h5_file = output_folder / f"{video_name}_tracked.h5"
            annotated_file = output_folder / f"{video_name}_tracked_annotated.mp4"
            already_processed = slp_file.exists() and h5_file.exists()
            if render:
                already_processed = already_processed and annotated_file.exists()
            if not already_processed:
                videos_filtered.append(video)

        self.videos_to_process = videos_filtered
        print(f"Filtered to {len(self.videos_to_process)} videos needing tracking.")

        self.videos_to_process = videos_filtered
        print(f"Filtered to {len(self.videos_to_process)} videos needing tracking.")

    def activate_conda(self):
        """
        Activate the conda environment where SLEAP is installed.

        Example usage:
            tracker.activate_conda()

        This is necessary to ensure the correct environment is used for tracking.
        """
        subprocess.run(["mamba", "activate", self.conda_env], check=True)

    def process_videos(self):
        """
        Process videos by running the SLEAP tracking command.

        Example usage:
            tracker.process_videos()
        """
        print(f"Processing {len(self.videos_to_process)} videos...")

        for video in self.videos_to_process:
            # Determine the output folder for this video
            output_folder = video.parent  # Save in the same directory as the video

            # Build the sleap-track command for tracking
            sleap_track_cmd = [
                "mamba",
                "run",
                "-n",
                self.conda_env,
                "sleap-track",
                str(video),
                "--model",
                str(self.model_path),
                "--output",
                str(output_folder / f"{video.stem}_tracked.slp"),
                "--batch_size",
                str(self.batch_size),
                # Add tracker if a centered instance model is specified
                *(
                    ["--tracking.tracker", self.tracker]
                    if self.model_centered_instance
                    else []
                ),
                # Add max_tracks and set max_tracking if specified
                *(["--max_tracking", "1"] if self.max_tracks else []),
                *(["--max_tracks", str(self.max_tracks)] if self.max_tracks else []),
                "--verbosity",
                "rich",
            ]
            if self.model_centered_instance:
                sleap_track_cmd.extend(["--model", str(self.model_centered_instance)])

            subprocess.run(sleap_track_cmd, check=True)

            # Convert the .slp file to .h5 format for analysis
            sleap_convert_cmd = [
                "mamba",
                "run",
                "-n",
                self.conda_env,
                "sleap-convert",
                str(output_folder / f"{video.stem}_tracked.slp"),
                "--format",
                "analysis",
            ]
            subprocess.run(sleap_convert_cmd, check=True)

        print("Video processing complete.")

    def run(self, video_extension=".mp4", render=False):
        """
        Main method to collect videos, filter tracked ones, and track remaining videos.

        Example usage:
            tracker.run()
        """
        self.collect_videos(video_extension=video_extension)
        self.filter_tracked_videos(render=render)
        if self.videos_to_process:
            self.process_videos()

            if render:
                h5_path = self.output_folder / f"{self.videos_to_process[0].stem}_tracked.h5"
                sleap_tracks = Sleap_Tracks(h5_path)
                if not hasattr(sleap_tracks, "video") or sleap_tracks.video is None:
                    print(f"Warning: No video path found in {h5_path}, skipping annotation.")
                else:
                    generate_annotated_video([sleap_tracks], save=True)
        else:
            print("No new videos to track.")

def main():
    parser = argparse.ArgumentParser(description="Run SLEAP tracking on video files.")
    parser.add_argument("model_path", type=str, help="Path to the trained SLEAP model for tracking")
    parser.add_argument("--data_folder", type=str, help="Directory containing videos to track")
    parser.add_argument("--model_centered_instance_path", type=str, help="Path to the centered instance model for tracking")
    parser.add_argument("--output_folder", type=str, help="Directory to store tracked .slp files")
    parser.add_argument("--conda_env", type=str, default="sleap", help="Conda environment where SLEAP is installed")
    parser.add_argument("--batch_size", type=int, default=16, help="Number of frames to process at once")
    parser.add_argument("--max_tracks", type=int, help="The maximum number of tracks to predict")
    parser.add_argument("--tracker", type=str, default="simple", help="The type of tracker to use")
    parser.add_argument("--video_filter", type=str, help="Optional filter for videos to process")
    parser.add_argument("--yaml_file", type=str, help="Path to a YAML file containing a list of directories to process")
    parser.add_argument("--video_extension", type=str, default=".mp4", help="Video file extension to process")
    parser.add_argument("--render", action="store_true", help="Generate annotated video after tracking")

    args = parser.parse_args()

    tracker = SleapTracker(
        model_path=args.model_path,
        data_folder=args.data_folder,
        model_centered_instance_path=args.model_centered_instance_path,
        output_folder=args.output_folder,
        conda_env=args.conda_env,
        batch_size=args.batch_size,
        max_tracks=args.max_tracks,
        tracker=args.tracker,
        video_filter=args.video_filter,
        yaml_file=args.yaml_file
    )

    tracker.run(video_extension=args.video_extension, render=args.render)

if __name__ == "__main__":
    main()