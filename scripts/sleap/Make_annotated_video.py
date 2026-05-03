import argparse
from pathlib import Path
from typing import List, Optional, Tuple
import cv2

import sys

# Add the root directory to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))
#print(sys.path)

from utils_behavior.Sleap_utils import Sleap_Tracks, generate_annotated_video


def process_video(
    h5_path: Path,
    video_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    time_range: Optional[Tuple[float, float]] = None,
    nodes: Optional[List[str]] = None,
    labels: bool = False,
    edges: bool = True,
    colorby: Optional[str] = None,
    smoothing: bool = False,
    smoothing_params: tuple = (221, 1),
) -> None:
    """
    Process a single video to create an annotated clip.

    Args:
        video_path (Path): Path to the input video file.
        h5_path (Path): Path to the corresponding SLEAP .h5 file.
        output_path (Optional[Path]): Path to save the output video. If None, it will be auto-generated.
        time_range (Optional[Tuple[float, float]]): Start and end times for the clip in seconds.
        nodes (Optional[List[str]]): List of specific nodes to annotate.
        labels (bool): Whether to include labels in the annotation.
        edges (bool): Whether to include edges in the annotation.
        colorby (Optional[str]): Attribute to color by.
    """
    sleap_tracks = Sleap_Tracks(
        h5_path, smoothed_tracks=smoothing, smoothing_params=smoothing_params
    )

    if video_path is None:
        video_path = sleap_tracks.video
        print(f"Inferred video path: {video_path}")

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if output_path is None:
        output_path = video_path.with_name(f"{video_path.stem}_annotated.mp4")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video FPS: {fps}")
    if fps <= 0:
        raise ValueError(f"Invalid FPS for video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames in video: {total_frames}")

    if time_range:
        start_time, end_time = time_range
        start_frame = int(start_time * fps) if start_time else 1
        end_frame = int(end_time * fps) if end_time else total_frames
    else:
        start_frame, end_frame = 1, total_frames

    generate_annotated_video(
        sleap_tracks_list=[sleap_tracks],  # Pass as a list
        video=str(video_path),  # Pass the video path as a string
        save=True,
        output_path=str(output_path),
        start=start_frame,
        end=end_frame,
        nodes=nodes,
        labels=labels,
        edges=edges,
        colorby=colorby,
    )

    print(f"Annotated video saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate annotated video clips from SLEAP tracking data."
    )
    parser.add_argument(
        "--videos",
        nargs="+",
        type=Path,
        help="Path(s) to input video file(s). If not provided, the video path will be inferred from the SLEAP .h5 file.",
    )
    parser.add_argument(
        "--h5", type=Path, required=True, help="Path to the SLEAP .h5 file"
    )
    parser.add_argument("--output", type=Path, help="Path to save the output video(s)")
    parser.add_argument(
        "--start", type=float, help="Start time for the clip in seconds"
    )
    parser.add_argument("--end", type=float, help="End time for the clip in seconds")
    parser.add_argument("--nodes", nargs="+", help="Specific nodes to annotate")
    parser.add_argument(
        "--labels", action="store_true", help="Include labels in the annotation"
    )
    parser.add_argument(
        "--no-edges",
        action="store_false",
        dest="edges",
        help="Exclude edges from the annotation",
    )
    parser.add_argument("--colorby", choices=["Nodes"], help="Attribute to color by")

    parser.add_argument(
        "--smoothing",
        type=bool,
        default=False,
        help="Apply smoothing to the tracks before generating the video",
    )
    parser.add_argument(
        "--smoothing_params",
        type=str,
        default="221, 1",
        help="Parameters for smoothing (window size, polyorder)",
    )

    args = parser.parse_args()

    smoothing_params = tuple(map(int, args.smoothing_params.split(",")))

    time_range = (
        (args.start, args.end)
        if args.start is not None or args.end is not None
        else None
    )

    # If no videos are provided, process the video inferred from the SLEAP .h5 file
    if args.videos is None:
        process_video(
            h5_path=args.h5,
            output_path=args.output,
            time_range=time_range,
            nodes=args.nodes,
            labels=args.labels,
            edges=args.edges,
            colorby=args.colorby,
            smoothing=args.smoothing,
            smoothing_params=smoothing_params,
        )
    else:
        for video_path in args.videos:
            if not video_path.exists():
                print(f"Warning: Video file not found: {video_path}")
                continue

            output_path = args.output if args.output else None
            if output_path and len(args.videos) > 1:
                output_path = output_path.with_name(
                    f"{video_path.stem}_{output_path.name}"
                )

            process_video(
                video_path=video_path,
                h5_path=args.h5,
                output_path=output_path,
                time_range=time_range,
                nodes=args.nodes,
                labels=args.labels,
                edges=args.edges,
                colorby=args.colorby,
                smoothing=args.smoothing,
                smoothing_params=smoothing_params,
            )


if __name__ == "__main__":
    main()
