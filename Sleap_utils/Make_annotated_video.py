import argparse
from pathlib import Path
from typing import List, Optional, Tuple

from utils_behavior.Sleap_utils import Sleap_Tracks, generate_annotated_video


def process_video(
    video_path: Path,
    h5_path: Path,
    output_path: Optional[Path] = None,
    time_range: Optional[Tuple[float, float]] = None,
    nodes: Optional[List[str]] = None,
    labels: bool = False,
    edges: bool = True,
    colorby: Optional[str] = None,
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
    sleap_tracks = Sleap_Tracks(h5_path, smoothed_tracks=False)

    if time_range:
        start_time, end_time = time_range
        sleap_tracks.filter_data(time_range)
        start_frame = int(start_time * sleap_tracks.fps) if start_time else None
        end_frame = int(end_time * sleap_tracks.fps) if end_time else None
    else:
        start_frame, end_frame = None, None

    if output_path is None:
        output_path = video_path.with_name(f"{video_path.stem}_annotated.mp4")

    generate_annotated_video(
        str(video_path),
        [sleap_tracks],
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
        "videos", nargs="+", type=Path, help="Path(s) to input video file(s)"
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

    args = parser.parse_args()

    time_range = (
        (args.start, args.end)
        if args.start is not None or args.end is not None
        else None
    )

    for video_path in args.videos:
        if not video_path.exists():
            print(f"Warning: Video file not found: {video_path}")
            continue

        output_path = args.output if args.output else None
        if output_path and len(args.videos) > 1:
            output_path = output_path.with_name(f"{video_path.stem}_{output_path.name}")

        process_video(
            video_path,
            args.h5,
            output_path,
            time_range,
            args.nodes,
            args.labels,
            args.edges,
            args.colorby,
        )


if __name__ == "__main__":
    main()
