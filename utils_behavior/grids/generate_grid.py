import math
import os
import subprocess
import json
from pathlib import Path
from operator import itemgetter
from itertools import groupby
import re
import time

FONT_FILE = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
VIDEO_EXT = "*.mp4"
BUNDLE_KEYWORD = "*bundle*.mp4"
ROTATE = False


def get_video_size(video_path):
    """
    Get the width and height of the video.

    Parameters
    ----------
    video_path : str
        The path to the video file.

    Returns
    -------
    tuple
        The width and height of the video.
    """
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=s=x:p=0",
            video_path,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.returncode != 0:
        return None
    width, height = map(int, result.stdout.decode("utf-8").strip().split("x"))
    return width, height


def create_blank_video(width, height, output_path, duration=1, fps=30):
    """
    Create a blank video with the specified width, height, and duration.

    Parameters
    ----------
    width : int
        The width of the video.
    height : int
        The height of the video.
    output_path : Path
        The path to the output video file.
    duration : int, optional
        The duration of the video in seconds (default is 1).
    fps : int, optional
        The frames per second of the video (default is 30).

    Returns
    -------
    None
    """
    subprocess.run(
        [
            "ffmpeg",
            "-f",
            "lavfi",
            "-i",
            f"color=c=black:s={width}x{height}:d={duration}:r={fps}",
            "-c:v",
            "libx264",
            "-t",
            str(duration),
            str(output_path),
        ]
    )


def resize_video(input_path, output_path, width, height):
    """
    Resize the video to the specified width and height.

    Parameters
    ----------
    input_path : Path
        The path to the input video file.
    output_path : Path
        The path to the output video file.
    width : int
        The width of the output video.
    height : int
        The height of the output video.

    Returns
    -------
    None
    """
    subprocess.run(
        [
            "ffmpeg",
            "-i",
            str(input_path),
            "-vf",
            f"scale={width}:{height}",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-preset",
            "slow",
            str(output_path),
        ]
    )


def create_grid_video(input_folder, output_path, keyword=None):
    """
    Creates a grid video from the videos in the given input folder.

    Parameters
    ----------
    input_folder : Path
        The path to the folder containing the input videos.
    output_path : Path
        The path to the output video.
    keyword : str, optional
        The keyword to use to find the input videos.

    Returns
    -------
    None
    """
    # Set the input folder path
    input_folder = Path(input_folder)

    # Get all video files from the input folder
    if keyword:
        input_files = list(input_folder.glob(f"*{keyword}*.mp4"))
        # Sort the input files by the number in their name
        input_files.sort(key=lambda f: int(f.stem.split("_")[1]))
    else:
        # If no keyword is provided, sort by numbers in file name
        input_files = sorted(
            list(input_folder.glob("*.mp4")),
            key=lambda f: int(re.findall(r"\d+", f.stem)[0]),
        )

    # Check if the video files are valid, meaning their size can be obtained
    valid_files = []
    for file in input_files:
        size = get_video_size(file)
        if size is not None:
            valid_files.append(file)
        else:
            print(f"Skipping video {file} due to invalid size")
    input_files = valid_files

    print(f"input_files: {input_files}")

    # Get the width and height of the first video
    width, height = get_video_size(input_files[0].as_posix())

    # Resize all videos to the same height
    resized_files = []
    for i, input_file in enumerate(input_files):
        resized_file = input_folder / f"resized_{i}.mp4"
        resize_video(input_file, resized_file, width, height)
        resized_files.append(resized_file)
    input_files = resized_files

    # Calculate the number of columns and rows for the grid layout
    num_videos = len(input_files)
    aspect_ratio = 16 / 9
    best_layout = (1, num_videos)
    best_diff = float("inf")
    for num_cols in range(1, num_videos + 1):
        num_rows = math.ceil(num_videos / num_cols)
        diff = abs(aspect_ratio - ((num_cols * width) / (num_rows * height)))
        if diff < best_diff:
            best_diff = diff
            best_layout = (num_rows, num_cols)
    num_rows, num_cols = best_layout

    print(f"num_videos: {num_videos}")
    print(f"num_cols: {num_cols}")
    print(f"num_rows: {num_rows}")

    # Create blank videos if necessary
    while num_videos < num_cols * num_rows:
        blank_video = input_folder / f"blank{num_videos}.mp4"
        create_blank_video(width, height, blank_video)
        input_files.append(blank_video)
        num_videos += 1

    # Set the scale factor
    scale_factor = 0.5

    # Create the filter_complex argument for the ffmpeg command
    filter_complex = "".join(
        f"[{i}:v]scale=iw*{scale_factor}:ih*{scale_factor}[s{i}];[s{i}]copy[v{i}];"
        for i in range(num_cols * num_rows)
    )
    for row in range(num_rows):
        row_inputs = "".join(f"[v{row * num_cols + col}]" for col in range(num_cols))
        filter_complex += f"{row_inputs}hstack=inputs={num_cols}[h{row}];"
    vstack_inputs = "".join(f"[h{row}]" for row in range(num_rows))
    filter_complex += f"{vstack_inputs}vstack=inputs={num_rows}[v]"
    filter_complex += f";[v]pad=ceil(iw/2)*2:ceil(ih/2)*2[v]"

    print(f"filter_complex : {filter_complex}")

    # Create the ffmpeg command arguments
    ffmpeg_args = ["ffmpeg"]
    for input_file in input_files:
        ffmpeg_args.extend(["-i", str(input_file)])
    ffmpeg_args.extend(
        ["-filter_complex", filter_complex, "-map", "[v]", str(output_path)]
    )

    # Run the ffmpeg command
    subprocess.run(ffmpeg_args)
