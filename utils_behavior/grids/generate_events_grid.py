import cv2
import os
import glob
import pandas as pd
import re
import random
import subprocess

import Utils


from moviepy.editor import VideoFileClip, CompositeVideoClip, vfx
import math
from pathlib import Path

DataPath = Utils.get_data_path()


def create_grid_video(input_folder, output_path, slow_factor=3.0):
    input_folder = Path(input_folder)
    input_files = sorted(list(input_folder.glob("*.mp4")))

    if not input_files:
        raise ValueError("No video files found in the input folder.")

    # Load video clips
    clips = [VideoFileClip(str(file)) for file in input_files]

    # Calculate grid dimensions
    num_videos = len(clips)
    aspect_ratio = 16 / 9
    num_cols = math.ceil(math.sqrt(num_videos * aspect_ratio))
    num_rows = math.ceil(num_videos / num_cols)

    # Calculate the size for each video in the grid
    max_width = max(clip.w for clip in clips)
    max_height = max(clip.h for clip in clips)

    # Resize all clips to the same size, slow them down, and hold the last frame
    max_duration = max(clip.duration for clip in clips)
    resized_clips = [
        clip.resize(width=max_width, height=max_height)
        .fx(vfx.speedx, factor=1.0 / slow_factor)
        .loop(duration=max_duration * slow_factor)
        for clip in clips
    ]

    # Create the grid
    grid = []
    for i in range(num_rows):
        for j in range(num_cols):
            index = i * num_cols + j
            if index < num_videos:
                clip = resized_clips[index]
                grid.append(clip.set_position((j * max_width, i * max_height)))

    # Create the final composite video
    final_clip = CompositeVideoClip(
        grid, size=(num_cols * max_width, num_rows * max_height)
    )

    # Write the output video
    final_clip.write_videofile(str(output_path))

    # Close all clips
    for clip in clips:
        clip.close()


def smooth_coordinates(data, window_size=5):
    """
    Smooth the coordinates using a rolling median.
    """
    return (
        data.rolling(window=window_size, center=True)
        .median()
        .fillna(method="bfill")
        .fillna(method="ffill")
    )


def generate_clip(data, fly_name, event, outpath, crop=False):
    """
    Make a video clip of a fly's event.
    """

    try:
        # Filter the data for the specific fly
        fly_data = data[data["fly"] == fly_name]

        # Parse the fly name to construct the folder path
        match = re.match(
            r"(\d{6}_TNT_Fine_\d+)_Videos_Tracked_arena(\d+)_corridor(\d+)", fly_name
        )
        if not match:
            raise ValueError(f"Invalid fly name format: {fly_name}")

        base_folder = match.group(1) + "_Videos_Tracked"
        arena = f"arena{match.group(2)}"
        corridor = f"corridor{match.group(3)}"
        folder_path = os.path.join(base_folder, arena, corridor)

        folder_path = os.path.join(DataPath, folder_path)

        # Debugging print statements
        print(f"Constructed folder path: {folder_path}")

        # Fetch the preprocessed video file
        video_files = glob.glob(os.path.join(folder_path, "*preprocessed*.mp4"))
        print(f"Found video files: {video_files}")

        if not video_files:
            raise FileNotFoundError(
                f"No preprocessed video file found in {folder_path}"
            )
        video_file = video_files[0]  # Assuming there's only one preprocessed video file

        # Filter the data for the specific event
        event_data = fly_data[fly_data["contact_index"] == event]

        # Open the video file
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {video_file}")

        # Get the frame rate of the video
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # Get start and end frames as the first and last frames of the event dataset
        start_frame = event_data["frame"].iloc[0]
        end_frame = event_data["frame"].iloc[-1]

        # Create a VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        # Get the video dimensions
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(
            f"Width: {width}, Height: {height}, FPS: {fps}, Start Frame: {start_frame}, End Frame: {end_frame}"
        )

        # Calculate cropping coordinates if crop is True
        if crop:
            # Smooth the coordinates
            event_data.loc[
                :,
                [
                    "x_Thorax",
                    "y_Thorax",
                    "x_centre_preprocessed",
                    "y_centre_preprocessed",
                ],
            ] = smooth_coordinates(
                event_data.loc[
                    :,
                    [
                        "x_Thorax",
                        "y_Thorax",
                        "x_centre_preprocessed",
                        "y_centre_preprocessed",
                    ],
                ]
            )

            x_coords = event_data[
                ["x_Thorax", "x_centre_preprocessed"]
            ].values.flatten()
            y_coords = event_data[
                ["y_Thorax", "y_centre_preprocessed"]
            ].values.flatten()
            x_min, x_max = round(x_coords.min() - 15), round(x_coords.max() + 15)
            y_min, y_max = round(y_coords.min() - 15), round(y_coords.max() + 15)
            x_min, x_max = max(0, x_min), min(width, x_max)
            y_min, y_max = max(0, y_min), min(height, y_max)
            print(
                f"Cropping coordinates: x_min={x_min}, x_max={x_max}, y_min={y_min}, y_max={y_max}"
            )
            crop_width = x_max - x_min
            crop_height = y_max - y_min
        else:
            crop_width = width
            crop_height = height

        # Create the output file
        out = cv2.VideoWriter(outpath, fourcc, fps, (crop_width, crop_height))

        try:
            # Go to the start frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            # Read the video frame by frame and write to the output file
            while cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    if crop:
                        frame = frame[y_min:y_max, x_min:x_max]
                    out.write(frame)
                    if cap.get(cv2.CAP_PROP_POS_FRAMES) >= end_frame:
                        break
                else:
                    break
        finally:
            # Release the video capture and writer objects
            cap.release()
            out.release()

        print(f"Successfully generated clip for {fly_name}, event {event}")

    except Exception as e:
        print(f"Error generating clip for {fly_name}, event {event}: {e}")

    return outpath


def process_dataset(
    data, events_data, output_dir, sample_size=None, crop=False, grid=False
):
    """
    Process the dataset and generate video clips for each combination of fly and contact index.
    """

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Group by fly and contact_index
    grouped = data.groupby(["fly", "contact_index"])

    # Sample the groups if sample_size is specified
    if sample_size:
        sampled_groups = random.sample(list(grouped), sample_size)
    else:
        sampled_groups = list(grouped)

    # Iterate over the sampled groups
    for (fly_name, contact_index), group in sampled_groups:
        # Generate the output file path
        outpath = os.path.join(output_dir, f"{fly_name}_event_{contact_index}.mp4")

        # Check if the clip already exists
        if os.path.exists(outpath):
            print(f"Clip already exists: {outpath}")
            continue

        # Generate the video clip
        try:
            generate_clip(events_data, fly_name, contact_index, outpath, crop)
        except Exception as e:
            print(
                f"Failed to generate video for {fly_name}, event {contact_index}: {e}"
            )

    if grid:
        # Create a grid folder in the output directory if it doesn't exist
        os.makedirs(os.path.join(output_dir, "grid"), exist_ok=True)

        create_grid_video(
            output_dir,
            os.path.join(output_dir, "grid", "grid.mp4"),
        )


def process_folder(
    input_dir,
    events_data_path,
    base_output_dir,
    sample_size=None,
    crop=False,
    grid=False,
):
    """
    Process all Feather files in the given directory and generate video clips for each.
    """

    # Ensure the base output directory exists
    os.makedirs(base_output_dir, exist_ok=True)

    # Iterate over all Feather files in the input directory
    for file in glob.glob(os.path.join(input_dir, "*.feather")):
        print(f"Processing file: {file}")

        try:
            # Load the dataset
            data = pd.read_feather(file)

            # Generate the output directory for this file
            file_name = os.path.splitext(os.path.basename(file))[0]
            output_dir = os.path.join(base_output_dir, file_name)

            # Load the events data
            events_data = pd.read_feather(events_data_path)

            # Process the dataset and generate video clips
            process_dataset(data, events_data, output_dir, sample_size, crop, grid)

            print(f"Finished processing file: {file}")

        except Exception as e:
            print(f"Failed to process file: {file}: {e}")


# Example usage
if __name__ == "__main__":
    input_dir = "/mnt/upramdya_data/MD/MultiMazeRecorder/Datasets/Skeleton_TNT/Cluster_data/250107_LooseContacts_Mapped"
    events_data_path = "/mnt/upramdya_data/MD/MultiMazeRecorder/Datasets/Skeleton_TNT/250106_FinalEventCutoffData_norm/contact_data/250106_Pooled_contact_data.feather"
    base_output_dir = "/mnt/upramdya_data/MD/MultiMazeRecorder/Datasets/Skeleton_TNT/BehaviorClusters/Loose_Contacts_Slowed"

    sample_size = 30
    crop = False
    grid = True

    # Process all Feather files in the input directory
    process_folder(
        input_dir, events_data_path, base_output_dir, sample_size, crop, grid
    )
