import cv2
import os
import glob
import pandas as pd
import re

import Utils

DataPath = Utils.get_data_path()


def generate_clip(data, fly_name, event, outpath):
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

        # Create the output file
        out = cv2.VideoWriter(outpath, fourcc, fps, (width, height))

        try:
            # Go to the start frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            # Read the video frame by frame and write to the output file
            while cap.isOpened():
                ret, frame = cap.read()
                if ret:
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


def process_dataset(data, events_data, output_dir):
    """
    Process the dataset and generate video clips for each combination of fly and contact index.
    """

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over all combinations of fly and contact index
    for _, row in data.iterrows():
        fly_name = row["fly"]
        contact_index = row["contact_index"]

        # Generate the output file path
        outpath = os.path.join(output_dir, f"{fly_name}_event_{contact_index}.mp4")

        # Generate the video clip
        try:
            generate_clip(events_data, fly_name, contact_index, outpath)
        except Exception as e:
            print(
                f"Failed to generate video for {fly_name}, event {contact_index}: {e}"
            )


def process_folder(input_dir, events_data_path, base_output_dir):
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
            process_dataset(data, events_data, output_dir)

            print(f"Finished processing file: {file}")

        except Exception as e:
            print(f"Failed to process file: {file}: {e}")


# Example usage
if __name__ == "__main__":
    input_dir = (
        "/mnt/upramdya_data/MD/MultiMazeRecorder/Datasets/Skeleton_TNT/Cluster_data"
    )
    events_data_path = "/mnt/upramdya_data/MD/MultiMazeRecorder/Datasets/Skeleton_TNT/241209_ContactData/241209_Pooled_contact_data.feather"
    base_output_dir = (
        "/mnt/upramdya_data/MD/MultiMazeRecorder/Datasets/Skeleton_TNT/BehaviorClusters"
    )

    # Process all Feather files in the input directory
    process_folder(input_dir, events_data_path, base_output_dir)
