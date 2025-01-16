from pathlib import Path
import pandas as pd
from typing import List, Optional
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# Constants
DATA_PATH = "/mnt/upramdya_data/MD/MultiMazeRecorder/Datasets/Coordinates/250106_Transformed_contact_data.feather"
NICKNAME_LIST = [
    "34497 (MZ19-GAL4)",
    "DDC-gal4",
    "Ple-Gal4.F a.k.a TH-Gal4",
    "MB504B (All PPL1)",
    "VT43924 (MB-APL)",
    "474 (MB-APL)",
]
OUTPUT_DIR = "/home/durrieu/Videos"  # Specify your desired output directory here


def load_dataset(file_path: str) -> pd.DataFrame:
    """Load dataset from feather file."""
    try:
        return pd.read_feather(file_path)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return pd.DataFrame()


def filter_by_nickname(data: pd.DataFrame, nickname: str) -> pd.DataFrame:
    """Filter dataset by nickname."""
    if "Nickname" not in data.columns:
        print("Error: 'Nickname' column not found in dataset.")
        return pd.DataFrame()
    return data.loc[data["Nickname"] == nickname]


def get_video_paths(flypaths: List[str]) -> List[Path]:
    """Get paths of preprocessed videos."""
    return [
        next(Path(flypath).glob("*preprocessed*.mp4"), None) for flypath in flypaths
    ]


def load_video(path: Path) -> Optional[cv2.VideoCapture]:
    """Load a single video clip."""
    if path and path.exists():
        try:
            cap = cv2.VideoCapture(str(path))
            if cap.isOpened():
                print(f"Successfully loaded video: {path}")
                return cap
            else:
                print(f"Error opening video {path}")
        except Exception as e:
            print(f"Error loading video {path}: {e}")
    else:
        print(f"Invalid or non-existent path: {path}")
    return None


def load_videos(video_paths: List[Path]) -> List[cv2.VideoCapture]:
    """Load video clips in parallel."""
    with ThreadPoolExecutor() as executor:
        videos = list(executor.map(load_video, video_paths))
    return [video for video in videos if video is not None]


def add_identifier(frame: np.ndarray, identifier: str, position="bottom") -> np.ndarray:
    """Add identifier text to a frame."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (255, 255, 255)
    thickness = 2
    size = cv2.getTextSize(identifier, font, font_scale, thickness)[0]
    if position == "bottom":
        x = 10
        y = frame.shape[0] - 10
    else:
        x = 10
        y = 10 + size[1]
    cv2.putText(
        frame, identifier, (x, y), font, font_scale, color, thickness, cv2.LINE_AA
    )
    return frame


def create_video_grid_with_features(
    videos: List[cv2.VideoCapture],
    identifiers: List[str],
    padding=10,
    grid_text="Video Grid Information",
) -> Optional[cv2.VideoWriter]:
    """Create a grid of videos with identifiers and features."""
    if not videos:
        print("No videos to create grid.")
        return None

    # Get the properties of the first video
    width = int(videos[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(videos[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = videos[0].get(cv2.CAP_PROP_FPS)

    # Create a VideoWriter object
    output_path = Path(OUTPUT_DIR) / "video_grid.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    while True:
        frames = []
        for video, identifier in zip(videos, identifiers):
            ret, frame = video.read()
            if not ret:
                break
            frame = add_identifier(frame, identifier)
            frames.append(frame)

        if len(frames) != len(videos):
            break

        # Create a grid of frames
        grid_frame = np.hstack(frames)
        out.write(grid_frame)

    for video in videos:
        video.release()
    out.release()
    return out


def generate_identifiers(data: pd.DataFrame, video_paths: List[Path]) -> List[str]:
    """Generate identifiers for each video based on the directory structure and dataset."""
    identifiers = []
    for path in video_paths:
        if path:
            corridor = path.parent.name
            arena = path.parent.parent.name
            date = data.loc[data["flypath"] == str(path.parent), "Date"].values[0]
            identifier = f"D{date} \n A{arena} \n C{corridor}"
            identifiers.append(identifier)
        else:
            identifiers.append("Unknown")
    return identifiers


def main(test=False):
    transformed_data = load_dataset(DATA_PATH)
    if transformed_data.empty:
        return

    mz19_data = filter_by_nickname(transformed_data, NICKNAME_LIST[0])
    if mz19_data.empty:
        return

    flypaths = mz19_data["flypath"].unique()
    video_paths = get_video_paths(flypaths)
    videos = load_videos(video_paths)

    print(f"Number of valid videos loaded: {len(videos)}")

    if not videos:
        print("No valid videos were loaded. Exiting.")
        return

    if test:
        # Trim videos to 20 seconds for testing
        for video in videos:
            video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            video.set(
                cv2.CAP_PROP_FRAME_COUNT,
                min(
                    20 * int(video.get(cv2.CAP_PROP_FPS)),
                    int(video.get(cv2.CAP_PROP_FRAME_COUNT)),
                ),
            )

    identifiers = generate_identifiers(mz19_data, video_paths)
    video_grid = create_video_grid_with_features(videos, identifiers)

    if video_grid:
        print("Video grid created successfully.")


if __name__ == "__main__":
    main(test=True)
