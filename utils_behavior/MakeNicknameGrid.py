from pathlib import Path
import pandas as pd
from typing import List, Optional

from utils_behavior import Ballpushing_utils

from moviepy.editor import VideoFileClip, clips_array, TextClip, CompositeVideoClip, vfx

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


def load_video(path: Path) -> Optional[VideoFileClip]:
    """Load a single video clip."""
    if path and path.exists():
        try:
            clip = VideoFileClip(str(path))
            print(f"Successfully loaded video: {path}")
            return clip
        except Exception as e:
            print(f"Error loading video {path}: {e}")
    else:
        print(f"Invalid or non-existent path: {path}")
    return None


def load_videos(video_paths: List[Path]) -> List[VideoFileClip]:
    """Load video clips in parallel."""
    with ThreadPoolExecutor() as executor:
        videos = list(executor.map(load_video, video_paths))
    return [video for video in videos if video is not None]


def create_video_grid(videos: List[VideoFileClip]) -> Optional[CompositeVideoClip]:
    """Create a grid of videos."""
    if not videos:
        print("No videos to create grid.")
        return None
    try:
        return clips_array([videos])
    except Exception as e:
        print(f"Error creating video grid: {e}")
        return None


def add_identifier(
    clip: VideoFileClip, identifier: str, position="bottom"
) -> CompositeVideoClip:
    txt_clip = TextClip(
        identifier, fontsize=24, color="white", bg_color="black", font="Arial"
    )
    txt_clip = txt_clip.set_position(position).set_duration(clip.duration)
    return CompositeVideoClip([clip, txt_clip])


def create_video_grid_with_features(
    videos: List[VideoFileClip],
    identifiers: List[str],
    padding=10,
    grid_text="Video Grid Information",
) -> CompositeVideoClip:
    # Add identifiers and padding to each video
    processed_videos = []
    for video, identifier in zip(videos, identifiers):
        video_with_id = add_identifier(video, identifier)
        padded_video = video_with_id.margin(
            top=padding, bottom=padding, left=padding, right=padding
        )
        processed_videos.append(padded_video)

    # Create the grid
    grid = clips_array([processed_videos])

    # Add text overlay to the grid
    txt_clip = TextClip(
        grid_text,
        fontsize=70,
        color="white",
        bg_color="rgba(0,0,0,0.5)",
        size=(grid.w, grid.h),
        font="Arial",
    )
    txt_clip = txt_clip.set_position("center").set_duration(grid.duration)

    final_video = CompositeVideoClip([grid, txt_clip])
    return final_video


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
        videos = [video.subclip(0, min(20, video.duration)) for video in videos]

    identifiers = generate_identifiers(mz19_data, video_paths)
    video_grid = create_video_grid_with_features(videos, identifiers)

    if video_grid:
        print("Video grid created successfully.")
        output_path = Path(OUTPUT_DIR) / (
            "test_video_grid.mp4" if test else "full_video_grid.mp4"
        )
        video_grid.write_videofile(str(output_path), fps=29)

    for video in videos:
        video.close()


if __name__ == "__main__":
    main(test=True)
