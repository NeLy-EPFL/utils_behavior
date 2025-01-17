import logging
from pathlib import Path
import pandas as pd
from typing import List, Optional
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration parameters
CONFIG = {
    "max_height": 510,
    "font_scale": 0.5,
    "line_spacing": 20,
    "horizontal_padding": 10,
    "vertical_padding": 50,
    "top_padding": 60,
    "bottom_padding": 50,
    "output_dir": "/mnt/upramdya_data/MD/Other_Videos/",
    "test_duration": 10,  # seconds
}

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


def load_dataset(file_path: str) -> pd.DataFrame:
    try:
        return pd.read_feather(file_path)
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return pd.DataFrame()


def filter_by_nickname(data: pd.DataFrame, nickname: str) -> pd.DataFrame:
    if "Nickname" not in data.columns:
        logger.error("Error: 'Nickname' column not found in dataset.")
        return pd.DataFrame()
    return data.loc[data["Nickname"] == nickname]


def get_video_paths(flypaths: List[str], preprocessed: bool = False) -> List[Path]:
    video_paths = []
    for flypath in flypaths:
        path = Path(flypath)
        try:
            if preprocessed:
                video = next(path.glob("*preprocessed*.mp4"), None)
            else:
                video = next(
                    (p for p in path.glob("*.mp4") if "preprocessed" not in p.name),
                    None,
                )
            video_paths.append(video)
        except StopIteration:
            logger.warning(f"No video found in path: {path}")
            video_paths.append(None)
    return video_paths


def load_video(path: Path) -> Optional[cv2.VideoCapture]:
    if path and path.exists():
        try:
            cap = cv2.VideoCapture(str(path))
            if cap.isOpened():
                logger.info(f"Successfully loaded video: {path}")
                return cap
            else:
                logger.error(f"Error opening video {path}")
        except Exception as e:
            logger.error(f"Error loading video {path}: {e}")
    else:
        logger.error(f"Invalid or non-existent path: {path}")
    return None


def load_videos(video_paths: List[Path]) -> List[cv2.VideoCapture]:
    with ThreadPoolExecutor() as executor:
        videos = list(executor.map(load_video, video_paths))
    return [video for video in videos if video is not None]


def add_identifier(frame: np.ndarray, identifier: str, position="bottom") -> np.ndarray:
    gpu_frame = cv2.cuda_GpuMat()
    gpu_frame.upload(frame)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = CONFIG["font_scale"]
    color = (255, 255, 255)
    thickness = 1
    line_height = int(cv2.getTextSize("A", font, font_scale, thickness)[0][1] * 1.5)
    y0 = (
        frame.shape[0] - CONFIG["bottom_padding"]
        if position == "bottom"
        else CONFIG["top_padding"] + line_height
    )

    frame = gpu_frame.download()
    for i, line in enumerate(identifier.split("\n")):
        y = y0 + i * CONFIG["line_spacing"]
        cv2.putText(
            frame, line, (10, y), font, font_scale, color, thickness, cv2.LINE_AA
        )
    gpu_frame.upload(frame)
    return gpu_frame.download()


def create_video_grid_with_features(
    videos: List[cv2.VideoCapture],
    identifiers: List[str],
    grid_text="Video Grid Information",
    test=False,
) -> Optional[cv2.VideoWriter]:
    """Create a grid of videos with identifiers and features using padding instead of resizing."""
    if not videos:
        logger.error("No videos to create grid.")
        return None

    # Get the properties of each video
    widths = [int(video.get(cv2.CAP_PROP_FRAME_WIDTH)) for video in videos]
    heights = [int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)) for video in videos]
    fps = videos[0].get(cv2.CAP_PROP_FPS)

    max_width = max(widths)
    max_height = min(
        max(heights), CONFIG["max_height"]
    )  # Crop videos exceeding max height

    # Calculate new dimensions with padding
    new_width = max_width * len(videos) + CONFIG["horizontal_padding"] * (
        len(videos) + 1
    )
    new_height = (
        max_height
        + CONFIG["top_padding"]
        + CONFIG["bottom_padding"]
        + 2 * CONFIG["vertical_padding"]
    )

    output_path = Path(CONFIG["output_dir"]) / (
        "test_video_grid.mp4" if test else "video_grid.mp4"
    )
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (new_width, new_height))

    frame_count = 0
    max_frames = int(CONFIG["test_duration"] * fps) if test else float("inf")

    while frame_count < max_frames:
        # Create a blank grid frame with padding
        grid_frame = np.zeros((new_height, new_width, 3), dtype=np.uint8)

        # Add title at the top
        cv2.putText(
            grid_frame,
            grid_text,
            (CONFIG["horizontal_padding"], CONFIG["top_padding"] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        for i, (video, identifier) in enumerate(zip(videos, identifiers)):
            ret, frame = video.read()
            if not ret:
                logger.warning(f"Could not read frame from video {i}")
                continue

            # Crop the frame if its height exceeds the maximum allowed height
            if frame.shape[0] > CONFIG["max_height"]:
                frame = frame[: CONFIG["max_height"], :, :]

            # Add padding to match the maximum width and height
            padded_frame = np.zeros((max_height, max_width, 3), dtype=np.uint8)
            y_offset = (max_height - frame.shape[0]) // 2
            x_offset = (max_width - frame.shape[1]) // 2
            padded_frame[
                y_offset : y_offset + frame.shape[0],
                x_offset : x_offset + frame.shape[1],
            ] = frame

            # Place the padded frame in the grid
            x_offset_grid = CONFIG["horizontal_padding"] * (i + 1) + max_width * i
            y_offset_grid = CONFIG["top_padding"] + CONFIG["vertical_padding"]
            grid_frame[
                y_offset_grid : y_offset_grid + max_height,
                x_offset_grid : x_offset_grid + max_width,
            ] = padded_frame

            # Add identifiers below each video after placing all frames
            identifier_y_start = y_offset_grid + max_height + CONFIG["vertical_padding"]
            for j, line in enumerate(identifier.split("\n")):
                y_position = identifier_y_start + j * CONFIG["line_spacing"]
                cv2.putText(
                    grid_frame,
                    line,
                    (x_offset_grid + 10, y_position),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    CONFIG["font_scale"],
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        out.write(grid_frame)
        frame_count += 1
        if frame_count % 100 == 0:
            logger.info(f"Processed {frame_count} frames")

    for video in videos:
        video.release()
    out.release()
    logger.info(f"Video written to {output_path}")
    return out


def generate_identifiers(data: pd.DataFrame, video_paths: List[Path]) -> List[str]:
    identifiers = []
    for path in video_paths:
        if path:
            corridor = path.parent.name
            arena = path.parent.parent.name
            date = data.loc[data["flypath"] == str(path.parent), "Date"].values[0]
            identifier = f"{date}\n{arena}\n{corridor}"
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

    logger.info(f"Number of valid videos loaded: {len(videos)}")

    if not videos:
        logger.error("No valid videos were loaded. Exiting.")
        return

    identifiers = generate_identifiers(mz19_data, video_paths)
    video_grid = create_video_grid_with_features(videos, identifiers, test=test)

    if video_grid:
        logger.info("Video grid created successfully.")


if __name__ == "__main__":
    main(test=True)
