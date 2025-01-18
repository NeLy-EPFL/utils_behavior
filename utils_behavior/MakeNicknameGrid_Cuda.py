import logging
from pathlib import Path
import pandas as pd
from typing import List, Optional
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import os
from tqdm import tqdm

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
    "output_dir": "/mnt/upramdya_data/MD/TNT_Screen_RawGrids/",
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
    "41744 (IR8a mutant)",
    "50742 (MB247-GAL4)",
    "MB247-Gal4",
    "854 (OK107-Gal4)",
    "MBON-11-GaL4 (MBON-γ1pedc>α/β)",
    "LC25",
    "LC4",
    "86666 (LH1139)",
    "86637 (LH2220)",
    "86705 (LH1668)",
    "86699 (LH123)",
    "SS52577-gal4 (PBG2‐9.s‐FBℓ3.b‐NO2V.b (PB))",
    "SS00078-gal4 (PBG2‐9.s‐FBℓ3.b‐NO2D.b (PB))",
    "SS02239-gal4 (P-F3LC patch line)",
    "MB312B (PAM-07)",
    "MB043B (PAM-11)",
    "PAM-01 (MB315C)",
    "MB063B (PAM-10)",
]

METADATA_COLUMNS = [
    "Brain region",
    "Nickname",
    "Genotype",
]


def ensure_output_directory_exists(output_dir: str):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Directory created: {output_dir}")
    else:
        logger.info(f"Directory already exists: {output_dir}")


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
    grid_text: str,
    output_filename: str,
    test=False,
) -> Optional[cv2.VideoWriter]:
    if not videos:
        logger.error("No videos to create grid.")
        return None

    try:
        widths, heights = zip(
            *[
                (
                    int(v.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    min(int(v.get(cv2.CAP_PROP_FRAME_HEIGHT)), CONFIG["max_height"]),
                )
                for v in videos
            ]
        )
        fps = videos[0].get(cv2.CAP_PROP_FPS)

        # Determine total number of frames to process
        total_frames = min(int(v.get(cv2.CAP_PROP_FRAME_COUNT)) for v in videos)
        if test:
            total_frames = min(total_frames, int(CONFIG["test_duration"] * fps))

        max_width, max_height = max(widths), max(heights)
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
            f"test_{output_filename}" if test else output_filename
        )
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (new_width, new_height))

        with tqdm(total=total_frames, desc="Processing frames") as pbar:
            for frame_count in range(total_frames):
                grid_frame = create_grid_frame(new_height, new_width, grid_text)

                for i, (video, identifier) in enumerate(zip(videos, identifiers)):
                    frame = read_and_process_frame(video, max_width, max_height)
                    if frame is None:
                        logger.warning(
                            f"End of video {i} reached before expected frame count."
                        )
                        continue

                    place_frame_in_grid(
                        grid_frame, frame, i, max_width, max_height, identifier
                    )

                out.write(grid_frame)
                pbar.update(1)

        return out

    except Exception as e:
        logger.error(f"Error in create_video_grid_with_features: {e}")
        return None

    finally:
        for video in videos:
            video.release()
        if "out" in locals():
            out.release()
        logger.info(f"Video written to {output_path}")


def create_grid_frame(height, width, grid_text):
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(
        frame,
        grid_text,
        (CONFIG["horizontal_padding"], CONFIG["top_padding"] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return frame


def read_and_process_frame(video, max_width, max_height):
    ret, frame = video.read()
    if not ret:
        return None

    frame = frame[: CONFIG["max_height"], :, :]  # Crop if needed

    # Use GPU for resizing
    gpu_frame = cv2.cuda_GpuMat()
    gpu_frame.upload(frame)
    gpu_frame = cv2.cuda.resize(gpu_frame, (max_width, max_height))
    return gpu_frame.download()


def place_frame_in_grid(grid_frame, frame, index, max_width, max_height, identifier):
    x_offset = CONFIG["horizontal_padding"] * (index + 1) + max_width * index
    y_offset = CONFIG["top_padding"] + CONFIG["vertical_padding"]

    grid_frame[y_offset : y_offset + max_height, x_offset : x_offset + max_width] = (
        frame
    )

    identifier_y_start = y_offset + max_height + CONFIG["vertical_padding"]
    for j, line in enumerate(identifier.split("\n")):
        y_position = identifier_y_start + j * CONFIG["line_spacing"]
        cv2.putText(
            grid_frame,
            line,
            (x_offset + 10, y_position),
            cv2.FONT_HERSHEY_SIMPLEX,
            CONFIG["font_scale"],
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


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


def generate_metadata(data: pd.DataFrame) -> str:

    # For each metadata column, get the first unique value and add it to the title along with the column name

    title_elements = []

    for column in METADATA_COLUMNS:

        value = data[column].values[0]
        title_elements.append(f"{column}: {value}")

    title = f"{'_'.join(title_elements)}"

    return title


def main(test=False, full_screen=False):
    transformed_data = load_dataset(DATA_PATH)
    if transformed_data.empty:
        return

    ensure_output_directory_exists(CONFIG["output_dir"])

    # If full screen, get all unique nicknames in the dataset, else use the NICKNAME_LIST

    if full_screen:
        NICKNAME_LIST = transformed_data["Nickname"].unique()

    else:
        NICKNAME_LIST = NICKNAME_LIST

    for nickname in NICKNAME_LIST:
        nickname_data = filter_by_nickname(transformed_data, nickname)
        if nickname_data.empty:
            continue

        flypaths = nickname_data["flypath"].unique()
        video_paths = get_video_paths(flypaths)
        videos = load_videos(video_paths)

        logger.info(f"Number of valid videos loaded: {len(videos)}")

        if not videos:
            logger.error("No valid videos were loaded. Skipping.")
            continue

        title = generate_metadata(nickname_data)
        output_filename = f"{nickname}_grid.mp4"

        # Check if the video grid already exists and if so, skip to the next nickname

        if os.path.exists(CONFIG["output_dir"] + output_filename):
            logger.info(f"Video grid already exists: {output_filename}")
            continue

        identifiers = generate_identifiers(nickname_data, video_paths)
        video_grid = create_video_grid_with_features(
            videos,
            grid_text=title,
            identifiers=identifiers,
            output_filename=output_filename,
            test=test,
        )

        if video_grid:
            logger.info(f"Video grid created successfully: {output_filename}")


if __name__ == "__main__":
    main(test=False)
