import logging
from pathlib import Path
import pandas as pd
from typing import List, Optional
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import os
from tqdm import tqdm
import re
import math
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration parameters
CONFIG = {
    "font_scale": 0.5,
    "line_spacing": 20,
    "horizontal_padding": 10,
    "vertical_padding": 50,
    "top_padding": 60,
    "bottom_padding": 50,
    "rotate_videos": True,
    "rotation_angle": cv2.ROTATE_90_CLOCKWISE,
    "output_dir": "/mnt/upramdya_data/MD/MagnetBlock/Grids",
    "max_grid_width": 3840,
    "max_grid_height": 2160,
    "min_cell_width": 320,
    "min_cell_height": 180,
    "aspect_ratio_tolerance": 0.2,
    "test_duration": 10,
}

MIN_DURATION = 1800  # 30 minutes

# Constants
DATA_PATH = "/mnt/upramdya_data/MD/MagnetBlock/Datasets/250213_coordinates.feather"

groupby = "label"

METADATA_COLUMNS = ["label"]


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


def filter_by_column(data: pd.DataFrame, column_name: str, value: str) -> pd.DataFrame:
    """Filter dataset by a specified column and value."""
    if column_name not in data.columns:
        logger.error(f"Error: '{column_name}' column not found in dataset.")
        return pd.DataFrame()
    return data.loc[data[column_name] == value]


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


def load_videos(video_paths: List[Path]) -> List[tuple[cv2.VideoCapture, Path]]:
    with ThreadPoolExecutor() as executor:
        videos = list(executor.map(load_video, video_paths))
    return [
        (video, path) for video, path in zip(videos, video_paths) if video is not None
    ]


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
        # Calculate aspect ratios considering rotation
        aspect_ratios = []
        for v in videos:
            if CONFIG["rotate_videos"]:
                ar = v.get(cv2.CAP_PROP_FRAME_HEIGHT) / v.get(cv2.CAP_PROP_FRAME_WIDTH)
            else:
                ar = v.get(cv2.CAP_PROP_FRAME_WIDTH) / v.get(cv2.CAP_PROP_FRAME_HEIGHT)
            aspect_ratios.append(ar)

        # Find median aspect ratio for grid calculation
        median_aspect = sorted(aspect_ratios)[len(aspect_ratios) // 2]
        num_videos = len(videos)

        # Dynamic grid calculation
        best_layout = None
        best_utilization = 0
        max_cols = min(
            num_videos, int(CONFIG["max_grid_width"] / CONFIG["min_cell_width"])
        )

        for cols in range(1, max_cols + 1):
            rows = math.ceil(num_videos / cols)

            # Calculate available space for cells
            avail_width = CONFIG["max_grid_width"] - CONFIG["horizontal_padding"] * (
                cols + 1
            )
            avail_height = (
                CONFIG["max_grid_height"]
                - CONFIG["top_padding"]
                - CONFIG["bottom_padding"]
                - CONFIG["vertical_padding"] * (rows + 1)
            )

            if avail_width <= 0 or avail_height <= 0:
                continue

            cell_width = avail_width / cols
            cell_height = avail_height / rows

            # Check minimum size constraints
            if (
                cell_width < CONFIG["min_cell_width"]
                or cell_height < CONFIG["min_cell_height"]
            ):
                continue

            # Check aspect ratio compatibility
            cell_ar = cell_width / cell_height
            if abs(cell_ar - median_aspect) > CONFIG["aspect_ratio_tolerance"]:
                continue

            # Calculate space utilization
            utilization = (
                (cell_width * cell_height)
                * num_videos
                / (CONFIG["max_grid_width"] * CONFIG["max_grid_height"])
            )
            if utilization > best_utilization:
                best_utilization = utilization
                best_layout = (cols, rows, cell_width, cell_height)

        # Original grid calculation attempt
        if not best_layout:
            logger.warning("No valid layout found - attempting video reduction")

            # Try removing videos until we find a workable layout
            reduction_factor = 0.9  # Remove 10% of videos each attempt
            max_attempts = 5
            original_count = num_videos

            for attempt in range(max_attempts):
                new_count = int(num_videos * (reduction_factor ** (attempt + 1)))
                if new_count < 4:  # Minimum videos to process
                    break

                logger.info(f"Attempting layout with {new_count} videos")

                # Randomly select subset of videos
                indices = np.random.choice(num_videos, new_count, replace=False)
                subset_videos = [videos[i] for i in indices]
                subset_aspects = [aspect_ratios[i] for i in indices]

                # Recalculate median aspect with subset
                median_aspect = sorted(subset_aspects)[len(subset_aspects) // 2]

                # Re-run grid calculation with subset
                best_layout = calculate_optimal_layout(
                    new_count,
                    median_aspect,
                    CONFIG["max_grid_width"],
                    CONFIG["max_grid_height"],
                )

                if best_layout:
                    logger.info(f"Found layout with {new_count} videos")
                    videos = subset_videos
                    identifiers = [identifiers[i] for i in indices]
                    break

            if not best_layout:
                logger.error(
                    f"Failed to find layout even after reducing to {new_count} videos"
                )
                return None

        cols, rows, cell_width, cell_height = best_layout
        cell_width, cell_height = int(cell_width), int(cell_height)

        # Calculate final grid dimensions
        grid_width = int(cols * cell_width + CONFIG["horizontal_padding"] * (cols + 1))
        grid_height = int(
            rows * cell_height
            + CONFIG["top_padding"]
            + CONFIG["bottom_padding"]
            + CONFIG["vertical_padding"] * (rows + 1)
        )

        # Initialize video writer
        fps = videos[0].get(cv2.CAP_PROP_FPS)
        output_path = Path(CONFIG["output_dir"]) / output_filename
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (grid_width, grid_height))

        # Handle sample/test duration
        total_frames = min(int(v.get(cv2.CAP_PROP_FRAME_COUNT)) for v in videos)
        if test:
            total_frames = min(
                total_frames,
                int(CONFIG["test_duration"] * fps),
            )

        with tqdm(total=total_frames, desc="Processing frames") as pbar:
            for frame_count in range(total_frames):
                grid_frame = create_grid_frame(grid_height, grid_width, grid_text)

                for idx, (video, identifier) in enumerate(zip(videos, identifiers)):
                    row = idx // cols
                    col = idx % cols

                    x_offset = (
                        CONFIG["horizontal_padding"] * (col + 1) + cell_width * col
                    )
                    y_offset = (
                        CONFIG["top_padding"]
                        + CONFIG["vertical_padding"]
                        + row * (cell_height + CONFIG["vertical_padding"])
                    )

                    frame = read_and_process_frame(video, cell_width, cell_height)
                    if frame is None:
                        continue

                    try:
                        grid_section = grid_frame[
                            y_offset : y_offset + cell_height,
                            x_offset : x_offset + cell_width,
                        ]
                        grid_section[:] = frame
                    except Exception as e:
                        logger.error(f"Frame placement failed: {e}")
                        continue

                    # Add identifier text
                    cv2.putText(
                        grid_frame,
                        identifier,
                        (
                            x_offset + 10,
                            (
                                y_offset + (cell_height + 10)
                                if CONFIG["rotate_videos"]
                                else y_offset + cell_height + 10
                            ),
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        CONFIG["font_scale"]
                        * (
                            1 if CONFIG["rotate_videos"] else 1
                        ),  # Smaller text when rotated
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
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


def calculate_optimal_layout(num_videos, median_aspect, max_width, max_height):
    best_layout = None
    best_utilization = 0
    max_cols = min(num_videos, int(max_width / CONFIG["min_cell_width"]))

    for cols in range(1, max_cols + 1):
        rows = math.ceil(num_videos / cols)

        # Relax constraints for reduced video count
        min_cell_w = CONFIG["min_cell_width"] * (0.9 ** (num_videos / 50))
        min_cell_h = CONFIG["min_cell_height"] * (0.9 ** (num_videos / 50))

        avail_width = max_width - CONFIG["horizontal_padding"] * (cols + 1)
        avail_height = (
            max_height
            - CONFIG["top_padding"]
            - CONFIG["bottom_padding"]
            - CONFIG["vertical_padding"] * (rows + 1)
        )

        if avail_width <= 0 or avail_height <= 0:
            continue

        cell_width = avail_width / cols
        cell_height = avail_height / rows

        if cell_width < min_cell_w or cell_height < min_cell_h:
            continue

        cell_ar = cell_width / cell_height
        if (
            abs(cell_ar - median_aspect) > CONFIG["aspect_ratio_tolerance"] * 1.5
        ):  # Increased tolerance
            continue

        utilization = (cell_width * cell_height) * num_videos / (max_width * max_height)
        if utilization > best_utilization:
            best_utilization = utilization
            best_layout = (cols, rows, cell_width, cell_height)

    return best_layout


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


def read_and_process_frame(
    video: cv2.VideoCapture, target_width: int, target_height: int
) -> Optional[np.ndarray]:
    ret, frame = video.read()
    if not ret:
        return None

    if CONFIG["rotate_videos"]:
        frame = cv2.rotate(frame, CONFIG["rotation_angle"])

    # Maintain aspect ratio with padding
    h, w = frame.shape[:2]
    aspect = w / h

    # Calculate scaled dimensions
    if aspect > (target_width / target_height):
        new_w = target_width
        new_h = int(target_width / aspect)
    else:
        new_h = target_height
        new_w = int(target_height * aspect)

    # Resize and pad
    try:
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        if new_w != target_width or new_h != target_height:
            frame = cv2.copyMakeBorder(
                frame,
                (target_height - new_h) // 2,
                (target_height - new_h + 1) // 2,
                (target_width - new_w) // 2,
                (target_width - new_w + 1) // 2,
                cv2.BORDER_CONSTANT,
                value=(0, 0, 0),
            )
        return cv2.resize(frame, (target_width, target_height))
    except Exception as e:
        logger.error(f"Frame processing error: {e}")
        return None


def place_frame_in_grid(grid_frame, frame, index, max_width, max_height, identifier):
    # Calculate position based on rotation state
    if CONFIG["rotate_videos"]:
        x_offset = CONFIG["horizontal_padding"] * (index + 1) + max_height * index
        y_offset = CONFIG["top_padding"] + CONFIG["vertical_padding"]
        text_x = x_offset + 10
        text_y = y_offset + max_width + CONFIG["vertical_padding"]
    else:
        x_offset = CONFIG["horizontal_padding"] * (index + 1) + max_width * index
        y_offset = CONFIG["top_padding"] + CONFIG["vertical_padding"]
        text_y = y_offset + max_height + CONFIG["vertical_padding"]

    # Place rotated frame
    if CONFIG["rotate_videos"]:
        grid_frame[
            y_offset : y_offset + max_width, x_offset : x_offset + max_height
        ] = frame
    else:
        grid_frame[
            y_offset : y_offset + max_height, x_offset : x_offset + max_width
        ] = frame

    # Add identifier text
    for j, line in enumerate(identifier.split("\n")):
        cv2.putText(
            grid_frame,
            line,
            (
                text_x if CONFIG["rotate_videos"] else x_offset + 10,
                text_y + j * CONFIG["line_spacing"],
            ),
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

            if CONFIG["rotate_videos"]:
                # Single line format for rotated videos
                identifier = f"{date}_{arena}_{corridor}"
            else:
                # Multi-line format
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


def sanitize_filename(filename: str) -> str:
    # Remove or replace problematic characters
    return re.sub(r'[<>:"/\\|?*]', "_", filename)


def validate_video_duration(video: cv2.VideoCapture, path: Path) -> bool:
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps if fps > 0 else 0

    if duration < MIN_DURATION:
        logger.warning(f"Excluding short video: {path.name} ({duration//60:.0f}min)")
        return False
    return True


def main(test=False, full_screen=False):
    transformed_data = load_dataset(DATA_PATH)
    if transformed_data.empty:
        print("empty data")
        return

    ensure_output_directory_exists(CONFIG["output_dir"])

    nickname_list = transformed_data[groupby].unique() if full_screen else []

    # if not nickname_list:
    #     logger.error("No experimental conditions found in dataset")
    #     return

    for nickname in nickname_list:
        nickname_data = filter_by_column(transformed_data, groupby, nickname)
        if nickname_data.empty:
            continue

        flypaths = nickname_data["flypath"].unique()
        video_paths = get_video_paths(flypaths)
        video_entries = load_videos(video_paths)

        valid_videos = [
            v for v, path in video_entries if validate_video_duration(v, path)
        ]
        if not valid_videos:
            continue

        title = generate_metadata(nickname_data)
        output_filename = sanitize_filename(f"{nickname}_grid.mp4")

        if Path(CONFIG["output_dir"], output_filename).exists():
            logger.info(f"Skipping existing: {output_filename}")
            continue

        identifiers = generate_identifiers(nickname_data, [p for _, p in video_entries])
        video_grid = create_video_grid_with_features(
            valid_videos,
            identifiers=identifiers,
            grid_text=title,
            output_filename=output_filename,
            test=test,
        )

        if video_grid:
            logger.info(f"Video grid created: {output_filename}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Enable test mode (10sec)")
    parser.add_argument(
        "--full-screen", action="store_true", help="Process all experimental conditions"
    )
    args = parser.parse_args()

    main(test=args.test, full_screen=args.full_screen)
