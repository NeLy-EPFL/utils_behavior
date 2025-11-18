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
import traceback

import argparse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# TODO: Fix messed up labels in vertical orientation
# Configuration parameters
CONFIG = {
    "font_scale": 0.5,
    "line_spacing": 20,
    "horizontal_padding": 10,
    "vertical_padding": 50,
    "top_padding": 60,
    "bottom_padding": 50,
    "rotate_videos": False,
    "rotation_angle": cv2.ROTATE_90_CLOCKWISE,  # Rotate videos by default
    "multiline_identifiers": False,  # True = use line breaks, False = use underscores
    "output_dir": "/mnt/upramdya_data/MD/F1_Tracks/F1_New_Grids/Cleaned",
    "max_grid_width": 3840,  # 1920,  # 3840,
    "max_grid_height": 2160,  # 1080,  # 2160,
    "min_cell_width": 320,
    "min_cell_height": 180,
    "aspect_ratio_tolerance": 0.2,
    "test_duration": 10,
    "test_start_time": 5000,  # Start time in seconds for test mode (None = beginning)
    "trial_mode": False,
    "trial_column": "trial",
    "clip_buffer": 2,  # seconds before/after trial
    "temp_clip_dir": "/tmp/video_clips",
    "force_single_row": False,  # Set to True to always use a single row for the grid
    "F1_experiments": (
        True
    ),  # Set to True for F1 corridor experiments (validates adjusted_time)
}

MIN_DURATION = 60  # 1 minute

# Constants
DATA_PATH = "/mnt/upramdya_data/MD/F1_Tracks/Datasets/251014_10_F1_coordinates_F1_New_Data/F1_coordinates/pooled_F1_coordinates.feather"
MAPPING_CSV_PATH = "/mnt/upramdya_data/MD/Region_map_250908.csv"  # Map if needed
MISSING_VIDEOS_PATH = None  # Path for missing videos list if needed

groupby = "F1_condition"

METADATA_COLUMNS = ["Date"]  # Arena and corridor will be extracted from flypath


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


def load_missing_videos_list() -> List[str]:
    """Load the list of missing video identifiers from the text file."""
    try:
        if not Path(MISSING_VIDEOS_PATH).exists():
            logger.warning(f"Missing videos file not found: {MISSING_VIDEOS_PATH}")
            return []

        with open(MISSING_VIDEOS_PATH, "r") as f:
            lines = f.readlines()

        # Skip header lines and extract identifiers
        missing_identifiers = []
        for line in lines:
            line = line.strip()
            if (
                line
                and not line.startswith("Missing Videos:")
                and not line.startswith("=")
            ):
                missing_identifiers.append(line)

        logger.info(f"Loaded {len(missing_identifiers)} missing video identifiers")
        return missing_identifiers
    except Exception as e:
        logger.error(f"Error loading missing videos list: {e}")
        return []


def create_nickname_mapping() -> dict:
    """Create mapping from nicknames/genotypes to simplified nicknames."""
    try:
        mapping_data = pd.read_csv(MAPPING_CSV_PATH)
        nickname_to_simplified = {}

        # Create mapping from both Nickname and Genotype columns
        for _, row in mapping_data.iterrows():
            if pd.notna(row.get("Nickname")) and pd.notna(
                row.get("Simplified Nickname")
            ):
                nickname_to_simplified[row["Nickname"]] = row["Simplified Nickname"]

            if pd.notna(row.get("Genotype")) and pd.notna(
                row.get("Simplified Nickname")
            ):
                nickname_to_simplified[row["Genotype"]] = row["Simplified Nickname"]

        logger.info(f"Created mapping for {len(nickname_to_simplified)} identifiers")
        return nickname_to_simplified
    except Exception as e:
        logger.error(f"Error creating nickname mapping: {e}")
        return {}


def filter_by_column(data: pd.DataFrame, column_name: str, value: str) -> pd.DataFrame:
    """Filter dataset by a specified column and value."""
    if column_name not in data.columns:
        logger.error(f"Error: '{column_name}' column not found in dataset.")
        return pd.DataFrame()
    return data.loc[data[column_name] == value]


def get_video_paths(flypaths: List[str], trial_data: pd.DataFrame) -> List[Path]:
    if CONFIG["trial_mode"]:
        # Extract trial clips for each video
        with ThreadPoolExecutor() as executor:
            all_clips = list(
                executor.map(
                    lambda p: extract_trial_clips(Path(p), trial_data), flypaths
                )
            )
        return [clip for sublist in all_clips for clip in sublist]
    else:
        video_paths = []
        for fp in flypaths:
            matches = list(Path(fp).glob("*.mp4"))
            if not matches:
                logger.error(f"No MP4 files found in directory: {fp}")
                continue
            video_paths.append(matches[0])  # Take first match or implement sorting
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
            logger.error(f"Error loading video {path}: {e}\n{traceback.format_exc()}")
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
    force_max=False,
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

        median_aspect = sorted(aspect_ratios)[len(aspect_ratios) // 2]
        num_videos = len(videos)

        # --- FORCE SINGLE ROW LOGIC ---
        if CONFIG.get("force_single_row", False):
            cols = num_videos
            rows = 1
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
            cell_width = avail_width / cols
            cell_height = avail_height
            # Optionally, allow cell_height to be smaller than min_cell_height if needed
            if cell_width < CONFIG["min_cell_width"]:
                logger.warning(
                    "Cell width below min_cell_width in forced single row mode."
                )
            if cell_height < CONFIG["min_cell_height"]:
                logger.warning(
                    "Cell height below min_cell_height in forced single row mode."
                )
            best_layout = (cols, rows, cell_width, cell_height)
        else:
            # Dynamic grid calculation
            best_layout = None
            best_utilization = 0
            max_cols = min(
                num_videos, int(CONFIG["max_grid_width"] / CONFIG["min_cell_width"])
            )

            for cols in range(1, max_cols + 1):
                rows = math.ceil(num_videos / cols)

                # Calculate available space for cells
                avail_width = CONFIG["max_grid_width"] - CONFIG[
                    "horizontal_padding"
                ] * (cols + 1)
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

        # If no valid layout and force_max is set, relax constraints
        if not best_layout and force_max:
            logger.warning(
                "No valid layout found - relaxing constraints to fit all videos (force_max)"
            )
            relax_factor = 0.8  # Reduce min cell size by 20% each attempt
            ar_tolerance_factor = 1.5  # Increase aspect ratio tolerance
            min_cell_width = CONFIG["min_cell_width"]
            min_cell_height = CONFIG["min_cell_height"]
            aspect_tol = CONFIG["aspect_ratio_tolerance"]
            max_attempts = 5
            for attempt in range(max_attempts):
                min_cell_width *= relax_factor
                min_cell_height *= relax_factor
                aspect_tol *= ar_tolerance_factor
                for cols in range(1, max_cols + 1):
                    rows = math.ceil(num_videos / cols)
                    avail_width = CONFIG["max_grid_width"] - CONFIG[
                        "horizontal_padding"
                    ] * (cols + 1)
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
                    if cell_width < min_cell_width or cell_height < min_cell_height:
                        continue
                    cell_ar = cell_width / cell_height
                    if abs(cell_ar - median_aspect) > aspect_tol:
                        continue
                    utilization = (
                        (cell_width * cell_height)
                        * num_videos
                        / (CONFIG["max_grid_width"] * CONFIG["max_grid_height"])
                    )
                    if utilization > best_utilization:
                        best_utilization = utilization
                        best_layout = (cols, rows, cell_width, cell_height)
                if best_layout:
                    logger.info(
                        f"Found forced layout with relaxed constraints (attempt {attempt+1})"
                    )
                    break
            if not best_layout:
                logger.error(
                    "Failed to find layout even after relaxing constraints with force_max"
                )
                return None

        # Original grid calculation attempt (video reduction)
        if not best_layout and not force_max:
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

        # Handle sample/test duration and start time
        total_frames = min(int(v.get(cv2.CAP_PROP_FRAME_COUNT)) for v in videos)
        start_frame = 0

        if test:
            # If test_start_time is specified, seek to that position
            if CONFIG.get("test_start_time") is not None:
                start_frame = int(CONFIG["test_start_time"] * fps)
                logger.info(
                    f"Test mode: Starting from {CONFIG['test_start_time']}s (frame {start_frame})"
                )

                # Seek all videos to the start position
                for video in videos:
                    video_frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                    if start_frame >= video_frame_count:
                        logger.warning(
                            f"Start time {CONFIG['test_start_time']}s exceeds video duration"
                        )
                        start_frame = max(
                            0, video_frame_count - int(CONFIG["test_duration"] * fps)
                        )
                    video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            # Calculate number of frames to process
            total_frames = int(CONFIG["test_duration"] * fps)
        else:
            # For full videos, adjust total_frames to account for start position
            total_frames = total_frames - start_frame

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

                    # Add identifier text (handle multi-line text)
                    text_x = x_offset + 10
                    text_y_base = (
                        y_offset + cell_height + 10
                        if not CONFIG["rotate_videos"]
                        else y_offset + cell_height + 10
                    )

                    # Split identifier into lines and render each line separately
                    for line_idx, line in enumerate(identifier.split("\n")):
                        cv2.putText(
                            grid_frame,
                            line,
                            (text_x, text_y_base + line_idx * CONFIG["line_spacing"]),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            CONFIG["font_scale"],
                            (255, 255, 255),
                            1,
                            cv2.LINE_AA,
                        )

                out.write(grid_frame)
                pbar.update(1)

        return out

    except Exception as e:
        logger.error(
            f"Error in create_video_grid_with_features: {e}\n{traceback.format_exc()}"
        )
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
        logger.error(f"Frame processing error: {e}\n{traceback.format_exc()}")
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
            # Extract corridor and arena from flypath
            corridor = path.parent.name
            arena = path.parent.parent.name

            # Find matching row in data to get date
            matching_rows = data.loc[data["flypath"] == str(path.parent)]
            if not matching_rows.empty:
                date = str(matching_rows["Date"].iloc[0])
            else:
                date = "Unknown"

            # Choose separator based on config
            if CONFIG.get("multiline_identifiers", False):
                # Multi-line format with line breaks
                identifier = f"{date}\n{arena}\n{corridor}"
            else:
                # Single line format with underscores
                identifier = f"{date}_{arena}_{corridor}"

            identifiers.append(identifier)
        else:
            identifiers.append("Unknown")
    return identifiers


def generate_metadata(data: pd.DataFrame) -> str:
    # Extract date from the data
    date = data["Date"].iloc[0] if "Date" in data.columns else "Unknown"

    # Extract arena and corridor info from flypath if available
    arena_corridor_info = ""
    if "flypath" in data.columns and not data["flypath"].empty:
        sample_path = Path(data["flypath"].iloc[0])
        arena = sample_path.parent.name
        corridor = sample_path.parent.parent.name
        arena_corridor_info = f"_{arena}_{corridor}"

    # Create title with available metadata
    title_elements = [f"Date: {date}"]

    # Add other interesting metadata if available
    for column in ["experiment", "Period", "FeedingState", "Orientation", "Light"]:
        if column in data.columns:
            unique_values = data[column].unique()
            if len(unique_values) == 1:  # Only add if all rows have the same value
                title_elements.append(f"{column}: {unique_values[0]}")

    title = " | ".join(title_elements)
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


def validate_f1_entry(group_data: pd.DataFrame, flypath: str) -> bool:
    """
    Validate that the fly actually entered the F1 corridor.
    Only performs validation if CONFIG["F1_experiments"] is True.

    Args:
        group_data: DataFrame with F1 coordinates data for the group
        flypath: Path to the fly's data directory

    Returns:
        bool: True if fly has positive adjusted_time (entered F1) or F1 check is disabled, False otherwise
    """
    # Skip F1 validation if not an F1 experiment
    if not CONFIG.get("F1_experiments", False):
        return True

    # Filter data for this specific flypath
    fly_data = group_data[group_data["flypath"] == flypath]

    if fly_data.empty:
        logger.warning(f"No data found for flypath: {flypath}")
        return False

    # Check if there are any positive adjusted_time values
    if "adjusted_time" not in fly_data.columns:
        logger.warning(f"No adjusted_time column for {flypath}")
        return False

    has_positive_time = (fly_data["adjusted_time"] >= 0).any()

    if not has_positive_time:
        logger.warning(f"Excluding fly that never entered F1 corridor: {flypath}")
        return False

    return True


def extract_trial_clips(video_path: Path, trial_data: pd.DataFrame) -> List[Path]:
    """Extract temporary clips for each trial using MoviePy"""
    from moviepy.editor import VideoFileClip

    clips = []
    video = VideoFileClip(str(video_path))

    for _, trial in trial_data.iterrows():
        start = max(0, trial["start_time"] - CONFIG["clip_buffer"])
        end = min(video.duration, trial["end_time"] + CONFIG["clip_buffer"])

        clip = video.subclip(start, end)
        temp_path = (
            Path(CONFIG["temp_clip_dir"])
            / f"{video_path.stem}_trial_{trial['trial_id']}.mp4"
        )
        clip.write_videofile(str(temp_path), logger=None)
        clips.append(temp_path)

    video.close()
    return clips


def main(test=False, full_screen=False):

    try:
        transformed_data = load_dataset(DATA_PATH)
        if transformed_data.empty:
            logger.error("Empty dataset loaded")
            return

        ensure_output_directory_exists(CONFIG["output_dir"])

        # Grouping logic based on mode
        if CONFIG["trial_mode"]:
            groups = transformed_data.groupby([CONFIG["trial_column"], groupby])
            logger.info(f"Processing {len(groups)} trial-group combinations")
        else:
            groups = transformed_data.groupby(groupby)
            logger.info(f"Processing {len(groups)} experimental groups")

        for group_key, group_data in groups:
            try:
                # Handle trial-mode metadata
                if CONFIG["trial_mode"]:
                    trial_id, nickname = group_key
                    output_filename = f"{nickname}_trial_{trial_id}_grid.mp4"
                    trial_times = group_data[
                        ["start_time", "end_time"]
                    ].drop_duplicates()
                else:
                    nickname = group_key
                    output_filename = f"{nickname}_grid.mp4"

                # Skip existing outputs
                output_path = Path(CONFIG["output_dir"]) / sanitize_filename(
                    output_filename
                )
                if output_path.exists():
                    logger.info(f"Skipping existing: {output_filename}")
                    continue

                # Get and validate video paths
                flypaths = group_data["flypath"].unique()

                invalid_paths = [fp for fp in flypaths if not Path(fp).exists()]
                if invalid_paths:
                    logger.error(f"Missing directories:\n{invalid_paths}")
                    continue

                video_paths = []
                for fp in flypaths:
                    path = Path(fp)
                    if (
                        path.is_file() and path.suffix == ".mp4"
                    ):  # Handle direct file paths
                        video_paths.append(path)
                    else:
                        matches = list(path.glob("*.mp4"))
                        if matches:
                            video_paths.append(sorted(matches)[0])  # Take latest file
                        else:
                            logger.warning(f"No videos found in {fp}")

                # Trial-specific processing
                if CONFIG["trial_mode"]:
                    # Extract trial clips in parallel
                    with ThreadPoolExecutor() as executor:
                        clip_paths = list(
                            executor.map(
                                lambda p: extract_trial_clips(Path(p), trial_times),
                                flypaths,
                            )
                        )
                    video_paths = [clip for sublist in clip_paths for clip in sublist]

                video_entries = load_videos(video_paths)
                valid_videos = [
                    v for v, path in video_entries if validate_video_duration(v, path)
                ]

                if not valid_videos:
                    logger.warning(f"No valid videos for group: {group_key}")
                    continue

                # Generate identifiers and metadata
                identifiers = generate_identifiers(
                    group_data, [p for _, p in video_entries]
                )
                # Main title is just the groupby value (nickname)
                title = nickname + (
                    f" | Trial {trial_id}" if CONFIG["trial_mode"] else ""
                )

                # Create video grid
                video_grid = create_video_grid_with_features(
                    valid_videos,
                    identifiers=identifiers,
                    grid_text=title,
                    output_filename=output_filename,
                    test=test,
                )

                if video_grid:
                    logger.info(f"Successfully created: {output_filename}")

            except Exception as e:
                logger.error(
                    f"Failed processing group {group_key}: {e}\n{traceback.format_exc()}"
                )
                continue
            finally:
                # Cleanup trial clips
                if CONFIG["trial_mode"] and "clip_paths" in locals():
                    for clip in clip_paths:
                        for c in clip:
                            c.unlink(missing_ok=True)
    except Exception as e:
        logger.error(f"Fatal error in main: {e}\n{traceback.format_exc()}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Enable test mode (10sec)")
    parser.add_argument(
        "--test-start",
        type=float,
        default=None,
        help="Start time in seconds for test mode (e.g., 5000 for 5000-5010s)",
    )
    parser.add_argument(
        "--full-screen", action="store_true", help="Process all experimental conditions"
    )
    parser.add_argument(
        "--trials", action="store_true", help="Enable trial-based clipping"
    )
    parser.add_argument(
        "--filter-values",
        type=str,
        default=None,
        help="Comma-separated list of values to process (e.g. Nicknames)",
    )
    parser.add_argument(
        "--force-max",
        action="store_true",
        help="Force using all videos in a group, even if it requires reducing cell size or relaxing aspect ratio.",
    )
    parser.add_argument(
        "--missing-only",
        action="store_true",
        help="Process only the missing videos identified by the check_rename_videos.py script.",
    )
    args = parser.parse_args()

    # Override config with command-line argument
    if args.test_start is not None:
        CONFIG["test_start_time"] = args.test_start

    # Parse filter values if provided
    filter_values = None
    if args.filter_values:
        filter_values = [v.strip() for v in args.filter_values.split(",") if v.strip()]

    def filtered_groups(groups, filter_values):
        if not filter_values:
            return groups
        for group_key, group_data in groups:
            # group_key can be a tuple if grouped by multiple columns
            if isinstance(group_key, tuple):
                key_val = group_key[0] if len(group_key) == 1 else group_key
            else:
                key_val = group_key
            if key_val in filter_values or (
                isinstance(key_val, tuple) and any(k in filter_values for k in key_val)
            ):
                yield (group_key, group_data)

    def main_with_filter(
        test=False, full_screen=False, force_max=False, missing_only=False
    ):
        try:
            transformed_data = load_dataset(DATA_PATH)
            if transformed_data.empty:
                logger.error("Empty dataset loaded")
                return

            ensure_output_directory_exists(CONFIG["output_dir"])

            # Load nickname mapping for simplified names
            nickname_mapping = create_nickname_mapping()

            # Load missing videos list if processing missing only
            missing_identifiers = []
            if missing_only:
                missing_identifiers = load_missing_videos_list()
                if not missing_identifiers:
                    logger.info("No missing videos to process")
                    return
                logger.info(f"Processing only missing videos: {missing_identifiers}")

            # Grouping logic based on mode
            if CONFIG["trial_mode"]:
                groups = transformed_data.groupby([CONFIG["trial_column"], groupby])
                logger.info(f"Processing {len(groups)} trial-group combinations")
            else:
                groups = transformed_data.groupby(groupby)
                logger.info(f"Processing {len(groups)} experimental groups")

            # Filter groups based on different criteria
            if missing_only:
                # Filter for missing identifiers (check both Nickname and Genotype)
                def is_missing_group(group_key, group_data):
                    if isinstance(group_key, tuple):
                        identifier = group_key[-1]  # Last element is the groupby key
                    else:
                        identifier = group_key

                    # Check if this identifier or any genotype in the group is missing
                    if identifier in missing_identifiers:
                        return True

                    # Also check genotypes in the group data
                    if "Genotype" in group_data.columns:
                        genotypes = group_data["Genotype"].unique()
                        if any(
                            genotype in missing_identifiers for genotype in genotypes
                        ):
                            return True

                    return False

                group_iter = [
                    (key, data) for key, data in groups if is_missing_group(key, data)
                ]
                logger.info(f"Found {len(group_iter)} missing groups to process")
            elif filter_values:
                group_iter = filtered_groups(groups, filter_values)
            else:
                group_iter = groups

            for group_key, group_data in group_iter:
                try:
                    # Handle trial-mode metadata
                    if CONFIG["trial_mode"]:
                        trial_id, nickname = group_key
                        # Use simplified nickname if available
                        display_name = nickname_mapping.get(nickname, nickname)
                        output_filename = f"{display_name}_trial_{trial_id}_grid.mp4"
                        trial_times = group_data[
                            ["start_time", "end_time"]
                        ].drop_duplicates()
                    else:
                        nickname = group_key
                        # Use simplified nickname if available, otherwise use original
                        display_name = nickname_mapping.get(nickname, nickname)
                        output_filename = f"{display_name}_grid.mp4"

                    # Skip existing outputs
                    output_path = Path(CONFIG["output_dir"]) / sanitize_filename(
                        output_filename
                    )
                    if output_path.exists():
                        logger.info(f"Skipping existing: {output_filename}")
                        continue

                    # Get and validate video paths
                    flypaths = group_data["flypath"].unique()

                    invalid_paths = [fp for fp in flypaths if not Path(fp).exists()]
                    if invalid_paths:
                        logger.error(f"Missing directories:\n{invalid_paths}")
                        continue

                    logger.info(f"Group {group_key}: Found {len(flypaths)} total flies")

                    # Validate F1 entry and collect video paths
                    video_paths = []
                    valid_flypaths = []
                    for fp in flypaths:
                        # First check if fly entered F1 corridor
                        if not validate_f1_entry(group_data, fp):
                            continue

                        path = Path(fp)
                        if (
                            path.is_file() and path.suffix == ".mp4"
                        ):  # Handle direct file paths
                            video_paths.append(path)
                            valid_flypaths.append(fp)
                        else:
                            matches = list(path.glob("*.mp4"))
                            if matches:
                                video_paths.append(
                                    sorted(matches)[0]
                                )  # Take latest file
                                valid_flypaths.append(fp)
                            else:
                                logger.warning(f"No videos found in {fp}")

                    # Check if we have any valid videos after F1 entry validation
                    if not video_paths:
                        logger.warning(
                            f"No valid F1-entering flies for group: {group_key}"
                        )
                        continue

                    logger.info(
                        f"Group {group_key}: {len(valid_flypaths)} flies entered F1 corridor ({len(flypaths) - len(valid_flypaths)} excluded)"
                    )

                    # Trial-specific processing
                    if CONFIG["trial_mode"]:
                        # Extract trial clips in parallel
                        with ThreadPoolExecutor() as executor:
                            clip_paths = list(
                                executor.map(
                                    lambda p: extract_trial_clips(Path(p), trial_times),
                                    flypaths,
                                )
                            )
                        video_paths = [
                            clip for sublist in clip_paths for clip in sublist
                        ]

                    video_entries = load_videos(video_paths)
                    valid_videos = [
                        v
                        for v, path in video_entries
                        if validate_video_duration(v, path)
                    ]

                    if not valid_videos:
                        logger.warning(f"No valid videos for group: {group_key}")
                        continue

                    # Generate identifiers and metadata
                    identifiers = generate_identifiers(
                        group_data, [p for _, p in video_entries]
                    )

                    # Use simplified name in title if available
                    title_name = (
                        display_name if "display_name" in locals() else nickname
                    )
                    # Main title is just the groupby value (display_name)
                    title = title_name + (
                        f" | Trial {trial_id}" if CONFIG["trial_mode"] else ""
                    )

                    # Create video grid
                    video_grid = create_video_grid_with_features(
                        valid_videos,
                        identifiers=identifiers,
                        grid_text=title,
                        output_filename=output_filename,
                        test=test,
                        force_max=force_max,
                    )

                    if video_grid:
                        logger.info(f"Successfully created: {output_filename}")

                except Exception as e:
                    logger.error(
                        f"Failed processing group {group_key}: {e}\n{traceback.format_exc()}"
                    )
                    continue
                finally:
                    # Cleanup trial clips
                    if CONFIG["trial_mode"] and "clip_paths" in locals():
                        for clip in clip_paths:
                            for c in clip:
                                c.unlink(missing_ok=True)
        except Exception as e:
            logger.error(f"Fatal error in main: {e}\n{traceback.format_exc()}")

    main_with_filter(
        test=args.test,
        full_screen=args.full_screen,
        force_max=args.force_max,
        missing_only=args.missing_only,
    )
