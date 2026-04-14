#!/usr/bin/env python3
"""
Script to create a grid video with skeleton tracking annotations.
Filters flies based on metadata and combines them into a single compressed video.
"""

from pathlib import Path
import json
import cv2
import numpy as np
from typing import List, Optional, Tuple
from tqdm import tqdm
import logging
import subprocess
import argparse
import sys

# Try to import Sleap_utils from utils_behavior package, or directly from same directory
try:
    from utils_behavior.Sleap_utils import Sleap_Tracks
except ImportError:
    # If utils_behavior is not installed, try importing from same directory
    sys.path.insert(0, str(Path(__file__).parent))
    from Sleap_utils import Sleap_Tracks

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
    "rotate_videos": False,  # Rotate videos 90 degrees clockwise
    "rotation_angle": cv2.ROTATE_90_CLOCKWISE,
    "output_dir": "/mnt/upramdya_data/MD/Example_videos",
    "max_grid_width": 3840,  # 4K resolution
    "max_grid_height": 2160,
    "min_cell_width": 200,
    "min_cell_height": 180,
    "aspect_ratio_tolerance": 0.2,
    "force_single_row": True,  # Force all videos in a single horizontal row
    # Compression settings - optimized for paper/sharing
    "reduce_framerate": False,  # Keep full framerate (29-30 fps)
    "compression_preset": "fast",
    "crf": 26,  # Higher CRF = smaller file (23-28 recommended)
    "use_ffmpeg_compression": True,
    "use_gpu_encoding": True,  # Use NVIDIA GPU for encoding (5-10x faster)
    "use_gpu_processing": True,  # Use GPU for frame operations (resize, rotate)
}


def check_gpu_support() -> bool:
    """Check if CUDA GPU is available for OpenCV."""
    try:
        return cv2.cuda.getCudaEnabledDeviceCount() > 0
    except (cv2.error, AttributeError):
        return False


def check_nvenc_support() -> bool:
    """Check if NVIDIA NVENC hardware encoding is available."""
    try:
        cmd = ["ffmpeg", "-hide_banner", "-encoders"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return "h264_nvenc" in result.stdout
    except Exception:
        return False


def ensure_output_directory_exists(output_dir: str):
    """Create output directory if it doesn't exist."""
    output_path = Path(output_dir)
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Directory created: {output_dir}")
    else:
        logger.info(f"Directory already exists: {output_dir}")


def load_metadata(base_folder: Path) -> dict:
    """Load metadata.json from the base folder."""
    metadata_path = base_folder / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {base_folder}")

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    logger.info(f"Loaded metadata from {metadata_path}")
    return metadata


def filter_arenas_by_feeding_state(metadata: dict, feeding_state: str) -> List[str]:
    """Filter arenas based on FeedingState value."""
    variables = metadata["Variable"]
    feeding_state_idx = variables.index("FeedingState")

    filtered_arenas = []
    for key, values in metadata.items():
        if key.startswith("Arena") and values[feeding_state_idx] == feeding_state:
            filtered_arenas.append(key.lower())  # e.g., "Arena1" -> "arena1"

    logger.info(
        f"Found {len(filtered_arenas)} arenas with FeedingState={feeding_state}"
    )
    return filtered_arenas


def collect_corridor_paths(
    base_folder: Path, arenas: List[str], max_corridors: int = 18
) -> List[Tuple[Path, Path, str]]:
    """
    Collect corridor video and tracking file paths.

    Returns:
        List of tuples: (video_path, tracking_h5_path, identifier)
    """
    corridor_data = []

    for arena in arenas:
        arena_path = base_folder / arena
        if not arena_path.exists():
            logger.warning(f"Arena path does not exist: {arena_path}")
            continue

        # Find all corridor directories
        corridor_dirs = sorted(
            [
                d
                for d in arena_path.iterdir()
                if d.is_dir() and d.name.startswith("corridor")
            ]
        )

        for corridor_dir in corridor_dirs:
            if len(corridor_data) >= max_corridors:
                break

            # Find video file - prefer original (non-preprocessed) video
            video_files = list(corridor_dir.glob("*.mp4"))
            video_file = None
            for vf in video_files:
                if "preprocessed" not in vf.name:
                    video_file = vf
                    break
            if video_file is None and video_files:
                video_file = video_files[0]

            if video_file is None:
                logger.warning(f"No video file found in {corridor_dir}")
                continue

            # Find full body tracking file
            tracking_files = list(corridor_dir.glob("*full_body.h5"))
            if not tracking_files:
                logger.warning(f"No full body tracking file found in {corridor_dir}")
                continue

            tracking_file = tracking_files[0]

            # Create identifier
            identifier = f"{arena}/{corridor_dir.name}"

            corridor_data.append((video_file, tracking_file, identifier))
            logger.info(f"Added corridor: {identifier}")

        if len(corridor_data) >= max_corridors:
            break

    logger.info(f"Collected {len(corridor_data)} corridors")
    return corridor_data


def reverse_preprocessing_transform(
    x: float,
    y: float,
    original_width: int,
    original_height: int,
    template_width: int = 96,
    template_height: int = 516,
    mask_padding: int = 20,
    crop_top: int = 74,
    crop_bottom: int = 0,
) -> tuple:
    """
    Reverse the preprocessing transformation applied to coordinates.
    Transforms coordinates from preprocessed video space back to original video space.

    Args:
        x, y: Coordinates in preprocessed video space
        original_width, original_height: Dimensions of the original video
        template_width, template_height: Target dimensions after preprocessing
        mask_padding: Padding added to left/right during preprocessing
        crop_top, crop_bottom: Pixels cropped from top/bottom during preprocessing

    Returns:
        (x_original, y_original): Coordinates in original video space
    """
    # Step 1: Remove padding
    x_unpadded = x - mask_padding
    y_unpadded = y

    # Step 2: Reverse cropping (add back the cropped top portion)
    y_uncropped = y_unpadded + crop_top
    x_uncropped = x_unpadded

    # Step 3: Reverse resizing
    x_original = x_uncropped * (original_width / template_width)
    y_original = y_uncropped * (original_height / template_height)

    return x_original, y_original


def binarise_frame(frame: np.ndarray) -> np.ndarray:
    """Detect the corridors in a frame using threshold and morphology.

    Args:
        frame: Input frame (BGR or grayscale)

    Returns:
        Binary mask of the corridor
    """
    if len(frame.shape) == 3:  # Convert to grayscale if needed
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame.copy()

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((60, 20), np.uint8)  # Smaller kernel to avoid losing details
    closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    return closing


def create_corridor_mask(
    binary_frame: np.ndarray, dilation_iterations: int = 1
) -> np.ndarray:
    """Create a mask from the binarized image with optional dilation.

    Args:
        binary_frame: Binary frame from binarise_frame
        dilation_iterations: Number of dilation iterations

    Returns:
        Dilated mask
    """
    if binary_frame.dtype != np.uint8:
        binary_frame = binary_frame.astype(np.uint8)

    # Dilate the binary frame to fill small gaps
    kernel = np.ones((5, 5), np.uint8)
    dilated_mask = cv2.dilate(binary_frame, kernel, iterations=dilation_iterations)

    return dilated_mask


def detect_corridor_bounds(frame: np.ndarray) -> tuple:
    """Detect the bounding box of the corridor in the frame.

    Args:
        frame: Input frame

    Returns:
        (x, y, w, h): Bounding box of the corridor
    """
    # Create binary mask to detect corridor
    binary = binarise_frame(frame)
    mask = create_corridor_mask(binary, dilation_iterations=2)

    # Find contours to get the corridor bounding box
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        # If no corridor detected, return full frame bounds
        return (0, 0, frame.shape[1], frame.shape[0])

    # Get the largest contour (the corridor)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    return (x, y, w, h)


def apply_corridor_mask(
    frame: np.ndarray, mask: np.ndarray, padding: int = 20, crop_top: int = 20
) -> np.ndarray:
    """Apply the corridor mask to the frame and crop/pad the result.

    Args:
        frame: Input frame
        mask: Binary mask from create_corridor_mask
        padding: Horizontal padding to add on sides
        crop_top: Pixels to crop from top

    Returns:
        Masked, cropped, and padded frame
    """
    # Apply mask
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

    # Crop from top
    if crop_top > 0:
        cropped_frame = masked_frame[crop_top:, :]
    else:
        cropped_frame = masked_frame

    # Add horizontal padding
    if padding > 0:
        padded_frame = cv2.copyMakeBorder(
            cropped_frame, 0, 0, padding, padding, cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )
    else:
        padded_frame = cropped_frame

    return padded_frame


def create_annotated_frame(
    frame: np.ndarray,
    sleap_tracks: Sleap_Tracks,
    frame_idx: int,
    target_width: int,
    target_height: int,
    identifier: str,
    use_gpu: bool,
    crop_offset: tuple = (0, 0),
    original_frame_dims: tuple = None,
) -> Optional[np.ndarray]:
    """
    Create a single annotated frame with skeleton tracking.

    Args:
        frame: Pre-read frame from video capture
        sleap_tracks: Sleap_Tracks object with tracking data
        frame_idx: Frame index to process (1-indexed)
        target_width: Target width for the cell
        target_height: Target height for the cell
        identifier: Text identifier for the corridor
        use_gpu: Whether to use GPU for resize operations
        crop_offset: (x, y) offset from cropping for coordinate adjustment
        original_frame_dims: (width, height) of the original frame before cropping

    Returns:
        Annotated frame as numpy array
    """
    crop_x, crop_y = crop_offset
    # Rotate if configured (always on CPU - it's fast enough)
    if CONFIG["rotate_videos"]:
        frame = cv2.rotate(frame, CONFIG["rotation_angle"])
    # Debug: Log frame info for first call
    logger.debug(
        f"Processing frame {frame_idx} for {identifier}, dimensions: {frame.shape}"
    )

    # Draw skeleton tracking on the frame with color coding
    # Note: Skeleton coordinates are from preprocessed videos, need to transform to original space

    # Get original video dimensions for coordinate transformation
    # Use the dimensions BEFORE cropping if provided
    if original_frame_dims is not None:
        original_width, original_height = original_frame_dims
    else:
        original_height, original_width = frame.shape[:2]
    template_width = 96
    template_height = 516

    # Define exact colors for each limb type (BGR format - converted from RGB)
    # Right legs: Red shades (darkest front -> lightest hind)
    LEG_RIGHT_FRONT = (49, 30, 186)  # RGB (186, 30, 49) -> BGR
    LEG_RIGHT_MIDDLE = (79, 86, 201)  # RGB (201, 86, 79) -> BGR
    LEG_RIGHT_REAR = (121, 133, 213)  # RGB (213, 133, 121) -> BGR
    # Left legs: Blue shades (darkest front -> lightest hind)
    LEG_LEFT_FRONT = (153, 115, 15)  # RGB (15, 115, 153) -> BGR
    LEG_LEFT_MIDDLE = (175, 141, 26)  # RGB (26, 141, 175) -> BGR
    LEG_LEFT_REAR = (203, 190, 117)  # RGB (117, 190, 203) -> BGR
    # Body: Gray
    BODY = (210, 210, 210)

    def get_limb_color(node1: str, node2: str, edge_idx: int) -> tuple:
        """Get color based on edge nodes.
        Node names: Lfront, Rfront, Rmid, Lmid, Rhind, Lhind, Thorax, Head, Abdomen

        Args:
            node1: First node name
            node2: Second node name
            edge_idx: Edge index (for fallback)

        Returns:
            BGR color tuple
        """
        # Check both nodes for leg identification
        nodes = node1.lower() + node2.lower()

        # Right legs
        if "rfront" in nodes:
            return LEG_RIGHT_FRONT
        elif "rmid" in nodes:
            return LEG_RIGHT_MIDDLE
        elif "rhind" in nodes:
            return LEG_RIGHT_REAR
        # Left legs
        elif "lfront" in nodes:
            return LEG_LEFT_FRONT
        elif "lmid" in nodes:
            return LEG_LEFT_MIDDLE
        elif "lhind" in nodes:
            return LEG_LEFT_REAR
        # Body parts (Thorax, Head, Abdomen)
        else:
            return BODY

    # Draw skeleton tracking on the frame
    skeleton_found = False

    for obj in sleap_tracks.objects:
        # Try to get pre-indexed frame data if available
        if hasattr(obj, "_frame_index"):
            row = obj._frame_index.get(frame_idx)
            if row is None:
                continue
        else:
            # Fallback for preview mode - query DataFrame directly
            matching_rows = obj.dataset[obj.dataset["frame"] == frame_idx]
            if matching_rows.empty:
                continue
            row = matching_rows.iloc[0].to_dict()

        skeleton_found = True

        # Draw edges (skeleton connections)
        if hasattr(sleap_tracks, "edge_names") and sleap_tracks.edge_names:
            logger.debug(
                f"Drawing {len(sleap_tracks.edge_names)} edges for frame {frame_idx}"
            )
            for edge_idx, edge in enumerate(sleap_tracks.edge_names):
                node1, node2 = edge

                x1_preprocessed = row.get(f"x_{node1}")
                y1_preprocessed = row.get(f"y_{node1}")
                x2_preprocessed = row.get(f"x_{node2}")
                y2_preprocessed = row.get(f"y_{node2}")

                if (
                    x1_preprocessed is not None
                    and y1_preprocessed is not None
                    and x2_preprocessed is not None
                    and y2_preprocessed is not None
                ):
                    if not (
                        np.isnan(x1_preprocessed)
                        or np.isnan(y1_preprocessed)
                        or np.isnan(x2_preprocessed)
                        or np.isnan(y2_preprocessed)
                    ):
                        # Transform coordinates from preprocessed to original space
                        x1, y1 = reverse_preprocessing_transform(
                            x1_preprocessed,
                            y1_preprocessed,
                            original_width,
                            original_height,
                        )
                        x2, y2 = reverse_preprocessing_transform(
                            x2_preprocessed,
                            y2_preprocessed,
                            original_width,
                            original_height,
                        )

                        # Adjust for crop offset
                        x1 -= crop_x
                        y1 -= crop_y
                        x2 -= crop_x
                        y2 -= crop_y

                        pt1 = (int(x1), int(y1))
                        pt2 = (int(x2), int(y2))

                        # Get color based on node names
                        color = get_limb_color(node1, node2, edge_idx)

                        cv2.line(frame, pt1, pt2, color, 1)

        # Skip drawing nodes - only draw connecting lines

    # Debug: Print if no skeleton data was found for this frame
    if not skeleton_found:
        logger.warning(
            f"No skeleton data found for frame {frame_idx} (identifier: {identifier})"
        )

    # Resize frame to target dimensions with aspect ratio preservation
    h, w = frame.shape[:2]
    aspect = w / h

    if aspect > (target_width / target_height):
        new_w = target_width
        new_h = int(target_width / aspect)
    else:
        new_h = target_height
        new_w = int(target_height * aspect)

    # Resize
    frame = cv2.resize(frame, (new_w, new_h))

    # Create padded frame with center alignment
    padded_frame = np.zeros((target_height, target_width, 3), dtype=np.uint8)

    # Center-align both horizontally and vertically
    y_offset = (target_height - new_h) // 2
    x_offset = (target_width - new_w) // 2

    padded_frame[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = frame

    return padded_frame


def preview_grid_frame(
    corridor_data: List[Tuple[Path, Path, str]],
    start_time: int = 0,
) -> None:
    """
    Display a single preview frame of the grid video.

    Args:
        corridor_data: List of (video_path, tracking_h5_path, identifier) tuples
        start_time: Start time in seconds (default: 0, from beginning)
    """
    if not corridor_data:
        logger.error("No corridor data provided")
        return

    # Check GPU support
    gpu_available = check_gpu_support()

    # Load all videos and tracking data (skip pre-indexing for speed)
    logger.info("Loading videos and tracking data for preview...")
    video_objects = []

    for video_path, tracking_path, identifier in corridor_data:
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.error(f"Failed to open video: {video_path}")
                continue

            sleap_tracks = Sleap_Tracks(str(tracking_path), object_type="fly")
            # Skip pre-indexing in preview mode - we only need one frame

            video_objects.append((cap, sleap_tracks, identifier))
            logger.info(f"Loaded: {identifier}")
        except Exception as e:
            logger.error(f"Error loading {identifier}: {e}")
            continue

    if not video_objects:
        logger.error("No videos loaded successfully")
        return

    # Calculate grid layout
    num_videos = len(video_objects)
    first_cap = video_objects[0][0]
    sample_width = int(first_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    sample_height = int(first_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if CONFIG["rotate_videos"]:
        sample_width, sample_height = sample_height, sample_width

    median_aspect = sample_width / sample_height

    if CONFIG.get("force_single_row", False):
        cols = num_videos
        rows = 1
        available_width = CONFIG["max_grid_width"] - CONFIG["horizontal_padding"] * (
            cols + 1
        )
        available_height = (
            CONFIG["max_grid_height"]
            - CONFIG["top_padding"]
            - CONFIG["bottom_padding"]
            - CONFIG["vertical_padding"] * 2
        )
        cell_width = available_width / cols
        cell_height = available_height
        cell_aspect = cell_width / cell_height
        if (
            abs(cell_aspect - median_aspect) / median_aspect
            > CONFIG["aspect_ratio_tolerance"]
        ):
            cell_height = cell_width / median_aspect
        best_layout = (cols, rows, cell_width, cell_height)
    else:
        best_layout = calculate_optimal_layout(
            num_videos,
            median_aspect,
            CONFIG["max_grid_width"],
            CONFIG["max_grid_height"],
        )

    if not best_layout:
        logger.error("Could not find valid grid layout")
        return

    cols, rows, cell_width, cell_height = best_layout
    cell_width, cell_height = int(cell_width), int(cell_height)

    logger.info(f"Grid layout: {cols}x{rows}, cell size: {cell_width}x{cell_height}")

    # Calculate final grid dimensions
    grid_width = int(cols * cell_width + CONFIG["horizontal_padding"] * (cols + 1))
    grid_height = int(
        rows * cell_height
        + CONFIG["top_padding"]
        + CONFIG["bottom_padding"]
        + CONFIG["vertical_padding"] * (rows + 1)
    )

    # Determine frame to preview
    video_fps = first_cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(start_time * video_fps)

    logger.info(f"Generating preview frame at {start_time}s (frame {start_frame})...")

    # Seek all videos to start_frame
    for cap, _, _ in video_objects:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Detect corridor bounds for cropping alignment
    logger.info("Detecting corridor bounds for alignment...")
    corridor_bounds = []
    for cap, _, identifier in video_objects:
        ret, frame = cap.read()
        if ret:
            bounds = detect_corridor_bounds(frame)
            corridor_bounds.append(bounds)
            logger.debug(f"{identifier}: corridor bounds {bounds}")
        else:
            corridor_bounds.append(
                (0, 0, frame.shape[1] if ret else 0, frame.shape[0] if ret else 0)
            )

    # Seek back to start frame
    for cap, _, _ in video_objects:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Cache GPU support check
    use_gpu = CONFIG.get("use_gpu_processing", False) and gpu_available

    # Create grid frame
    grid_frame = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)

    # Process single frame
    for idx, (cap, sleap_tracks, identifier) in enumerate(video_objects):
        ret, frame = cap.read()
        if not ret:
            logger.warning(f"Could not read frame from {identifier}")
            continue

        # Crop frame to corridor bounds for alignment
        crop_offset = (0, 0)
        original_dims = (
            frame.shape[1],
            frame.shape[0],
        )  # (width, height) before cropping
        if idx < len(corridor_bounds):
            x, y, w, h = corridor_bounds[idx]
            if w > 0 and h > 0:
                frame = frame[y : y + h, x : x + w]
                crop_offset = (x, y)

        # Create annotated frame directly
        # SLEAP uses 1-indexed frames, so add 1 to the frame index
        annotated_frame = create_annotated_frame(
            frame,
            sleap_tracks,
            start_frame + 1,
            cell_width,
            cell_height,
            identifier,
            use_gpu,
            crop_offset,
            original_dims,
        )

        # Calculate position in grid
        row = idx // cols
        col = idx % cols
        y = CONFIG["top_padding"] + row * (cell_height + CONFIG["vertical_padding"])
        x = CONFIG["horizontal_padding"] + col * (
            cell_width + CONFIG["horizontal_padding"]
        )

        # Place in grid
        grid_frame[y : y + cell_height, x : x + cell_width] = annotated_frame

    # Release captures
    for cap, _, _ in video_objects:
        cap.release()

    # Display the frame
    logger.info("Displaying preview frame. Press any key to close...")
    cv2.imshow("Grid Preview - Press any key to close", grid_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    logger.info("Preview complete!")


def create_annotated_grid_video(
    corridor_data: List[Tuple[Path, Path, str]],
    output_filename: str,
    start_time: int = 0,
    duration: int = 120,
) -> None:
    """
    Create a grid video with skeleton tracking annotations.

    Args:
        corridor_data: List of (video_path, tracking_h5_path, identifier) tuples
        output_filename: Name of the output video file
        start_time: Start time in seconds (default: 0, from beginning)
        duration: Duration in seconds (default: 120, 2 minutes)
    """
    if not corridor_data:
        logger.error("No corridor data provided")
        return

    ensure_output_directory_exists(CONFIG["output_dir"])

    # Check GPU support
    gpu_available = check_gpu_support()
    gpu_encoding_available = check_nvenc_support()

    if CONFIG.get("use_gpu_encoding") and gpu_encoding_available:
        logger.info("NVIDIA NVENC GPU encoding enabled")
    elif CONFIG.get("use_gpu_encoding"):
        logger.warning("GPU encoding requested but NVENC not available, using CPU")
        CONFIG["use_gpu_encoding"] = False

    # Load all videos and tracking data
    logger.info("Loading videos and tracking data...")
    video_objects = []

    for video_path, tracking_path, identifier in corridor_data:
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.error(f"Failed to open video: {video_path}")
                continue

            sleap_tracks = Sleap_Tracks(str(tracking_path), object_type="fly")

            # Pre-index SLEAP data for O(1) frame lookups
            logger.info(f"Pre-indexing SLEAP data for {identifier}...")
            for obj in sleap_tracks.objects:
                # Build frame index: frame_number -> row dict
                obj._frame_index = {
                    int(row["frame"]): row.to_dict()
                    for _, row in obj.dataset.iterrows()
                }

            video_objects.append((cap, sleap_tracks, identifier))
            logger.info(f"Loaded: {identifier}")
        except Exception as e:
            logger.error(f"Error loading {identifier}: {e}")
            continue

    if not video_objects:
        logger.error("No videos loaded successfully")
        return

    # Calculate grid layout
    num_videos = len(video_objects)

    # Get video dimensions (after rotation)
    first_cap = video_objects[0][0]
    sample_width = int(first_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    sample_height = int(first_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if CONFIG["rotate_videos"]:
        sample_width, sample_height = sample_height, sample_width

    median_aspect = sample_width / sample_height

    # Calculate optimal grid layout
    if CONFIG.get("force_single_row", False):
        # Force single row layout
        cols = num_videos
        rows = 1

        # Calculate cell dimensions for single row
        available_width = CONFIG["max_grid_width"] - CONFIG["horizontal_padding"] * (
            cols + 1
        )
        available_height = (
            CONFIG["max_grid_height"]
            - CONFIG["top_padding"]
            - CONFIG["bottom_padding"]
            - CONFIG["vertical_padding"] * 2
        )

        cell_width = available_width / cols
        cell_height = available_height

        # Adjust to maintain aspect ratio if needed
        cell_aspect = cell_width / cell_height
        if (
            abs(cell_aspect - median_aspect) / median_aspect
            > CONFIG["aspect_ratio_tolerance"]
        ):
            # Adjust cell height to match aspect ratio
            cell_height = cell_width / median_aspect

        best_layout = (cols, rows, cell_width, cell_height)
        logger.info(f"Forced single row layout: {cols} columns")
    else:
        best_layout = calculate_optimal_layout(
            num_videos,
            median_aspect,
            CONFIG["max_grid_width"],
            CONFIG["max_grid_height"],
        )

    if not best_layout:
        logger.error("Could not find valid grid layout")
        return

    cols, rows, cell_width, cell_height = best_layout
    cell_width, cell_height = int(cell_width), int(cell_height)

    logger.info(f"Grid layout: {cols}x{rows}, cell size: {cell_width}x{cell_height}")

    # Calculate final grid dimensions
    grid_width = int(cols * cell_width + CONFIG["horizontal_padding"] * (cols + 1))
    grid_height = int(
        rows * cell_height
        + CONFIG["top_padding"]
        + CONFIG["bottom_padding"]
        + CONFIG["vertical_padding"] * (rows + 1)
    )

    # Initialize video writer
    fps = first_cap.get(cv2.CAP_PROP_FPS)

    if CONFIG.get("reduce_framerate", False):
        fps = fps / 2
        frame_skip = 2
    else:
        frame_skip = 1

    output_path = Path(CONFIG["output_dir"]) / output_filename

    # Use temporary file if ffmpeg compression is enabled
    if CONFIG.get("use_ffmpeg_compression", True):
        write_path = str(output_path.with_suffix(".temp.mp4"))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    else:
        write_path = str(output_path)
        fourcc = cv2.VideoWriter_fourcc(*"H264")

    out = cv2.VideoWriter(write_path, fourcc, fps, (grid_width, grid_height))

    # Determine frame range based on start_time and duration
    video_fps = first_cap.get(
        cv2.CAP_PROP_FPS
    )  # Original fps before potential reduction
    max_frames = min(
        int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap, _, _ in video_objects
    )

    start_frame = int(start_time * video_fps)
    end_frame = int((start_time + duration) * video_fps)
    end_frame = min(end_frame, max_frames)

    total_frames = end_frame - start_frame

    logger.info(
        f"Processing frames {start_frame} to {end_frame} ({duration}s) at {fps} fps..."
    )

    # Seek all videos to start_frame ONCE (not on every frame)
    for cap, _, _ in video_objects:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Detect corridor bounds for cropping alignment
    logger.info("Detecting corridor bounds for alignment...")
    corridor_bounds = []
    for cap, _, identifier in video_objects:
        ret, frame = cap.read()
        if ret:
            bounds = detect_corridor_bounds(frame)
            corridor_bounds.append(bounds)
            logger.debug(f"{identifier}: corridor bounds {bounds}")
        else:
            corridor_bounds.append(None)

    # Seek back to start frame
    for cap, _, _ in video_objects:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Cache GPU support check (don't check on every frame)
    use_gpu = CONFIG.get("use_gpu_processing", False) and gpu_available

    # Allocate grid frame ONCE, reuse it
    grid_frame = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)

    # Process frames
    with tqdm(total=total_frames // frame_skip, desc="Creating grid video") as pbar:
        for frame_offset in range(0, total_frames, frame_skip):
            frame_idx = start_frame + frame_offset

            # Reset grid frame (faster than reallocating)
            grid_frame.fill(0)

            # Generate and place annotated frames (sequential reading, no seeking!)
            for idx, (cap, sleap_tracks, identifier) in enumerate(video_objects):
                # Read frame sequentially
                ret, raw_frame = cap.read()
                if not ret:
                    continue

                # Skip frames if reduce_framerate is enabled
                if frame_skip > 1 and frame_offset % frame_skip != 0:
                    continue

                # Crop frame to corridor bounds for alignment
                crop_offset = (0, 0)
                original_dims = (
                    raw_frame.shape[1],
                    raw_frame.shape[0],
                )  # (width, height) before cropping
                if idx < len(corridor_bounds) and corridor_bounds[idx] is not None:
                    x, y, w, h = corridor_bounds[idx]
                    if w > 0 and h > 0:
                        raw_frame = raw_frame[y : y + h, x : x + w]
                        crop_offset = (x, y)

                annotated_frame = create_annotated_frame(
                    raw_frame,
                    sleap_tracks,
                    frame_idx + 1,  # 1-indexed for SLEAP data
                    cell_width,
                    cell_height,
                    identifier,
                    use_gpu,
                    crop_offset,
                    original_dims,
                )

                if annotated_frame is None:
                    continue

                # Calculate position in grid
                row = idx // cols
                col = idx % cols

                x_pos = CONFIG["horizontal_padding"] + col * (
                    cell_width + CONFIG["horizontal_padding"]
                )
                y_pos = (
                    CONFIG["top_padding"]
                    + CONFIG["vertical_padding"]
                    + row * (cell_height + CONFIG["vertical_padding"])
                )

                # Place frame in grid
                grid_frame[y_pos : y_pos + cell_height, x_pos : x_pos + cell_width] = (
                    annotated_frame
                )

            out.write(grid_frame)
            pbar.update(1)

    out.release()

    # Release all video captures
    for cap, _, _ in video_objects:
        cap.release()

    # Compress with ffmpeg if enabled
    if CONFIG.get("use_ffmpeg_compression", True):
        logger.info("Compressing video with ffmpeg...")

        # Use GPU encoding if available
        if CONFIG.get("use_gpu_encoding") and check_nvenc_support():
            ffmpeg_cmd = [
                "ffmpeg",
                "-i",
                write_path,
                "-c:v",
                "h264_nvenc",
                "-preset",
                "p4",  # NVENC preset (p1-p7, p4 is balanced)
                "-cq",
                str(CONFIG["crf"]),  # CQ mode for NVENC
                "-y",
                str(output_path),
            ]
            logger.info("Using NVIDIA NVENC GPU encoding")
        else:
            ffmpeg_cmd = [
                "ffmpeg",
                "-i",
                write_path,
                "-c:v",
                "libx264",
                "-preset",
                CONFIG["compression_preset"],
                "-crf",
                str(CONFIG["crf"]),
                "-y",
                str(output_path),
            ]
            logger.info("Using CPU encoding")

        try:
            subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
            Path(write_path).unlink()  # Delete temporary file
            logger.info(f"Video saved to: {output_path}")
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg compression failed: {e}")
            logger.info(f"Temporary file saved as: {write_path}")
    else:
        logger.info(f"Video saved to: {output_path}")


def calculate_optimal_layout(num_videos, median_aspect, max_width, max_height):
    """Calculate the optimal grid layout."""
    best_layout = None
    best_utilization = 0
    max_cols = min(num_videos, int(max_width / CONFIG["min_cell_width"]))

    for cols in range(1, max_cols + 1):
        rows = (num_videos + cols - 1) // cols

        # Calculate cell dimensions
        available_width = max_width - CONFIG["horizontal_padding"] * (cols + 1)
        available_height = (
            max_height
            - CONFIG["top_padding"]
            - CONFIG["bottom_padding"]
            - CONFIG["vertical_padding"] * (rows + 1)
        )

        cell_width = available_width / cols
        cell_height = available_height / rows

        # Check minimum size constraints
        if (
            cell_width < CONFIG["min_cell_width"]
            or cell_height < CONFIG["min_cell_height"]
        ):
            continue

        # Check aspect ratio
        cell_aspect = cell_width / cell_height
        aspect_diff = abs(cell_aspect - median_aspect) / median_aspect

        if aspect_diff > CONFIG["aspect_ratio_tolerance"]:
            continue

        # Calculate utilization
        utilization = (cell_width * cell_height * num_videos) / (max_width * max_height)

        if utilization > best_utilization:
            best_utilization = utilization
            best_layout = (cols, rows, cell_width, cell_height)

    return best_layout


def main():
    parser = argparse.ArgumentParser(
        description="Create a grid video with skeleton tracking annotations"
    )
    parser.add_argument(
        "--base-folder",
        type=str,
        required=True,
        help="Base folder containing metadata.json and arena directories",
    )
    parser.add_argument(
        "--feeding-state",
        type=str,
        default="starved_noWater",
        help="FeedingState value to filter for (default: starved_noWater)",
    )
    parser.add_argument(
        "--num-corridors",
        type=int,
        default=18,
        help="Number of corridors to include in the grid (default: 18)",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Output filename (default: auto-generated)",
    )
    parser.add_argument(
        "--start-time",
        type=int,
        default=0,
        help="Start time in seconds (default: 0, from beginning)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=120,
        help="Duration in seconds (default: 120, 2 minutes)",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=None,
        help="Override CRF value for compression (23-28, higher=smaller file)",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU acceleration (both processing and encoding)",
    )
    parser.add_argument(
        "--no-gpu-encoding",
        action="store_true",
        help="Disable GPU encoding (use CPU encoding)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: display a single preview frame instead of creating the full video",
    )

    args = parser.parse_args()

    # Override config with command-line arguments
    if args.crf is not None:
        CONFIG["crf"] = args.crf

    if args.no_gpu:
        CONFIG["use_gpu_processing"] = False
        CONFIG["use_gpu_encoding"] = False
        logger.info("GPU acceleration disabled by user")

    if args.no_gpu_encoding:
        CONFIG["use_gpu_encoding"] = False
        logger.info("GPU encoding disabled by user")

    # Load metadata and filter arenas
    base_folder = Path(args.base_folder)
    metadata = load_metadata(base_folder)
    filtered_arenas = filter_arenas_by_feeding_state(metadata, args.feeding_state)

    if not filtered_arenas:
        logger.error(f"No arenas found with FeedingState={args.feeding_state}")
        return

    # Collect corridor paths
    corridor_data = collect_corridor_paths(
        base_folder, filtered_arenas, args.num_corridors
    )

    if not corridor_data:
        logger.error("No valid corridors found")
        return

    # Generate output filename
    if args.output_name:
        output_filename = args.output_name
    else:
        # Extract genotype from metadata if available
        genotype = "unknown"
        if metadata and filtered_arenas:
            variables = metadata["Variable"]
            genotype_idx = (
                variables.index("Genotype") if "Genotype" in variables else None
            )
            if genotype_idx is not None:
                first_arena_key = (
                    f"Arena{filtered_arenas[0][5:]}"  # Convert arena1 -> Arena1
                )
                genotype = (
                    metadata.get(first_arena_key, [None])[genotype_idx] or "unknown"
                )

        output_filename = f"{genotype}_lowRes_annotatedGrid_{args.feeding_state}_{len(corridor_data)}flies_{args.duration}s.mp4"

    # Test mode or full video creation
    if args.test:
        logger.info("Running in test/preview mode...")
        preview_grid_frame(corridor_data, args.start_time)
    else:
        # Create the grid video
        create_annotated_grid_video(
            corridor_data, output_filename, args.start_time, args.duration
        )

    logger.info("Done!")


if __name__ == "__main__":
    main()
