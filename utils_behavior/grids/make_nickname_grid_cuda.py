import logging
from pathlib import Path
import pandas as pd
from typing import List, Optional
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import os
import subprocess
from tqdm import tqdm
import re
import math
import traceback

import argparse
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Sanitisation helpers (Dataverse-safe ASCII filenames) ---
_GREEK_MAP: dict[str, str] = {
    "α": "alpha",
    "β": "beta",
    "γ": "gamma",
    "δ": "delta",
    "ε": "epsilon",
    "ζ": "zeta",
    "η": "eta",
    "θ": "theta",
    "ι": "iota",
    "κ": "kappa",
    "λ": "lambda",
    "μ": "mu",
    "ν": "nu",
    "ξ": "xi",
    "ο": "omicron",
    "π": "pi",
    "ρ": "rho",
    "σ": "sigma",
    "τ": "tau",
    "υ": "upsilon",
    "φ": "phi",
    "χ": "chi",
    "ψ": "psi",
    "ω": "omega",
    "\u2032": "p",  # ′ prime → p
    "\u2019": "",  # right single quote → removed
    "'": "p",  # ASCII apostrophe as prime → p
}


def sanitize_for_dataverse(name: str) -> str:
    """Sanitize a name for Dataverse upload: Greek → ASCII, no spaces/parens/non-ASCII."""
    for ch, rep in _GREEK_MAP.items():
        name = name.replace(ch, rep)
    name = name.replace(" ", "-")
    name = re.sub(r"[()]", "", name)
    name = re.sub(r"[^\x00-\x7F]", "", name)
    return name


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
    "output_dir": "/mnt/upramdya_data/MD/F1_Tracks/TNT_F1_Grids_Light",
    "max_grid_width": 3840,  # 4K resolution - fits more cells in grid
    "max_grid_height": 2160,  # 4K resolution - fits more cells in grid
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
    # Performance/Quality settings - DEFAULT: 4k_high_quality preset
    "reduce_framerate": (
        False
    ),  # Set to True to halve the framerate (e.g., 30fps -> 15fps) - saves 50% space but reduces smoothness
    "compression_preset": (
        "medium"
    ),  # Options: "ultrafast", "fast", "medium", "slow" - 'medium' gives better compression
    "crf": (
        23
    ),  # Constant Rate Factor (18-28): 23=good quality (default). 20=better, 28=pixelated, 18=best
    "use_ffmpeg_compression": (
        True
    ),  # Use ffmpeg for proper compression (recommended, much smaller files)
    "use_gpu_encoding": (
        False
    ),  # Use NVIDIA NVENC hardware acceleration for ffmpeg compression (5-10x faster)
}

MIN_DURATION = 60  # 1 minute

# --- Experiment profiles ---
# Each profile sets data source, groupby column, output directory,
# and the grid layout parameters that differ between experiment types.
EXPERIMENT_PROFILES: dict[str, dict] = {
    "tnt_screen": {
        # Data
        "data_path": (
            "/mnt/upramdya_data/MD/Ballpushing_TNTScreen/Datasets/250414_summary_TNT_screen_Data/summary/pooled_summary.feather"
        ),
        "mapping_csv_path": "/mnt/upramdya_data/MD/Region_map_260506.csv",
        "groupby": "Nickname",
        # Output
        "output_dir": "/mnt/upramdya_data/MD/TNT_Screen_RawGrids",
        # Layout — 1080p single-row for regular multi-maze corridors
        "max_grid_width": 1920,
        "max_grid_height": 1080,
        "force_single_row": True,
        "rotate_videos": False,
        "F1_experiments": False,
    },
    "F1": {
        # Data
        "data_path": (
            "/mnt/upramdya_data/MD/F1_Tracks/Datasets/260123_16_F1_coordinates_F1_TNT_Full_Data/F1_coordinates/pooled_F1_coordinates.feather"
        ),
        "mapping_csv_path": "/mnt/upramdya_data/MD/Region_map_260506.csv",
        "groupby": ["Genotype", "Pretraining"],
        # Output
        "output_dir": "/mnt/upramdya_data/MD/F1_Tracks/TNT_F1_Grids_Light",
        # Layout — 4K multi-row for F1 corridors
        "max_grid_width": 3840,
        "max_grid_height": 2160,
        "force_single_row": False,
        "rotate_videos": False,
        "F1_experiments": True,
    },
    "ballpushing_learning": {
        # Data — PR flies learning-trials dataset; all dates → single grid
        "data_path": (
            "/mnt/upramdya_data/MD/BallPushing_Learning/Datasets/250318_Datasets/250320_Annotated_data.feather"
        ),
        "mapping_csv_path": "/mnt/upramdya_data/MD/Region_map_260506.csv",
        "groupby": None,  # None = entire dataset as one group
        "group_name": "LearningTrials",  # output filename stem
        # Output
        "output_dir": "/mnt/upramdya_data/MD/BallPushing_Learning/Grids",
        # Layout — 4K multi-row, rotated corridors (ball-pushing setup)
        "max_grid_width": 3840,
        "max_grid_height": 2160,
        "force_single_row": False,
        "rotate_videos": True,
        "F1_experiments": False,
    },
}


def check_nvenc_support() -> bool:
    """Check if NVIDIA NVENC hardware encoding is available."""
    try:
        cmd = ["ffmpeg", "-hide_banner", "-encoders"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return "h264_nvenc" in result.stdout
    except Exception:
        return False


def get_video_duration(video_path: Path) -> float:
    """Get video duration in seconds using ffprobe."""
    info = get_video_info_ffprobe(video_path)
    return info["duration"] if info else 0.0


def get_video_info_ffprobe(video_path: Path) -> Optional[dict]:
    """Return width, height, fps, duration for a video file via ffprobe."""
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "format=duration:stream=width,height,r_frame_rate",
            "-of",
            "json",
            str(video_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        stream = data.get("streams", [{}])[0]
        fmt = data.get("format", {})
        fps_str = stream.get("r_frame_rate", "30/1")
        num, den = map(int, fps_str.split("/")) if "/" in fps_str else (30, 1)
        fps = num / den if den else 30.0
        return {
            "width": int(stream.get("width", 0)),
            "height": int(stream.get("height", 0)),
            "fps": fps,
            "duration": float(fmt.get("duration", 0)),
        }
    except Exception as e:
        logger.warning(f"Could not get info for {video_path}: {e}")
        return None


def detect_corridor_crop(
    video_path: Path,
    row_threshold: int = 80,
    n_samples: int = 5,
) -> Optional[tuple]:
    """
    Detect the vertical content region of a corridor video.

    Samples n_samples frames from the middle half of the video, takes the
    element-wise max (so a moving fly doesn't shadow rows), then finds the
    first and last row whose mean brightness exceeds ``row_threshold``.

    Returns (y_start, content_h, raw_w) — full frame width is kept; only
    vertical cropping is done to remove black/dim background at top/bottom.
    Returns None on failure.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            return None
        raw_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        raw_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        start_f = total // 4
        end_f = 3 * total // 4
        indices = [
            int(start_f + (end_f - start_f) * i / n_samples) for i in range(n_samples)
        ]
        frames = []
        for fi in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        if not frames:
            return None
        projection = np.max(np.stack(frames, axis=0), axis=0)
        row_means = projection.mean(axis=1)
        bright = np.where(row_means > row_threshold)[0]
        if len(bright) == 0:
            return None
        y_start = int(bright[0])
        y_end = int(bright[-1])
        content_h = y_end - y_start + 1
        return (y_start, content_h, raw_w)
    except Exception as e:
        logger.warning(f"Corridor crop detection failed for {video_path.name}: {e}")
        return None
    finally:
        cap.release()


def _compute_layout(num_videos: int, ar: float) -> Optional[tuple]:
    """Compute (cols, rows, cell_w, cell_h) using CONFIG constraints."""
    if CONFIG.get("force_single_row"):
        cols = num_videos
        rows = 1
        avail_w = CONFIG["max_grid_width"] - CONFIG["horizontal_padding"] * (cols + 1)
        avail_h = (
            CONFIG["max_grid_height"]
            - CONFIG["top_padding"]
            - CONFIG["bottom_padding"]
            - CONFIG["vertical_padding"] * (rows + 1)
        )
        return (cols, rows, avail_w / cols, avail_h)

    best_layout = None
    best_util = 0
    max_cols = min(num_videos, int(CONFIG["max_grid_width"] / CONFIG["min_cell_width"]))
    for cols in range(1, max_cols + 1):
        rows = math.ceil(num_videos / cols)
        avail_w = CONFIG["max_grid_width"] - CONFIG["horizontal_padding"] * (cols + 1)
        avail_h = (
            CONFIG["max_grid_height"]
            - CONFIG["top_padding"]
            - CONFIG["bottom_padding"]
            - CONFIG["vertical_padding"] * (rows + 1)
        )
        if avail_w <= 0 or avail_h <= 0:
            continue
        cw, ch = avail_w / cols, avail_h / rows
        if cw < CONFIG["min_cell_width"] or ch < CONFIG["min_cell_height"]:
            continue
        if abs(cw / ch - ar) > CONFIG["aspect_ratio_tolerance"]:
            continue
        util = (
            cw
            * ch
            * num_videos
            / (CONFIG["max_grid_width"] * CONFIG["max_grid_height"])
        )
        if util > best_util:
            best_util = util
            best_layout = (cols, rows, cw, ch)
    return best_layout


def _run_ffmpeg_with_progress(
    cmd: list,
    total_frames: int,
    label: str = "",
) -> subprocess.CompletedProcess:
    """
    Run an ffmpeg command while showing a tqdm progress bar driven by
    ``-progress pipe:1`` frame-count output.

    Returns a CompletedProcess-like object with .returncode and .stderr.
    ``total_frames`` is used to set the bar maximum; pass 0 to get a spinner.
    """
    # Inject -progress pipe:1 -nostats before the output path (last element)
    progress_cmd = list(cmd)
    progress_cmd[-1:] = ["-progress", "pipe:1", "-nostats", progress_cmd[-1]]

    stderr_lines: list = []
    bar = tqdm(
        total=total_frames if total_frames > 0 else None,
        unit="fr",
        desc=label,
        leave=False,
        dynamic_ncols=True,
    )
    last_frame = 0
    try:
        proc = subprocess.Popen(
            progress_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        # Read stdout (progress lines) and stderr concurrently via threads
        import threading

        def _drain_stderr():
            for line in proc.stderr:
                stderr_lines.append(line)

        t = threading.Thread(target=_drain_stderr, daemon=True)
        t.start()

        for line in proc.stdout:
            line = line.strip()
            if line.startswith("frame="):
                try:
                    frame = int(line.split("=", 1)[1])
                    bar.update(frame - last_frame)
                    last_frame = frame
                except ValueError:
                    pass

        proc.wait()
        t.join()
    finally:
        bar.close()

    return subprocess.CompletedProcess(
        args=progress_cmd,
        returncode=proc.returncode,
        stdout="",
        stderr="".join(stderr_lines),
    )


def generate_grid_ffmpeg(
    video_paths: List[Path],
    identifiers: List[str],
    grid_text: str,
    output_path: Path,
    test: bool = False,
    compress: bool = False,
    max_size_gb: float = 2.5,
    use_gpu: bool = False,
) -> bool:
    """
    Assemble a video grid entirely with ffmpeg (xstack + drawtext filters).
    No OpenCV CUDA required. Supports GPU-accelerated encode via NVENC.

    Layout mirrors the OpenCV backend: title bar at top, cells in rows/cols,
    per-cell label (date / arena / corridor) drawn below each clip.
    """
    if not video_paths:
        logger.error("No video paths provided")
        return False

    info = get_video_info_ffprobe(video_paths[0])
    if not info:
        logger.error(f"Cannot read video info from {video_paths[0]}")
        return False

    fps = info["fps"]

    # --- Per-video corridor crop detection ----------------------------------
    # Detect the actual corridor content region (strips dim background rows).
    # Returns (y_start, content_h, raw_w) per video; falls back to full frame.
    logger.info("Detecting corridor content regions…")
    crop_results: List[tuple] = []  # (y_start, content_h, raw_w)
    raw_h_list: List[int] = []  # raw frame height per video (for rotated layout)
    for p in video_paths:
        vi = get_video_info_ffprobe(p)
        raw_w = vi["width"] if vi else 0
        raw_h = vi["height"] if vi else 0
        raw_h_list.append(raw_h)
        cp = detect_corridor_crop(p)
        if cp:
            y_start, content_h, raw_w = cp
            crop_results.append((y_start, content_h, raw_w))
            logger.debug(
                f"  {p.name}: y_start={y_start} content_h={content_h} raw={raw_w}×{raw_h}"
            )
        else:
            crop_results.append((0, raw_h, raw_w))

    raw_widths = [r[2] for r in crop_results]
    content_hs = [r[1] for r in crop_results]

    # cell_h = min detected content height — crops background from all corridors.
    # cell_w = min raw width — wider corridors are center-cropped to this width
    #          so every cell is identical and separators are always uniform.
    cell_h = (min(content_hs) // 2) * 2
    cell_w = (min(raw_widths) // 2) * 2

    # Thin separator between columns
    sep_px = 2

    num_videos = len(video_paths)

    # A rough label_h estimate (3 lines × line_spacing) for layout purposes;
    # the precise value is recalculated below after _wrap_label is defined.
    _label_h_est = CONFIG["line_spacing"] * 3

    # ---------------------------------------------------------------------------
    # Layout: compute (cols, rows) and effective cell size (eff_cell_w × eff_cell_h).
    #
    # For ROTATED videos (transpose=1 swaps axes, making portrait corridors lie
    # landscape) we cannot use native pixel dims as the cell size — they are
    # tiny (e.g. 94×500 → 500×94 after rotation) and would never fill 4 K.
    # Instead we derive the target cell size from the aspect ratio and find
    # the best (cols, rows) that maximises grid utilisation via scaling.
    #
    # For NON-ROTATED videos we keep the original approach: use native cropped
    # dimensions as the cell size and tile them at that size.
    # ---------------------------------------------------------------------------
    if CONFIG["rotate_videos"]:
        # Median raw dims (robust to outlier videos)
        _sorted_rh = sorted(raw_h_list)
        _sorted_rw = sorted(raw_widths)
        _med_rh = _sorted_rh[len(_sorted_rh) // 2]  # becomes width after transpose
        _med_rw = _sorted_rw[len(_sorted_rw) // 2]  # becomes height after transpose
        ar = _med_rh / max(_med_rw, 1)  # width / height ratio in landscape

        if CONFIG.get("force_single_row"):
            cols, rows = num_videos, 1
            _avail_w = (CONFIG["max_grid_width"] - sep_px * num_videos) / num_videos
            _avail_h = CONFIG["max_grid_height"] - _label_h_est
            # Scale to fit slot while preserving AR
            if _avail_w / max(_avail_h, 1) > ar:
                _ch = _avail_h
                _cw = _ch * ar
            else:
                _cw = _avail_w
                _ch = _cw / ar
        else:
            best_layout = None
            best_util = 0.0
            for c in range(1, num_videos + 1):
                r = math.ceil(num_videos / c)
                # Available space per cell (accounting for separators / labels)
                _avail_w_c = (CONFIG["max_grid_width"] - sep_px * c) / c
                _avail_h_c = (CONFIG["max_grid_height"] - _label_h_est * r) / r
                if _avail_w_c <= 0 or _avail_h_c <= 0:
                    continue
                # Scale to fit while preserving AR
                if _avail_w_c / _avail_h_c > ar:
                    # height-constrained
                    _ch = _avail_h_c
                    _cw = _ch * ar
                else:
                    # width-constrained
                    _cw = _avail_w_c
                    _ch = _cw / ar
                _total_w = c * (_cw + sep_px)
                _total_h = r * (_ch + _label_h_est)
                if (
                    _total_w <= CONFIG["max_grid_width"]
                    and _total_h <= CONFIG["max_grid_height"]
                ):
                    _util = (
                        _cw
                        * _ch
                        * num_videos
                        / (CONFIG["max_grid_width"] * CONFIG["max_grid_height"])
                    )
                    if _util > best_util:
                        best_util = _util
                        best_layout = (c, r, _cw, _ch)

            if not best_layout:
                logger.error(
                    "No valid grid layout found — try --force-max or fewer videos"
                )
                return False
            cols, rows, _cw, _ch = best_layout

        eff_cell_w = (int(_cw) // 2) * 2
        eff_cell_h = (int(_ch) // 2) * 2

    else:
        # Non-rotated: native cropped pixel dimensions define the cell size.
        eff_cell_w, eff_cell_h = cell_w, cell_h

        if CONFIG.get("force_single_row"):
            cols, rows = num_videos, 1
        else:
            _slot_w = eff_cell_w + sep_px
            _slot_h_est = eff_cell_h + _label_h_est
            _max_cols = max(1, CONFIG["max_grid_width"] // _slot_w)
            best_layout = None
            best_util = 0.0
            for c in range(1, _max_cols + 1):
                r = math.ceil(num_videos / c)
                total_w = c * _slot_w
                total_h = r * _slot_h_est
                if (
                    total_w <= CONFIG["max_grid_width"]
                    and total_h <= CONFIG["max_grid_height"]
                ):
                    util = (total_w * total_h) / (
                        CONFIG["max_grid_width"] * CONFIG["max_grid_height"]
                    )
                    if util > best_util:
                        best_util = util
                        best_layout = (c, r)
            if not best_layout:
                logger.error(
                    "No valid grid layout found — try --force-max or fewer videos"
                )
                return False
            cols, rows = best_layout

    title_h = max(2, (CONFIG["top_padding"] // 2) * 2)

    # NVDEC: need ≥144px in both dimensions (corridors are ~96px — use CPU decode)
    _NVDEC_MIN = 144
    can_hwdec = use_gpu and cell_w >= _NVDEC_MIN and cell_h >= _NVDEC_MIN
    if use_gpu and not can_hwdec:
        logger.info(
            f"Corridor dimensions {cell_w}×{cell_h} below NVDEC minimum — "
            "hardware decode disabled, GPU encode (h264_nvenc) still active"
        )

    logger.info(
        f"Grid layout: {cols}×{rows}, corridor {eff_cell_w}×{eff_cell_h} (post-rotation), "
        f"sep={sep_px}px, {num_videos} videos — output {output_path.name}"
    )

    def _esc(text: str) -> str:
        """Escape special characters for ffmpeg drawtext."""
        return text.replace("\\", "\\\\").replace("'", "\u2019").replace(":", "\\:")

    def _wrap_label(text: str, max_chars: int) -> str:
        """Hard-wrap each newline-separated segment to fit max_chars per line."""
        result = []
        for segment in text.split("\n"):
            while len(segment) > max_chars:
                wrap_at = max_chars
                for sep in (" ", "-", "_"):
                    pos = segment.rfind(sep, 0, max_chars)
                    if pos > 0:
                        wrap_at = pos + 1
                        break
                result.append(segment[:wrap_at].rstrip("_- "))
                segment = segment[wrap_at:]
            result.append(segment)
        return "\n".join(result)

    # Approximate monospace character width at fontsize=16 for label wrapping.
    _label_fontsize = 16
    _char_w_px = _label_fontsize * 0.6  # ~9.6 px/char for Mono
    _max_chars = max(6, int(eff_cell_w / _char_w_px))

    # label_h from worst-case wrapped line count across all labels.
    _max_label_lines = max(
        (len(_wrap_label(lbl, _max_chars).split("\n")) for lbl in identifiers),
        default=3,
    )
    label_h = max(2, ((_max_label_lines * CONFIG["line_spacing"]) // 2) * 2)

    # Build per-input flags and filter_complex
    filter_parts: List[str] = []
    cmd_inputs: List[str] = []

    for i, (path, label) in enumerate(zip(video_paths, identifiers)):
        if can_hwdec:
            cmd_inputs += ["-hwaccel", "cuda"]
        if test:
            raw_start = CONFIG.get("test_start_time") or 0
            duration = get_video_duration(path)
            test_dur = CONFIG["test_duration"]
            if duration > 0:
                t_start = min(raw_start, max(0.0, duration - test_dur))
            else:
                t_start = 0
            cmd_inputs += ["-ss", str(t_start), "-t", str(test_dur)]
        cmd_inputs += ["-i", str(path)]

        rotate = "transpose=1," if CONFIG["rotate_videos"] else ""

        if CONFIG["rotate_videos"]:
            # After transpose=1 the frame is eff_cell_w×eff_cell_h (from raw dims).
            # Use scale to guarantee EXACTLY eff_cell_w×eff_cell_h output — this
            # is what the xstack slot dimensions are based on, so any size mismatch
            # would cause cells to overlap.  For same-setup corridors the scale
            # is effectively a no-op; for edge cases it rescales cleanly.
            cell_filter = (
                f"transpose=1,"
                f"scale={eff_cell_w}:{eff_cell_h},"
                f"setsar=1,"
                f"pad=iw:ih+{label_h}:0:0:black"
            )
        else:
            # Non-rotated: center-crop to uniform cell_w×cell_h using detection offsets.
            y_start, content_h, raw_w = crop_results[i]
            x_off = (raw_w - cell_w) // 2
            cell_filter = (
                f"crop={eff_cell_w}:{eff_cell_h}:{x_off}:{y_start},"
                f"setsar=1,"
                f"pad=iw:ih+{label_h}:0:0:black"
            )
        # Per-line centered drawtext
        label_lines = _wrap_label(label, _max_chars).split("\n")
        for j, line in enumerate(label_lines):
            y_txt = eff_cell_h + 8 + j * CONFIG["line_spacing"]
            cell_filter += (
                f",drawtext=text='{_esc(line)}'"
                f":x=(w-tw)/2:y={y_txt}"
                f":fontsize={_label_fontsize}:fontcolor=white:font=Mono"
            )
        filter_parts.append(f"[{i}:v]{cell_filter}[cell{i}]")

    # xstack: sep_px-wide black gap between columns
    slot_h = eff_cell_h + label_h
    slot_w = eff_cell_w + sep_px
    xstack_layout = "|".join(
        f"{(i % cols) * slot_w}_{(i // cols) * slot_h}" for i in range(num_videos)
    )
    xstack_inputs = "".join(f"[cell{i}]" for i in range(num_videos))
    filter_parts.append(
        f"{xstack_inputs}xstack=inputs={num_videos}:layout={xstack_layout}:fill=black[grid]"
    )

    # Title bar at top
    filter_parts.append(
        f"[grid]pad=iw:ih+{title_h}:0:{title_h}:black,"
        f"drawtext=text='{_esc(grid_text)}':x=10:y=10:"
        f"fontsize=22:fontcolor=white:font=Mono[out]"
    )

    filter_complex = "; ".join(filter_parts)

    # Encoding arguments
    if compress:
        duration_s = (
            CONFIG["test_duration"] if test else get_video_duration(video_paths[0])
        )
        bitrate = calculate_target_bitrate(
            duration_s * num_videos / num_videos, max_size_gb
        )
        # For single-pass with bitrate control
        if use_gpu:
            enc_args = [
                "-c:v",
                "h264_nvenc",
                "-preset",
                "p4",
                "-b:v",
                f"{bitrate}k",
                "-maxrate",
                f"{int(bitrate * 1.5)}k",
                "-bufsize",
                f"{bitrate * 2}k",
            ]
        else:
            enc_args = [
                "-c:v",
                "libx264",
                "-preset",
                CONFIG.get("compression_preset", "medium"),
                "-b:v",
                f"{bitrate}k",
                "-maxrate",
                f"{int(bitrate * 1.5)}k",
                "-bufsize",
                f"{bitrate * 2}k",
            ]
    else:
        if use_gpu:
            enc_args = [
                "-c:v",
                "h264_nvenc",
                "-preset",
                "p4",
                "-cq",
                str(CONFIG.get("crf", 23)),
            ]
        else:
            enc_args = [
                "-c:v",
                "libx264",
                "-preset",
                CONFIG.get("compression_preset", "medium"),
                "-crf",
                str(CONFIG.get("crf", 23)),
            ]

    cmd = (
        ["ffmpeg"]
        + cmd_inputs
        + ["-filter_complex", filter_complex, "-map", "[out]"]
        + enc_args
        + ["-pix_fmt", "yuv420p", "-r", str(fps), "-an", "-y", str(output_path)]
    )

    logger.info("Running ffmpeg grid assembly…")
    total_frames = (
        int(get_video_duration(video_paths[0]) * fps)
        if not test
        else int(CONFIG["test_duration"] * fps)
    )
    result = _run_ffmpeg_with_progress(
        cmd, total_frames=total_frames, label=output_path.stem
    )

    if result.returncode != 0:
        logger.error(f"ffmpeg failed:\n{result.stderr[-3000:]}")
        return False

    size_gb = output_path.stat().st_size / (1024**3)
    logger.info(f"✓ Grid created: {output_path.name} ({size_gb:.2f} GB)")

    # Strict size enforcement for compress mode: two-pass retry
    if compress and size_gb > max_size_gb:
        excess_mb = (size_gb - max_size_gb) * 1000
        logger.warning(f"Exceeds limit by {excess_mb:.0f} MB — re-encoding (two-pass)…")
        temp = output_path.with_suffix(".tmp.mp4")
        output_path.rename(temp)
        ok = compress_to_size_limit(
            temp,
            output_path,
            max_size_gb=max_size_gb,
            preset=CONFIG.get("compression_preset", "medium"),
            use_gpu=use_gpu,
        )
        temp.unlink(missing_ok=True)
        if not ok:
            logger.error("Two-pass retry also failed")
            return False

    return True


def calculate_target_bitrate(
    duration: float, max_size_gb: float, safety_margin: float = 0.90
) -> int:
    """Calculate target video bitrate (kbps) to stay within max_size_gb."""
    if duration <= 0:
        return 5000
    max_size_bits = max_size_gb * 1024 * 1024 * 1024 * 8
    video_bitrate_bps = (max_size_bits / duration) * safety_margin
    return max(1000, min(50000, int(video_bitrate_bps / 1000)))


def compress_to_size_limit(
    input_path: Path,
    output_path: Path,
    max_size_gb: float = 2.5,
    preset: str = "medium",
    use_gpu: bool = False,
) -> bool:
    """
    Two-pass bitrate-constrained encode targeting max_size_gb.
    If the output still exceeds the limit, retries once with a proportionally
    lower bitrate (same strategy as compress_videos_yaml.py).
    """
    duration = get_video_duration(input_path)
    if duration <= 0:
        logger.error(f"Cannot compress {input_path.name}: unreadable duration")
        return False

    target_bitrate = calculate_target_bitrate(duration, max_size_gb)
    logger.info(
        f"  Target bitrate: {target_bitrate / 1000:.1f} Mbps "
        f"(max {max_size_gb} GB, {duration / 60:.1f} min)"
    )

    pass_log = str(output_path.parent / f"ffmpeg2pass_{output_path.stem}")

    def _run_two_pass(bitrate: int) -> bool:
        try:
            if use_gpu:
                nvenc_preset = {
                    "ultrafast": "p1",
                    "fast": "p2",
                    "medium": "p4",
                    "slow": "p5",
                }.get(preset, "p4")
                subprocess.run(
                    [
                        "ffmpeg",
                        "-hwaccel",
                        "cuda",
                        "-i",
                        str(input_path),
                        "-c:v",
                        "h264_nvenc",
                        "-preset",
                        nvenc_preset,
                        "-b:v",
                        f"{bitrate}k",
                        "-maxrate",
                        f"{int(bitrate * 1.5)}k",
                        "-bufsize",
                        f"{bitrate * 2}k",
                        "-pass",
                        "1",
                        "-passlogfile",
                        pass_log,
                        "-an",
                        "-f",
                        "null",
                        "/dev/null",
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=7200,
                )
                subprocess.run(
                    [
                        "ffmpeg",
                        "-hwaccel",
                        "cuda",
                        "-i",
                        str(input_path),
                        "-c:v",
                        "h264_nvenc",
                        "-preset",
                        nvenc_preset,
                        "-b:v",
                        f"{bitrate}k",
                        "-maxrate",
                        f"{int(bitrate * 1.5)}k",
                        "-bufsize",
                        f"{bitrate * 2}k",
                        "-pass",
                        "2",
                        "-passlogfile",
                        pass_log,
                        "-movflags",
                        "+faststart",
                        "-an",
                        "-y",
                        str(output_path),
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=7200,
                )
            else:
                subprocess.run(
                    [
                        "ffmpeg",
                        "-i",
                        str(input_path),
                        "-c:v",
                        "libx264",
                        "-preset",
                        preset,
                        "-b:v",
                        f"{bitrate}k",
                        "-maxrate",
                        f"{int(bitrate * 1.5)}k",
                        "-bufsize",
                        f"{bitrate * 2}k",
                        "-pass",
                        "1",
                        "-passlogfile",
                        pass_log,
                        "-threads",
                        str(os.cpu_count()),
                        "-tune",
                        "grain",
                        "-an",
                        "-f",
                        "null",
                        "/dev/null",
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=7200,
                )
                subprocess.run(
                    [
                        "ffmpeg",
                        "-i",
                        str(input_path),
                        "-c:v",
                        "libx264",
                        "-preset",
                        preset,
                        "-b:v",
                        f"{bitrate}k",
                        "-maxrate",
                        f"{int(bitrate * 1.5)}k",
                        "-bufsize",
                        f"{bitrate * 2}k",
                        "-pass",
                        "2",
                        "-passlogfile",
                        pass_log,
                        "-threads",
                        str(os.cpu_count()),
                        "-tune",
                        "grain",
                        "-movflags",
                        "+faststart",
                        "-an",
                        "-y",
                        str(output_path),
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=7200,
                )
            return True
        except Exception as e:
            logger.error(f"Two-pass encoding error: {e}")
            return False
        finally:
            for f in output_path.parent.glob(f"ffmpeg2pass_{output_path.stem}*"):
                try:
                    f.unlink()
                except Exception:
                    pass

    if not _run_two_pass(target_bitrate):
        return False

    actual_gb = output_path.stat().st_size / (1024**3)
    logger.info(f"  Output size: {actual_gb:.2f} GB")

    if actual_gb > max_size_gb:
        excess_mb = (actual_gb - max_size_gb) * 1000
        logger.warning(
            f"  Exceeds limit by {excess_mb:.0f} MB — re-encoding at lower bitrate"
        )
        corrected_bitrate = int(target_bitrate / (actual_gb / max_size_gb) * 0.92)
        output_path.unlink()
        if not _run_two_pass(corrected_bitrate):
            return False
        actual_gb = output_path.stat().st_size / (1024**3)
        if actual_gb > max_size_gb:
            logger.error(f"  Still exceeds limit after retry ({actual_gb:.2f} GB)")
            return False

    logger.info(f"  ✓ Compression done: {actual_gb:.2f} GB")
    return True


# Constants
DATA_PATH = "/mnt/upramdya_data/MD/F1_Tracks/Datasets/260123_16_F1_coordinates_F1_TNT_Full_Data/F1_coordinates/pooled_F1_coordinates.feather"
MAPPING_CSV_PATH = "/mnt/upramdya_data/MD/Region_map_250908.csv"  # Map if needed
MISSING_VIDEOS_PATH = None  # Path for missing videos list if needed

groupby = ["Genotype", "Pretraining"]

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


def create_nickname_mapping(csv_path: str = None) -> dict:
    """Create mapping from nicknames/genotypes to simplified nicknames."""
    path = csv_path or MAPPING_CSV_PATH
    try:
        mapping_data = pd.read_csv(path)
        nickname_to_simplified = {}

        for _, row in mapping_data.iterrows():
            simplified = row.get("Simplified Nickname")
            if pd.isna(simplified):
                continue

            # Direct CSV columns already used by old code
            for col in ("Nickname", "Genotype"):
                key = row.get(col)
                if pd.notna(key):
                    nickname_to_simplified[key] = simplified

            # Old Simplified Nickname matches feather Nickname in many cases
            # (e.g. 'MBON-08-GaL4  MBON-09-GaL4 ' or 'MB310C')
            old = row.get("Old Simplified Nickname")
            if pd.notna(old) and old:
                old = str(old).strip()  # strip trailing spaces present in the CSV
                nickname_to_simplified[old] = simplified

                # Feather often stores '{old} ({simplified})' as Nickname
                # (e.g. 'MBON-01-GaL4 (MBON-γ5β′2a)')
                constructed = f"{old} ({simplified})"
                nickname_to_simplified[constructed] = simplified

        logger.info(f"Created mapping for {len(nickname_to_simplified)} identifiers")
        return nickname_to_simplified
    except Exception as e:
        logger.error(f"Error creating nickname mapping: {e}")
        return {}


def lookup_simplified(nickname: str, mapping: dict) -> str:
    """
    Resolve a feather Nickname to its Simplified Nickname.

    Tries in order:
    1. Direct mapping lookup.
    2. Parenthetical content: '51978 (AstA-GAL4 3M)' → look up 'AstA-GAL4 3M'.
       Handles stock-number-prefixed keys where the content is the Old Simplified Nickname.
    3. Prefix before first '(': 'R12B01-gal4 (R4d - distal (EB))' → look up 'R12B01-gal4'.
       Handles cases where only the GAL4 stock name (Old Simplified) is in the mapping.
    4. Newline normalization: 'JON-AB\\nJO-15"' → look up 'JON-AB'.
       Handles multi-line Nickname strings in the feather.
    5. Case-insensitive scan of the mapping (slow but rare fallback).
    6. Falls back to the raw nickname if nothing matches.
    """
    if nickname in mapping:
        return mapping[nickname]

    # Try stripped key (feather strings sometimes carry trailing spaces)
    stripped = nickname.strip()
    if stripped != nickname and stripped in mapping:
        return mapping[stripped]

    # 2. Parenthetical content (last group)
    m = re.search(r"\((.+)\)$", nickname.strip())
    if m:
        content = m.group(1)
        if content in mapping:
            return mapping[content]

    # 3. Prefix before first '(' (strip trailing whitespace)
    paren_pos = nickname.find("(")
    if paren_pos > 0:
        prefix = nickname[:paren_pos].strip()
        if prefix in mapping:
            return mapping[prefix]

    # 4. First line of a multi-line nickname
    if "\n" in nickname:
        first_line = nickname.split("\n")[0].strip()
        if first_line in mapping:
            return mapping[first_line]

    # 5. Case-insensitive scan (last resort)
    lower = nickname.lower()
    for key, val in mapping.items():
        if key.lower() == lower:
            return val

    return nickname


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
    compress: bool = False,
    max_size_gb: float = 2.5,
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

        # Initialize video writer with optimized settings
        fps = videos[0].get(cv2.CAP_PROP_FPS)

        # Optionally reduce framerate for faster processing
        if CONFIG.get("reduce_framerate", False):
            fps = fps / 2
            frame_skip = 2
        else:
            frame_skip = 1

        output_path = Path(CONFIG["output_dir"]) / output_filename

        # Use temporary file if ffmpeg compression is enabled
        if CONFIG.get("use_ffmpeg_compression", True):
            # Use .avi for temp file - MJPG codec works with AVI, not MP4
            temp_output = output_path.parent / f"temp_{output_path.stem}.avi"
            final_output = output_path
            write_path = str(temp_output)
            # MJPG works well with AVI for fast temporary encoding
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        else:
            write_path = str(output_path)
            final_output = None
            # Use mp4v for final output if not using ffmpeg
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        out = cv2.VideoWriter(write_path, fourcc, fps, (grid_width, grid_height))

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
                # Skip frames if reducing framerate
                if CONFIG.get("reduce_framerate", False) and frame_count % 2 != 0:
                    # Still need to read frames to keep videos in sync
                    for video in videos:
                        video.read()
                    pbar.update(1)
                    continue

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

        # Compress with ffmpeg if enabled
        if CONFIG.get("use_ffmpeg_compression", True) and final_output:
            try:
                # Close the temp video first
                out.release()
                out = None  # Prevent double release in finally block

                if compress:
                    # Size-limited two-pass encoding (guarantees <= max_size_gb)
                    logger.info(
                        f"Compressing with size limit of {max_size_gb} GB "
                        f"(two-pass bitrate-constrained)..."
                    )
                    success = compress_to_size_limit(
                        temp_output,
                        final_output,
                        max_size_gb=max_size_gb,
                        preset=CONFIG.get("compression_preset", "medium"),
                        use_gpu=CONFIG.get("use_gpu_encoding", False),
                    )
                    if success:
                        temp_output.unlink(missing_ok=True)
                    else:
                        logger.error(
                            "Size-limited compression failed; keeping temp file"
                        )
                        if temp_output.exists():
                            temp_output.rename(final_output)
                else:
                    # CRF-based encoding (original quality-target behavior)
                    use_gpu = CONFIG.get("use_gpu_encoding", False)
                    encoder = "GPU (NVENC)" if use_gpu else "CPU (libx264)"
                    logger.info(
                        f"Compressing video with ffmpeg using {encoder} (CRF/CQ={CONFIG['crf']})..."
                    )
                    preset = CONFIG.get("compression_preset", "fast")
                    crf = CONFIG.get("crf", 28)

                    if use_gpu:
                        nvenc_preset_map = {
                            "ultrafast": "p1",
                            "fast": "p2",
                            "medium": "p4",
                            "slow": "p6",
                            "slower": "p7",
                        }
                        nvenc_preset = nvenc_preset_map.get(preset, "p4")
                        cmd = [
                            "ffmpeg",
                            "-hwaccel",
                            "cuda",
                            "-i",
                            str(temp_output),
                            "-c:v",
                            "h264_nvenc",
                            "-preset",
                            nvenc_preset,
                            "-cq",
                            str(crf),
                            "-pix_fmt",
                            "yuv420p",
                            "-y",
                            str(final_output),
                        ]
                    else:
                        cmd = [
                            "ffmpeg",
                            "-i",
                            str(temp_output),
                            "-c:v",
                            "libx264",
                            "-preset",
                            preset,
                            "-crf",
                            str(crf),
                            "-pix_fmt",
                            "yuv420p",
                            "-y",
                            str(final_output),
                        ]

                    result = subprocess.run(cmd, capture_output=True, text=True)
                    if result.returncode == 0:
                        temp_output.unlink()
                        logger.info(f"Compression successful: {final_output.name}")
                        final_size_mb = final_output.stat().st_size / (1024 * 1024)
                        logger.info(f"Final file size: {final_size_mb:.1f} MB")
                    else:
                        logger.error(f"FFmpeg compression failed: {result.stderr}")
                        temp_output.rename(final_output)
            except Exception as e:
                logger.error(f"Error during ffmpeg compression: {e}")
                if temp_output.exists():
                    temp_output.rename(final_output)

        return out

    except Exception as e:
        logger.error(
            f"Error in create_video_grid_with_features: {e}\n{traceback.format_exc()}"
        )
        return None
    finally:
        for video in videos:
            video.release()
        if "out" in locals() and out is not None:
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

    # Resize — GPU if available, CPU fallback
    try:
        gpu_frame = cv2.cuda_GpuMat()
        gpu_frame.upload(frame)
        gpu_resized = cv2.cuda.resize(
            gpu_frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR
        )
        frame = gpu_resized.download()
    except (cv2.error, AttributeError):
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Pad to exact target size
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
    """Build per-cell labels: date / arena / corridor extracted from flypath."""
    identifiers = []
    for path in video_paths:
        if path:
            # flypath structure: .../date/arena/corridor/
            corridor = path.parent.name
            arena = path.parent.parent.name

            matching_rows = data.loc[data["flypath"] == str(path.parent)]
            if not matching_rows.empty:
                date = str(matching_rows["Date"].iloc[0])
            else:
                date = "Unknown"

            identifier = f"{date}\n{arena}\n{corridor}"
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
                    trial_id, genotype, pretraining = group_key
                    output_filename = (
                        f"{genotype}_{pretraining}_trial_{trial_id}_grid.mp4"
                    )
                    trial_times = group_data[
                        ["start_time", "end_time"]
                    ].drop_duplicates()
                else:
                    genotype, pretraining = group_key
                    output_filename = f"{genotype}_{pretraining}_grid.mp4"

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
                # Main title includes genotype and pretraining
                if CONFIG["trial_mode"]:
                    title = f"{genotype} - {pretraining} | Trial {trial_id}"
                else:
                    title = f"{genotype} - {pretraining}"

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
        "--genotypes",
        type=str,
        default=None,
        help="Comma-separated list of genotype patterns to match (substring matching, e.g., 'TNT' matches TNTxLC16-1)",
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
    parser.add_argument(
        "--quality-preset",
        type=str,
        choices=[
            "4k_high",
            "4k_reduced_fps",
            "1440p",
            "1440p_reduced_fps",
            "1080p_better",
            "1080p_fast",
        ],
        default=None,
        help="Quality preset: 4k_high (4K CRF23), 4k_reduced_fps (4K CRF23 15fps, RECOMMENDED), "
        "1440p (2560x1440 CRF23), 1440p_reduced_fps (1440p CRF23 15fps), "
        "1080p_better (1080p CRF20), 1080p_fast (1080p CRF28)",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=None,
        help="Override CRF value (18-28): lower=better quality, higher=smaller files",
    )
    parser.add_argument(
        "--reduced-fps",
        action="store_true",
        help="Enable reduced framerate (halves FPS, saves ~50%% space)",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        choices=["4k", "1440p", "1080p"],
        default=None,
        help="Override output resolution: 4k (3840x2160), 1440p (2560x1440), 1080p (1920x1080)",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use NVIDIA GPU hardware acceleration (NVENC) for ffmpeg compression - 5-10x faster",
    )
    parser.add_argument(
        "--profile",
        type=str,
        choices=list(EXPERIMENT_PROFILES.keys()),
        default=None,
        help=(
            "Experiment profile to use. Sets data path, groupby column, output dir, "
            "and grid layout automatically. "
            + " | ".join(
                f"{k}: {v['output_dir']}" for k, v in EXPERIMENT_PROFILES.items()
            )
        ),
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["ffmpeg", "opencv"],
        default="ffmpeg",
        help=(
            "Grid assembly backend. 'ffmpeg' (default): pure ffmpeg xstack, no CUDA "
            "OpenCV required, GPU-accelerated via NVDEC/NVENC. "
            "'opencv': frame-by-frame Python loop with optional CUDA OpenCV."
        ),
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Randomly sample at most N videos per group when the full set is too "
            "large for a valid grid layout. Uses a fixed seed for reproducibility."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Print the list of groups that would be processed (output filename, "
            "video count) without generating any files."
        ),
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        help=(
            "Apply size-limited two-pass compression to the output. "
            "Output will be saved as *_grid_compressed.mp4. "
            "Use --max-size to set the size limit (default 2.5 GB)."
        ),
    )
    parser.add_argument(
        "--max-size",
        type=float,
        default=2.5,
        help="Maximum output file size in GB when --compress is used (default: 2.5).",
    )
    args = parser.parse_args()

    # Apply experiment profile (sets data paths, groupby, output dir, and layout)
    active_data_path = DATA_PATH
    active_mapping_csv = MAPPING_CSV_PATH
    active_groupby = groupby
    active_group_name = None
    active_output_dir = CONFIG["output_dir"]

    if args.profile:
        profile = EXPERIMENT_PROFILES[args.profile]
        active_data_path = profile["data_path"]
        active_mapping_csv = profile["mapping_csv_path"]
        active_groupby = profile["groupby"]
        active_group_name = profile.get("group_name", None)
        active_output_dir = profile["output_dir"]
        for key in [
            "max_grid_width",
            "max_grid_height",
            "force_single_row",
            "rotate_videos",
            "F1_experiments",
        ]:
            if key in profile:
                CONFIG[key] = profile[key]
        logger.info(f"Applied profile '{args.profile}':")
        logger.info(f"  data_path:  {active_data_path}")
        logger.info(f"  output_dir: {active_output_dir}")
        logger.info(f"  groupby:    {active_groupby}")
        logger.info(
            f"  grid size:  {CONFIG['max_grid_width']}x{CONFIG['max_grid_height']}, "
            f"force_single_row={CONFIG['force_single_row']}"
        )

    # Apply quality preset if specified
    QUALITY_PRESETS = {
        "4k_high": {
            "max_grid_width": 3840,
            "max_grid_height": 2160,
            "crf": 23,
            "reduce_framerate": False,
            "compression_preset": "medium",
        },
        "4k_reduced_fps": {
            "max_grid_width": 3840,
            "max_grid_height": 2160,
            "crf": 23,
            "reduce_framerate": True,
            "compression_preset": "fast",
        },
        "1440p": {
            "max_grid_width": 2560,
            "max_grid_height": 1440,
            "crf": 23,
            "reduce_framerate": False,
            "compression_preset": "medium",
        },
        "1440p_reduced_fps": {
            "max_grid_width": 2560,
            "max_grid_height": 1440,
            "crf": 23,
            "reduce_framerate": True,
            "compression_preset": "fast",
        },
        "1080p_better": {
            "max_grid_width": 1920,
            "max_grid_height": 1080,
            "crf": 20,
            "reduce_framerate": False,
            "compression_preset": "medium",
        },
        "1080p_fast": {
            "max_grid_width": 1920,
            "max_grid_height": 1080,
            "crf": 28,
            "reduce_framerate": True,
            "compression_preset": "fast",
        },
    }

    if args.quality_preset:
        preset = QUALITY_PRESETS[args.quality_preset]
        CONFIG.update(preset)
        logger.info(f"Applied quality preset: {args.quality_preset}")
        logger.info(
            f"  Resolution: {CONFIG['max_grid_width']}x{CONFIG['max_grid_height']}"
        )
        logger.info(f"  CRF: {CONFIG['crf']}")
        logger.info(f"  Reduced FPS: {CONFIG['reduce_framerate']}")

    # Check GPU support if requested
    if args.gpu:
        if check_nvenc_support():
            CONFIG["use_gpu_encoding"] = True
            logger.info("✓ NVIDIA NVENC hardware acceleration enabled")
        else:
            logger.warning("✗ NVENC not available, falling back to CPU encoding")
            logger.warning(
                "  Make sure you have NVIDIA GPU and recent ffmpeg with NVENC support"
            )

    # Override config with command-line arguments
    if args.test_start is not None:
        CONFIG["test_start_time"] = args.test_start

    # Individual parameter overrides (take precedence over presets)
    if args.crf is not None:
        CONFIG["crf"] = args.crf
        logger.info(f"Override CRF: {args.crf}")

    if args.reduced_fps:
        CONFIG["reduce_framerate"] = True
        logger.info("Enabled reduced framerate")

    if args.resolution:
        resolutions = {"4k": (3840, 2160), "1440p": (2560, 1440), "1080p": (1920, 1080)}
        width, height = resolutions[args.resolution]
        CONFIG["max_grid_width"] = width
        CONFIG["max_grid_height"] = height
        logger.info(f"Override resolution: {width}x{height}")

    # Parse filter values if provided
    filter_values = None
    if args.filter_values:
        filter_values = [v.strip() for v in args.filter_values.split(",") if v.strip()]

    # Parse genotype filter if provided
    genotype_filter = None
    if args.genotypes:
        genotype_filter = [v.strip() for v in args.genotypes.split(",") if v.strip()]
        logger.info(f"Filtering for genotypes: {genotype_filter}")

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

    def filtered_genotypes(groups, genotype_list):
        """Filter groups to include genotypes matching any pattern (substring matching)."""
        if not genotype_list:
            return groups
        for group_key, group_data in groups:
            # For groupby=['Genotype', 'Pretraining'], group_key is (genotype, pretraining)
            # For trial mode, group_key is (trial_id, genotype, pretraining)
            if CONFIG["trial_mode"]:
                # Extract genotype from trial mode tuple
                genotype = group_key[1] if len(group_key) > 1 else None
            else:
                # Extract genotype from normal mode tuple
                genotype = group_key[0] if isinstance(group_key, tuple) else group_key

            # Substring matching: check if any pattern is contained in the genotype
            if genotype and any(pattern in genotype for pattern in genotype_list):
                yield (group_key, group_data)

    def main_with_filter(
        test=False,
        full_screen=False,
        force_max=False,
        missing_only=False,
        genotypes=None,
        compress=False,
        max_size_gb=2.5,
        dry_run=False,
        backend="ffmpeg",
        active_data_path=DATA_PATH,
        active_mapping_csv=MAPPING_CSV_PATH,
        active_output_dir=CONFIG["output_dir"],
        active_groupby=groupby,
    ):
        try:
            transformed_data = load_dataset(active_data_path)
            if transformed_data.empty:
                logger.error("Empty dataset loaded")
                return

            ensure_output_directory_exists(active_output_dir)

            # Load nickname mapping for simplified names (using active CSV)
            nickname_mapping = create_nickname_mapping(active_mapping_csv)

            # Load missing videos list if processing missing only
            missing_identifiers = []
            if missing_only:
                missing_identifiers = load_missing_videos_list()
                if not missing_identifiers:
                    logger.info("No missing videos to process")
                    return
                logger.info(f"Processing only missing videos: {missing_identifiers}")

            # Grouping logic based on mode
            if active_groupby is None:
                # Entire dataset is one group; use the profile-supplied name
                _name = active_group_name or "all"
                groups = [(_name, transformed_data)]
                logger.info(f"Processing 1 experimental group ('{_name}', all rows)")
            elif CONFIG["trial_mode"]:
                groups = transformed_data.groupby(
                    [CONFIG["trial_column"], active_groupby]
                )
                logger.info(f"Processing {len(groups)} trial-group combinations")
            else:
                groups = transformed_data.groupby(active_groupby)
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
            elif genotypes:
                # Filter by genotypes first, then optionally by other filter values
                group_iter = list(filtered_genotypes(groups, genotypes))
                if filter_values:
                    group_iter = list(filtered_groups(group_iter, filter_values))
            elif filter_values:
                group_iter = list(filtered_groups(groups, filter_values))
            else:
                group_iter = list(groups)

            for group_key, group_data in tqdm(
                group_iter,
                desc="Groups",
                unit="grp",
                dynamic_ncols=True,
            ):
                try:
                    # Handle trial-mode metadata
                    grid_suffix = "_grid_compressed.mp4" if compress else "_grid.mp4"
                    if CONFIG["trial_mode"]:
                        trial_id, genotype, pretraining = group_key
                        display_genotype = lookup_simplified(genotype, nickname_mapping)
                        display_name = sanitize_for_dataverse(display_genotype)
                        output_filename = f"{display_name}_{pretraining}_trial_{trial_id}{grid_suffix}"
                        title = f"{display_name} - {pretraining} | Trial {trial_id}"
                        trial_times = group_data[
                            ["start_time", "end_time"]
                        ].drop_duplicates()
                    elif isinstance(group_key, str):
                        # Single-key groupby (e.g., tnt_screen "Nickname")
                        nickname = group_key
                        simplified = lookup_simplified(nickname, nickname_mapping)
                        display_name = sanitize_for_dataverse(simplified)
                        output_filename = f"{display_name}{grid_suffix}"
                        title = display_name
                    else:
                        genotype, pretraining = group_key
                        display_genotype = lookup_simplified(genotype, nickname_mapping)
                        display_name = sanitize_for_dataverse(display_genotype)
                        output_filename = f"{display_name}_{pretraining}{grid_suffix}"
                        title = f"{display_name} - {pretraining}"

                    # Skip existing outputs
                    output_path = Path(active_output_dir) / sanitize_filename(
                        output_filename
                    )

                    # --dry-run: describe what would be done and move on
                    if dry_run:
                        status = "EXISTS" if output_path.exists() else "PENDING"
                        logger.info(
                            f"[DRY RUN] {status:7s}  {output_filename}  "
                            f"(key={group_key!r})"
                        )
                        continue

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

                    # Optionally cap the number of videos to allow a valid layout
                    if args.max_videos and len(video_paths) > args.max_videos:
                        import random

                        rng = random.Random(42)
                        paired = list(zip(video_paths, valid_flypaths))
                        paired = rng.sample(paired, args.max_videos)
                        video_paths, valid_flypaths = map(list, zip(*paired))
                        logger.info(
                            f"Group {group_key}: sampled {args.max_videos} videos "
                            f"(from {len(flypaths)} total, seed=42)"
                        )

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

                    # Create video grid — dispatch to selected backend
                    CONFIG["output_dir"] = active_output_dir
                    if backend == "ffmpeg":
                        use_gpu = check_nvenc_support()
                        ok = generate_grid_ffmpeg(
                            video_paths=video_paths,
                            identifiers=identifiers,
                            grid_text=title,
                            output_path=output_path,
                            test=test,
                            compress=compress,
                            max_size_gb=max_size_gb,
                            use_gpu=use_gpu,
                        )
                        video_grid = ok
                    else:
                        video_grid = create_video_grid_with_features(
                            valid_videos,
                            identifiers=identifiers,
                            grid_text=title,
                            output_filename=output_filename,
                            test=test,
                            force_max=force_max,
                            compress=compress,
                            max_size_gb=max_size_gb,
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
        genotypes=genotype_filter,
        compress=args.compress,
        max_size_gb=args.max_size,
        dry_run=args.dry_run,
        backend=args.backend,
        active_data_path=active_data_path,
        active_mapping_csv=active_mapping_csv,
        active_output_dir=active_output_dir,
        active_groupby=active_groupby,
    )
