#!/usr/bin/env python3
"""
Compress large MP4 videos using ffmpeg with H.264 encoding.

This script processes all MP4 files in a directory and creates compressed versions
using the same quality settings as the grid generation (CRF=23, medium preset).
"""

import argparse
import subprocess
from pathlib import Path
import logging
import sys
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            f"compression_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        ),
    ],
)
logger = logging.getLogger(__name__)


def get_video_info(video_path: Path) -> dict:
    """Get video duration and size information."""
    try:
        # Get duration using ffprobe
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration,size",
            "-of",
            "default=noprint_wrappers=1",
            str(video_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        info = {}
        for line in result.stdout.strip().split("\n"):
            if "=" in line:
                key, value = line.split("=")
                info[key] = value

        return {
            "duration": float(info.get("duration", 0)),
            "size_bytes": int(info.get("size", 0)),
            "size_gb": int(info.get("size", 0)) / (1024**3),
        }
    except Exception as e:
        logger.warning(f"Could not get info for {video_path.name}: {e}")
        return {
            "duration": 0,
            "size_bytes": video_path.stat().st_size,
            "size_gb": video_path.stat().st_size / (1024**3),
        }


def check_nvenc_support() -> bool:
    """Check if NVIDIA NVENC hardware encoding is available."""
    try:
        cmd = ["ffmpeg", "-hide_banner", "-encoders"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return "h264_nvenc" in result.stdout
    except Exception:
        return False


def compress_video(
    input_path: Path,
    output_path: Path,
    crf: int = 23,
    preset: str = "medium",
    overwrite: bool = False,
    use_gpu: bool = False,
) -> bool:
    """
    Compress a video using ffmpeg with H.264 encoding.

    Args:
        input_path: Path to input video
        output_path: Path to output compressed video
        crf: Constant Rate Factor (18-28, lower = better quality, 23 is good default)
        preset: Compression preset (ultrafast, fast, medium, slow, slower)
        overwrite: Whether to overwrite existing output files
        use_gpu: Use NVIDIA NVENC hardware acceleration if available

    Returns:
        True if compression succeeded, False otherwise
    """
    if output_path.exists() and not overwrite:
        logger.warning(f"Output already exists: {output_path.name}")
        return False

    logger.info(f"Compressing: {input_path.name}")

    # Get input file info
    input_info = get_video_info(input_path)
    logger.info(f"  Input size: {input_info['size_gb']:.2f} GB")
    if input_info["duration"] > 0:
        logger.info(f"  Duration: {input_info['duration']/60:.1f} minutes")

    try:
        # Build ffmpeg command based on GPU/CPU mode
        if use_gpu:
            # NVIDIA NVENC hardware encoding
            # preset mapping: p1 (fastest) to p7 (slowest)
            # For NVENC, we use -cq (constant quality) instead of -crf
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
                "cuda",  # Enable CUDA hardware acceleration
                "-i",
                str(input_path),
                "-c:v",
                "h264_nvenc",  # NVIDIA hardware encoder
                "-preset",
                nvenc_preset,
                "-cq",
                str(crf),  # Constant quality (like CRF for CPU)
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                "-y" if overwrite else "-n",
                str(output_path),
            ]
            logger.info(f"  Using GPU acceleration (NVENC, preset={nvenc_preset})")
        else:
            # CPU encoding with libx264
            cmd = [
                "ffmpeg",
                "-i",
                str(input_path),
                "-c:v",
                "libx264",
                "-preset",
                preset,
                "-crf",
                str(crf),
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",  # Copy or re-encode audio
                "-b:a",
                "128k",  # Audio bitrate
                "-y" if overwrite else "-n",  # Overwrite or skip existing
                str(output_path),
            ]
            logger.info(f"  Using CPU encoding (libx264, preset={preset})")

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        # Get output file info
        output_info = get_video_info(output_path)
        logger.info(f"  Output size: {output_info['size_gb']:.2f} GB")

        # Calculate compression ratio
        if input_info["size_bytes"] > 0:
            ratio = (1 - output_info["size_bytes"] / input_info["size_bytes"]) * 100
            saved_gb = input_info["size_gb"] - output_info["size_gb"]
            logger.info(
                f"  Compression: {ratio:.1f}% reduction ({saved_gb:.2f} GB saved)"
            )

        logger.info(f"✓ Successfully compressed: {output_path.name}")
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"✗ FFmpeg error for {input_path.name}:")
        logger.error(f"  {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"✗ Error compressing {input_path.name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Compress MP4 videos using H.264 encoding",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compress all videos in a folder with default settings
  python compress_existing_videos.py /path/to/videos
  
  # Use GPU acceleration (much faster!)
  python compress_existing_videos.py /path/to/videos --gpu
  
  # Use custom CRF and save to different directory
  python compress_existing_videos.py /path/to/videos --crf 28 --output-dir /path/to/compressed
  
  # GPU with higher quality, slower preset
  python compress_existing_videos.py /path/to/videos --gpu --crf 20 --preset slow
  
  # Add suffix to output filenames
  python compress_existing_videos.py /path/to/videos --suffix _compressed
        """,
    )

    parser.add_argument(
        "input_dir", type=str, help="Directory containing MP4 videos to compress"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: same as input, creates 'compressed' subfolder)",
    )

    parser.add_argument(
        "--suffix",
        type=str,
        default="_compressed",
        help="Suffix to add to compressed filenames (default: _compressed)",
    )

    parser.add_argument(
        "--crf",
        type=int,
        default=23,
        help="Constant Rate Factor: 18-28, lower=better quality (default: 23)",
    )

    parser.add_argument(
        "--preset",
        type=str,
        default="medium",
        choices=["ultrafast", "fast", "medium", "slow", "slower"],
        help="Compression preset (default: medium)",
    )

    parser.add_argument(
        "--pattern",
        type=str,
        default="*.mp4",
        help="File pattern to match (default: *.mp4)",
    )

    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing compressed files"
    )

    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use NVIDIA GPU hardware acceleration (NVENC) - much faster",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without actually compressing",
    )

    args = parser.parse_args()

    # Validate input directory
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)

    if not input_dir.is_dir():
        logger.error(f"Input path is not a directory: {input_dir}")
        sys.exit(1)

    # Set up output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = input_dir / "compressed"
        output_dir.mkdir(exist_ok=True)

    # Check GPU support if requested
    if args.gpu:
        if check_nvenc_support():
            logger.info("✓ NVIDIA NVENC hardware acceleration available")
        else:
            logger.warning("✗ NVENC not available, falling back to CPU encoding")
            logger.warning(
                "  Make sure you have NVIDIA GPU and recent ffmpeg with NVENC support"
            )
            args.gpu = False

    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Settings: CRF/CQ={args.crf}, preset={args.preset}")
    logger.info(f"Acceleration: {'GPU (NVENC)' if args.gpu else 'CPU (libx264)'}")
    logger.info(f"File pattern: {args.pattern}")

    # Find all matching video files
    video_files = sorted(input_dir.glob(args.pattern))

    if not video_files:
        logger.warning(f"No files matching '{args.pattern}' found in {input_dir}")
        sys.exit(0)

    logger.info(f"Found {len(video_files)} video(s) to process")
    print()

    # Process each video
    successful = 0
    failed = 0
    skipped = 0
    total_input_size = 0
    total_output_size = 0

    for i, video_path in enumerate(video_files, 1):
        logger.info(f"[{i}/{len(video_files)}] Processing: {video_path.name}")

        # Generate output filename
        output_name = f"{video_path.stem}{args.suffix}{video_path.suffix}"
        output_path = output_dir / output_name

        if args.dry_run:
            logger.info(f"  Would compress: {video_path} -> {output_path}")
            continue

        # Check if already exists
        if output_path.exists() and not args.overwrite:
            logger.info(f"  Skipping (already exists): {output_path}")
            skipped += 1
            continue

            # Get input size for totals,
            use_gpu = args.gpu
        input_info = get_video_info(video_path)
        total_input_size += input_info["size_bytes"]

        # Compress the video
        success = compress_video(
            video_path,
            output_path,
            crf=args.crf,
            preset=args.preset,
            overwrite=args.overwrite,
        )

        if success:
            successful += 1
            output_info = get_video_info(output_path)
            total_output_size += output_info["size_bytes"]
        else:
            failed += 1

        print()  # Blank line between videos

    # Summary
    logger.info("=" * 60)
    logger.info("COMPRESSION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total videos: {len(video_files)}")
    logger.info(f"Successfully compressed: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Skipped (already exist): {skipped}")

    if successful > 0 and not args.dry_run:
        total_input_gb = total_input_size / (1024**3)
        total_output_gb = total_output_size / (1024**3)
        total_saved_gb = total_input_gb - total_output_gb
        total_ratio = (
            (1 - total_output_size / total_input_size) * 100
            if total_input_size > 0
            else 0
        )

        logger.info(f"Total input size: {total_input_gb:.2f} GB")
        logger.info(f"Total output size: {total_output_gb:.2f} GB")
        logger.info(
            f"Total saved: {total_saved_gb:.2f} GB ({total_ratio:.1f}% reduction)"
        )

    logger.info("=" * 60)


if __name__ == "__main__":
    main()
