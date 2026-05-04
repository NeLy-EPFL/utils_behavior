"""
Test compression settings on actual video files from the dataset.
Creates 10-second samples from real videos with various configurations.

Usage:
    python test_real_video_compression.py --video /path/to/video.mp4
    python test_real_video_compression.py --auto  # Uses sample from dataset
"""

import cv2
import subprocess
import time
import logging
import argparse
from pathlib import Path
from dataclasses import dataclass
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


@dataclass
class CompressionConfig:
    name: str
    description: str
    scale_factor: float  # Multiply original resolution by this
    use_ffmpeg: bool
    ffmpeg_preset: str = "fast"
    crf: int = 23
    reduce_framerate: bool = False


# Compression configurations to test
CONFIGS = [
    CompressionConfig(
        name="original_4k_mp4v",
        description="ORIGINAL PRE-OPTIMIZATION: Full res (4K/original), mp4v codec, no compression",
        scale_factor=1.0,
        use_ffmpeg=False,
    ),
    CompressionConfig(
        name="original_1080p_mp4v",
        description="Half resolution only: 1080p, mp4v codec, no compression",
        scale_factor=0.5,
        use_ffmpeg=False,
    ),
    CompressionConfig(
        name="optimized_crf28",
        description="Optimized: Half res, H.264 CRF=28, fast preset",
        scale_factor=0.5,
        use_ffmpeg=True,
        crf=28,
        ffmpeg_preset="fast",
    ),
    CompressionConfig(
        name="optimized_crf32",
        description="Optimized: Half res, H.264 CRF=32 (smaller files)",
        scale_factor=0.5,
        use_ffmpeg=True,
        crf=32,
        ffmpeg_preset="fast",
    ),
    CompressionConfig(
        name="optimized_reduced_fps",
        description="Optimized: Half res, CRF=28, reduced FPS",
        scale_factor=0.5,
        use_ffmpeg=True,
        crf=28,
        reduce_framerate=True,
    ),
    # New modes based on feedback: need higher res for cell fitting + better quality
    CompressionConfig(
        name="4k_high_quality",
        description="NEW: Full 4K res, H.264 CRF=23 (fits more cells, good quality)",
        scale_factor=1.0,
        use_ffmpeg=True,
        crf=23,
        ffmpeg_preset="medium",
    ),
    CompressionConfig(
        name="4k_reduced_fps",
        description="NEW: Full 4K, H.264 CRF=23, 15fps (fits cells, saves space)",
        scale_factor=1.0,
        use_ffmpeg=True,
        crf=23,
        ffmpeg_preset="fast",
        reduce_framerate=True,
    ),
    CompressionConfig(
        name="1440p_high_quality",
        description="NEW: 1440p (2560x1440), H.264 CRF=23 (middle ground)",
        scale_factor=0.67,  # ~1440p from typical 4K source
        use_ffmpeg=True,
        crf=23,
        ffmpeg_preset="medium",
    ),
    CompressionConfig(
        name="1440p_reduced_fps",
        description="NEW: 1440p, H.264 CRF=23, 15fps (balanced: cells + size)",
        scale_factor=0.67,
        use_ffmpeg=True,
        crf=23,
        ffmpeg_preset="fast",
        reduce_framerate=True,
    ),
    CompressionConfig(
        name="1080p_better_quality",
        description="NEW: 1080p, H.264 CRF=20 (better quality than CRF=28)",
        scale_factor=0.5,
        use_ffmpeg=True,
        crf=20,
        ffmpeg_preset="medium",
    ),
]


def extract_sample(
    video_path: Path, start_sec: float, duration_sec: float, output_path: Path
):
    """Extract a sample from the video using ffmpeg (fastest method)."""
    cmd = [
        "ffmpeg",
        "-ss",
        str(start_sec),
        "-i",
        str(video_path),
        "-t",
        str(duration_sec),
        "-c",
        "copy",
        "-y",
        str(output_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"Failed to extract sample: {result.stderr}")
        return False

    return True


def process_video_sample(
    input_path: Path,
    output_path: Path,
    config: CompressionConfig,
    original_fps: float,
    original_size: tuple,
) -> dict:
    """Process video sample with given configuration."""

    logger.info(f"\n{'='*60}")
    logger.info(f"Testing: {config.name}")
    logger.info(f"{config.description}")
    logger.info(f"{'='*60}")

    start_time = time.time()

    # Calculate new resolution
    orig_width, orig_height = original_size
    new_width = int(orig_width * config.scale_factor)
    new_height = int(orig_height * config.scale_factor)
    # Make sure dimensions are even (required for H.264)
    new_width = new_width if new_width % 2 == 0 else new_width - 1
    new_height = new_height if new_height % 2 == 0 else new_height - 1

    # Adjust FPS
    fps = original_fps
    if config.reduce_framerate:
        fps = original_fps / 2

    if not config.use_ffmpeg:
        # Use OpenCV - read, process, write
        cap = cv2.VideoCapture(str(input_path))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (new_width, new_height))

        frame_count = 0
        frame_skip = 2 if config.reduce_framerate else 1

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if config.reduce_framerate and frame_count % frame_skip != 0:
                frame_count += 1
                continue

            if config.scale_factor != 1.0:
                frame = cv2.resize(frame, (new_width, new_height))

            out.write(frame)
            frame_count += 1

        cap.release()
        out.release()

        encoding_time = time.time() - start_time
        compress_time = 0

    else:
        # Use ffmpeg for better compression
        temp_path = output_path.parent / f"temp_{output_path.name}"

        # Build ffmpeg command
        cmd = [
            "ffmpeg",
            "-i",
            str(input_path),
            "-vf",
            f"scale={new_width}:{new_height}",
            "-c:v",
            "libx264",
            "-preset",
            config.ffmpeg_preset,
            "-crf",
            str(config.crf),
            "-pix_fmt",
            "yuv420p",
        ]

        if config.reduce_framerate:
            cmd.extend(["-r", str(fps)])

        cmd.extend(["-y", str(output_path)])

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"FFmpeg encoding failed: {result.stderr}")
            return None

        encoding_time = time.time() - start_time
        compress_time = encoding_time  # All in one step with ffmpeg

    # Get file size
    file_size_mb = output_path.stat().st_size / (1024 * 1024)

    # Calculate metrics
    cap = cv2.VideoCapture(str(output_path))
    actual_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    duration_sec = actual_frames / actual_fps if actual_fps > 0 else 0
    cap.release()

    bitrate_mbps = (file_size_mb * 8) / duration_sec if duration_sec > 0 else 0

    results = {
        "name": config.name,
        "description": config.description,
        "file_size_mb": file_size_mb,
        "encoding_time_sec": encoding_time,
        "bitrate_mbps": bitrate_mbps,
        "resolution": f"{new_width}x{new_height}",
        "fps": actual_fps,
        "duration_sec": duration_sec,
    }

    logger.info(f"\n✓ Results:")
    logger.info(f"  File size: {file_size_mb:.2f} MB")
    logger.info(f"  Encoding time: {encoding_time:.2f}s")
    logger.info(f"  Bitrate: {bitrate_mbps:.2f} Mbps")
    logger.info(f"  Resolution: {results['resolution']}")
    logger.info(f"  FPS: {actual_fps:.1f}")

    return results


def find_sample_video():
    """Try to find a video file from the dataset."""
    data_path = Path(
        "/mnt/upramdya_data/MD/F1_Tracks/Datasets/260123_16_F1_coordinates_F1_TNT_Full_Data/F1_coordinates/pooled_F1_coordinates.feather"
    )

    if not data_path.exists():
        return None

    # Load dataset to find a video path
    try:
        import pandas as pd

        df = pd.read_feather(data_path)

        if "flypath" in df.columns:
            # Get first flypath
            flypath = Path(df["flypath"].iloc[0])

            # Look for mp4 file
            if flypath.exists():
                videos = list(flypath.glob("*.mp4"))
                if videos:
                    return videos[0]
    except Exception as e:
        logger.warning(f"Could not auto-detect video: {e}")

    return None


def main():
    parser = argparse.ArgumentParser(description="Test video compression settings")
    parser.add_argument("--video", type=str, help="Path to video file to test")
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Automatically find a sample video from the dataset",
    )
    parser.add_argument(
        "--start", type=float, default=60.0, help="Start time in seconds (default: 60)"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Sample duration in seconds (default: 10)",
    )

    args = parser.parse_args()

    # Find video file
    if args.video:
        video_path = Path(args.video)
    elif args.auto:
        video_path = find_sample_video()
        if not video_path:
            logger.error("Could not auto-detect video. Please specify --video path")
            return
        logger.info(f"Auto-detected video: {video_path}")
    else:
        logger.error("Please specify --video or --auto")
        return

    if not video_path.exists():
        logger.error(f"Video not found: {video_path}")
        return

    # Create output directory
    output_dir = Path("compression_test_real_video")
    output_dir.mkdir(exist_ok=True)

    logger.info("=" * 60)
    logger.info("REAL VIDEO COMPRESSION TEST")
    logger.info("=" * 60)
    logger.info(f"Source video: {video_path.name}")
    logger.info(f"Sample: {args.start}s to {args.start + args.duration}s")
    logger.info("")

    # Get video info
    cap = cv2.VideoCapture(str(video_path))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    logger.info(f"Original video specs:")
    logger.info(f"  Resolution: {original_width}x{original_height}")
    logger.info(f"  FPS: {original_fps}")

    # Extract sample
    sample_path = output_dir / "sample_original.mp4"
    logger.info(f"\nExtracting {args.duration}s sample...")
    if not extract_sample(video_path, args.start, args.duration, sample_path):
        logger.error("Failed to extract sample")
        return

    # Run tests
    results = []
    for config in CONFIGS:
        output_path = output_dir / f"{config.name}.mp4"
        result = process_video_sample(
            sample_path,
            output_path,
            config,
            original_fps,
            (original_width, original_height),
        )
        if result:
            results.append(result)

    # Comparison
    if results:
        logger.info("\n" + "=" * 80)
        logger.info("COMPARISON SUMMARY")
        logger.info("=" * 80)

        df = pd.DataFrame(results)
        df = df.sort_values("file_size_mb")

        # Add comparison columns (compare to original full resolution)
        baseline_size = (
            df[df["name"].str.contains("original_4k")]["file_size_mb"].iloc[0]
            if any(df["name"].str.contains("original_4k"))
            else df["file_size_mb"].max()
        )
        df["vs_original_%"] = df["file_size_mb"] / baseline_size * 100
        df["savings_%"] = 100 - df["vs_original_%"]

        print("\n")
        print(
            df[
                [
                    "name",
                    "file_size_mb",
                    "savings_%",
                    "encoding_time_sec",
                    "bitrate_mbps",
                    "resolution",
                    "fps",
                ]
            ].to_string(index=False)
        )

        # Save results
        csv_path = output_dir / "real_video_comparison.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"\n✓ Results saved to: {csv_path}")

        # Recommendations
        logger.info("\n" + "=" * 80)
        logger.info("RECOMMENDATIONS")
        logger.info("=" * 80)

        # Show original baseline
        logger.info(f"\n📌 ORIGINAL PRE-OPTIMIZATION BASELINE:")
        logger.info(
            f"   File size: {baseline_size:.2f} MB (full resolution, mp4v codec)"
        )

        best_savings = df.loc[df["savings_%"].idxmax()]
        logger.info(f"\n💾 BEST COMPRESSION: {best_savings['name']}")
        logger.info(f"   File size: {best_savings['file_size_mb']:.2f} MB")
        logger.info(f"   Savings: {best_savings['savings_%']:.1f}% vs original")

        fastest = df.loc[df["encoding_time_sec"].idxmin()]
        logger.info(f"\n⚡ FASTEST: {fastest['name']}")
        logger.info(f"   Encoding time: {fastest['encoding_time_sec']:.2f}s")

        # Show current optimized performance
        optimized = df[df["name"] == "optimized_crf28"]
        if not optimized.empty:
            logger.info(f"\n✅ CURRENT OPTIMIZED SETTINGS (CRF=28):")
            logger.info(f"   File size: {optimized['file_size_mb'].iloc[0]:.2f} MB")
            logger.info(
                f"   Savings: {optimized['savings_%'].iloc[0]:.1f}% vs original"
            )
            logger.info(
                f"   Encoding time: {optimized['encoding_time_sec'].iloc[0]:.2f}s"
            )

        logger.info(f"\n✓ Test videos saved in: {output_dir.absolute()}")
        logger.info("\nYou can review the video quality by watching the files in:")
        logger.info(f"  {output_dir.absolute()}")


if __name__ == "__main__":
    main()
