"""
Test script to compare different video compression settings.
Creates 10-second samples with various configurations and reports:
- File size
- Encoding time
- Compression ratio

Usage:
    python test_compression_settings.py
"""

import cv2
import numpy as np
import subprocess
import time
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


@dataclass
class CompressionTest:
    name: str
    description: str
    output_resolution: tuple  # (width, height)
    codec: str  # For OpenCV VideoWriter
    use_ffmpeg: bool
    ffmpeg_preset: str = "fast"
    crf: int = 23
    reduce_framerate: bool = False


# Define test configurations
TESTS = [
    CompressionTest(
        name="original_4k_mp4v",
        description="ORIGINAL PRE-OPTIMIZATION: 4K (3840x2160), mp4v codec, no compression",
        output_resolution=(3840, 2160),
        codec="mp4v",
        use_ffmpeg=False,
    ),
    CompressionTest(
        name="original_1080p_mp4v",
        description="Half resolution only: 1080p, mp4v codec, no compression",
        output_resolution=(1920, 1080),
        codec="mp4v",
        use_ffmpeg=False,
    ),
    CompressionTest(
        name="current_default",
        description="Current optimized (1080p, ffmpeg H.264, CRF=28)",
        output_resolution=(1920, 1080),
        codec="MJPG",  # Temp codec
        use_ffmpeg=True,
        crf=28,
        ffmpeg_preset="fast",
    ),
    CompressionTest(
        name="high_quality",
        description="High quality (1080p, ffmpeg H.264, CRF=23)",
        output_resolution=(1920, 1080),
        codec="MJPG",
        use_ffmpeg=True,
        crf=23,
        ffmpeg_preset="medium",
    ),
    CompressionTest(
        name="small_files",
        description="Small files (1080p, ffmpeg H.264, CRF=32)",
        output_resolution=(1920, 1080),
        codec="MJPG",
        use_ffmpeg=True,
        crf=32,
        ffmpeg_preset="fast",
    ),
    CompressionTest(
        name="reduced_fps",
        description="Reduced FPS (1080p, 15fps, CRF=28)",
        output_resolution=(1920, 1080),
        codec="MJPG",
        use_ffmpeg=True,
        crf=28,
        ffmpeg_preset="fast",
        reduce_framerate=True,
    ),
    # New modes based on feedback: need higher res for cell fitting + better quality
    CompressionTest(
        name="4k_high_quality",
        description="NEW: 4K resolution, H.264 CRF=23 (fits more cells, good quality)",
        output_resolution=(3840, 2160),
        codec="MJPG",
        use_ffmpeg=True,
        crf=23,
        ffmpeg_preset="medium",
    ),
    CompressionTest(
        name="4k_reduced_fps",
        description="NEW: 4K resolution, H.264 CRF=23, 15fps (fits cells, saves space)",
        output_resolution=(3840, 2160),
        codec="MJPG",
        use_ffmpeg=True,
        crf=23,
        ffmpeg_preset="fast",
        reduce_framerate=True,
    ),
    CompressionTest(
        name="1440p_high_quality",
        description="NEW: 1440p (2560x1440), H.264 CRF=23 (middle ground)",
        output_resolution=(2560, 1440),
        codec="MJPG",
        use_ffmpeg=True,
        crf=23,
        ffmpeg_preset="medium",
    ),
    CompressionTest(
        name="1440p_reduced_fps",
        description="NEW: 1440p, H.264 CRF=23, 15fps (balanced: cells + size)",
        output_resolution=(2560, 1440),
        codec="MJPG",
        use_ffmpeg=True,
        crf=23,
        ffmpeg_preset="fast",
        reduce_framerate=True,
    ),
    CompressionTest(
        name="1080p_better_quality",
        description="NEW: 1080p, H.264 CRF=20 (better quality than CRF=28)",
        output_resolution=(1920, 1080),
        codec="MJPG",
        use_ffmpeg=True,
        crf=20,
        ffmpeg_preset="medium",
    ),
]


def create_test_frames(num_frames: int, resolution: tuple, fps: float) -> list:
    """Generate synthetic test frames with movement and text."""
    width, height = resolution
    frames = []

    logger.info(f"Generating {num_frames} test frames at {width}x{height}...")

    for i in range(num_frames):
        # Create a base frame with gradient
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Add moving gradient
        progress = i / num_frames
        gradient = np.linspace(0, 255, width, dtype=np.uint8)
        frame[:, :, 0] = gradient
        frame[:, :, 1] = int(255 * progress)
        frame[:, :, 2] = 255 - gradient

        # Add moving circle
        center_x = int(width * progress)
        center_y = height // 2
        cv2.circle(frame, (center_x, center_y), 100, (255, 255, 255), -1)

        # Add frame number
        cv2.putText(
            frame,
            f"Frame {i}/{num_frames}",
            (50, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (0, 0, 0),
            3,
        )

        # Add timestamp
        timestamp = f"Time: {i/fps:.2f}s"
        cv2.putText(
            frame, timestamp, (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2
        )

        frames.append(frame)

    return frames


def write_video_opencv(
    frames: list, output_path: Path, fps: float, codec: str, resolution: tuple
) -> float:
    """Write frames using OpenCV VideoWriter."""
    start_time = time.time()

    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(str(output_path), fourcc, fps, resolution)

    for frame in frames:
        # Resize if needed
        if frame.shape[1] != resolution[0] or frame.shape[0] != resolution[1]:
            frame = cv2.resize(frame, resolution)
        out.write(frame)

    out.release()

    return time.time() - start_time


def compress_with_ffmpeg(
    input_path: Path, output_path: Path, crf: int, preset: str
) -> float:
    """Compress video using ffmpeg."""
    start_time = time.time()

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
        "-y",
        str(output_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"FFmpeg failed: {result.stderr}")
        return -1

    # Delete temp file
    input_path.unlink()

    return time.time() - start_time


def run_compression_test(
    test: CompressionTest, frames: list, original_fps: float, output_dir: Path
) -> dict:
    """Run a single compression test and return results."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing: {test.name}")
    logger.info(f"Description: {test.description}")
    logger.info(f"{'='*60}")

    # Adjust FPS if needed
    fps = original_fps
    test_frames = frames
    if test.reduce_framerate:
        fps = original_fps / 2
        test_frames = frames[::2]  # Skip every other frame
        logger.info(f"Reduced FPS: {original_fps} -> {fps}")

    # Create output path
    if test.use_ffmpeg:
        temp_path = output_dir / f"{test.name}_temp.avi"
        final_path = output_dir / f"{test.name}.mp4"
    else:
        final_path = output_dir / f"{test.name}.mp4"
        temp_path = None

    # Write video
    logger.info(f"Writing {len(test_frames)} frames at {test.output_resolution}...")
    write_time = write_video_opencv(
        test_frames,
        temp_path if test.use_ffmpeg else final_path,
        fps,
        test.codec,
        test.output_resolution,
    )

    # Compress if needed
    compress_time = 0
    if test.use_ffmpeg:
        logger.info(
            f"Compressing with ffmpeg (CRF={test.crf}, preset={test.ffmpeg_preset})..."
        )
        compress_time = compress_with_ffmpeg(
            temp_path, final_path, test.crf, test.ffmpeg_preset
        )
        if compress_time < 0:
            return None

    # Get file size
    file_size_mb = final_path.stat().st_size / (1024 * 1024)
    total_time = write_time + compress_time

    # Calculate metrics
    duration_sec = len(test_frames) / fps
    bitrate_mbps = (file_size_mb * 8) / duration_sec
    pixels = test.output_resolution[0] * test.output_resolution[1]
    megapixels = pixels / 1_000_000

    results = {
        "name": test.name,
        "description": test.description,
        "file_size_mb": file_size_mb,
        "write_time_sec": write_time,
        "compress_time_sec": compress_time,
        "total_time_sec": total_time,
        "bitrate_mbps": bitrate_mbps,
        "resolution": f"{test.output_resolution[0]}x{test.output_resolution[1]}",
        "megapixels": megapixels,
        "fps": fps,
    }

    logger.info(f"\n✓ Results:")
    logger.info(f"  File size: {file_size_mb:.2f} MB")
    logger.info(f"  Write time: {write_time:.2f}s")
    if compress_time > 0:
        logger.info(f"  Compress time: {compress_time:.2f}s")
    logger.info(f"  Total time: {total_time:.2f}s")
    logger.info(f"  Bitrate: {bitrate_mbps:.2f} Mbps")
    logger.info(f"  Resolution: {results['resolution']} ({megapixels:.1f} MP)")
    logger.info(f"  FPS: {fps}")

    return results


def main():
    """Run all compression tests and compare results."""
    output_dir = Path("compression_test_results")
    output_dir.mkdir(exist_ok=True)

    logger.info("=" * 60)
    logger.info("VIDEO COMPRESSION COMPARISON TEST")
    logger.info("=" * 60)
    logger.info("This test will create 10-second sample videos with different")
    logger.info("compression settings and compare file sizes and encoding times.")
    logger.info("")

    # Test parameters
    TEST_DURATION = 10  # seconds
    ORIGINAL_FPS = 30.0
    BASE_RESOLUTION = (1920, 1080)

    num_frames = int(TEST_DURATION * ORIGINAL_FPS)

    # Generate test frames
    frames = create_test_frames(num_frames, BASE_RESOLUTION, ORIGINAL_FPS)

    # Run all tests
    results = []
    for test in TESTS:
        result = run_compression_test(test, frames, ORIGINAL_FPS, output_dir)
        if result:
            results.append(result)

    # Create comparison table
    if results:
        logger.info("\n" + "=" * 80)
        logger.info("COMPARISON SUMMARY")
        logger.info("=" * 80)

        df = pd.DataFrame(results)

        # Sort by file size
        df = df.sort_values("file_size_mb")

        # Add relative columns (compare to original 4K)
        baseline = (
            df[df["name"] == "original_4k_mp4v"]["file_size_mb"].iloc[0]
            if "original_4k_mp4v" in df["name"].values
            else df["file_size_mb"].max()
        )
        df["size_vs_original"] = df["file_size_mb"] / baseline
        df["savings_%"] = (1 - df["size_vs_original"]) * 100

        # Display table
        print("\n")
        print(df.to_string(index=False))

        # Save to CSV
        csv_path = output_dir / "comparison_results.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"\n✓ Results saved to: {csv_path}")

        # Print recommendations
        logger.info("\n" + "=" * 80)
        logger.info("RECOMMENDATIONS")
        logger.info("=" * 80)

        smallest = df.iloc[0]
        logger.info(f"\n📦 SMALLEST FILE: {smallest['name']}")
        logger.info(f"   {smallest['description']}")
        logger.info(
            f"   Size: {smallest['file_size_mb']:.2f} MB ({smallest['savings_%']:.1f}% savings vs original 4K)"
        )

        fastest = df.loc[df["total_time_sec"].idxmin()]
        logger.info(f"\n⚡ FASTEST ENCODING: {fastest['name']}")
        logger.info(f"   {fastest['description']}")
        logger.info(f"   Total time: {fastest['total_time_sec']:.2f}s")

        best_balance = df.loc[(df["file_size_mb"] * df["total_time_sec"]).idxmin()]
        logger.info(f"\n⚖️  BEST BALANCE: {best_balance['name']}")
        logger.info(f"   {best_balance['description']}")
        logger.info(
            f"   Size: {best_balance['file_size_mb']:.2f} MB, Time: {best_balance['total_time_sec']:.2f}s"
        )

        # Show comparison with original
        logger.info(f"\n📊 COMPARISON WITH ORIGINAL (4K, mp4v):")
        logger.info(f"   Original size: {baseline:.2f} MB")
        current_default = df[df["name"] == "current_default"]
        if not current_default.empty:
            logger.info(
                f"   Current optimized: {current_default['file_size_mb'].iloc[0]:.2f} MB ({current_default['savings_%'].iloc[0]:.1f}% smaller)"
            )

        logger.info(f"\n✓ Test videos saved in: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
