#!/usr/bin/env python3
"""
Compress videos from a YAML file list while maintaining quality and size constraints.

This script processes videos listed in a YAML file and creates compressed versions
that do not exceed a specified maximum file size (default 2.5 GB).

Encoding strategy:
- Default is single-pass ABR with a tight maxrate cap. Final size lands within
  ~5% of target, which the retry-with-proportional-correction handles. This
  avoids the full extra encode "analysis" pass that doubles runtime.
- --two-pass switches to two-pass for stricter size accuracy. On GPU this uses
  NVENC's built-in -multipass fullres (one ffmpeg invocation, internal two-pass);
  on CPU it runs libx264/libx265 two-pass.

YAML file format:
---
videos:
  - /path/to/video1.mp4
  - /path/to/video2.mp4

Or with custom output paths:
---
videos:
  - input: /path/to/video1.mp4
    output: /path/to/compressed/video1_compressed.mp4

Usage:
    python compress_videos_yaml.py videos.yaml --gpu
    python compress_videos_yaml.py videos.yaml --gpu --hevc
    python compress_videos_yaml.py videos.yaml --two-pass --parallel 4
"""

import argparse
import subprocess
from pathlib import Path
import logging
import sys
from datetime import datetime
import yaml
from typing import List, Dict
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            f"video_compression_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        ),
    ],
)
logger = logging.getLogger(__name__)


def get_video_info(video_path: Path) -> dict:
    """Get comprehensive video information using ffprobe."""
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "format=duration,size,bit_rate:stream=width,height,r_frame_rate,codec_name",
            "-of",
            "json",
            str(video_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)

        format_info = data.get("format", {})
        stream_info = data.get("streams", [{}])[0]

        fps_str = stream_info.get("r_frame_rate", "30/1")
        if "/" in fps_str:
            num, den = map(int, fps_str.split("/"))
            fps = num / den if den != 0 else 30.0
        else:
            fps = float(fps_str)

        duration = float(format_info.get("duration", 0))
        size_bytes = int(format_info.get("size", video_path.stat().st_size))

        return {
            "duration": duration,
            "size_bytes": size_bytes,
            "size_gb": size_bytes / (1024**3),
            "bitrate": int(format_info.get("bit_rate", 0)),
            "width": int(stream_info.get("width", 0)),
            "height": int(stream_info.get("height", 0)),
            "fps": fps,
            "codec": stream_info.get("codec_name", "unknown"),
        }
    except Exception as e:
        logger.warning(f"Could not get full info for {video_path.name}: {e}")
        size_bytes = video_path.stat().st_size
        return {
            "duration": 0,
            "size_bytes": size_bytes,
            "size_gb": size_bytes / (1024**3),
            "bitrate": 0,
            "width": 0,
            "height": 0,
            "fps": 30.0,
            "codec": "unknown",
        }


def calculate_target_bitrate(
    duration: float,
    max_size_gb: float,
    audio_bitrate_kbps: int = 0,
    safety_margin: float = 0.90,
) -> int:
    """Calculate target video bitrate to achieve desired file size."""
    if duration <= 0:
        logger.warning("Invalid duration, using default bitrate")
        return 5000

    max_size_bits = max_size_gb * 1024 * 1024 * 1024 * 8
    total_bitrate_bps = max_size_bits / duration
    audio_bitrate_bps = audio_bitrate_kbps * 1000
    video_bitrate_bps = (total_bitrate_bps - audio_bitrate_bps) * safety_margin
    video_bitrate_kbps = int(video_bitrate_bps / 1000)

    return max(1000, min(50000, video_bitrate_kbps))


def check_nvenc_support() -> bool:
    """Check if NVIDIA NVENC hardware encoding is available."""
    try:
        cmd = ["ffmpeg", "-hide_banner", "-encoders"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return "h264_nvenc" in result.stdout
    except Exception:
        return False


def _run_ffmpeg(cmd: list, label: str) -> None:
    """Run ffmpeg, surfacing stderr on failure."""
    logger.info(f"  {label}")
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=7200)
    except subprocess.CalledProcessError as e:
        tail = (e.stderr or "").strip().splitlines()[-15:]
        raise RuntimeError(f"ffmpeg failed ({label}):\n" + "\n".join(tail)) from e


def compress_video_bitrate(
    input_path: Path,
    output_path: Path,
    target_bitrate_kbps: int,
    preset: str = "medium",
    use_gpu: bool = False,
    use_hevc: bool = False,
    tune: str = "grain",
    two_pass: bool = False,
) -> bool:
    """Compress video using bitrate-constrained encoding.

    Default is single-pass ABR with a tight maxrate cap (~5% size accuracy).
    With two_pass=True: NVENC uses -multipass fullres (one ffmpeg run, internal
    two-pass); libx264/libx265 use classic two-pass (two ffmpeg runs).
    """
    # Tight maxrate so peaks can't blow the size cap; small bufsize keeps it honest.
    maxrate_kbps = int(target_bitrate_kbps * 1.10)
    bufsize_kbps = int(target_bitrate_kbps * 1.50)
    pass_log = str(output_path.parent / f"ffmpeg2pass_{output_path.stem}")

    try:
        if use_gpu:
            nvenc_preset_map = {
                "ultrafast": "p1",
                "fast": "p2",
                "medium": "p4",
                "slow": "p5",
                "slower": "p6",
            }
            nvenc_preset = nvenc_preset_map.get(preset, "p4")
            codec = "hevc_nvenc" if use_hevc else "h264_nvenc"

            cmd = [
                "ffmpeg",
                "-hwaccel", "cuda",
                "-i", str(input_path),
                "-c:v", codec,
                "-preset", nvenc_preset,
                "-rc", "vbr",
                "-b:v", f"{target_bitrate_kbps}k",
                "-maxrate", f"{maxrate_kbps}k",
                "-bufsize", f"{bufsize_kbps}k",
            ]
            if two_pass:
                # NVENC's internal two-pass: one ffmpeg invocation, much cheaper
                # than running the full encode twice.
                cmd += ["-multipass", "fullres"]
            cmd += [
                "-movflags", "+faststart",
                "-an", "-y", str(output_path),
            ]
            label = "Encoding (NVENC two-pass)" if two_pass else "Encoding (NVENC single-pass)"
            _run_ffmpeg(cmd, label)

        elif use_hevc:
            x265_tune_map = {"film": "grain", "animation": "animation", "grain": "grain"}
            x265_tune = x265_tune_map.get(tune)
            tune_suffix = f":tune={x265_tune}" if x265_tune else ""
            vbv = (
                f":vbv-maxrate={maxrate_kbps}:vbv-bufsize={bufsize_kbps}"
            )

            if two_pass:
                _run_ffmpeg(
                    [
                        "ffmpeg", "-i", str(input_path),
                        "-c:v", "libx265",
                        "-preset", preset,
                        "-b:v", f"{target_bitrate_kbps}k",
                        "-x265-params", f"pass=1:stats={pass_log}.x265log{tune_suffix}",
                        "-threads", str(min(os.cpu_count() or 1, 16)),
                        "-an", "-f", "null", "/dev/null",
                    ],
                    "Pass 1/2: x265 stats",
                )
                _run_ffmpeg(
                    [
                        "ffmpeg", "-i", str(input_path),
                        "-c:v", "libx265",
                        "-preset", preset,
                        "-b:v", f"{target_bitrate_kbps}k",
                        "-x265-params",
                        f"pass=2:stats={pass_log}.x265log{vbv}{tune_suffix}",
                        "-threads", str(min(os.cpu_count() or 1, 16)),
                        "-movflags", "+faststart",
                        "-an", "-y", str(output_path),
                    ],
                    "Pass 2/2: x265 encode",
                )
            else:
                _run_ffmpeg(
                    [
                        "ffmpeg", "-i", str(input_path),
                        "-c:v", "libx265",
                        "-preset", preset,
                        "-b:v", f"{target_bitrate_kbps}k",
                        "-x265-params", f"{vbv.lstrip(':')}{tune_suffix}",
                        "-threads", str(min(os.cpu_count() or 1, 16)),
                        "-movflags", "+faststart",
                        "-an", "-y", str(output_path),
                    ],
                    "Encoding (x265 single-pass)",
                )

        else:
            base = [
                "ffmpeg", "-i", str(input_path),
                "-c:v", "libx264",
                "-preset", preset,
                "-b:v", f"{target_bitrate_kbps}k",
                "-maxrate", f"{maxrate_kbps}k",
                "-bufsize", f"{bufsize_kbps}k",
                "-threads", str(os.cpu_count()),
                "-tune", tune,
            ]
            if two_pass:
                _run_ffmpeg(
                    base + [
                        "-pass", "1", "-passlogfile", pass_log,
                        "-an", "-f", "null", "/dev/null",
                    ],
                    "Pass 1/2: x264 stats",
                )
                _run_ffmpeg(
                    base + [
                        "-pass", "2", "-passlogfile", pass_log,
                        "-movflags", "+faststart",
                        "-an", "-y", str(output_path),
                    ],
                    "Pass 2/2: x264 encode",
                )
            else:
                _run_ffmpeg(
                    base + [
                        "-movflags", "+faststart",
                        "-an", "-y", str(output_path),
                    ],
                    "Encoding (x264 single-pass)",
                )

        return True
    except Exception as e:
        logger.error(f"Encoding error: {e}")
        return False
    finally:
        for log_file in Path(pass_log).parent.glob(f"ffmpeg2pass_{output_path.stem}*"):
            try:
                log_file.unlink()
            except Exception:
                pass


def compress_video(
    input_path: Path,
    output_path: Path,
    max_size_gb: float = 2.5,
    preset: str = "medium",
    use_gpu: bool = False,
    use_hevc: bool = False,
    tune: str = "grain",
    overwrite: bool = False,
    two_pass: bool = False,
) -> bool:
    """
    Compress a video using bitrate-constrained encoding.
    If output exceeds the size limit, re-encodes with a proportionally reduced bitrate.
    """
    if output_path.exists() and not overwrite:
        logger.warning(f"Output already exists (use --overwrite): {output_path.name}")
        return False

    logger.info(f"Processing: {input_path.name}")

    input_info = get_video_info(input_path)
    logger.info(
        f"  Input: {input_info['size_gb']:.2f} GB, "
        f"{input_info['width']}x{input_info['height']}, "
        f"{input_info['duration']/60:.1f} min, "
        f"{input_info['fps']:.1f} fps"
    )

    # Abort if video is unreadable (corrupt file, missing moov atom, etc.)
    if input_info["duration"] == 0:
        logger.error(
            f"  \u2717 Cannot encode: duration is 0 (corrupt or unreadable file), skipping"
        )
        return False

    target_bitrate = calculate_target_bitrate(input_info["duration"], max_size_gb)
    logger.info(
        f"  Target bitrate: {target_bitrate/1000:.1f} Mbps (max {max_size_gb} GB)"
    )

    if not compress_video_bitrate(
        input_path, output_path, target_bitrate, preset, use_gpu, use_hevc, tune, two_pass
    ):
        return False

    # Verify output - STRICT size enforcement
    output_size_gb = output_path.stat().st_size / (1024**3)
    logger.info(f"  Output: {output_size_gb:.2f} GB")

    if output_size_gb > max_size_gb:
        excess_mb = (output_size_gb - max_size_gb) * 1000
        logger.warning(
            f"  ⚠ Exceeds {max_size_gb} GB by {excess_mb:.0f} MB - re-encoding..."
        )

        # Proportional correction based on actual overshoot
        actual_ratio = output_size_gb / max_size_gb
        corrected_bitrate = int(target_bitrate / actual_ratio * 0.92)
        logger.info(f"  Corrected bitrate: {corrected_bitrate/1000:.1f} Mbps")

        output_path.unlink()

        if not compress_video_bitrate(
            input_path, output_path, corrected_bitrate, preset, use_gpu, use_hevc, tune, two_pass
        ):
            logger.error(f"  ✗ Re-encoding failed")
            return False

        output_size_gb = output_path.stat().st_size / (1024**3)
        logger.info(f"  Final output: {output_size_gb:.2f} GB")

        if output_size_gb > max_size_gb:
            logger.error(
                f"  ✗ Still exceeds limit (by {(output_size_gb - max_size_gb)*1000:.0f} MB)"
            )
            logger.error(f"  This video may need manual processing")
            return False

    ratio = (1 - output_size_gb / input_info["size_gb"]) * 100
    saved_gb = input_info["size_gb"] - output_size_gb
    logger.info(f"  Compression: {ratio:.1f}% reduction ({saved_gb:.2f} GB saved)")
    logger.info(f"✓ Successfully compressed: {output_path.name}")

    return True


def load_video_list(yaml_path: Path) -> List[Dict[str, str]]:
    """Load video list from YAML file."""
    try:
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)

        if not data or "videos" not in data:
            raise ValueError("YAML file must contain a 'videos' key")

        videos = []
        for item in data["videos"]:
            if isinstance(item, str):
                videos.append({"input": item, "output": None})
            elif isinstance(item, dict):
                if "input" not in item:
                    raise ValueError(f"Video entry must have 'input' key: {item}")
                videos.append(
                    {"input": item["input"], "output": item.get("output", None)}
                )
            else:
                raise ValueError(f"Invalid video entry format: {item}")

        logger.info(f"Loaded {len(videos)} videos from {yaml_path.name}")
        return videos
    except Exception as e:
        logger.error(f"Error loading YAML file: {e}")
        raise


def process_video_task(task_args):
    """Worker function for parallel processing."""
    (
        index,
        input_path,
        output_path,
        max_size_gb,
        preset,
        use_gpu,
        use_hevc,
        tune,
        overwrite,
        two_pass,
    ) = task_args

    # Re-initialize logging in worker process (forked file handlers may not work correctly)
    logging.getLogger().handlers = []
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logger.info(f"\n[{index}]")
    success = compress_video(
        input_path, output_path, max_size_gb, preset, use_gpu, use_hevc, tune, overwrite, two_pass
    )
    return (index, success, input_path.name)


def main():
    parser = argparse.ArgumentParser(
        description="Compress videos from YAML list with quality and size constraints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compress videos listed in YAML file (max 2.5 GB each)
  python compress_videos_yaml.py videos.yaml --gpu
  
  # Use H.265 for better compression
  python compress_videos_yaml.py videos.yaml --hevc --gpu
  
  # Parallel processing (4 workers for CPU, 2 for GPU)
  python compress_videos_yaml.py videos.yaml --parallel 4
  
  # Microscopy/screen recording optimization
  python compress_videos_yaml.py videos.yaml --tune animation --gpu
        """,
    )

    parser.add_argument(
        "yaml_file", type=str, help="YAML file containing list of videos"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: same as input)",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="_compressed",
        help="Suffix to add to filenames (default: _compressed)",
    )
    parser.add_argument(
        "--max-size",
        type=float,
        default=2.5,
        help="Maximum output file size in GB (default: 2.5)",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="medium",
        choices=["ultrafast", "fast", "medium", "slow", "slower"],
        help="Compression preset (default: medium for speed)",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use NVIDIA GPU hardware acceleration (NVENC)",
    )
    parser.add_argument(
        "--hevc",
        action="store_true",
        help="Use H.265/HEVC (40-50%% smaller, slower encode)",
    )
    parser.add_argument(
        "--tune",
        type=str,
        default="grain",
        choices=["film", "animation", "grain", "stillimage", "fastdecode"],
        help="FFmpeg tune for content type (default: grain, suited for high-detail/noisy sources)",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        metavar="N",
        help="Process N videos in parallel (default: 1)",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing compressed files"
    )
    parser.add_argument(
        "--two-pass",
        action="store_true",
        help="Use two-pass encoding for stricter size accuracy (slower). "
        "Default is single-pass with a tight maxrate cap, which is ~2x faster "
        "and lands within ~5%% of target.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without compressing",
    )

    args = parser.parse_args()

    yaml_path = Path(args.yaml_file)
    if not yaml_path.exists():
        logger.error(f"YAML file not found: {yaml_path}")
        sys.exit(1)

    try:
        video_list = load_video_list(yaml_path)
    except Exception as e:
        logger.error(f"Failed to load video list: {e}")
        sys.exit(1)

    if args.hevc and args.tune in ("stillimage", "fastdecode"):
        logger.warning(f"--tune {args.tune} is not supported with --hevc, ignoring")

    if args.gpu:
        if check_nvenc_support():
            logger.info("✓ NVIDIA NVENC hardware acceleration available")
        else:
            logger.warning("✗ NVENC not available, falling back to CPU")
            args.gpu = False

    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Validate parallel settings
    max_workers = args.parallel
    if max_workers > 1:
        if args.gpu:
            max_workers = min(max_workers, 2)
            logger.info(f"GPU mode: limiting to {max_workers} parallel workers")
        else:
            recommended = max(1, os.cpu_count() // 2)
            if max_workers > recommended:
                logger.warning(
                    f"--parallel {max_workers} may cause thrashing (recommended: {recommended})"
                )

    codec_info = "H.265/HEVC" if args.hevc else "H.264"
    pass_info = "two-pass" if args.two_pass else "single-pass"
    logger.info(f"\nCompressing {len(video_list)} videos")
    logger.info(f"Codec: {codec_info} ({pass_info}), Preset: {args.preset}, Tune: {args.tune}")
    logger.info(f"Max size: {args.max_size} GB, Parallel: {max_workers}")
    logger.info("-" * 80)

    successful = 0
    failed = 0
    skipped = 0

    tasks = []
    for i, video_entry in enumerate(video_list, 1):
        input_path = Path(video_entry["input"])

        if not input_path.exists():
            logger.error(f"✗ Input not found: {input_path}")
            failed += 1
            continue

        if video_entry["output"]:
            output_path = Path(video_entry["output"])
        elif args.output_dir:
            output_path = (
                Path(args.output_dir)
                / f"{input_path.stem}{args.suffix}{input_path.suffix}"
            )
        else:
            output_path = (
                input_path.parent / f"{input_path.stem}{args.suffix}{input_path.suffix}"
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Skip if already processed (fast file size check)
        task_overwrite = args.overwrite
        if output_path.exists() and not args.overwrite:
            output_size_gb = output_path.stat().st_size / (1024**3)
            if output_size_gb <= args.max_size:
                logger.info(
                    f"⏭ Skipping (done, {output_size_gb:.2f} GB): {input_path.name}"
                )
                skipped += 1
                continue
            else:
                logger.info(
                    f"♻ Re-processing (was {output_size_gb:.2f} GB): {input_path.name}"
                )
                task_overwrite = True  # oversized output must be replaced

        if args.dry_run:
            logger.info(
                f"[{i}/{len(video_list)}] Would compress: {input_path} → {output_path}"
            )
            continue

        tasks.append(
            (
                i,
                input_path,
                output_path,
                args.max_size,
                args.preset,
                args.gpu,
                args.hevc,
                args.tune,
                task_overwrite,
                args.two_pass,
            )
        )

    if args.dry_run:
        logger.info(f"\nDry run complete. Would process {len(tasks)} videos.")
        sys.exit(0)

    # Process videos
    if max_workers == 1:
        for task in tasks:
            index, success, name = process_video_task(task)
            if success:
                successful += 1
            else:
                logger.error(f"✗ Failed: {name}")
                failed += 1
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_video_task, task): task for task in tasks
            }
            for future in as_completed(futures):
                try:
                    index, success, name = future.result()
                    if success:
                        successful += 1
                    else:
                        logger.error(f"✗ Failed: {name}")
                        failed += 1
                except Exception as e:
                    task = futures[future]
                    logger.error(f"✗ Error: {task[1].name} - {e}")
                    failed += 1

    logger.info("\n" + "=" * 80)
    logger.info("COMPRESSION SUMMARY")
    logger.info("=" * 80)
    logger.info(
        f"Total: {len(video_list)} | Successful: {successful} | Skipped: {skipped} | Failed: {failed}"
    )
    logger.info("=" * 80)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
