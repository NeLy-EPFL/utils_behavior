#!/usr/bin/env python3
"""
Find all MP4 files containing 'grid' in their name and create a YAML list.

Usage:
    python find_grid_videos.py /mnt/upramdya_data/MD
    python find_grid_videos.py /mnt/upramdya_data/MD --output grid_videos.yaml
    python find_grid_videos.py /mnt/upramdya_data/MD --pattern "*grid*.mp4" --recursive
"""

import argparse
from pathlib import Path
import yaml
import sys


def find_videos(
    directory: Path,
    pattern: str = "*grid*.mp4",
    recursive: bool = True,
    case_sensitive: bool = False
) -> list:
    """
    Find all video files matching the pattern.
    
    Args:
        directory: Directory to search
        pattern: Glob pattern to match (default: *grid*.mp4)
        recursive: Search recursively in subdirectories
        case_sensitive: Whether pattern matching is case-sensitive
        
    Returns:
        List of absolute paths to matching videos
    """
    videos = []
    
    if recursive:
        # Recursive search
        for video_path in directory.rglob(pattern):
            if video_path.is_file():
                videos.append(str(video_path.absolute()))
        
        # If not case-sensitive, also search for uppercase variants
        if not case_sensitive:
            for video_path in directory.rglob(pattern.replace('grid', 'Grid')):
                if video_path.is_file():
                    path_str = str(video_path.absolute())
                    if path_str not in videos:
                        videos.append(path_str)
            for video_path in directory.rglob(pattern.replace('grid', 'GRID')):
                if video_path.is_file():
                    path_str = str(video_path.absolute())
                    if path_str not in videos:
                        videos.append(path_str)
    else:
        # Non-recursive search
        for video_path in directory.glob(pattern):
            if video_path.is_file():
                videos.append(str(video_path.absolute()))
        
        # If not case-sensitive, also search for uppercase variants
        if not case_sensitive:
            for video_path in directory.glob(pattern.replace('grid', 'Grid')):
                if video_path.is_file():
                    path_str = str(video_path.absolute())
                    if path_str not in videos:
                        videos.append(path_str)
            for video_path in directory.glob(pattern.replace('grid', 'GRID')):
                if video_path.is_file():
                    path_str = str(video_path.absolute())
                    if path_str not in videos:
                        videos.append(path_str)
    
    return sorted(videos)


def main():
    parser = argparse.ArgumentParser(
        description="Find MP4 files with 'grid' in name and create YAML list",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Find all grid videos in directory (recursive)
  python find_grid_videos.py /mnt/upramdya_data/MD
  
  # Save to specific output file
  python find_grid_videos.py /mnt/upramdya_data/MD --output my_videos.yaml
  
  # Non-recursive search (only in specified directory)
  python find_grid_videos.py /mnt/upramdya_data/MD --no-recursive
  
  # Custom pattern
  python find_grid_videos.py /mnt/upramdya_data/MD --pattern "*Grid*.mp4"
  
  # Case-sensitive search
  python find_grid_videos.py /mnt/upramdya_data/MD --case-sensitive
        """
    )
    
    parser.add_argument(
        "directory",
        type=str,
        help="Directory to search for videos"
    )
    
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="grid_videos.yaml",
        help="Output YAML file (default: grid_videos.yaml)"
    )
    
    parser.add_argument(
        "--pattern",
        type=str,
        default="*grid*.mp4",
        help="Glob pattern to match (default: *grid*.mp4)"
    )
    
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Don't search recursively in subdirectories"
    )
    
    parser.add_argument(
        "--case-sensitive",
        action="store_true",
        help="Case-sensitive pattern matching"
    )
    
    args = parser.parse_args()
    
    # Validate directory
    directory = Path(args.directory)
    if not directory.exists():
        print(f"Error: Directory not found: {directory}", file=sys.stderr)
        sys.exit(1)
    
    if not directory.is_dir():
        print(f"Error: Not a directory: {directory}", file=sys.stderr)
        sys.exit(1)
    
    # Find videos
    print(f"Searching for videos matching '{args.pattern}' in {directory}")
    if not args.no_recursive:
        print("  (searching recursively in subdirectories)")
    
    videos = find_videos(
        directory,
        pattern=args.pattern,
        recursive=not args.no_recursive,
        case_sensitive=args.case_sensitive
    )
    
    if not videos:
        print(f"\nNo videos found matching pattern: {args.pattern}")
        sys.exit(0)
    
    print(f"\nFound {len(videos)} videos:")
    for i, video in enumerate(videos, 1):
        # Show relative path if possible, otherwise absolute
        try:
            rel_path = Path(video).relative_to(directory)
            print(f"  {i:3d}. {rel_path}")
        except ValueError:
            print(f"  {i:3d}. {video}")
    
    # Create YAML structure
    yaml_data = {
        'videos': videos
    }
    
    # Write YAML file
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)
    
    print(f"\n✓ Created YAML file: {output_path}")
    print(f"\nTo compress these videos, run:")
    print(f"  python utils_behavior/compress_videos_yaml.py {output_path}")
    print(f"  python utils_behavior/compress_videos_yaml.py {output_path} --gpu")
    print(f"  python utils_behavior/compress_videos_yaml.py {output_path} --max-size 2.5 --gpu")


if __name__ == "__main__":
    main()
