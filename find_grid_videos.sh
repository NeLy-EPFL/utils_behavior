#!/bin/bash
# Find all MP4 files with 'grid' in their name and create a YAML file
# Usage: ./find_grid_videos.sh /mnt/upramdya_data/MD [output.yaml]

if [ $# -lt 1 ]; then
    echo "Usage: $0 <directory> [output_file.yaml]"
    echo "Example: $0 /mnt/upramdya_data/MD grid_videos.yaml"
    exit 1
fi

SEARCH_DIR="$1"
OUTPUT_FILE="${2:-grid_videos.yaml}"

if [ ! -d "$SEARCH_DIR" ]; then
    echo "Error: Directory not found: $SEARCH_DIR"
    exit 1
fi

echo "Searching for MP4 files with 'grid' in name in: $SEARCH_DIR"
echo "This may take a while for large directories..."

# Find all matching videos (case-insensitive)
VIDEOS=$(find "$SEARCH_DIR" -type f -iname "*grid*.mp4" 2>/dev/null | sort)

# Count videos
VIDEO_COUNT=$(echo "$VIDEOS" | grep -c .)

if [ -z "$VIDEOS" ] || [ "$VIDEO_COUNT" -eq 0 ]; then
    echo "No videos found matching '*grid*.mp4'"
    exit 0
fi

echo "Found $VIDEO_COUNT videos"

# Create YAML file
echo "videos:" > "$OUTPUT_FILE"

# Add each video path to YAML
echo "$VIDEOS" | while read -r VIDEO_PATH; do
    echo "  - $VIDEO_PATH" >> "$OUTPUT_FILE"
done

echo ""
echo "✓ Created YAML file: $OUTPUT_FILE"
echo ""
echo "Video list preview (first 10):"
head -n 11 "$OUTPUT_FILE"
if [ "$VIDEO_COUNT" -gt 10 ]; then
    echo "  ... and $((VIDEO_COUNT - 10)) more"
fi

echo ""
echo "To compress these videos, run:"
echo "  python utils_behavior/compress_videos_yaml.py $OUTPUT_FILE"
echo "  python utils_behavior/compress_videos_yaml.py $OUTPUT_FILE --gpu"
echo "  python utils_behavior/compress_videos_yaml.py $OUTPUT_FILE --max-size 2.5 --gpu"
