#!/bin/bash
# Convert F1_Tracks "ball" SLEAP .slp files to analysis .h5 (uv-native, no conda).
#
# - Removes stale "ball" .h5 files that are not the *_processed.h5 output.
# - Converts each "ball" .slp to <stem>.h5 via sleap_io in an isolated uv env
#   (uv run --with sleap-io), then renames to <stem>_processed.h5.
#
# Requires `uv` on PATH. Run: bash scripts/sleap/convert_h5.sh
set -euo pipefail

DATA_DIR="/mnt/upramdya_data/MD/F1_Tracks/Videos/"
export PATH="$HOME/.local/bin:$PATH"

# slp -> analysis .h5 using sleap_io (independent of any conda env's libraries).
convert_slp() {
    local slp="$1" out="$2"
    env -u LD_LIBRARY_PATH -u PYTHONPATH \
        uv run --no-project --with sleap-io python -c \
        'import sys, sleap_io as sio; sio.save_analysis_h5(sio.load_file(sys.argv[1]), sys.argv[2])' \
        "$slp" "$out"
}

echo "Removing stale 'ball' .h5 files (not *_processed.h5)..."
find "$DATA_DIR" -type f -name "*ball*.h5" ! -name "*_processed.h5" -exec rm -v {} \;

echo "Searching for 'ball' .slp files in $DATA_DIR..."
mapfile -t SLP_FILES < <(find "$DATA_DIR" -type f -name "*ball*.slp")

if [ "${#SLP_FILES[@]}" -eq 0 ]; then
    echo "No SLP files found for conversion."
    exit 0
fi

for SLP_FILE in "${SLP_FILES[@]}"; do
    H5_FILE="${SLP_FILE%.slp}.h5"
    PROCESSED_H5_FILE="${H5_FILE%.h5}_processed.h5"
    echo "Converting $SLP_FILE -> $PROCESSED_H5_FILE ..."
    if convert_slp "$SLP_FILE" "$H5_FILE"; then
        mv "$H5_FILE" "$PROCESSED_H5_FILE"
        echo "  done."
    else
        echo "  ERROR converting $SLP_FILE."
    fi
done

echo "All conversions completed."
