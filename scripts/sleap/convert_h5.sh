#!/bin/bash

# Define the data directory
DATA_DIR="/mnt/upramdya_data/MD/F1_Tracks/Videos/"

# Activate the Conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate sleap_dev

# Remove H5 files that have "ball" in their name and do not have "_processed" in their name
echo "Removing H5 files that have 'ball' in their name and do not have '_processed' in their name..."
find "$DATA_DIR" -type f -name "*ball*.h5" ! -name "*_processed.h5" -exec rm {} \;

# Find all SLP files that have "ball" in their name in the data directory, excluding directories that contain any H5 file with "_processed" in its name
echo "Searching for SLP files in $DATA_DIR..."
SLP_FILES=$(find "$DATA_DIR" -type f -name "*ball*.slp" ! -path "*/$(find "$DATA_DIR" -type f -name "*_processed*.h5" -exec dirname {} \; | sort -u | tr '\n' ':' | sed 's/:$//')/*")

# Check if any files were found
if [ -z "$SLP_FILES" ]; then
    echo "No SLP files found for conversion."
    exit 0
fi

# Loop through each SLP file and convert it to H5 format
for SLP_FILE in $SLP_FILES; do
    echo "Converting $SLP_FILE to .h5 format..."
    sleap-convert "$SLP_FILE" --format analysis
    if [ $? -eq 0 ]; then
        H5_FILE="${SLP_FILE%.slp}.h5"
        PROCESSED_H5_FILE="${H5_FILE%.h5}_processed.h5"
        mv "$H5_FILE" "$PROCESSED_H5_FILE"
        echo "Conversion of $SLP_FILE completed successfully and renamed to $PROCESSED_H5_FILE."
    else
        echo "Error converting $SLP_FILE."
    fi
done

echo "All conversions completed."
