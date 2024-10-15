#!/bin/bash

# Define the data directory
DATA_DIR="/mnt/upramdya_data/MD/F1_Tracks/Videos/"

# Activate the Conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate sleap_dev

# Find all SLP files that have "ball" in their name in the data directory
SLP_FILES=$(find "$DATA_DIR" -type f -name "*ball*.slp")

# Loop through each SLP file and convert it to H5 format
for SLP_FILE in $SLP_FILES; do
    echo "Converting $SLP_FILE to .h5 format..."
    sleap-convert "$SLP_FILE" --format analysis
    if [ $? -eq 0 ]; then
        echo "Conversion of $SLP_FILE completed successfully."
    else
        echo "Error converting $SLP_FILE."
    fi
done

echo "All conversions completed."
