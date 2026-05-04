import h5py
import os
import subprocess
from pathlib import Path

def is_h5_file_valid(file_path):
    """
    Check if an h5 file is valid and readable.

    Parameters:
    file_path (str): The path to the h5 file.

    Returns:
    bool: True if the file is valid and readable, False otherwise.
    """
    try:
        with h5py.File(file_path, 'r') as f:
            # Try to read a small part of the file to check its validity
            _ = f.keys()
        return True
    except Exception as e:
        print(f"Invalid h5 file: {file_path}. Error: {e}")
        return False
    
def regenerate_invalid_h5_files(h5_files, slp_file):
    """
    Check the validity of h5 files, remove invalid files, and regenerate them using sleap-convert.

    Parameters:
    h5_files (list): List of paths to the h5 files.
    slp_file (str): The path to the slp file.
    """
    for h5_file in h5_files:
        if not is_h5_file_valid(h5_file):
            # Remove the invalid h5 file
            os.remove(h5_file)
            print(f"Removed invalid h5 file: {h5_file}")

            # Regenerate the h5 file using sleap-convert
            command = f"sleap-convert {slp_file} --format analysis"
            try:
                subprocess.run(command, shell=True, check=True)
                print(f"Regenerated h5 file: {h5_file}")
            except subprocess.CalledProcessError as e:
                print(f"Error regenerating h5 file: {h5_file}. Error: {e}")

def generate_h5_file(slp_file):
    """
    Generate an h5 file from an slp file using sleap-convert.

    Parameters:
    slp_file (str): The path to the slp file.
    """
    command = f"sleap-convert {slp_file} --format analysis"
    try:
        subprocess.run(command, shell=True, check=True)
        print(f"Generated h5 file from slp file: {slp_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error generating h5 file from slp file: {slp_file}. Error: {e}")

# Example usage
# h5_files = ["path/to/file1.h5", "path/to/file2.h5"]
# slp_file = "path/to/file.slp"
# regenerate_invalid_h5_files(h5_files, slp_file)

DataPath = Path("/mnt/upramdya_data/MD/F1_Tracks/Videos/")

# Find all video folders in the data path; which are folders with .mp4 files. We use rglob
video_folders = [folder.parent for folder in DataPath.rglob("*.mp4")]

print(video_folders)

for folder in video_folders:
    # Check if there's a .h5 file with "ball" in the name
    ball_h5_files = list(folder.rglob("*ball*.h5"))
    
    # Check if there's a .slp file with "ball" in the name
    slp_files = list(folder.rglob("*ball*.slp"))
    
    if slp_files:
        slp_file = slp_files[0]  # Assuming there's only one .slp file per folder

        if ball_h5_files:
            for ball_h5_file in ball_h5_files:
                # Check if the h5 file is valid and readable
                if not is_h5_file_valid(ball_h5_file):
                    regenerate_invalid_h5_files([ball_h5_file], slp_file)
        else:
            # If the .h5 file is missing, generate it
            generate_h5_file(slp_file)
    else:
        print(f"No .slp file found in folder: {folder}")