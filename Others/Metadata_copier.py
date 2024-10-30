import shutil
from pathlib import Path

data_path = Path("/mnt/upramdya_data/MD/F1_Tracks/Videos/")
metadata_path = Path("/mnt/upramdya_data/MD/MultiMazeRecorder/Videos")

# Iterate over each folder in data_path
for data_folder in data_path.iterdir():
    if data_folder.is_dir():
        # Find the corresponding folder in metadata_path
        metadata_folder = metadata_path / data_folder.name
        
        # Check if the metadata folder exists and contains Metadata.json
        metadata_file = metadata_folder / "Metadata.json"
        if metadata_file.exists():
            # Copy the Metadata.json file to the data folder
            shutil.copy(metadata_file, data_folder / "Metadata.json")
            print(f"Copied Metadata.json to {data_folder}")
        else:
            print(f"Metadata.json not found in {metadata_folder}")