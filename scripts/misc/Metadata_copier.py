import shutil
from pathlib import Path

data_path = Path("/mnt/upramdya_data/MD/F1_Tracks/Videos/")
metadata_path = Path("/mnt/upramdya_data/MD/MultiMazeRecorder/Videos")

# Iterate over each folder in data_path
for data_folder in data_path.iterdir():
    if data_folder.is_dir():
        # Find the corresponding folder in metadata_path
        metadata_folder = metadata_path / data_folder.name
        
        # List of files to copy
        files_to_copy = ["Metadata.json", "fps.npy", "duration.npy"]
        
        for file_name in files_to_copy:
            # Check if the file exists in the metadata folder
            metadata_file = metadata_folder / file_name
            if metadata_file.exists():
                # Copy the file to the data folder
                shutil.copy(metadata_file, data_folder / file_name)
                print(f"Copied {file_name} to {data_folder}")
            else:
                print(f"{file_name} not found in {metadata_folder}")