from CheckProgress import load_directories_from_yaml

from utils_behavior import Utils

from pathlib import Path

yaml_file = Path("/home/durrieu/sleap_tools/folders_to_process.yaml")

directories = load_directories_from_yaml(yaml_file)


data_path = Utils.get_data_path()

tnt_folders = [
    folder
    for folder in data_path.iterdir()
    if folder.is_dir() and "TNT_Fine" in folder.name
]

tnt_folders

# Find any directory that is in one list but not in the other

missing_folders = [folder for folder in tnt_folders if folder not in directories]

print(f"Missing folders: {missing_folders}")
