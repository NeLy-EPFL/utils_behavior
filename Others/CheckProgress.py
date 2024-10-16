from pathlib import Path
import yaml


# Load the directory list from the YAML file
def load_directories_from_yaml(yaml_file):
    with open(yaml_file, 'r') as file:
        directories = yaml.safe_load(file).get('directories', [])
        if directories:
            print(f"Loaded {len(directories)} directories from YAML file.")
            return [Path(d) for d in directories]
    return []

# Find how many videos in the directories have _preprocessed in their name (preprocessed videos) and compute the ratio of preprocessed videos to (total videos - preprocessed videos)

def count_preprocessed_videos(directories):
    total_videos = 0
    preprocessed_videos = 0
    for directory in directories:
        for video in directory.rglob("*.mp4"):
            total_videos += 1
            if "_preprocessed" in video.stem:
                preprocessed_videos += 1
    return preprocessed_videos, total_videos, preprocessed_videos / (total_videos - preprocessed_videos) if total_videos - preprocessed_videos != 0 else 0

# Example usage
yaml_file = Path("/home/durrieu/sleap_tools/folders_to_process.yaml")

directories = load_directories_from_yaml(yaml_file)
preprocessed_videos, total_videos, ratio = count_preprocessed_videos(directories)

print(f"Total videos to process: {total_videos - preprocessed_videos}")
print(f"Preprocessed videos: {preprocessed_videos}")
print(f"Ratio of preprocessed videos : {ratio}")

# Also check how many slp files with _full_body in their name are in the directories

def count_full_body_slp_files(directories):
    full_body_slp_files = 0
    for directory in directories:
        for slp_file in directory.rglob("*_full_body*.slp"):
            full_body_slp_files += 1
    return full_body_slp_files

full_body_slp_files = count_full_body_slp_files(directories)
print(f"Full body slp files: {full_body_slp_files}")