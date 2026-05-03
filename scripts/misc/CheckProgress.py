from pathlib import Path
import yaml


# Load the directory list from the YAML file
def load_directories_from_yaml(yaml_file):
    with open(yaml_file, "r") as file:
        directories = yaml.safe_load(file).get("directories", [])
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
                # print(f"Found preprocessed video: {video}")
                preprocessed_videos += 1
    return (
        preprocessed_videos,
        total_videos,
        (
            preprocessed_videos / (total_videos - preprocessed_videos)
            if total_videos - preprocessed_videos != 0
            else 0
        ),
    )


# Example usage
yaml_file = Path("/home/durrieu/sleap_tools/magnet_block_folders.yaml")

directories = load_directories_from_yaml(yaml_file)
preprocessed_videos, total_videos, ratio = count_preprocessed_videos(directories)

print(f"Total videos to process: {total_videos - preprocessed_videos}")
print(f"Preprocessed videos: {preprocessed_videos}")
print(f"Ratio of preprocessed videos : {ratio}")

# Also check how many slp files with _full_body in their name are in the directories


def count_full_body_slp_files(directories):
    full_body_slp_files = 0
    for directory in directories:
        for slp_file in directory.rglob("*_full*.slp"):
            print(f"Found full body slp file: {slp_file}")
            full_body_slp_files += 1
    return full_body_slp_files


full_body_slp_files = count_full_body_slp_files(directories)
print(f"Full body slp files: {full_body_slp_files}")

# Find folders that contain a _preprocessed video but no _full_body slp file


# Find folders that contain a _preprocessed video but no _full_body slp file
def find_folders_with_preprocessed_video_but_no_full_body_slp(directories):
    folders = []
    for directory in directories:
        for subdir in directory.rglob("*"):
            if subdir.is_dir():
                preprocessed_videos = list(subdir.glob("*_preprocessed*.mp4"))
                full_body_slp_files = list(subdir.glob("*_full*.slp"))
                if preprocessed_videos and not full_body_slp_files:
                    print(
                        f"Found {subdir} with preprocessed video but no full body slp"
                    )

                    # Remove the preprocessed video
                    # for video in preprocessed_videos:
                    #     print(f"Removing {video}")
                    #     #video.unlink()
                    folders.append(subdir)
    return folders


folders_with_preprocessed_video_but_no_full_body_slp = (
    find_folders_with_preprocessed_video_but_no_full_body_slp(directories)
)

print(
    f"Folders missing a slp file but with preprocessed folder:{folders_with_preprocessed_video_but_no_full_body_slp}"
)


def find_last_folders(directories):
    last_folders = []
    for directory in directories:
        for subdir in directory.iterdir():
            if subdir.is_dir():
                # Check if the subdir contains any subdirectories
                subdirs = [d for d in subdir.iterdir() if d.is_dir()]
                if not subdirs:
                    # If there are no subdirectories, check for video and SLP files
                    video_files = list(subdir.glob("*.mp4"))
                    slp_files = list(subdir.glob("*.slp"))
                    if video_files or slp_files:
                        last_folders.append(subdir)
                else:
                    # Recursively check the subdirectories
                    last_folders.extend(find_last_folders([subdir]))
    return last_folders


# Find folders that contain more than one _preprocessed video
def find_folders_with_multiple_preprocessed_videos(directories):
    folders = []
    for directory in directories:
        preprocessed_videos = list(directory.glob("*_preprocessed*.mp4"))
        if len(preprocessed_videos) > 1:
            folders.append(directory)
    return folders


# Find folders that are missing a _full_body slp file
def find_folders_missing_full_body_slp(directories):
    folders = []
    for directory in directories:
        full_body_slp_files = list(directory.glob("*full*.slp"))
        if not full_body_slp_files:
            folders.append(directory)
    return folders


# Find the last folders
last_folders = find_last_folders(directories)

# Perform the checks in the last folders
folders_with_multiple_preprocessed_videos = (
    find_folders_with_multiple_preprocessed_videos(last_folders)
)

# For each folder with multiple preprocessed videos, find the videos that have "annotated" in their name and remove them


def remove_annotated_videos(folders):
    for folder in folders:
        annotated_videos = list(folder.glob("*annotated*.mp4"))
        for video in annotated_videos:
            print(f"Removing {video}")
            video.unlink()


# remove_annotated_videos(folders_with_multiple_preprocessed_videos)

folders_missing_full_body_slp = find_folders_missing_full_body_slp(last_folders)

print(
    f"Folders with multiple preprocessed videos: {folders_with_multiple_preprocessed_videos}"
)
# print(f"Folders missing full body slp files: {folders_missing_full_body_slp}")

# Find folders that contain more than one .slp file with "full" in their name


# def find_folders_with_multiple_full_slp_files(directories):
#     folders = []
#     for directory in directories:
#         full_slp_files = list(directory.rglob("*full*.slp"))
#         if len(full_slp_files) > 1:
#             folders.append(directory)
#     return folders


# folders_with_multiple_full_slp_files = find_folders_with_multiple_full_slp_files(
#     directories
# )

# print(f"Folders with multiple full slp files: {folders_with_multiple_full_slp_files}")

# Now find .h5 and .slp files that have "full" but not "preprocessed" in their name


# def find_files_with_full_but_not_preprocessed(directories):
#     files = []
#     for directory in directories:
#         for file in directory.rglob("*full*.h5"):
#             if "preprocessed" not in file.stem:
#                 # Remove the file
#                 print(f"Removing {file}")
#                 file.unlink()
#         for file in directory.rglob("*full*.slp"):
#             if "preprocessed" not in file.stem:
#                 print(f"Removing {file}")
#                 file.unlink()
#     return files


# files_with_full_but_not_preprocessed = find_files_with_full_but_not_preprocessed(
#     directories
# )

# print(f"Files with full but not preprocessed: {files_with_full_but_not_preprocessed}")
