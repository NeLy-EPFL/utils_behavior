from pathlib import Path

def check_video_and_slp_files(data_dir):
    data_dir = Path(data_dir)
    
    # Dictionary to store the results
    results = {}

    # Iterate through all directories and subdirectories
    for subdir in data_dir.rglob('*'):
        if subdir.is_dir():
            # Check for .mp4 files in the current directory
            mp4_files = list(subdir.glob('*.mp4'))
            
            # Check for .slp files with "ball" in their name in the current directory
            slp_files = list(subdir.glob('*ball*.slp'))
            
            # Check for .h5 files with "ball" and "processed" in their names in the current directory
            ball_h5_files = list(subdir.glob('*ball*processed*.h5'))
            
            # Check for .h5 files with "fly" in their names in the current directory
            fly_h5_files = list(subdir.glob('*fly*.h5'))
            
            # Store the result regardless of whether both are found
            results[subdir] = {
                'mp4_files': mp4_files,
                'slp_files': slp_files,
                'ball_h5_files': ball_h5_files,
                'fly_h5_files': fly_h5_files
            }
    
    return results

data_dir = "/mnt/upramdya_data/MD/F1_Tracks/Videos/"
results = check_video_and_slp_files(data_dir)

# Initialize counters
total_videos = 0
total_slp_files = 0
total_ball_h5_files = 0
total_fly_h5_files = 0
videos_without_slp = []
videos_without_ball_h5 = []
videos_without_fly_h5 = []

# Print the results and count the files
for subdir, files in results.items():
    print(f"Directory: {subdir}")
    print("MP4 Files:")
    for mp4_file in files['mp4_files']:
        print(f"  - {mp4_file}")
        total_videos += 1
    print("SLP Files:")
    for slp_file in files['slp_files']:
        print(f"  - {slp_file}")
        total_slp_files += 1
    print("Ball H5 Files:")
    for ball_h5_file in files['ball_h5_files']:
        print(f"  - {ball_h5_file}")
        total_ball_h5_files += 1
    print("Fly H5 Files:")
    for fly_h5_file in files['fly_h5_files']:
        print(f"  - {fly_h5_file}")
        total_fly_h5_files += 1
    print()

    # Check for videos without associated .slp files
    if files['mp4_files'] and not files['slp_files']:
        videos_without_slp.extend(files['mp4_files'])
    
    # Check for videos without associated ball .h5 files
    if files['mp4_files'] and not files['ball_h5_files']:
        videos_without_ball_h5.extend(files['mp4_files'])
    
    # Check for videos without associated fly .h5 files
    if files['mp4_files'] and not files['fly_h5_files']:
        videos_without_fly_h5.extend(files['mp4_files'])

# Print summary
print(f"Total videos: {total_videos}")
print(f"Total .slp files: {total_slp_files}")
print(f"Total ball .h5 files: {total_ball_h5_files}")
print(f"Total fly .h5 files: {total_fly_h5_files}")
print(f"Videos without associated .slp files: {len(videos_without_slp)}")
for video in videos_without_slp:
    print(f"  - {video}")
print(f"Videos without associated ball .h5 files: {len(videos_without_ball_h5)}")
for video in videos_without_ball_h5:
    print(f"  - {video}")
print(f"Videos without associated fly .h5 files: {len(videos_without_fly_h5)}")
for video in videos_without_fly_h5:
    print(f"  - {video}")