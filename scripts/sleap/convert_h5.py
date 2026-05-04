from pathlib import Path
import os

DataPath = Path("/mnt/upramdya_data/MD/F1_Tracks/Videos/")

# Find all video folders in the data path; which are folders with .mp4 files. We use rglob

video_folders = [folder.parent for folder in DataPath.rglob("*.mp4")]

print (video_folders)


for folder in video_folders:
    slp_files = list(folder.rglob("*.slp"))
    # In each video folder, check if there is ball tracking data (slp file with "processed" and "ball" in the name)
    for slp_file in slp_files:
        if "processed" in slp_file.name and "ball" in slp_file.name:
            print(slp_file)
            #break
        
        else:
            print("No ball tracking data found in folder", folder)
            
            # call the terminal command to convert the slp file to h5 file : sleap-convert "SLP_FILE" --format analysis
            os.system(f"sleap-convert {slp_file} --format analysis")
            
    # In each video folder, check if there is fly tracking data (slp file with "fly" in the name)
    
    for slp_file in slp_files:
        if "fly" in slp_file.name:
            print(slp_file)
            #break
        
        else:
            print("No fly tracking data found in folder", folder)
            
            # call the terminal command to convert the slp file to h5 file : sleap-convert "SLP_FILE" --format analysis
            os.system(f"sleap-convert {slp_file} --format analysis")
            
# Finally, get total number of video folders and total number of each h5 file type (ball and fly) in the data path

# Get total number of video folders
print("Total number of video folders:", len(video_folders))

# Get total number of ball h5 files
print("Total number of ball h5 files:", len(list(DataPath.rglob("*ball*.h5"))))

# Get total number of fly h5 files
print("Total number of fly h5 files:", len(list(DataPath.rglob("*fly*.h5"))))
        