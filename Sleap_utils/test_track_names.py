#!/usr/bin/env python3
"""
Test script to check if track names are being extracted from H5 files.
"""

import sys
sys.path.append('/home/matthias/utils_behavior')

from utils_behavior.Sleap_utils import Sleap_Tracks
import h5py
from pathlib import Path

def check_h5_file_contents(h5_file_path):
    """Check what's inside an H5 file."""
    print(f"\n=== Checking H5 file contents: {h5_file_path} ===")
    
    try:
        with h5py.File(h5_file_path, "r") as f:
            print("Available keys:", list(f.keys()))
            
            if "track_names" in f:
                track_names = [x.decode("utf-8") for x in f["track_names"][:]]
                print("Track names found:", track_names)
            else:
                print("No track_names key found")
                
            if "track_occupancy" in f:
                print("Track occupancy shape:", f["track_occupancy"].shape)
            else:
                print("No track_occupancy key found")
                
            print("Number of tracks:", len(f["tracks"][:]))
            
            # Show first few track points to understand structure
            tracks = f["tracks"][:]
            if len(tracks) > 0:
                print(f"First track shape: {tracks[0][0].shape}")
                print(f"Sample track data shape: {len(tracks)} tracks")
            
    except Exception as e:
        print(f"Error reading H5 file: {e}")

def test_sleap_tracks_with_track_names(h5_file_path):
    """Test the Sleap_Tracks class with track name extraction."""
    print(f"\n=== Testing Sleap_Tracks class with: {h5_file_path} ===")
    
    try:
        # Load with debug to see what's happening
        sleap_tracks = Sleap_Tracks(h5_file_path, object_type="ball", debug=True)
        
        print(f"\nResults:")
        print(f"Track names attribute: {sleap_tracks.track_names}")
        print(f"Track occupancy attribute: {sleap_tracks.track_occupancy}")
        print(f"Number of objects: {len(sleap_tracks.objects)}")
        
        # Show object naming
        for i, obj in enumerate(sleap_tracks.objects):
            sample_row = obj.dataset.iloc[0] if len(obj.dataset) > 0 else None
            if sample_row is not None:
                print(f"Object {i+1}: {sample_row['object']}, track_id: {sample_row.get('track_id', 'N/A')}")
        
    except Exception as e:
        print(f"Error testing Sleap_Tracks: {e}")

if __name__ == "__main__":
    # Test with the processed ball files
    test_files = [
        "/mnt/upramdya_data/MD/F1_Tracks/Videos/250904_F1_New_Videos_Checked/arena1/Right/Right_tracked_ball_processed.000_Right.analysis.h5",
        "/mnt/upramdya_data/MD/F1_Tracks/Videos/250904_F1_New_Videos_Checked/arena1/Left/Left_tracked_ball_processed.000_Left.analysis.h5"
    ]
    
    for test_file in test_files:
        if Path(test_file).exists():
            check_h5_file_contents(test_file)
            test_sleap_tracks_with_track_names(test_file)
        else:
            print(f"File not found: {test_file}")
