from utils_behavior import Sleap_utils
from utils_behavior import HoloviewsTemplates
from utils_behavior import HoloviewsPlots
from utils_behavior import Utils
from utils_behavior import Processing
from utils_behavior import Ballpushing_utils

import importlib
from pathlib import Path
import json
from matplotlib import pyplot as plt
import pandas as pd
import holoviews as hv
from holoviews import opts
hv.extension('bokeh')
import seaborn as sns
from shiny import App, Inputs, Outputs, Session, reactive, render, req, ui
import numpy as np
import h5py
import re

# ==================================================================
# CONFIGURATION SECTION - EDIT THESE VALUES TO MODIFY BEHAVIOR
# ==================================================================
CONFIG = {
    "PATHS": {
        "data_root": Path("/mnt/upramdya_data/MD/MultiMazeRecorder/Datasets"),
        "output_summary_dir": "250220_StdContacts_Ctrl",
        "output_data_dir": "250220_StdContacts_Ctrl_Data",
    },
    "PROCESSING": {
        "experiment_filter": "TNT_Fine",    # Filter for experiment folders
        "success_cutoff": False,             # Enable success cutoff filtering
        "success_method": "final_event",     # Cutoff method selection
        "pooled_prefix": "250220_pooled",    # Base name for combined datasets
        "metrics": ["standardized_contacts"] # Metrics to process (add/remove as needed)
    }
}
# ==================================================================
# MAIN PROCESSING SCRIPT
# ==================================================================

# Build derived paths from configuration
output_summary = CONFIG["PATHS"]["data_root"] / CONFIG["PATHS"]["output_summary_dir"]
output_data = CONFIG["PATHS"]["data_root"] / CONFIG["PATHS"]["output_data_dir"]

# Create output directories
output_summary.mkdir(parents=True, exist_ok=True)
output_data.mkdir(parents=True, exist_ok=True)

# Get list of experiment folders to process
Exp_folders = [
    folder for folder in CONFIG["PATHS"]["data_root"].iterdir()
    if folder.is_dir() and CONFIG["PROCESSING"]["experiment_filter"] in folder.name
]
print(f"Folders to analyze: {[f.name for f in Exp_folders]}")

# Create metric subdirectories
for metric in CONFIG["PROCESSING"]["metrics"]:
    (output_data / metric).mkdir(exist_ok=True)

# Main processing loop
for folder in Exp_folders:
    exp_name = folder.name
    experiment_pkl_path = output_summary / f"{exp_name}.pkl"
    
    # Check if datasets already exist
    datasets_exist = all(
        (output_data / metric / f"{exp_name}_{metric}.feather").exists()
        for metric in CONFIG["PROCESSING"]["metrics"]
    )
    
    if datasets_exist:
        print(f"All datasets for {exp_name} exist. Skipping.")
        continue

    # Load or create experiment
    if experiment_pkl_path.exists():
        try:
            experiment = Ballpushing_utils.load_object(experiment_pkl_path)
            print(f"Loaded existing experiment: {exp_name}")
        except Exception as e:
            print(f"Error loading {exp_name}: {str(e)}")
            continue
    else:
        try:
            experiment = Ballpushing_utils.Experiment(
                folder,
                success_cutoff=CONFIG["PROCESSING"]["success_cutoff"],
                success_cutoff_method=CONFIG["PROCESSING"]["success_method"]
            )
            Ballpushing_utils.save_object(experiment, experiment_pkl_path)
            print(f"Created new experiment: {exp_name}")
        except Exception as e:
            print(f"Error creating {exp_name}: {str(e)}")
            continue

    # Generate datasets for each metric
    for metric in CONFIG["PROCESSING"]["metrics"]:
        dataset_path = output_data / metric / f"{exp_name}_{metric}.feather"
        
        if dataset_path.exists():
            continue
            
        try:
            dataset = Ballpushing_utils.Dataset(experiment, dataset_type=metric)
            if not dataset.data.empty:
                dataset.data.to_feather(dataset_path)
                print(f"Saved {metric} dataset for {exp_name}")
            else:
                print(f"Empty {metric} data for {exp_name}")
        except Exception as e:
            print(f"Error generating {metric} for {exp_name}: {str(e)}")
            continue

# Create pooled datasets
for metric in CONFIG["PROCESSING"]["metrics"]:
    pooled_path = output_data / metric / f"{CONFIG['PROCESSING']['pooled_prefix']}_{metric}.feather"
    
    if not pooled_path.exists():
        try:
            metric_files = list((output_data / metric).glob("*.feather"))
            if not metric_files:
                print(f"No {metric} files found for pooling")
                continue
                
            pooled_df = pd.concat([pd.read_feather(f) for f in metric_files])
            pooled_df.to_feather(pooled_path)
            print(f"Created pooled {metric} dataset: {pooled_path.name}")
        except Exception as e:
            print(f"Error pooling {metric}: {str(e)}")
    else:
        print(f"Pooled {metric} dataset already exists")

print("Processing complete!")