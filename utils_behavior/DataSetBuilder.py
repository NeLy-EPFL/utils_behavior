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

hv.extension("bokeh")
import seaborn as sns
from shiny import App, Inputs, Outputs, Session, reactive, render, req, ui
import numpy as np
import h5py
import re

import gc
import psutil
import os

# ==================================================================
# CONFIGURATION SECTION - EDIT THESE VALUES TO MODIFY BEHAVIOR
# ==================================================================
CONFIG = {
    "PATHS": {
        "data_root": Path("/mnt/upramdya_data/MD/MultiMazeRecorder/Videos/"),
        "dataset_dir": Path("/mnt/upramdya_data/MD/BallPushing_Learning/Datasets/"),
        "output_summary_dir": "250326_StdContacts_Ctrl_300frames",
        "output_data_dir": "250326_StdContacts_Ctrl_300frames_Data",
        "excluded_folders": [],
        "config_path": "config.json",
    },
    "PROCESSING": {
        "experiment_filter": "Learning",  # Filter for experiment folders
        "pooled_prefix": "250326_pooled",  # Base name for combined datasets
        "metrics": [
            "standardized_contacts"
        ],  # Metrics to process (add/remove as needed)
    },
}


def config_to_dict(config):
    return {
        "experiment_type": config.experiment_type,
        "time_range": config.time_range,
        "success_cutoff": config.success_cutoff,
        "success_cutoff_method": config.success_cutoff_method,
        "tracks_smoothing": config.tracks_smoothing,
        "log_missing": config.log_missing,
        "log_path": str(config.log_path),
        "keep_idle": config.keep_idle,
        "downsampling_factor": config.downsampling_factor,
        "random_exclude_interactions": config.random_exclude_interactions,
        "random_interaction_map": config.random_interaction_map,
        "interaction_threshold": config.interaction_threshold,
        "gap_between_events": config.gap_between_events,
        "events_min_length": config.events_min_length,
        "frames_before_onset": config.frames_before_onset,
        "frames_after_onset": config.frames_after_onset,
        "dead_threshold": config.dead_threshold,
        "adjusted_events_normalisation": config.adjusted_events_normalisation,
        "significant_threshold": config.significant_threshold,
        "aha_moment_threshold": config.aha_moment_threshold,
        "success_direction_threshold": config.success_direction_threshold,
        "final_event_threshold": config.final_event_threshold,
        "final_event_F1_threshold": config.final_event_F1_threshold,
        "max_event_threshold": config.max_event_threshold,
        "template_width": config.template_width,
        "template_height": config.template_height,
        "padding": config.padding,
        "y_crop": config.y_crop,
        "contact_nodes": config.contact_nodes,
        "contact_threshold": config.contact_threshold,
        "gap_between_contacts": config.gap_between_contacts,
        "contact_min_length": config.contact_min_length,
        "skeleton_tracks_smoothing": config.skeleton_tracks_smoothing,
        "hidden_value": config.hidden_value,
    }


def log_memory_usage(label):
    """Log current memory usage"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    print(f"Memory usage at {label}: {memory_info.rss / 1024 / 1024:.2f} MB")


# ==================================================================
# MAIN PROCESSING SCRIPT
# ==================================================================

# Build derived paths from configuration
output_summary = CONFIG["PATHS"]["dataset_dir"] / CONFIG["PATHS"]["output_summary_dir"]
output_data = CONFIG["PATHS"]["dataset_dir"] / CONFIG["PATHS"]["output_data_dir"]

# Create output directories
output_summary.mkdir(parents=True, exist_ok=True)
output_data.mkdir(parents=True, exist_ok=True)

# Get list of experiment folders to process
Exp_folders = [
    folder
    for folder in CONFIG["PATHS"]["data_root"].iterdir()
    if folder.is_dir() and CONFIG["PROCESSING"]["experiment_filter"] in folder.name
]

# Exclude folders based on config list
if CONFIG["PATHS"]["excluded_folders"]:
    Exp_folders = [
        folder
        for folder in Exp_folders
        if folder.name not in CONFIG["PATHS"]["excluded_folders"]
    ]

print(f"Folders to analyze: {[f.name for f in Exp_folders]}")

# Create metric subdirectories
for metric in CONFIG["PROCESSING"]["metrics"]:
    (output_data / metric).mkdir(exist_ok=True)

# Main processing loop
checkpoint_file = output_summary / "processing_checkpoint.json"

# Load checkpoint if exists
if checkpoint_file.exists():
    with open(checkpoint_file, "r") as f:
        processed_folders = set(json.load(f))
    print(
        f"Resuming from checkpoint, {len(processed_folders)} folders already processed"
    )
else:
    processed_folders = set()

# Process each folder
for folder in Exp_folders:
    exp_name = folder.name
    experiment_pkl_path = output_summary / f"{exp_name}.pkl"

    # Skip if already processed
    if exp_name in processed_folders:
        print(f"Skipping already processed experiment: {exp_name}")
        continue

    log_memory_usage(f"Before processing {exp_name}")

    # Load or create experiment
    try:
        if experiment_pkl_path.exists():
            experiment = Ballpushing_utils.load_object(experiment_pkl_path)
            print(f"Loaded existing experiment: {exp_name}")
        else:
            # Process one experiment completely
            experiment = Ballpushing_utils.Experiment(folder)
            Ballpushing_utils.save_object(experiment, experiment_pkl_path)
            print(f"Created new experiment: {exp_name}")

            # Save config if first experiment
            if not (output_summary / CONFIG["PATHS"]["config_path"]).exists():
                config_dict = config_to_dict(experiment.config)
                config_json_path = output_summary / CONFIG["PATHS"]["config_path"]
                with open(config_json_path, "w") as config_file:
                    json.dump(config_dict, config_file, indent=4)
                print(f"Saved config for {exp_name} to {config_json_path}")

        # Generate and save datasets
        all_metrics_processed = True
        for metric in CONFIG["PROCESSING"]["metrics"]:
            dataset_path = output_data / metric / f"{exp_name}_{metric}.feather"

            if dataset_path.exists():
                print(f"Dataset {dataset_path} already exists. Skipping.")
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
                all_metrics_processed = False

        # Mark as processed only if all metrics were processed
        if all_metrics_processed:
            processed_folders.add(exp_name)
            # Save checkpoint after each successful experiment
            with open(checkpoint_file, "w") as f:
                json.dump(list(processed_folders), f)

    except Exception as e:
        print(f"Error processing {exp_name}: {str(e)}")

    # Explicit cleanup
    del experiment
    gc.collect()
    log_memory_usage(f"After processing {exp_name}")

# Create pooled datasets
for metric in CONFIG["PROCESSING"]["metrics"]:
    pooled_path = (
        output_data
        / metric
        / f"{CONFIG['PROCESSING']['pooled_prefix']}_{metric}.feather"
    )

    if not pooled_path.exists():
        try:
            metric_files = list((output_data / metric).glob("*.feather"))
            if not metric_files:
                print(f"No {metric} files found for pooling")
                continue

            # Use chunking for pooling to avoid loading all files at once
            chunk_size = 5  # Adjust based on file sizes
            total_chunks = (len(metric_files) + chunk_size - 1) // chunk_size

            first_chunk = True
            for chunk_idx in range(total_chunks):
                start_idx = chunk_idx * chunk_size
                end_idx = min((chunk_idx + 1) * chunk_size, len(metric_files))
                chunk_files = metric_files[start_idx:end_idx]

                print(f"Processing chunk {chunk_idx+1}/{total_chunks} for {metric}")
                chunk_df = pd.concat([pd.read_feather(f) for f in chunk_files])

                if first_chunk:
                    # First chunk: create the file
                    chunk_df.to_feather(pooled_path)
                    first_chunk = False
                else:
                    # Subsequent chunks: append to existing
                    existing_df = pd.read_feather(pooled_path)
                    combined_df = pd.concat([existing_df, chunk_df])
                    combined_df.to_feather(pooled_path)
                    del existing_df, combined_df

                del chunk_df
                gc.collect()
                log_memory_usage(f"After pooling chunk {chunk_idx+1} for {metric}")

            print(f"Created pooled {metric} dataset: {pooled_path.name}")
        except Exception as e:
            print(f"Error pooling {metric}: {str(e)}")
    else:
        print(f"Pooled {metric} dataset already exists")

print("Processing complete!")
