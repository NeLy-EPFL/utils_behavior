import pandas as pd
import numpy as np
import pyarrow.feather as feather
from scipy.fft import fft
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import time
import logging
from pathlib import Path
import json
from tqdm import tqdm
import pyarrow
import pyarrow.ipc as ipc
import gc
import psutil
import os

# Configuration section
config = {
    "input_path": "/mnt/upramdya_data/MD/BallPushing_Learning/Datasets/250326_StdContacts_Ctrl_300frames_Data/standardized_contacts/250326_pooled_standardized_contacts.feather",
    "output_path": "/mnt/upramdya_data/MD/BallPushing_Learning/Datasets/250326_StdContacts_Ctrl_300frames_Data/Transformed/250313_pooled_standardized_contacts_Transformed.feather",
    "features": [
        "derivatives",
        "relative_positions",
        "statistical_measures",
        "fourier",
        "frame_features",
        "keypoints",
    ],
    "n_jobs": min(os.cpu_count(), 8),
    "frames_per_event": None,
    "batch_size": 100000,
    "cleanup_temp_files": True,
    "retry_failed": True,
}

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# Memory monitoring function
def check_memory(threshold=80):
    """Monitor memory usage and trigger GC if above threshold percent"""
    memory_percent = psutil.virtual_memory().percent
    if memory_percent > threshold:
        gc.collect()
        return True
    return False


def get_optimal_batch_size(default_size=10000, min_size=1000):
    """Determine optimal batch size based on available system memory"""
    # Get available memory in GB
    available_memory_gb = psutil.virtual_memory().available / (1024**3)

    # Conservative approach: adjust batch size based on available memory
    # Rough estimate: ~10MB per 1000 rows (adjust based on your specific data)
    memory_based_size = int(available_memory_gb * 100000)  # ~100K rows per GB

    # More conservative when running over SSH
    if "SSH_CONNECTION" in os.environ:
        memory_based_size = memory_based_size // 2

    # Constrain between min_size and default_size
    optimal_size = max(min_size, min(default_size, memory_based_size))
    logging.info(
        f"Optimal batch size determined: {optimal_size} rows (available memory: {available_memory_gb:.2f} GB)"
    )
    return optimal_size


def batch_load_data(input_path):
    """Load data in batches using PyArrow IPC reader with a fixed chunk strategy"""
    import pyarrow.ipc as ipc

    with tqdm(desc="Loading data batches") as pbar:
        with ipc.open_file(input_path) as reader:
            num_batches = reader.num_record_batches
            pbar.total = num_batches

            for i in range(num_batches):
                batch = reader.get_batch(i)
                df_batch = batch.to_pandas()
                yield df_batch
                check_memory()  # Still monitor memory and GC if needed
                pbar.update(1)


# Adaptive worker count based on available resources
def get_optimal_workers():
    """Determine optimal worker count based on system resources"""
    cpu_count = os.cpu_count()
    memory_gb = psutil.virtual_memory().total / (1024**3)

    # More conservative when running over SSH
    if "SSH_CONNECTION" in os.environ:
        return min(cpu_count // 2, max(1, int(memory_gb // 4)))
    else:
        return min(cpu_count, max(1, int(memory_gb // 2)))


def cleanup_batch_files(output_dir):
    """Delete temporary batch files after successful concatenation"""
    batch_files = [
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if f.startswith("batch_")
    ]

    logging.info(f"Cleaning up {len(batch_files)} temporary batch files")
    for file in tqdm(batch_files, desc="Cleaning up batch files"):
        os.remove(file)
    logging.info("Cleanup complete")


num_cores = os.cpu_count()
n_jobs = min(num_cores, 10)  # Limit to 4 CPU cores to avoid crashing the computer


def save_config(config, savepath):
    """Save configuration settings to a JSON file."""
    config_path = Path(savepath).with_suffix(".json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"Configuration saved to {config_path}")


def transpose_keypoints(data, frames_per_event=None):
    """Transpose keypoints with proper X/Y coordinate pairing."""
    x_fly_pattern = r"^x_.*_fly$"
    y_fly_pattern = r"^y_.*_fly$"
    centre_pattern = r"^[xy]_centre_preprocessed$"

    keypoint_columns = data.filter(
        regex=f"({x_fly_pattern}|{y_fly_pattern}|{centre_pattern})"
    ).columns
    transposed = {}

    data = data.assign(
        frame_count=data.groupby(["fly", "event_id", "adjusted_frame"]).cumcount()
    )

    # Filter frames by the frames_per_event parameter if provided
    if frames_per_event:
        unique_frames = sorted(data["adjusted_frame"].unique())
        allowed_frames = unique_frames[:frames_per_event]
        data = data[data["adjusted_frame"].isin(allowed_frames)]

    for keypoint in keypoint_columns:
        parts = keypoint.split("_")
        axis = parts[0]

        if "fly" in parts:
            keypoint_name = "_".join(parts[1:-1])
        elif "preprocessed" in parts:
            keypoint_name = "_".join(parts[1:-1])

        keypoint_df = data.pivot(
            index=["fly", "event_id"],
            columns=["adjusted_frame", "frame_count"],
            values=keypoint,
        )

        keypoint_df.columns = [
            f"{keypoint_name}_frame{frame}_{axis}" for (frame, _) in keypoint_df.columns
        ]

        transposed.update(keypoint_df.iloc[0].to_dict())

    return transposed


def calculate_frame_features(group, frames_per_event=None):
    """Calculate per-frame velocity and angles using original coordinates."""
    features = {}
    group = group.sort_values("adjusted_frame").reset_index(drop=True)

    # Filter frames by the frames_per_event parameter if provided
    if frames_per_event:
        unique_frames = sorted(group["adjusted_frame"].unique())
        allowed_frames = unique_frames[:frames_per_event]
        group = group[group["adjusted_frame"].isin(allowed_frames)]

    # Now only process the allowed frames
    available_frames = group["adjusted_frame"].unique()

    base_keypoints = {
        "Head",
        "Thorax",
        "Abdomen",
        "Rfront",
        "Lfront",
        "Rmid",
        "Lmid",
        "Rhind",
        "Lhind",
        "Rwing",
        "Lwing",
        "centre",
    }

    for kp in base_keypoints:
        x_col = f"x_{kp}"
        y_col = f"y_{kp}"

        if x_col not in group.columns or y_col not in group.columns:
            continue

        x_vals = group[x_col].values
        y_vals = group[y_col].values

        dx = np.diff(x_vals, prepend=x_vals[0])
        dy = np.diff(y_vals, prepend=y_vals[0])

        velocities = np.hypot(dx, dy)
        angles = np.degrees(np.arctan2(dx, dy)) % 360
        angular_velocity = np.diff(angles, prepend=angles[0])

        for i, frame in enumerate(group["adjusted_frame"]):
            features[f"{kp}_frame{frame}_velocity"] = velocities[i]
            features[f"{kp}_frame{frame}_angle"] = angles[i]
            features[f"{kp}_frame{frame}_angular_velocity"] = angular_velocity[i]

    return features


def calculate_derivatives(group, keypoint_columns, fly_relative=True):
    """Calculate velocity and acceleration derivatives."""
    if fly_relative:
        fly_relative_columns = [col for col in keypoint_columns if "_fly" in col]
    else:
        fly_relative_columns = keypoint_columns
    velocities = group[fly_relative_columns].diff()
    accelerations = velocities.diff()
    return (
        {f"{col}_vel_mean": velocities[col].mean() for col in fly_relative_columns}
        | {f"{col}_vel_std": velocities[col].std() for col in fly_relative_columns}
        | {f"{col}_acc_mean": accelerations[col].mean() for col in fly_relative_columns}
        | {f"{col}_acc_std": accelerations[col].std() for col in fly_relative_columns}
    )


def calculate_relative_positions(group, keypoint_columns, fly_start_map):
    """Calculate relative positions and distances."""
    fly = group["fly"].iloc[0]
    overall_start = fly_start_map[fly]

    initial_x = group["x_centre_preprocessed"].iloc[0]
    initial_y = group["y_centre_preprocessed"].iloc[0]
    group = group.copy()
    group.loc[:, "euclidean_distance"] = np.sqrt(
        (group["x_centre_preprocessed"] - initial_x) ** 2
        + (group["y_centre_preprocessed"] - initial_y) ** 2
    )

    initial_positions = group[keypoint_columns].iloc[0]
    displacements = group[keypoint_columns] - initial_positions

    median_euclidean_distance = group["euclidean_distance"].median()

    initial_distance = group["euclidean_distance"].iloc[0]
    final_distance = group["euclidean_distance"].iloc[-1]
    direction = 1 if final_distance > initial_distance else -1

    final_x = group["x_centre_preprocessed"].iloc[-1]
    final_y = group["y_centre_preprocessed"].iloc[-1]
    raw_displacement = np.sqrt((final_x - initial_x) ** 2 + (final_y - initial_y) ** 2)

    contact_start_x = group["x_centre_preprocessed"].iloc[0]
    contact_start_y = group["y_centre_preprocessed"].iloc[0]
    contact_end_x = group["x_centre_preprocessed"].iloc[-1]
    contact_end_y = group["y_centre_preprocessed"].iloc[-1]

    start_distance = np.sqrt(
        (contact_start_x - overall_start["overall_x_start"]) ** 2
        + (contact_start_y - overall_start["overall_y_start"]) ** 2
    )

    end_distance = np.sqrt(
        (contact_end_x - overall_start["overall_x_start"]) ** 2
        + (contact_end_y - overall_start["overall_y_start"]) ** 2
    )

    return (
        {f"{col}_disp_mean": displacements[col].mean() for col in keypoint_columns}
        | {f"{col}_disp_std": displacements[col].std() for col in keypoint_columns}
        | {
            "median_euclidean_distance": median_euclidean_distance,
            "direction": direction,
            "raw_displacement": raw_displacement,
            "start_distance": start_distance,
            "end_distance": end_distance,
        }
    )


def calculate_statistical_measures(group, keypoint_columns, fly_relative=True):
    """Calculate statistical measures for keypoints."""
    if fly_relative:
        fly_relative_columns = [col for col in keypoint_columns if "_fly" in col]
    else:
        fly_relative_columns = keypoint_columns
    return (
        {f"{col}_mean": group[col].mean() for col in fly_relative_columns}
        | {f"{col}_std": group[col].std() for col in fly_relative_columns}
        | {f"{col}_skew": group[col].skew() for col in fly_relative_columns}
        | {f"{col}_kurt": group[col].kurtosis() for col in fly_relative_columns}
    )


def calculate_fourier_features(group, keypoint_columns, fly_relative=True):
    """Calculate Fourier features for keypoints."""
    if fly_relative:
        fly_relative_columns = [col for col in keypoint_columns if "_fly" in col]
    else:
        fly_relative_columns = keypoint_columns
    fft_results = {}
    for col in fly_relative_columns:
        fft_vals = fft(group[col].values)
        dominant_freq = np.abs(fft_vals[1 : len(fft_vals) // 2]).argmax() + 1
        fft_results[f"{col}_dom_freq"] = dominant_freq
        fft_results[f"{col}_dom_freq_magnitude"] = np.abs(fft_vals[dominant_freq])
    return fft_results


def calculate_tsfresh_features(group, keypoint_columns, n_jobs):
    """Calculate TSFresh features for keypoints."""
    tsfresh_data = group[keypoint_columns].copy()
    tsfresh_data["id"] = 0  # Single id for the group
    tsfresh_data["frame"] = group["frame"].values

    extracted_features = extract_features(
        tsfresh_data,
        column_id="id",
        column_sort="frame",
        default_fc_parameters=ComprehensiveFCParameters(),
        n_jobs=n_jobs,
    )
    return extracted_features.iloc[0].to_dict()


def get_fly_initial_positions(data):
    """Get first recorded position for each fly in entire dataset."""
    return (
        data.groupby("fly")[["x_centre_preprocessed", "y_centre_preprocessed"]]
        .first()
        .rename(
            columns={
                "x_centre_preprocessed": "overall_x_start",
                "y_centre_preprocessed": "overall_y_start",
            }
        )
        .reset_index()
    )


def process_group(
    fly,
    event_id,
    event_type,
    group,
    features,
    keypoint_columns,
    metadata_columns,
    fly_start_map,
    n_jobs,
    frames_per_event=None,
):
    """Process a single group of data."""
    if frames_per_event:
        group = group.iloc[:frames_per_event]

    duration = group["frame"].max() - group["frame"].min() + 1
    metadata = group[metadata_columns].iloc[0]
    row = {
        "duration": duration,
        "fly": fly,
        "event_id": event_id,
        "event_type": event_type if "event_type" in group.columns else None,
        "start": group["time"].iloc[0],
        "end": group["time"].iloc[-1],
        "start_frame": group["frame"].iloc[0],
        "end_frame": group["frame"].iloc[-1],
    }
    if "derivatives" in features:
        row.update(calculate_derivatives(group, keypoint_columns))
    if "relative_positions" in features:
        row.update(calculate_relative_positions(group, keypoint_columns, fly_start_map))
    if "statistical_measures" in features:
        row.update(calculate_statistical_measures(group, keypoint_columns))
    if "fourier" in features:
        row.update(calculate_fourier_features(group, keypoint_columns))
    if "frame_features" in features:
        row.update(calculate_frame_features(group, frames_per_event=frames_per_event))
    if "keypoints" in features:
        row.update(transpose_keypoints(group, frames_per_event=frames_per_event))
    if "tsfresh" in features:
        row.update(calculate_tsfresh_features(group, keypoint_columns, n_jobs=n_jobs))
    row.update(metadata.to_dict())
    return row


def transform_data_batch(
    data_batch,
    features,
    n_jobs=num_cores,
    output_dir=None,
    frames_per_event=None,
    processed_groups=None,
):
    """Process a batch of data with error tracking"""
    keypoint_columns = data_batch.filter(regex="^(x|y)_").columns
    all_metadata_columns = [
        "flypath",
        "experiment",
        "event_type",
        "Nickname",
        "Brain region",
        "Date",
        "Genotype",
        "Period",
        "FeedingState",
        "Orientation",
        "Light",
        "Crossing",
        "event_id",
        "trial",
    ]

    metadata_columns = [
        col for col in all_metadata_columns if col in data_batch.columns
    ]
    fly_initial_positions = get_fly_initial_positions(data_batch)
    fly_start_map = fly_initial_positions.set_index("fly")[
        ["overall_x_start", "overall_y_start"]
    ].to_dict("index")

    # Group data
    groups = (
        list(data_batch.groupby(["fly", "event_id", "event_type"]))
        if "event_type" in data_batch.columns
        else list(data_batch.groupby(["fly", "event_id"]))
    )

    batch_data = []
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = {}  # Map futures to their group keys for error tracking
        for group_key, group in groups:
            # Generate a consistent group ID
            group_id = "_".join(str(k) for k in group_key)

            if processed_groups and group_id in processed_groups:
                continue

            future = executor.submit(
                process_group,
                *group_key,
                group,
                features,
                keypoint_columns,
                metadata_columns,
                fly_start_map,
                n_jobs,
                frames_per_event,
            )
            futures[future] = group_key

        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Processing groups"
        ):
            try:
                batch_data.append(future.result())
                # Garbage collect periodically
                if len(batch_data) % 50 == 0:
                    gc.collect()
            except Exception as e:
                # Get the group key for the failed future
                group_key = futures[future]
                group_id = "_".join(str(k) for k in group_key)

                # Log the error
                error_msg = str(e)
                logging.error(f"Error processing group {group_id}: {error_msg}")

                # Track the failed group
                track_failed_group(output_dir, *group_key, error_msg)

    if batch_data:
        batch_df = pd.DataFrame(batch_data)
        batch_filename = os.path.join(output_dir, f"batch_{time.time()}.feather")
        feather.write_feather(batch_df, batch_filename)
        logging.info(f"Saved batch to {batch_filename}")

        # Update processed groups
        if processed_groups is not None:
            with open(os.path.join(output_dir, "processed_groups.txt"), "a") as f:
                for group_key, _ in groups:
                    group_id = "_".join(str(k) for k in group_key)
                    if group_id not in processed_groups:
                        f.write(f"{group_id}\n")


def concatenate_batches(output_dir, output_path, cleanup=True):
    """Concatenate all batch files into a single DataFrame with progress bar."""
    batch_files = [
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if f.startswith("batch_")
    ]

    logging.info(f"Concatenating {len(batch_files)} batch files")
    batch_dfs = []

    for batch_file in tqdm(batch_files, desc="Reading batch files"):
        batch_dfs.append(pd.read_feather(batch_file))

    concatenated_df = pd.concat(batch_dfs, ignore_index=True)

    logging.info(f"Writing concatenated data to {output_path}")
    feather.write_feather(concatenated_df, output_path)
    logging.info(f"Concatenated all batches and saved to {output_path}")

    # Cleanup temporary files if requested
    if cleanup:
        cleanup_batch_files(output_dir)


def track_failed_group(output_dir, fly, event_id, event_type, error_msg):
    """Record failed group processing in an error log"""
    error_file = os.path.join(output_dir, "failed_groups.txt")
    with open(error_file, "a") as f:
        f.write(f"{fly}_{event_id}_{event_type}|{error_msg}\n")


def load_failed_groups(output_dir):
    """Load list of failed groups from error log"""
    failed_groups = {}
    error_file = os.path.join(output_dir, "failed_groups.txt")

    if os.path.exists(error_file):
        with open(error_file, "r") as f:
            for line in f:
                parts = line.strip().split("|", 1)
                if len(parts) == 2:
                    group_key, error_msg = parts
                    failed_groups[group_key] = error_msg

    return failed_groups


def retry_failed_groups(
    input_path, output_dir, features, n_jobs, frames_per_event=None
):
    """Retry processing previously failed groups"""
    failed_groups = load_failed_groups(output_dir)

    if not failed_groups:
        logging.info("No failed groups to retry")
        return 0

    logging.info(f"Attempting to retry {len(failed_groups)} failed groups")

    # Load the data for these specific groups
    all_data = pd.read_feather(input_path)

    retried_count = 0
    batch_data = []

    for group_key in tqdm(failed_groups, desc="Retrying failed groups"):
        try:
            fly, event_id, event_type = group_key.split("_")
            event_id = int(event_id)

            # Filter data for this specific group
            if "event_type" in all_data.columns:
                group_data = all_data[
                    (all_data["fly"] == fly)
                    & (all_data["event_id"] == event_id)
                    & (all_data["event_type"] == event_type)
                ]
            else:
                group_data = all_data[
                    (all_data["fly"] == fly) & (all_data["event_id"] == event_id)
                ]

            if len(group_data) == 0:
                logging.warning(f"Could not find data for group {group_key}")
                continue

            # Process this group with minimal parallelization to avoid issues
            keypoint_columns = all_data.filter(regex="^(x|y)_").columns
            all_metadata_columns = [
                "flypath",
                "experiment",
                "event_type",
                "Nickname",
                "Brain region",
                "Date",
                "Genotype",
                "Period",
                "FeedingState",
                "Orientation",
                "Light",
                "Crossing",
                "event_id",
            ]
            metadata_columns = [
                col for col in all_metadata_columns if col in all_data.columns
            ]

            # Get fly initial positions for this specific fly
            fly_data = all_data[all_data["fly"] == fly]
            fly_initial_positions = get_fly_initial_positions(fly_data)
            fly_start_map = fly_initial_positions.set_index("fly")[
                ["overall_x_start", "overall_y_start"]
            ].to_dict("index")

            # Process this group
            result = process_group(
                fly,
                event_id,
                event_type,
                group_data,
                features,
                keypoint_columns,
                metadata_columns,
                fly_start_map,
                n_jobs=1,  # Use single thread for retries to avoid issues
                frames_per_event=frames_per_event,
            )

            batch_data.append(result)
            retried_count += 1

            # Remove from failed groups file
            with open(os.path.join(output_dir, "failed_groups.txt"), "r") as f:
                failed_lines = f.readlines()

            with open(os.path.join(output_dir, "failed_groups.txt"), "w") as f:
                for line in failed_lines:
                    if not line.startswith(f"{group_key}|"):
                        f.write(line)

        except Exception as e:
            logging.error(f"Retry failed for group {group_key}: {str(e)}")
            # Update the error message
            with open(os.path.join(output_dir, "failed_groups.txt"), "a") as f:
                f.write(f"{group_key}|RETRY_FAILED: {str(e)}\n")

    # Save the retried data as a batch
    if batch_data:
        batch_df = pd.DataFrame(batch_data)
        batch_filename = os.path.join(
            output_dir, f"retried_batch_{time.time()}.feather"
        )
        feather.write_feather(batch_df, batch_filename)
        logging.info(f"Saved {retried_count} retried groups to {batch_filename}")

    return retried_count


def main(
    input_path,
    output_path,
    features,
    n_jobs=None,
    frames_per_event=None,
    cleanup_temp_files=True,
    retry_failed=True,
):
    """Streamlined main function with fixed batch processing"""

    if n_jobs is None:
        n_jobs = get_optimal_workers()

    logging.info(f"Using {n_jobs} workers based on system resources")

    # Create output directory
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Track processed groups for resumability
    processed_groups = set()
    if os.path.exists(os.path.join(output_dir, "processed_groups.txt")):
        with open(os.path.join(output_dir, "processed_groups.txt"), "r") as f:
            processed_groups = set(line.strip() for line in f)

    # Process batches with fixed batch size from PyArrow
    for batch_idx, data_batch in enumerate(batch_load_data(input_path), 1):
        logging.info(f"Processing batch {batch_idx} with {len(data_batch)} rows")

        # Process the batch
        transform_data_batch(
            data_batch,
            features,
            n_jobs=n_jobs,
            output_dir=output_dir,
            frames_per_event=frames_per_event,
            processed_groups=processed_groups,
        )

        # Still use garbage collection between batches
        gc.collect()

    # Retry failed groups if requested
    if retry_failed:
        failed_groups = load_failed_groups(output_dir)
        if failed_groups:
            logging.info(
                f"Found {len(failed_groups)} failed groups. Attempting to retry..."
            )
            retry_count = retry_failed_groups(
                input_path, output_dir, features, max(1, n_jobs // 2), frames_per_event
            )
            logging.info(
                f"Successfully retried {retry_count} of {len(failed_groups)} failed groups"
            )

    # After all batches are processed, concatenate them
    logging.info("All batches processed. Concatenating results...")
    concatenate_batches(output_dir, output_path, cleanup=cleanup_temp_files)

    # Report any remaining failed groups
    remaining_failed = load_failed_groups(output_dir)
    if remaining_failed:
        failures_file = os.path.join(output_dir, "final_failures.txt")
        logging.warning(
            f"{len(remaining_failed)} groups failed processing. See {failures_file} for details."
        )
        with open(failures_file, "w") as f:
            for group_key, error_msg in remaining_failed.items():
                f.write(f"{group_key}: {error_msg}\n")
    else:
        logging.info("All groups processed successfully.")

    logging.info(f"Final output saved to {output_path}")


if __name__ == "__main__":
    # Simplify configuration
    config.setdefault("retry_failed", True)

    # Check if directory exists and if not create it
    savedir = Path(config["output_path"]).parent
    savedir.mkdir(parents=True, exist_ok=True)

    # Save configuration settings
    save_config(config, config["output_path"])

    # Main function with simplified parameters
    main(
        input_path=config["input_path"],
        output_path=config["output_path"],
        features=config["features"],
        n_jobs=config["n_jobs"],
        frames_per_event=config["frames_per_event"],
        cleanup_temp_files=config.get("cleanup_temp_files", True),
        retry_failed=config.get("retry_failed", True),
    )
