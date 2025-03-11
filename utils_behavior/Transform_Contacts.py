import pandas as pd
import numpy as np
import pyarrow.feather as feather
from scipy.fft import fft
from scipy.signal import find_peaks
from sklearn.feature_selection import VarianceThreshold
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import time
import logging
from pathlib import Path
import json

# Configuration section
config = {
    "input_path": "/mnt/upramdya_data/MD/Ballpushing_Exploration/Datasets/250307_StdContacts_Ctrl_noOverlap_Data_cutoff/standardized_contacts/250307_pooled_standardized_contacts.feather",
    "output_path": "/mnt/upramdya_data/MD/Ballpushing_Exploration/Datasets/250307_StdContacts_Ctrl_noOverlap_Data_cutoff/Transformed/250305_Pooled_FeedingState_Transformed_200frames.feather",
    "features": [
        "derivatives",
        "relative_positions",
        "statistical_measures",
        "fourier",
        "frame_features",
        "keypoints",
        # "tsfresh",
    ],
    "n_jobs": min(os.cpu_count(), 8),
    "test_rows": None,
    "chunk_size": None,
    "frames_per_event": 200,
}

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

num_cores = os.cpu_count()
n_jobs = min(num_cores, 10)  # Limit to 4 CPU cores to avoid crashing the computer

def save_config(config, savepath):
    """Save configuration settings to a JSON file."""
    config_path = Path(savepath).with_suffix('.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Configuration saved to {config_path}")

def transpose_keypoints(data):
    """Transpose keypoints with proper X/Y coordinate pairing."""
    x_fly_pattern = r"^x_.*_fly$"
    y_fly_pattern = r"^y_.*_fly$"
    centre_pattern = r"^[xy]_centre_preprocessed$"

    keypoint_columns = data.filter(
        regex=f"({x_fly_pattern}|{y_fly_pattern}|{centre_pattern})"
    ).columns
    transposed = {}

    data = data.assign(frame_count=data.groupby(["fly", "adjusted_frame"]).cumcount())

    for keypoint in keypoint_columns:
        parts = keypoint.split("_")
        axis = parts[0]

        if "fly" in parts:
            keypoint_name = "_".join(parts[1:-1])
        elif "preprocessed" in parts:
            keypoint_name = "_".join(parts[1:-1])

        keypoint_df = data.pivot(
            index=["fly"],
            columns=["adjusted_frame", "frame_count"],
            values=keypoint,
        )

        keypoint_df.columns = [
            f"{keypoint_name}_frame{frame}_{axis}"
            for (frame, _) in keypoint_df.columns
        ]

        transposed.update(keypoint_df.iloc[0].to_dict())

    return transposed

def calculate_frame_features(group):
    """Calculate per-frame velocity and angles using original coordinates."""
    features = {}
    group = group.sort_values("adjusted_frame").reset_index(drop=True)

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
        row.update(calculate_frame_features(group))
    if "keypoints" in features:
        row.update(transpose_keypoints(group))
    if "tsfresh" in features:
        row.update(calculate_tsfresh_features(group, keypoint_columns, n_jobs=n_jobs))
    row.update(metadata.to_dict())
    return row

def transform_data(data, features, n_jobs=num_cores, chunk_size=None, output_dir=None, frames_per_event=None):
    """Transform the data by processing each group and extracting features."""
    keypoint_columns = data.filter(regex="^(x|y)_").columns
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

    metadata_columns = [col for col in all_metadata_columns if col in data.columns]

    fly_initial_positions = get_fly_initial_positions(data)

    fly_start_map = fly_initial_positions.set_index("fly")[
        ["overall_x_start", "overall_y_start"]
    ].to_dict("index")

    groups = (
        list(data.groupby(["fly", "event_id", "event_type"]))
        if "event_type" in data.columns
        else list(data.groupby(["fly", "event_id"]))
    )

    total_groups = len(groups)
    logging.info(f"Processing {total_groups} groups with {n_jobs} workers")
    start_time = time.time()

    if chunk_size is None:
        chunk_size = total_groups

    existing_chunks = set(
        int(f.split("_")[1].split(".")[0])
        for f in os.listdir(output_dir)
        if f.startswith("chunk_")
    )
    total_chunks = (total_groups + chunk_size - 1) // chunk_size

    for chunk_index in range(0, total_groups, chunk_size):
        chunk_number = chunk_index // chunk_size + 1
        if chunk_number in existing_chunks:
            logging.info(f"Skipping chunk {chunk_number} as it already exists")
            continue

        chunk = groups[chunk_index : chunk_index + chunk_size]
        chunk_data = []
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = [
                executor.submit(
                    process_group,
                    fly,
                    event_id,
                    event_type,
                    group,
                    features,
                    keypoint_columns,
                    metadata_columns,
                    fly_start_map,
                    n_jobs,
                    frames_per_event,
                )
                for (fly, event_id, event_type), group in chunk
            ]

            for i, future in enumerate(as_completed(futures), 1):
                chunk_data.append(future.result())
                if i % 100 == 0 or i == len(chunk):
                    logging.info(
                        f"Processed {i}/{len(chunk)} groups in chunk {chunk_number}"
                    )

        chunk_df = pd.DataFrame(chunk_data)
        chunk_filename = os.path.join(output_dir, f"chunk_{chunk_number}.feather")
        feather.write_feather(chunk_df, chunk_filename)
        logging.info(f"Saved chunk {chunk_number} to {chunk_filename}")

    end_time = time.time()
    logging.info(
        f"Data transformation completed in {end_time - start_time:.2f} seconds"
    )
    logging.info(
        f"Processed {len(os.listdir(output_dir))} chunks out of {total_chunks} total chunks"
    )

def concatenate_chunks(output_dir, output_path):
    """Concatenate all chunk files into a single DataFrame."""
    chunk_files = [
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if f.startswith("chunk_")
    ]
    chunk_dfs = [pd.read_feather(chunk_file) for chunk_file in chunk_files]
    concatenated_df = pd.concat(chunk_dfs, ignore_index=True)
    feather.write_feather(concatenated_df, output_path)
    logging.info(f"Concatenated all chunks and saved to {output_path}")

def feature_selection(data):
    """Perform feature selection by removing low-variance and highly correlated features."""
    logging.info("Starting feature selection")
    start_time = time.time()

    numeric_columns = data.select_dtypes(include=[np.number]).columns
    non_numeric_columns = data.select_dtypes(exclude=[np.number]).columns

    logging.info(
        f"Found {len(numeric_columns)} numeric columns and {len(non_numeric_columns)} non-numeric columns"
    )

    numeric_data = data[numeric_columns]

    logging.info("Applying variance threshold")
    selector = VarianceThreshold(threshold=0.01)
    data_var = selector.fit_transform(numeric_data)
    selected_features = numeric_data.columns[selector.get_support(indices=True)]
    selected_numeric_data = pd.DataFrame(
        data_var, columns=selected_features, index=data.index
    )

    logging.info(
        f"Removed {len(numeric_columns) - len(selected_features)} low-variance features"
    )

    logging.info("Removing highly correlated features")
    corr_matrix = selected_numeric_data.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    final_numeric_data = selected_numeric_data.drop(columns=to_drop)

    logging.info(
        f"Removed {len(selected_features) - len(final_numeric_data.columns)} highly correlated features"
    )

    final_data = pd.concat([final_numeric_data, data[non_numeric_columns]], axis=1)

    end_time = time.time()
    logging.info(f"Feature selection completed in {end_time - start_time:.2f} seconds")
    logging.info(f"Final dataset shape: {final_data.shape}")

    return final_data

def main(
    input_path, output_path, features, n_jobs=num_cores, test_rows=None, chunk_size=None, frames_per_event=None
):
    """Main function to load data, transform it, and perform feature selection."""
    logging.info(f"Loading data from {input_path}...")
    data = pd.read_feather(input_path)
    logging.info(f"Loaded data shape: {data.shape}")

    if test_rows:
        data = data.head(test_rows)
        logging.info(
            f"Using a subset of {test_rows} rows for testing. New shape: {data.shape}"
        )

    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logging.info(f"Transforming data using features: {features}")
    transform_data(
        data, features, n_jobs=n_jobs, chunk_size=chunk_size, output_dir=output_dir, frames_per_event=frames_per_event
    )

    logging.info("Concatenating chunks...")
    concatenate_chunks(output_dir, output_path)

    logging.info("Selecting features...")
    transformed_data = pd.read_feather(output_path)
    selected_data = feature_selection(transformed_data)

    selected_path = output_path.replace(".feather", "_Selected.feather")
    feather.write_feather(selected_data, selected_path)

    logging.info("Transformation complete.")
    logging.info(f"Final data shape: {selected_data.shape}")

if __name__ == "__main__":
    # Check if directory exists and if not create it
    savedir = Path(config["output_path"]).parent
    savedir.mkdir(parents=True, exist_ok=True)

    # Save configuration settings
    save_config(config, config["output_path"])

    # Load your data
    logging.info(f"Loading data from {config['input_path']}...")
    data = pd.read_feather(config["input_path"])

    # Main function to load data, transform it, and perform feature selection
    main(
        input_path=config["input_path"],
        output_path=config["output_path"],
        features=config["features"],
        n_jobs=config["n_jobs"],
        test_rows=config["test_rows"],
        chunk_size=config["chunk_size"],
        frames_per_event=config["frames_per_event"],
    )