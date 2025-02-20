import pandas as pd
import numpy as np
import pyarrow.feather as feather
from scipy.fft import fft
from scipy.signal import find_peaks
from tsfresh import extract_features
from tsfresh.feature_extraction import ComprehensiveFCParameters
from sklearn.feature_selection import VarianceThreshold
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import time
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

num_cores = os.cpu_count()
n_jobs = min(num_cores, 10)  # Limit to 4 CPU cores to avoid crashing the computer


def transpose_keypoints(data):
    """
    Transpose keypoints so that each keypoint from each frame is a separate column
    """
    
    # Extract fly relative keypoints, that are called x and y_keypointname_fly"
    fly_relative_keypoints = data.filter(regex="^(x|y)_.*_fly$").columns
    
    # For each keypoint, transform the data so that each keypoint x and y from each frame is a separate column
    
    # Build a dictionary to store the transformed data
    keypoints_dataset = {}
    
    for keypoint in fly_relative_keypoints:
        keypoint_name = keypoint.split("_")[1]
        keypoint_data = data.pivot(index="frame", columns="fly", values=keypoint)
        keypoint_data.columns = keypoint_data.columns.droplevel()
        keypoint_data.columns.name = None
        keypoint_data = keypoint_data.add_prefix(f"{keypoint_name}_")
        keypoints_dataset[keypoint_name] = keypoint_data
   
        
    return keypoints_dataset

def calculate_instantaneous_features(group, keypoint_columns):
    velocities = group[keypoint_columns].diff()
    accelerations = velocities.diff()
    
    features = {}
    for col in keypoint_columns:
        # Raw time series values
        features.update({
            f"{col}_velocity_ts": velocities[col].values,
            f"{col}_acceleration_ts": accelerations[col].values
        })
        
        # Time-series statistics
        features.update({
            f"{col}_vel_max": velocities[col].max(),
            f"{col}_vel_min": velocities[col].min(),
            f"{col}_vel_peak_freq": len(find_peaks(velocities[col])[0])/len(velocities),
            f"{col}_acc_jerk": accelerations[col].diff().abs().mean()
        })
        
    return features
    

def calculate_derivatives(group, keypoint_columns):
    velocities = group[keypoint_columns].diff()
    accelerations = velocities.diff()
    return (
        {f"{col}_vel_mean": velocities[col].mean() for col in keypoint_columns}
        | {f"{col}_vel_std": velocities[col].std() for col in keypoint_columns}
        | {f"{col}_acc_mean": accelerations[col].mean() for col in keypoint_columns}
        | {f"{col}_acc_std": accelerations[col].std() for col in keypoint_columns}
    )


def calculate_relative_positions(group, keypoint_columns, fly_start_map):

    fly = group["fly"].iloc[0]
    overall_start = fly_start_map[fly]

    # Calculate the Euclidean distance between each frame and the initial frame
    initial_x = group["x_centre_preprocessed"].iloc[0]
    initial_y = group["y_centre_preprocessed"].iloc[0]
    group["euclidean_distance"] = np.sqrt(
        (group["x_centre_preprocessed"] - initial_x) ** 2
        + (group["y_centre_preprocessed"] - initial_y) ** 2
    )

    initial_positions = group[keypoint_columns].iloc[0]
    displacements = group[keypoint_columns] - initial_positions

    # Calculate median euclidean distance
    median_euclidean_distance = group["euclidean_distance"].median()

    # Calculate direction
    initial_distance = group["euclidean_distance"].iloc[0]
    final_distance = group["euclidean_distance"].iloc[-1]
    direction = 1 if final_distance > initial_distance else -1

    # Calculate raw displacement for centre_preprocessed keypoint
    final_x = group["x_centre_preprocessed"].iloc[-1]
    final_y = group["y_centre_preprocessed"].iloc[-1]
    raw_displacement = np.sqrt((final_x - initial_x) ** 2 + (final_y - initial_y) ** 2)

    # Contact positions
    contact_start_x = group["x_centre_preprocessed"].iloc[0]
    contact_start_y = group["y_centre_preprocessed"].iloc[0]
    contact_end_x = group["x_centre_preprocessed"].iloc[-1]
    contact_end_y = group["y_centre_preprocessed"].iloc[-1]

    # Calculate distances
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


def calculate_statistical_measures(group, keypoint_columns):
    return (
        {f"{col}_mean": group[col].mean() for col in keypoint_columns}
        | {f"{col}_std": group[col].std() for col in keypoint_columns}
        | {f"{col}_skew": group[col].skew() for col in keypoint_columns}
        | {f"{col}_kurt": group[col].kurtosis() for col in keypoint_columns}
    )


def calculate_fourier_features(group, keypoint_columns):
    fft_results = {}
    for col in keypoint_columns:
        fft_vals = fft(group[col].values)
        dominant_freq = np.abs(fft_vals[1 : len(fft_vals) // 2]).argmax() + 1
        fft_results[f"{col}_dom_freq"] = dominant_freq
        fft_results[f"{col}_dom_freq_magnitude"] = np.abs(fft_vals[dominant_freq])
    return fft_results


def calculate_tsfresh_features(group, keypoint_columns, n_jobs):
    tsfresh_data = group[keypoint_columns].copy()
    tsfresh_data["id"] = 0  # Single id for the group
    tsfresh_data["frame"] = group["frame"].values

    # Extract features using tsfresh
    extracted_features = extract_features(
        tsfresh_data,
        column_id="id",
        column_sort="frame",
        default_fc_parameters=ComprehensiveFCParameters(),
        n_jobs=n_jobs,
    )
    return extracted_features.iloc[0].to_dict()


def get_fly_initial_positions(data):
    """Get first recorded position for each fly in entire dataset"""
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
    contact_index,
    group,
    features,
    keypoint_columns,
    metadata_columns,
    fly_start_map,
    n_jobs,
):
    duration = group["frame"].max() - group["frame"].min() + 1
    metadata = group[metadata_columns].iloc[0]
    row = {
        "duration": duration,
        "fly": fly,
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
    if "tsfresh" in features:
        row.update(calculate_tsfresh_features(group, keypoint_columns, n_jobs=n_jobs))
    row.update(metadata.to_dict())
    return row


def transform_data(data, features, n_jobs=num_cores, chunk_size=None, output_dir=None):
    keypoint_columns = data.filter(regex="^(x|y)_").columns
    metadata_columns = [
        "flypath",
        "experiment",
        "Nickname",
        "Brain region",
        "Date",
        "Genotype",
        "Period",
        "FeedingState",
        "Orientation",
        "Light",
        "Crossing",
        "contact_index",
    ]
    # Precompute fly initial positions
    fly_initial_positions = get_fly_initial_positions(data)

    # Create dictionary for faster lookups
    fly_start_map = fly_initial_positions.set_index("fly")[
        ["overall_x_start", "overall_y_start"]
    ].to_dict("index")

    groups = list(data.groupby(["fly", "contact_index"]))
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
                    contact_index,
                    group,
                    features,
                    keypoint_columns,
                    metadata_columns,
                    fly_start_map,  # Pass the precomputed map
                    n_jobs,
                )
                for (fly, contact_index), group in chunk
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
    input_path, output_path, features, n_jobs=num_cores, test_rows=None, chunk_size=None
):
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
        data, features, n_jobs=n_jobs, chunk_size=chunk_size, output_dir=output_dir
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
    input_path = "/mnt/upramdya_data/MD/MultiMazeRecorder/Datasets/Skeleton_TNT/240120_short_contacts_no_cutoff_no_downsample_Data/contact_data/250106_Pooled_contact_data.feather"
    output_path = "/mnt/upramdya_data/MD/MultiMazeRecorder/Datasets/Skeleton_TNT/240204_short_contacts_no_cutoff_no_downsample_Data/Transformed_contacts_nocutoff_flexible_rule.feather"

    # input_path = "/mnt/upramdya_data/MD/MultiMazeRecorder/Datasets/Skeleton_TNT/250106_FinalEventCutoffData_norm/contact_data/250106_Pooled_contact_data.feather"
    # output_path = ""/mnt/upramdya_data/MD/MultiMazeRecorder/Datasets/Skeleton_TNT/250107_Transform/250107_Transformed_contact_data_rawdisp.feather""
    features = [
        "derivatives",
        "relative_positions",
        # "statistical_measures",
        # "fourier",
        # "tsfresh",
    ]

    num_cores = os.cpu_count()
    n_jobs = min(num_cores, 10)  # Limit to 4 CPU cores to avoid crashing the computer

    test_rows = 100  # Use a small subset for testing

    main(
        input_path,
        output_path,
        features,
        n_jobs=n_jobs,
        test_rows=None,
        chunk_size=None,  # Adjust chunk size as needed
    )
