import pandas as pd
import numpy as np
import pyarrow.feather as feather
from scipy.fft import fft
from scipy.signal import find_peaks

# from tsfresh import extract_features
# from tsfresh.feature_extraction import ComprehensiveFCParameters
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
    """Transpose keypoints with proper X/Y coordinate pairing"""
    # Get all fly-relative keypoints (both X and Y)
    x_fly_pattern = r"^x_.*_fly$"
    y_fly_pattern = r"^y_.*_fly$"
    # y_body_pattern = r"^y_(Head|Thorax|Abdomen|Rfront|Lfront|Rmid|Lmid|Rhind|Lhind|Rwing|Lwing)$"
    centre_pattern = r"^[xy]_centre_preprocessed$"

    # Combine patterns using regex OR
    keypoint_columns = data.filter(
        regex=f"({x_fly_pattern}|{y_fly_pattern}|{centre_pattern})"
    ).columns
    transposed = {}

    print(f"Keypoint columns: {keypoint_columns}")

    # Create unique frame identifiers
    data = data.assign(frame_count=data.groupby(["fly", "adjusted_frame"]).cumcount())

    for keypoint in keypoint_columns:
        # Split into components
        parts = keypoint.split("_")
        axis = parts[0]

        # Handle different naming conventions
        if "fly" in parts:
            # X and Y-coordinate pattern: x_Head_fly -> Head
            keypoint_name = "_".join(parts[1:-1])
        elif "preprocessed" in parts:
            # Centre coordinates: x_centre_preprocessed -> centre
            keypoint_name = "_".join(parts[1:-1])

        # Add event_type to the pivot index
        keypoint_df = data.pivot(
            index=[
                "fly",
            ],  # Modified index, removed event_type
            columns=["adjusted_frame", "frame_count"],
            values=keypoint,
        )

        # Update column naming to include event type
        keypoint_df.columns = [
            f"{keypoint_name}_frame{frame}_{axis}"  # Removed event_type from column name
            for (frame, _) in keypoint_df.columns
        ]

        transposed.update(keypoint_df.iloc[0].to_dict())

    return transposed


def calculate_frame_features(group):
    """Calculate per-frame velocity and angles using original coordinates"""
    features = {}
    group = group.sort_values("adjusted_frame").reset_index(drop=True)

    # Base keypoints to process (from original columns)
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

        # Calculate velocity components
        dx = np.diff(x_vals, prepend=x_vals[0])
        dy = np.diff(y_vals, prepend=y_vals[0])

        # Calculate velocity magnitude and angles
        velocities = np.hypot(dx, dy)
        angles = np.degrees(np.arctan2(dx, dy)) % 360

        # Store event type in features
        for i, frame in enumerate(group["adjusted_frame"]):
            features[f"{kp}_frame{frame}_velocity"] = velocities[i]
            features[f"{kp}_frame{frame}_angle"] = angles[i]

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
    event_id,
    event_type,
    # contact_index,
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


def transform_data(data, features, n_jobs=num_cores, chunk_size=None, output_dir=None):
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
        # "contact_index",
    ]

    # Filter metadata columns to only include those present in the data
    metadata_columns = [col for col in all_metadata_columns if col in data.columns]

    # Precompute fly initial positions
    fly_initial_positions = get_fly_initial_positions(data)

    # Create dictionary for faster lookups
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
                    fly_start_map,  # Pass the precomputed map
                    n_jobs,
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
    input_path = "/mnt/upramdya_data/MD/Ballpushing_Exploration/Datasets/250227_StdContacts_Ctrl_Data/standardized_contacts/250228_pooled_standardized_contacts.feather"
    output_path = "/mnt/upramdya_data/MD/Ballpushing_Exploration/Datasets/250227_StdContacts_Ctrl_Data/Transformed/250228_Pooled_FeedingState_Transformed.feather"

    # input_path = "/mnt/upramdya_data/MD/MultiMazeRecorder/Datasets/Skeleton_TNT/250106_FinalEventCutoffData_norm/contact_data/250106_Pooled_contact_data.feather"
    # output_path = ""/mnt/upramdya_data/MD/MultiMazeRecorder/Datasets/Skeleton_TNT/250107_Transform/250107_Transformed_contact_data_rawdisp.feather""
    features = [
        # "derivatives",
        # "relative_positions",
        # "statistical_measures",
        # "fourier",
        "frame_features",
        "keypoints",
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
