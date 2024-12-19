import pandas as pd
import numpy as np
import pyarrow.feather as feather
from scipy.fft import fft
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


def calculate_derivatives(group, keypoint_columns):
    velocities = group[keypoint_columns].diff()
    accelerations = velocities.diff()
    return (
        {f"{col}_vel_mean": velocities[col].mean() for col in keypoint_columns}
        | {f"{col}_vel_std": velocities[col].std() for col in keypoint_columns}
        | {f"{col}_acc_mean": accelerations[col].mean() for col in keypoint_columns}
        | {f"{col}_acc_std": accelerations[col].std() for col in keypoint_columns}
    )


def calculate_relative_positions(group, keypoint_columns):
    initial_positions = group[keypoint_columns].iloc[0]
    displacements = group[keypoint_columns] - initial_positions
    return {
        f"{col}_disp_mean": displacements[col].mean() for col in keypoint_columns
    } | {f"{col}_disp_std": displacements[col].std() for col in keypoint_columns}


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


def process_group(
    fly, contact_index, group, features, keypoint_columns, metadata_columns, n_jobs
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
        row.update(calculate_relative_positions(group, keypoint_columns))
    if "statistical_measures" in features:
        row.update(calculate_statistical_measures(group, keypoint_columns))
    if "fourier" in features:
        row.update(calculate_fourier_features(group, keypoint_columns))
    if "tsfresh" in features:
        row.update(calculate_tsfresh_features(group, keypoint_columns, n_jobs=n_jobs))
    row.update(metadata.to_dict())
    return row


def transform_data(data, features, n_jobs=num_cores, chunk_size=None):
    transformed_data = []
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

    groups = list(data.groupby(["fly", "contact_index"]))
    total_groups = len(groups)
    logging.info(f"Processing {total_groups} groups with {n_jobs} workers")
    start_time = time.time()

    if chunk_size is None:
        # Process all groups at once without chunking
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
                    n_jobs,
                )
                for (fly, contact_index), group in groups
            ]
            for i, future in enumerate(as_completed(futures), 1):
                transformed_data.append(future.result())
                if i % 100 == 0 or i == total_groups:
                    logging.info(
                        f"Processed {i}/{total_groups} groups ({i/total_groups:.2%})"
                    )
    else:
        # Process groups in chunks
        for i in range(0, total_groups, chunk_size):
            chunk = groups[i : i + chunk_size]
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
                        n_jobs,
                    )
                    for (fly, contact_index), group in chunk
                ]
                for j, future in enumerate(as_completed(futures), 1):
                    transformed_data.append(future.result())
                    if (i + j) % 100 == 0 or (i + j) == total_groups:
                        logging.info(
                            f"Processed {i + j}/{total_groups} groups ({(i + j)/total_groups:.2%})"
                        )

    end_time = time.time()
    logging.info(
        f"Data transformation completed in {end_time - start_time:.2f} seconds"
    )
    return pd.DataFrame(transformed_data)


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

    logging.info(f"Transforming data using features: {features}")
    transformed_data = transform_data(
        data, features, n_jobs=n_jobs, chunk_size=chunk_size
    )
    logging.info(f"Transformed data shape: {transformed_data.shape}")

    feather.write_feather(transformed_data, output_path)

    logging.info("Selecting features...")
    selected_data = feature_selection(transformed_data)

    logging.info(f"Saving transformed data to {output_path}...")

    selected_path = output_path.replace(".feather", "_Selected.feather")

    feather.write_feather(selected_data, selected_path)

    logging.info("Transformation complete.")
    logging.info(f"Final data shape: {selected_data.shape}")


if __name__ == "__main__":
    input_path = "/mnt/upramdya_data/MD/MultiMazeRecorder/Datasets/Skeleton_TNT/241218_FinalEventCutoffData_norm/contact_data/241209_Pooled_contact_data.feather"
    output_path = "/mnt/upramdya_data/MD/MultiMazeRecorder/Datasets/Skeleton_TNT/241219_Transformed_contact_data_derivative.feather"

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
        chunk_size=None,
    )
