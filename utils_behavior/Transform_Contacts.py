import pandas as pd
import numpy as np
import pyarrow.feather as feather
from scipy.fft import fft
from tsfresh import extract_features
from tsfresh.feature_extraction import ComprehensiveFCParameters
from sklearn.feature_selection import VarianceThreshold
from concurrent.futures import ProcessPoolExecutor, as_completed

def calculate_derivatives(group, keypoint_columns):
    velocities = group[keypoint_columns].diff()
    accelerations = velocities.diff()
    return {
        f"{col}_vel_mean": velocities[col].mean() for col in keypoint_columns
    } | {
        f"{col}_vel_std": velocities[col].std() for col in keypoint_columns
    } | {
        f"{col}_acc_mean": accelerations[col].mean() for col in keypoint_columns
    } | {
        f"{col}_acc_std": accelerations[col].std() for col in keypoint_columns
    }

def calculate_relative_positions(group, keypoint_columns):
    initial_positions = group[keypoint_columns].iloc[0]
    displacements = group[keypoint_columns] - initial_positions
    return {
        f"{col}_disp_mean": displacements[col].mean() for col in keypoint_columns
    } | {
        f"{col}_disp_std": displacements[col].std() for col in keypoint_columns
    }

def calculate_statistical_measures(group, keypoint_columns):
    return {
        f"{col}_mean": group[col].mean() for col in keypoint_columns
    } | {
        f"{col}_std": group[col].std() for col in keypoint_columns
    } | {
        f"{col}_skew": group[col].skew() for col in keypoint_columns
    } | {
        f"{col}_kurt": group[col].kurtosis() for col in keypoint_columns
    }

def calculate_fourier_features(group, keypoint_columns):
    fft_results = {}
    for col in keypoint_columns:
        fft_vals = fft(group[col].values)
        dominant_freq = np.abs(fft_vals[1:len(fft_vals)//2]).argmax() + 1
        fft_results[f"{col}_dom_freq"] = dominant_freq
        fft_results[f"{col}_dom_freq_magnitude"] = np.abs(fft_vals[dominant_freq])
    return fft_results

def calculate_tsfresh_features(group, keypoint_columns, n_jobs=1):
    tsfresh_data = group[['frame'] + keypoint_columns.tolist()]
    tsfresh_data['id'] = 0  # Single id for the group
    extracted_features = extract_features(tsfresh_data, column_id='id', column_sort='frame', default_fc_parameters=ComprehensiveFCParameters(), n_jobs=n_jobs)
    extracted_features.columns = [f"tsfresh_{col}" for col in extracted_features.columns]
    return extracted_features.iloc[0].to_dict()

def process_group(fly, contact_index, group, features, keypoint_columns, metadata_columns, n_jobs):
    duration = group["frame"].max() - group["frame"].min() + 1
    metadata = group[metadata_columns].iloc[0]
    row = {"duration": duration, "fly": fly}
    if 'derivatives' in features:
        row.update(calculate_derivatives(group, keypoint_columns))
    if 'relative_positions' in features:
        row.update(calculate_relative_positions(group, keypoint_columns))
    if 'statistical_measures' in features:
        row.update(calculate_statistical_measures(group, keypoint_columns))
    if 'fourier' in features:
        row.update(calculate_fourier_features(group, keypoint_columns))
    if 'tsfresh' in features:
        row.update(calculate_tsfresh_features(group, keypoint_columns, n_jobs=n_jobs))
    row.update(metadata.to_dict())
    return row

def transform_data(data, features, n_jobs=1):
    if n_jobs < 1 and n_jobs != -1:
        n_jobs = 1  # Set to 1 if n_jobs is invalid
    elif n_jobs == -1:
        n_jobs = None  # Use all available CPU cores

    transformed_data = []
    keypoint_columns = data.filter(regex="^(x|y)_").columns
    metadata_columns = ['experiment', 'Nickname', 'Brain region', 'Date', 'Genotype', 'Period', 'FeedingState', 'Orientation', 'Light', 'Crossing', 'contact_index']

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = [executor.submit(process_group, fly, contact_index, group, features, keypoint_columns, metadata_columns, n_jobs) 
                   for (fly, contact_index), group in data.groupby(['fly', 'contact_index'])]
        
        for future in as_completed(futures):
            transformed_data.append(future.result())

    return pd.DataFrame(transformed_data)

def feature_selection(data):
    # Remove features with low variance
    selector = VarianceThreshold(threshold=0.01)
    data_var = selector.fit_transform(data)
    selected_features = data.columns[selector.get_support(indices=True)]
    data = pd.DataFrame(data_var, columns=selected_features)

    # Remove highly correlated features
    corr_matrix = data.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    data = data.drop(columns=to_drop)

    return data

def main():
    input_path = "/mnt/upramdya_data/MD/MultiMazeRecorder/Datasets/Skeleton_TNT/241209_ContactData/241209_Pooled_contact_data.feather"
    output_path = "/mnt/upramdya_data/MD/MultiMazeRecorder/Datasets/Skeleton_TNT/241210_Transformed_contact_data_Complete.feather"
    
    # List of features to include in the transformation
    features = ['derivatives', 'relative_positions', 'statistical_measures', 'fourier', 'tsfresh']
    n_jobs = -1  # Use all available CPU cores

    print(f"Loading data from {input_path}...")
    data = pd.read_feather(input_path)

    print(f"Transforming data using features: {features}")
    transformed_data = transform_data(data, features, n_jobs=n_jobs)

    print("Selecting features...")
    selected_data = feature_selection(transformed_data)

    print(f"Saving transformed data to {output_path}...")
    feather.write_feather(selected_data, output_path)

    print("Transformation complete.")
    print(f"Transformed data shape: {selected_data.shape}")
    print(f"Columns: {selected_data.columns.tolist()}")

if __name__ == "__main__":
    main()