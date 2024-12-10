import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pyarrow.feather as feather
import logging

logging.basicConfig(level=logging.INFO)

def prepare_features(data):
    # Assuming all columns except metadata are features
    metadata_columns = ['experiment', 'Nickname', 'Brain region', 'Date', 'Genotype', 'Period', 'FeedingState', 'Orientation', 'Light', 'Crossing', 'contact_index', 'duration', 'fly']
    feature_columns = data.columns.difference(metadata_columns)
    return data[feature_columns].values

def run_pca_and_save(data, explained_variance_threshold=0.95, savepath=None):
    logging.info("Preparing features...")
    features = prepare_features(data)
    metadata = data.drop(columns=features.columns)

    logging.info("Normalizing features...")
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features)

    logging.info("Applying PCA...")
    pca = PCA()
    pca_results = pca.fit_transform(features_normalized)

    # Determine the number of components to keep
    cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.argmax(cumulative_explained_variance >= explained_variance_threshold) + 1

    logging.info(f"Number of components to keep: {n_components}")

    # Truncate PCA results to keep only the required number of components
    pca_results = pca_results[:, :n_components]

    logging.info("PCA completed.")

    # Combine PCA results with metadata
    pca_df = pd.DataFrame(pca_results, columns=[f"PCA Component {i+1}" for i in range(n_components)])
    combined_df = pd.concat([pca_df, metadata.reset_index(drop=True)], axis=1)

    if savepath:
        feather.write_feather(combined_df, savepath)
        logging.info(f"PCA results saved to {savepath}")

    return combined_df

if __name__ == "__main__":
    data_path = "/mnt/upramdya_data/MD/MultiMazeRecorder/Datasets/Skeleton_TNT/241209_Transformed_contact_data.feather"
    pca_savepath = "/mnt/upramdya_data/MD/MultiMazeRecorder/Datasets/Skeleton_TNT/PCA/241210_pca_data_transformed.feather"

    logging.info(f"Loading transformed data from {data_path}...")
    try:
        data = pd.read_feather(data_path)
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        exit(1)

    combined_df = run_pca_and_save(data, explained_variance_threshold=0.95, savepath=pca_savepath)

    logging.info(combined_df.head())
    logging.info(combined_df.columns)