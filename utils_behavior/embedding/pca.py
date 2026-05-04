import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pyarrow.feather as feather
import logging

logging.basicConfig(level=logging.INFO)


def exclude_data(data, column, values):
    """
    Exclude rows from the DataFrame where the specified column has any of the specified values.

    Parameters:
    data (pd.DataFrame): The input DataFrame.
    column (str): The column to check for exclusion.
    values (list): The list of values to exclude.

    Returns:
    pd.DataFrame: The DataFrame with the specified rows excluded.
    """
    return data[~data[column].isin(values)]


def prepare_features(data):
    # Identify non-numeric columns (metadata)
    non_numeric_columns = data.select_dtypes(exclude=[np.number]).columns
    # Add some numeric metadata columns to add to non_numeric_columns: contact_index, Crossing,start
    non_numeric_columns = non_numeric_columns.append(
        pd.Index(["contact_index", "Crossing", "start"])
    )

    feature_columns = data.columns.difference(non_numeric_columns)
    return data[feature_columns].values, feature_columns


def run_pca_and_save(data, explained_variance_threshold=0.95, savepath=None):
    logging.info("Preparing features...")
    features, feature_columns = prepare_features(data)
    metadata = data.drop(columns=feature_columns)

    logging.info("Normalizing features...")
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features)

    logging.info("Applying PCA...")
    pca = PCA()
    pca_results = pca.fit_transform(features_normalized)

    # Determine the number of components to keep
    cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components = (
        np.argmax(cumulative_explained_variance >= explained_variance_threshold) + 1
    )

    logging.info(f"Number of components to keep: {n_components}")

    # Truncate PCA results to keep only the required number of components
    pca_results = pca_results[:, :n_components]

    logging.info("PCA completed.")

    # Combine PCA results with metadata
    pca_df = pd.DataFrame(
        pca_results, columns=[f"PCA Component {i+1}" for i in range(n_components)]
    )
    combined_df = pd.concat([pca_df, metadata.reset_index(drop=True)], axis=1)

    if savepath:
        feather.write_feather(combined_df, savepath)
        logging.info(f"PCA results saved to {savepath}")

    return combined_df


if __name__ == "__main__":
    data_path = "/mnt/upramdya_data/MD/MultiMazeRecorder/Datasets/Skeleton_TNT/241220_Transform/241220_Transformed_contact_data_mintsfresh_Selected.feather"
    pca_savepath = "/mnt/upramdya_data/MD/MultiMazeRecorder/Datasets/Skeleton_TNT/PCA/241220_pca_data_transformed_New.feather"

    logging.info(f"Loading transformed data from {data_path}...")
    try:
        data = pd.read_feather(data_path)
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        exit(1)

    # Exclude specific Genotype values
    data = exclude_data(data, "Genotype", ["M6", "M7", "PR", "CS"])

    combined_df = run_pca_and_save(
        data, explained_variance_threshold=0.95, savepath=pca_savepath
    )

    logging.info(combined_df.head())
    logging.info(combined_df.columns)
