import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import pyarrow.feather as feather

def prepare_features(data, strategy):
    if strategy in ['derivative', 'relative_positions']:
        return data.filter(regex="^(x|y)_").values
    elif strategy == 'position_list':
        # Flatten the position lists into a single feature vector for each contact event
        return np.array([np.concatenate([np.array(data[col].iloc[i]) for col in data.filter(regex="^(x|y)_").columns]) 
                         for i in range(len(data))])
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

def run_pca_and_save(data, strategy, explained_variance_threshold=0.95, savepath=None):
    print("Preparing features...")
    features = prepare_features(data, strategy)
    metadata = data.drop(columns=data.filter(regex="^(x|y)_").columns)

    print("Applying PCA...")
    pca = PCA()
    pca_results = pca.fit_transform(features)

    # Determine the number of components to keep
    cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.argmax(cumulative_explained_variance >= explained_variance_threshold) + 1

    print(f"Number of components to keep: {n_components}")

    # Truncate PCA results to keep only the required number of components
    pca_results = pca_results[:, :n_components]

    print("PCA completed.")

    # Combine PCA results with metadata
    pca_df = pd.DataFrame(pca_results, columns=[f"PCA Component {i+1}" for i in range(n_components)])
    combined_df = pd.concat([pca_df, metadata.reset_index(drop=True)], axis=1)

    if savepath:
        feather.write_feather(combined_df, savepath)
        print(f"PCA results saved to {savepath}")

    return combined_df

if __name__ == "__main__":
    data_path = "/mnt/upramdya_data/MD/MultiMazeRecorder/Datasets/Skeleton_TNT/241209_Transformed_contact_data.feather"
    pca_savepath = "/mnt/upramdya_data/MD/MultiMazeRecorder/Datasets/Skeleton_TNT/PCA/241210_pca_data_transformed.feather"
    strategy = 'derivative'  # Must match the strategy used in transform_data.py

    print(f"Loading transformed data from {data_path}...")
    data = pd.read_feather(data_path)

    combined_df = run_pca_and_save(data, strategy, explained_variance_threshold=0.95, savepath=pca_savepath)

    print(combined_df.head())
    print(combined_df.columns)