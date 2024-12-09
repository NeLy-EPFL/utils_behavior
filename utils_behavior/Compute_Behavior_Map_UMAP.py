import pandas as pd
from sklearn.decomposition import IncrementalPCA, PCA
import umap
import pyarrow.feather as feather
import time

def compute_behavior_map(
    data,
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
    explained_variance_threshold=0.95,
    batch_size=1000,
    savepath=None,
):

    print("Applying Incremental PCA...")

    # Extract features and metadata
    features = data.filter(regex="^(x|y)_").values
    metadata = data.drop(columns=data.filter(regex="^(x|y)_").columns)
    contact_indices = (
        data["contact_index"]
        if "contact_index" in data.columns
        else pd.Series([None] * len(data))
    )

    # Compute the explained variance ratio using PCA
    pca = PCA()
    pca.fit(features)
    explained_variance_ratio = pca.explained_variance_ratio_

    # Determine the number of components to keep based on the explained variance threshold
    cumulative_explained_variance = explained_variance_ratio.cumsum()
    n_components_pca = (
        cumulative_explained_variance < explained_variance_threshold
    ).sum() + 1

    print(f"Number of components to keep: {n_components_pca}")

    # Apply Incremental PCA with the determined number of components
    ipca = IncrementalPCA(n_components=n_components_pca, batch_size=batch_size)
    for batch in range(0, features.shape[0], batch_size):
        ipca.partial_fit(features[batch : batch + batch_size])

    pca_results = ipca.transform(features)

    print("Incremental PCA completed. Starting UMAP...")

    # Create UMAP object
    umap_model = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric='euclidean',
        verbose=True,
        low_memory=True  # Enable low memory mode
    )

    # Fit and transform the data
    umap_results = umap_model.fit_transform(pca_results)

    umap_df = pd.DataFrame(
        umap_results, columns=[f"UMAP Component {i+1}" for i in range(n_components)]
    )
    metadata_df = metadata.reset_index(drop=True)

    result_df = pd.concat(
        [
            umap_df,
            metadata_df,
            pd.DataFrame({"contact_index": contact_indices}).reset_index(drop=True),
        ],
        axis=1,
    )

    if savepath:
        result_df.to_csv(savepath, index=False)
        print(f"Behavior map saved to {savepath}")

    return result_df

# Example usage
if __name__ == "__main__":

    data_path = "/mnt/upramdya_data/MD/MultiMazeRecorder/Datasets/Skeleton_TNT/Contact_data/241206_Pooled_contact_data.csv"

    # Load your data
    print(f"Loading data from {data_path}...")
    data = pd.read_csv(data_path, low_memory=False, dtype={32: str, 33: str})

    # Compute the behavior map
    behavior_map = compute_behavior_map(
        data,
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        explained_variance_threshold=0.95,
        batch_size=1000,
        savepath="/mnt/upramdya_data/MD/MultiMazeRecorder/Datasets/Skeleton_TNT/UMAP/241206_behavior_map.csv",
    )

    print(behavior_map)