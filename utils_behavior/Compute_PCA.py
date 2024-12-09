import pandas as pd
from sklearn.decomposition import IncrementalPCA, PCA
import pyarrow.feather as feather

def run_pca_and_save(
    data,
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
    n_components = (
        cumulative_explained_variance < explained_variance_threshold
    ).sum() + 1

    print(f"Number of components to keep: {n_components}")

    # Apply Incremental PCA with the determined number of components
    ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
    for batch in range(0, features.shape[0], batch_size):
        ipca.partial_fit(features[batch : batch + batch_size])

    pca_results = ipca.transform(features)

    print("Incremental PCA completed.")

    # Combine PCA results with metadata and contact indices
    pca_df = pd.DataFrame(pca_results, columns=[f"PCA Component {i+1}" for i in range(n_components)])
    combined_df = pd.concat([pca_df, metadata.reset_index(drop=True)], axis=1)
    combined_df["contact_index"] = contact_indices.reset_index(drop=True)

    # Save the combined results to a Feather file
    if savepath:
        feather.write_feather(combined_df, savepath)
        print(f"PCA results saved to {savepath}")

    return combined_df

# Example usage for Step 1
if __name__ == "__main__":
    data_path = "/mnt/upramdya_data/MD/MultiMazeRecorder/Datasets/Skeleton_TNT/241209_ContactData/241209_Pooled_contact_data.feather"
    pca_savepath = "/mnt/upramdya_data/MD/MultiMazeRecorder/Datasets/Skeleton_TNT/PCA/241209_pca_data.feather"

    # Load your data
    print(f"Loading data from {data_path}...")
    data = pd.read_feather(data_path)

    # Run PCA and save the results
    combined_df = run_pca_and_save(
        data,
        explained_variance_threshold=0.95,
        batch_size=1000,
        savepath=pca_savepath,
    )

    print(combined_df)