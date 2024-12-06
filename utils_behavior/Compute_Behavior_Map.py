import pandas as pd
from sklearn.decomposition import IncrementalPCA, PCA
from openTSNE import TSNE
from openTSNE.callbacks import Callback
from tqdm.auto import tqdm
import time


class ProgressCallback:
    def __init__(self, n_iter):
        self.n_iter = n_iter
        self.start_time = time.time()

    def __call__(self, iteration, error, local_error):
        elapsed_time = time.time() - self.start_time
        print(
            f"Iteration {iteration}/{self.n_iter}, Error: {error:.4f}, Time: {elapsed_time:.2f}s"
        )

    def close(self):
        print("t-SNE completed.")


def compute_behavior_map(
    data,
    perplexity=30,
    n_iter=3000,
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

    print("Incremental PCA completed. Starting t-SNE...")

    # Create the progress callback
    progress_callback = ProgressCallback(n_iter)

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=n_iter,
        random_state=42,
        n_jobs=-1,
        callbacks=[progress_callback],
        callbacks_every_iters=1,
    )
    tsne_results = tsne.fit(pca_results)

    # Close the progress bar
    progress_callback.close()

    tsne_df = pd.DataFrame(
        tsne_results, columns=["t-SNE Component 1", "t-SNE Component 2"]
    )
    metadata_df = metadata.reset_index(drop=True)

    result_df = pd.concat(
        [
            tsne_df,
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
        perplexity=30,
        n_iter=3000,
        explained_variance_threshold=0.95,
        batch_size=1000,
        savepath="/mnt/upramdya_data/MD/MultiMazeRecorder/Datasets/Skeleton_TNT/TSNE/241206_behavior_map.csv",
    )

    print(behavior_map)
