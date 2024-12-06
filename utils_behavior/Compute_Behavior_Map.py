import pandas as pd

# from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
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
    pca_components=50,
    savepath=None,
):

    print("Applying PCA...")

    # Extract features and metadata
    features = data.filter(regex="^(x|y)_").values
    metadata = data.drop(columns=data.filter(regex="^(x|y)_").columns)
    contact_indices = (
        data["contact_index"]
        if "contact_index" in data.columns
        else pd.Series([None] * len(data))
    )

    # Adjust PCA components if necessary
    n_features = features.shape[1]
    pca_components = min(pca_components, n_features)

    pca = PCA(n_components=pca_components)
    pca_results = pca.fit_transform(features)

    print("PCA completed. Starting t-SNE...")

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
    data = pd.read_csv(data_path)

    # Compute the behavior map
    behavior_map = compute_behavior_map(
        data,
        perplexity=30,
        n_iter=3000,
        pca_components=50,
        savepath="/mnt/upramdya_data/MD/MultiMazeRecorder/Datasets/Skeleton_TNT/TSNE/241206_behavior_map.csv",
    )

    print(behavior_map)
