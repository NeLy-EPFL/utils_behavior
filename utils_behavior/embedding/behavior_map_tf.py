import pandas as pd
import numpy as np
from sklearn.decomposition import IncrementalPCA, PCA
import pyarrow.feather as feather
from tsnecuda import TSNE as TSNE_CUDA
import time

class ProgressCallback:
    def __init__(self, n_iter):
        self.n_iter = n_iter
        self.start_time = time.time()

    def __call__(self, iteration, error):
        elapsed_time = time.time() - self.start_time
        print(f"Iteration {iteration}/{self.n_iter}, Error: {error:.4f}, Time: {elapsed_time:.2f}s")

    def close(self):
        print("t-SNE completed.")

def data_generator(data_path, batch_size):
    for chunk in pd.read_csv(data_path, chunksize=batch_size):
        yield chunk.filter(regex="^(x|y)_").values.astype(np.float32)

def compute_behavior_map(
    data_path,
    perplexity=30,
    n_iter=3000,
    explained_variance_threshold=0.95,
    batch_size=1000,
    savepath=None,
):
    print("Starting PCA...")

    # Load a small sample to determine the number of components
    sample_data = next(data_generator(data_path, batch_size))
    pca = PCA()
    pca.fit(sample_data)
    explained_variance_ratio = pca.explained_variance_ratio_

    # Determine the number of components to keep based on the explained variance threshold
    cumulative_explained_variance = explained_variance_ratio.cumsum()
    n_components = (cumulative_explained_variance < explained_variance_threshold).sum() + 1

    print(f"Number of components to keep: {n_components}")

    # Initialize Incremental PCA with the determined number of components
    ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)

    # Apply Incremental PCA in batches
    pca_results = []
    for batch in data_generator(data_path, batch_size):
        ipca.partial_fit(batch)
        pca_results.append(ipca.transform(batch))

    pca_results = np.concatenate(pca_results, axis=0)
    print(f"PCA completed. Reduced dimensions: {pca_results.shape[1]}")

    print("Starting t-SNE...")

    # Create TSNE_CUDA object
    tsne_cuda = TSNE_CUDA(
        n_components=2,
        perplexity=perplexity,
        n_iter=n_iter,
        verbose=1
    )

    # Fit and transform the data
    start_time = time.time()
    tsne_results = tsne_cuda.fit_transform(pca_results)
    end_time = time.time()
    print(f"t-SNE completed in {end_time - start_time:.2f} seconds")

    # Create DataFrame with results
    tsne_df = pd.DataFrame(
        tsne_results, columns=["t-SNE Component 1", "t-SNE Component 2"]
    )

    if savepath:
        feather.write_feather(tsne_df, savepath)
        print(f"Behavior map saved to {savepath}")

    return tsne_df

# Example usage
if __name__ == "__main__":
    data_path = "/mnt/upramdya_data/MD/MultiMazeRecorder/Datasets/Skeleton_TNT/Contact_data/241206_Pooled_contact_data.csv"

    # Compute the behavior map
    behavior_map = compute_behavior_map(
        data_path,
        perplexity=30,
        n_iter=3000,
        explained_variance_threshold=0.95,
        batch_size=1000,
        savepath="/mnt/upramdya_data/MD/MultiMazeRecorder/Datasets/Skeleton_TNT/TSNE/241206_behavior_map.feather",
    )

    print(behavior_map.head())