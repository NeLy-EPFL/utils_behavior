import pandas as pd
from tsnecuda import TSNE
import umap
import pyarrow.feather as feather
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

def run_tsne_or_umap(
    combined_data,
    method="tsne",
    perplexity=30,
    n_iter=3000,
    n_neighbors=15,
    min_dist=0.1,
    savepath=None,
):

    # Extract PCA data, metadata, and contact indices
    pca_data = combined_data.filter(regex="^PCA Component").values
    metadata = combined_data.drop(columns=combined_data.filter(regex="^PCA Component").columns)
    contact_indices = combined_data["contact_index"]

    if method == "tsne":
        print("Starting t-SNE with CUDA...")

        # Create the progress callback
        progress_callback = ProgressCallback(n_iter)

        tsne = TSNE(
            n_components=2,  # tsnecuda only supports n_components=2
            perplexity=perplexity,
            n_iter=n_iter,
            learning_rate=200,
            verbose=1
        )
        results = tsne.fit_transform(pca_data)

        # Close the progress bar
        progress_callback.close()

        results_df = pd.DataFrame(
            results, columns=["t-SNE Component 1", "t-SNE Component 2"]
        )

    elif method == "umap":
        print("Starting UMAP...")

        # Create UMAP object
        umap_model = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=2,  # UMAP supports more than 2 components, but we keep it consistent
            metric='euclidean',
            verbose=True
        )

        # Fit and transform the data
        results = umap_model.fit_transform(pca_data)

        results_df = pd.DataFrame(
            results, columns=["UMAP Component 1", "UMAP Component 2"]
        )

    # Combine results with metadata and contact indices
    combined_results_df = pd.concat([results_df, metadata.reset_index(drop=True), pd.DataFrame({"contact_index": contact_indices}).reset_index(drop=True)], axis=1)

    if savepath:
        feather.write_feather(combined_results_df, savepath)
        print(f"Results saved to {savepath}")

    return combined_results_df

# Example usage for Step 2
if __name__ == "__main__":
    pca_data_path = "/mnt/upramdya_data/MD/MultiMazeRecorder/Datasets/Skeleton_TNT/PCA/241206_pca_data.feather"
    results_savepath = "/mnt/upramdya_data/MD/MultiMazeRecorder/Datasets/Skeleton_TNT/TSNE/241206_behavior_map.feather"

    # Load PCA data
    print(f"Loading PCA data from {pca_data_path}...")
    combined_data = feather.read_feather(pca_data_path)

    # Run t-SNE or UMAP on the PCA data
    results_df = run_tsne_or_umap(
        combined_data,
        method="tsne",  # Change to "umap" if you want to use UMAP
        perplexity=30,
        n_iter=3000,
        n_neighbors=15,
        min_dist=0.1,
        savepath=results_savepath,
    )

    print(results_df)