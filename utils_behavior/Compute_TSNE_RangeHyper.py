import pandas as pd
from tsnecuda import TSNE
import umap
import pyarrow.feather as feather
import time
import numpy as np
from sklearn.model_selection import train_test_split


def stratified_subsample(data, target_size, stratify_column="contact_index"):
    sampling_ratio = target_size / len(data)
    stratified_sample, _ = train_test_split(
        data, train_size=sampling_ratio, stratify=data[stratify_column], random_state=42
    )
    return stratified_sample


class ProgressCallback:
    def __init__(self, n_iter):
        self.n_iter = n_iter
        self.start_time = time.time()

    def __call__(self, iteration, error):
        elapsed_time = time.time() - self.start_time
        print(
            f"Iteration {iteration}/{self.n_iter}, Error: {error:.4f}, Time: {elapsed_time:.2f}s"
        )

    def close(self):
        print("t-SNE completed.")


def run_tsne(
    pca_data,
    metadata,
    contact_indices,
    perplexity,
    n_iter,
    learning_rate,
    savepath=None,
):

    print(
        f"Starting t-SNE with CUDA (perplexity={perplexity}, n_iter={n_iter}, learning_rate={learning_rate})..."
    )

    # Create the progress callback
    progress_callback = ProgressCallback(n_iter)

    tsne = TSNE(
        n_components=2,  # tsnecuda only supports n_components=2
        perplexity=perplexity,
        n_iter=n_iter,
        learning_rate=learning_rate,
        verbose=1,
    )
    results = tsne.fit_transform(pca_data)

    # Close the progress bar
    progress_callback.close()

    results_df = pd.DataFrame(
        results, columns=["t-SNE Component 1", "t-SNE Component 2"]
    )

    # Combine results with metadata and contact indices
    combined_results_df = pd.concat(
        [
            results_df,
            metadata.reset_index(drop=True),
            pd.DataFrame({"contact_index": contact_indices}).reset_index(drop=True),
        ],
        axis=1,
    )

    # Ensure there are no duplicate column names
    combined_results_df = combined_results_df.loc[
        :, ~combined_results_df.columns.duplicated()
    ]

    if savepath:
        feather.write_feather(combined_results_df, savepath)
        print(f"Results saved to {savepath}")

    return combined_results_df


if __name__ == "__main__":
    pca_data_path = "/mnt/upramdya_data/MD/MultiMazeRecorder/Datasets/Skeleton_TNT/PCA/241209_pca_data.feather"
    results_savepath_template = "/mnt/upramdya_data/MD/MultiMazeRecorder/Datasets/Skeleton_TNT/TSNE/241209_behavior_map_perplexity_{perplexity}_niter_{n_iter}_lr_{learning_rate}.feather"

    # Load PCA data
    print(f"Loading PCA data from {pca_data_path}...")
    combined_data = feather.read_feather(pca_data_path)

    # Subsample the data
    target_size = 1000000  # Adjust this value based on your GPU capabilities
    subsampled_data = stratified_subsample(combined_data, target_size)
    print(f"Subsampled data size: {len(subsampled_data)}")

    # Extract PCA data, metadata, and contact indices
    pca_data = subsampled_data.filter(regex="^PCA Component").values
    metadata = subsampled_data.drop(
        columns=subsampled_data.filter(regex="^PCA Component").columns
    )
    contact_indices = subsampled_data["contact_index"]

    # Define hyperparameter sets
    hyperparameter_sets = [
        {"perplexity": 100, "n_iter": 10000, "learning_rate": 200},
        {"perplexity": 100, "n_iter": 10000, "learning_rate": 100},
        {"perplexity": 100, "n_iter": 10000, "learning_rate": 50},
        # Add more hyperparameter sets as needed
    ]

    # Run t-SNE with different hyperparameters
    for params in hyperparameter_sets:
        perplexity = params["perplexity"]
        n_iter = params["n_iter"]
        learning_rate = params["learning_rate"]
        results_savepath = results_savepath_template.format(
            perplexity=perplexity, n_iter=n_iter, learning_rate=learning_rate
        )

        results_df = run_tsne(
            pca_data,
            metadata,
            contact_indices,
            perplexity=perplexity,
            n_iter=n_iter,
            learning_rate=learning_rate,
            savepath=results_savepath,
        )

        print(results_df)
