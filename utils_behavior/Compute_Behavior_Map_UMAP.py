import pandas as pd
from sklearn.decomposition import IncrementalPCA, PCA
import umap
import time
from pathlib import Path
import json

# Configuration section
config = {
    "data_path": "/mnt/upramdya_data/MD/Ballpushing_Exploration/Datasets/250307_StdContacts_Ctrl_noOverlap_Data_cutoff/Transformed/250305_Pooled_FeedingState_Transformed.feather",
    "savepath": "/mnt/upramdya_data/MD/Ballpushing_Exploration/Datasets/250307_StdContacts_Ctrl_noOverlap_Data_cutoff/UMAP/250305_Pooled_FeedingState_InteractionsOnly.feather",
    "n_neighbors": 15,
    "min_dist": 0.1,
    "n_components": 2,
    "explained_variance_threshold": 0.95,
    "batch_size": 1000,
    "use_pca": False,
    "feature_groups": [
        "tracking",
        # "frame",
        # "derivatives",
        # "statistical",
        # "fourier"
    ],
    "include_ball": False,
    "include_random_events": False,
}

print(f"Configuration: {config}")

def save_config(config, savepath):
    """Save configuration settings to a JSON file."""
    config_path = Path(savepath).with_suffix('.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Configuration saved to {config_path}")

def compute_behavior_map(
    data,
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
    explained_variance_threshold=0.95,
    batch_size=1000,
    savepath=None,
    use_pca=True,
    feature_groups=["tracking", "frame"],
    include_ball=False,
    include_random_events=True,
):
    """
    Compute a behavior map using UMAP.

    Parameters:
    - data (pd.DataFrame): The input data.
    - n_neighbors (int): The size of local neighborhood (in terms of number of neighboring sample points) used for manifold approximation.
    - min_dist (float): The effective minimum distance between embedded points.
    - n_components (int): The number of dimensions of the embedded space.
    - explained_variance_threshold (float): The threshold for explained variance to determine the number of PCA components.
    - batch_size (int): The batch size for incremental PCA.
    - savepath (str): The path to save the resulting UMAP embedding.
    - use_pca (bool): Whether to use PCA for dimensionality reduction before UMAP.
    - feature_groups (list): The list of feature groups to include.
    - include_ball (bool): Whether to include ball features.
    - include_random_events (bool): Whether to include random events.
    """
    # Filter data to only include interaction events if include_random_events is False
    if not include_random_events:
        data = data[data["event_type"] == "interaction"]
        print(f"Filtered data to only include interaction events. New shape: {data.shape}")

    # Feature selection configuration
    feature_config = {
        "tracking": [r"_frame\d+_x$", r"_frame\d+_y$"],
        "frame": [r"_frame\d+_velocity$", r"_frame\d+_angle$"],
        "derivatives": [r"_vel_mean$", r"_vel_std$", r"_acc_mean$", r"_acc_std$"],
        "statistical": [r"_mean$", r"_std$", r"_skew$", r"_kurt$"],
        "fourier": [r"_dom_freq$", r"_dom_freq_magnitude$"],
    }

    # Create combined regex pattern
    regex_parts = []
    for group in feature_groups:
        regex_parts.extend(feature_config.get(group, []))
    feature_pattern = "|".join(regex_parts)

    # Extract features and metadata
    feature_columns = data.filter(regex=feature_pattern).columns.tolist()

    if len(feature_columns) == 0:
        raise ValueError("No features found matching the selected feature groups")

    # Add non-regex based features explicitly
    additional_features = []
    if "derivatives" in feature_groups:
        additional_features.extend(
            [
                col
                for col in data.columns
                if any(x in col for x in ["_vel_", "_acc_"])
                and col not in feature_columns
            ]
        )

    if "statistical" in feature_groups:
        additional_features.extend(
            [
                col
                for col in data.columns
                if any(x in col for x in ["_mean", "_std", "_skew", "_kurt"])
                and col not in feature_columns
            ]
        )

    if "fourier" in feature_groups:
        additional_features.extend(
            [
                col
                for col in data.columns
                if "dom_freq" in col and col not in feature_columns
            ]
        )

    if not include_ball:
        print("Excluding ball features...")

        # Remove features that have "centre" in their name
        feature_columns = [col for col in feature_columns if "centre" not in col]
        additional_features = [
            col for col in additional_features if "centre" not in col
        ]

    if len(feature_columns) == 0 and len(additional_features) == 0:
        raise ValueError("No features found matching the selected feature groups")

    feature_columns = feature_columns + additional_features
    features = data[feature_columns].values

    event_indices = (
        data["event_id"]
        if "event_id" in data.columns
        else pd.Series([None] * len(data))
    )
    metadata = data.drop(columns=feature_columns).drop(
        columns=["event_id"], errors="ignore"
    )

    print(f"Processing {len(features)} samples with {len(feature_columns)} features...")

    # Save the list of features
    feature_list = pd.DataFrame(feature_columns)
    feature_list.to_csv(savepath.replace(".feather", "_features.csv"), index=False)

    print(f"Metadata columns: {metadata.columns}")

    # Save the metadata
    metadata.to_csv(savepath.replace(".feather", "_metadata.csv"), index=False)

    # Replace NaNs with 9999
    features = features.astype(float)
    features = pd.DataFrame(features)
    features = features.fillna(9999)
    features = features.values

    # PCA processing
    if use_pca:
        print("Applying Incremental PCA...")
        pca = PCA().fit(features)
        cumulative_variance = pca.explained_variance_ratio_.cumsum()
        n_components_pca = (
            cumulative_variance < explained_variance_threshold
        ).sum() + 1

        ipca = IncrementalPCA(n_components=n_components_pca, batch_size=batch_size)
        for batch in range(0, len(features), batch_size):
            ipca.partial_fit(features[batch : batch + batch_size])
        processed_features = ipca.transform(features)
    else:
        print("Skipping PCA...")
        processed_features = features

    # UMAP processing
    print("Running UMAP...")
    umap_model = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric="euclidean",
        verbose=True,
        low_memory=True,
    )
    umap_results = umap_model.fit_transform(processed_features)

    # Compile results
    result_df = pd.concat(
        [
            pd.DataFrame(
                umap_results, columns=[f"UMAP{i+1}" for i in range(n_components)]
            ),
            metadata.reset_index(drop=True),
            pd.DataFrame({"event_id": event_indices}),
        ],
        axis=1,
    )

    if savepath:
        result_df.to_feather(savepath)
        print(f"Saved to {savepath}")

    return result_df

if __name__ == "__main__":
    # Check if directory exists and if not create it
    savedir = Path(config["savepath"]).parent
    savedir.mkdir(parents=True, exist_ok=True)

    # Save configuration settings
    save_config(config, config["savepath"])

    # Load your data
    print(f"Loading data from {config['data_path']}...")
    data = pd.read_feather(config["data_path"])

    # Compute the behavior map
    behavior_map = compute_behavior_map(
        data,
        n_neighbors=config["n_neighbors"],
        min_dist=config["min_dist"],
        n_components=config["n_components"],
        explained_variance_threshold=config["explained_variance_threshold"],
        batch_size=config["batch_size"],
        savepath=config["savepath"],
        use_pca=config["use_pca"],
        feature_groups=config["feature_groups"],
        include_ball=config["include_ball"],
        include_random_events=config["include_random_events"],
    )

    print(behavior_map)