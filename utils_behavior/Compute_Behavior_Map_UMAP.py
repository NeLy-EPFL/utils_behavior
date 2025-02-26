import pandas as pd
from sklearn.decomposition import IncrementalPCA, PCA
import umap
import time

def compute_behavior_map(
    data,
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
    explained_variance_threshold=0.95,
    batch_size=1000,
    savepath=None,
    use_pca=True,
    feature_groups=['tracking', 'frame'],
    include_ball=False,
):
    # Feature selection configuration
    feature_config = {
        'tracking': [r'_frame\d+_x$', r'_frame\d+_y$'],
        'frame': [r'_frame\d+_velocity$', r'_frame\d+_angle$']
    }
    
    if not include_ball:
        print("Excluding ball features...")
        
        # Remove features that have "centre" in their name
        feature_config['tracking'] = [f for f in feature_config['tracking'] if 'centre' not in f]
        feature_config['frame'] = [f for f in feature_config['frame'] if 'centre' not in f]
    
    # Create combined regex pattern
    regex_parts = []
    for group in feature_groups:
        regex_parts.extend(feature_config.get(group, []))
    feature_pattern = '|'.join(regex_parts)
    
    # Extract features and metadata
    feature_columns = data.filter(regex=feature_pattern).columns
    if len(feature_columns) == 0:
        raise ValueError("No features found matching the selected feature groups")
        
    features = data[feature_columns].values
    event_indices = data['event_id'] if 'event_id' in data.columns else pd.Series([None]*len(data))
    metadata = data.drop(columns=feature_columns).drop(columns=['event_id'], errors='ignore')
    
    print(f"Processing {len(features)} samples with {len(feature_columns)} features...")
    
    # Save the list of features
    feature_list = pd.DataFrame(feature_columns)
    feature_list.to_csv(savepath.replace('.feather', '_features.csv'), index=False)
    
    print(f"Metadata columns: {metadata.columns}")
    
    # save the metadata
    
    metadata.to_csv(savepath.replace('.feather', '_metadata.csv'), index=False)
    
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
        n_components_pca = (cumulative_variance < explained_variance_threshold).sum() + 1
        
        ipca = IncrementalPCA(n_components=n_components_pca, batch_size=batch_size)
        for batch in range(0, len(features), batch_size):
            ipca.partial_fit(features[batch:batch+batch_size])
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
        metric='euclidean',
        verbose=True,
        low_memory=True
    )
    umap_results = umap_model.fit_transform(processed_features)

    # Compile results
    result_df = pd.concat([
        pd.DataFrame(umap_results, columns=[f"UMAP{i+1}" for i in range(n_components)]),
        metadata.reset_index(drop=True),
        pd.DataFrame({'event_id': event_indices})
    ], axis=1)

    if savepath:
        result_df.to_feather(savepath)
        print(f"Saved to {savepath}")

    return result_df

# Example usage
if __name__ == "__main__":

    data_path = "/mnt/upramdya_data/MD/MultiMazeRecorder/Datasets/250220_StdContacts_Ctrl_Data/Transformed/230704_FeedingState_1_AM_Videos_Tracked_standardized_contacts_Transformed.feather"

    # Load your data
    print(f"Loading data from {data_path}...")
    data = pd.read_feather(data_path)

    # Compute the behavior map
    behavior_map = compute_behavior_map(
        data,
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        explained_variance_threshold=0.95,
        batch_size=1000,
        savepath="/mnt/upramdya_data/MD/MultiMazeRecorder/Datasets/250220_StdContacts_Ctrl_Data/UMAP/240225_UMAPTest.feather",
        use_pca=False,
        include_ball=False
    )

    print(behavior_map)