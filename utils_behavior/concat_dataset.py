# This code is used to find .feather files in a directory, import them as pandas dataframes, and concatenate them into a single dataframe. The concatenated dataframe is then saved as a new .feather file. The code is used to combine datasets for different experiments and metrics into a single pooled dataset for further analysis.

import pandas as pd
from pathlib import Path

# Define the directory containing the .feather files

data_dir = Path(
    "/mnt/upramdya_data/MD/Ballpushing_Exploration/Datasets/250227_StdContacts_Ctrl_Data/standardized_contacts"
)

# List all .feather files in the directory

feather_files = list(data_dir.glob("*.feather"))

# Import each .feather file as a pandas dataframe and concatenate them into a single dataframe

dfs = [pd.read_feather(file) for file in feather_files]

concatenated_df = pd.concat(dfs)

# Save the concatenated dataframe as a new .feather file

output_file = data_dir / "250228_pooled_standardized_contacts.feather"
concatenated_df.to_feather(output_file)

print(f"Concatenated dataframe saved to {output_file}")
