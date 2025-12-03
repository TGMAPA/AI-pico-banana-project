"""
This file will get as input the CSV that contains the mapping of local input paths and output online paths. 
THis output paths will be mapped with the local stored outputs to have the output local path mapped with each dataset 
(should be already filtered) samples.
"""

# Import modules
import pandas as pd
import os


# input csv map from where this script will be mapping its input images with its new local output path  
CSV_INPUT = "data/open-image-mapping-resources/source-info/filtered_sft_with_local_source_image_path.csv"

# csv map that will be created and will contained every filtered sample with its local input and output paths (IO)
CSV_OUTPUT = "data/open-image-mapping-resources/source-info/filtered_dataset_IO_local.csv"

# Local directory that contains output images
OUTPUT_FOLDER = "data/openimage_source_images/output" 

# Read csv
df = pd.read_csv(CSV_INPUT)

# Function for creating local path
def build_local_output_path(path):
    if pd.isna(path):
        return None
    filename = os.path.basename(path)  # 32.png
    return os.path.join(OUTPUT_FOLDER, filename)

# Create new column into csv
df["local_output_image"] = df["output_image"].apply(build_local_output_path)

# Save new csv
df.to_csv(CSV_OUTPUT, index=False)

print("New column added succesfully: 'local_output_image'.")
print(f"File saved in : {CSV_OUTPUT}")
