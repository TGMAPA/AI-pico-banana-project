"""
This file will turn a jsonl map file into a classic csv
"""


# Import modules
import pandas as pd
import json


# Jsonl non filtered file path with mapped inputs
SFT_WITH_LOCAL_SOURCE_IMAGE_PATH_JSONL_PATH = "../source-info/sft_with_local_source_image_path.jsonl"

# CSV file created path
SFT_WITH_LOCAL_SOURCE_IMAGE_PATH_CSV_PATH = "../source-info/dataset_with_local_source_image_path.csv"

"""
    Convert a JSONL (JSON Lines) file to CSV and save it.
    
    Parameters
    ----------
    jsonl_path : str
        Path to the input JSONL file.
    csv_path : str
        Path where the CSV file will be saved.
    """
def jsonl_to_csv(jsonl_path: str, csv_path: str) -> None:
    
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            # Parse each line as a separate JSON object
            records.append(json.loads(line))

    # Convert list of dicts to DataFrame
    df = pd.DataFrame(records)

    # Save DataFrame as CSV
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"CSV file saved at: {csv_path}")

jsonl_to_csv(SFT_WITH_LOCAL_SOURCE_IMAGE_PATH_JSONL_PATH, SFT_WITH_LOCAL_SOURCE_IMAGE_PATH_CSV_PATH)