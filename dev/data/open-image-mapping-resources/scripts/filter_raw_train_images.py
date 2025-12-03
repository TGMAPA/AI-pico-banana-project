"""
This file will delete output images that arent in the filteres local map that will be used.
The main reason for this is to free space of the current data

"""

# Import modules
import os
import csv


# Csv path where "local_input_image" column is contained (this file should be already filtered)
CSV_PATH = r"data/open-image-mapping-resources/source-info/filtered_sft_with_local_source_image_path.csv"

# Path where input images are located for its proper cleaning
BASE_IMAGE_PATH = r"data/openimage_source_images/input/"

# Directories where deletion need tu execute recursive search and deletion
IMAGE_DIRS = [
    BASE_IMAGE_PATH + r"train_0",
    BASE_IMAGE_PATH + r"train_1",
]

# Image formats to search
VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Function that reads csv and extract every input path names contained in the "local_input_image" column
def load_valid_filenames_from_csv(csv_path):
    valid_names = set()

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "local_input_image" not in reader.fieldnames:
            raise ValueError("Column 'local_input_image' doesnt exits")

        for row in reader:
            path = row.get("local_input_image", "")
            if not path:
                continue

            # Get file basename
            basename = os.path.basename(path)
            if basename:
                valid_names.add(basename)

    return valid_names

# Function that iterates over every input image folder and deletes every unvalid file
def clean_image_dirs(image_dirs, valid_names):
    total_files = 0
    to_delete = 0

    for base_dir in image_dirs:
        print(f"\nChecking directory : {base_dir}")

        for root, _, files in os.walk(base_dir):
            for fname in files:
                total_files += 1
                ext = os.path.splitext(fname)[1].lower()

                # Only get images
                if ext not in VALID_EXTS:
                    continue

                if fname not in valid_names:
                    to_delete += 1
                    full_path = os.path.join(root, fname)

                    try:
                        os.remove(full_path)
                    except Exception as e:
                        print(f"[ERROR] Image couldnt be deleted {full_path}: {e}")

if __name__ == "__main__":
    print(f"Loading valid file names from : {CSV_PATH}")
    valid_filenames = load_valid_filenames_from_csv(CSV_PATH)
    print(f"Total files from csv : {len(valid_filenames)}")

    clean_image_dirs(IMAGE_DIRS, valid_filenames)
