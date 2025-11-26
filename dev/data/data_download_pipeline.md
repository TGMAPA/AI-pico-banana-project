


# Apple Pico-Banana Repo Download (https://github.com/apple/pico-banana-400k/tree/main?tab=readme-ov-file)

```bash
# Install awscli if you don't have it (https://aws.amazon.com/cli/)
# Download Open Images packed files 
aws s3 --no-sign-request --endpoint-url https://s3.amazonaws.com cp s3://open-images-dataset/tar/train_0.tar.gz . 
aws s3 --no-sign-request --endpoint-url https://s3.amazonaws.com cp s3://open-images-dataset/tar/train_1.tar.gz . 

# Create folder for extracted images 
mkdir openimage_source_images

# Extract the tar files 
tar -xvzf train_0.tar.gz -C openimage_source_images
tar -xvzf train_1.tar.gz -C openimage_source_images

# Download metadata CSV (ImageID ↔ OriginalURL mapping)  
wget https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv

# Map urls to local paths
python map_openimage_url_to_local.py #please modify variable is_multi_turn and file paths as needed
```

# Filter desired Categories from sft_with_local_source_image_path.jsonl

```bash
python filter_editCats_sft.py
```

# Filter train images to delete extra samples and free disk space

```bash
python filter_raw_train_images.py
```

# Filter train images to delete extra samples and free disk space

```bash
python filter_raw_train_images.py
```

# Download Outputs for filtered train samples

```bash
python download_edits.py
```

# Add local outputs paths with filtered local map (filtered_sft_with_local_source_image_path) and "filtered_dataset_IO_loca.csv" will be created

```bash
python map_localOutputs_to_filtered_local_sft.py
```