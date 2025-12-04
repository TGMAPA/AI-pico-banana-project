# Apple Pico-Banana Dataset Download 
- (https://github.com/apple/pico-banana-400k/tree/main?tab=readme-ov-file)
  
Verify file paths in code for local execution.
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

# Download SFT jsonl base map
wget https://ml-site.cdn-apple.com/datasets/pico-banana-300k/nb/jsonl/sft.jsonl

# Download SFT manifest (output images urls)
wget https://ml-site.cdn-apple.com/datasets/pico-banana-300k/nb/manifest/sft_manifest.txt

# Map urls to local paths
python map_openimage_url_to_local.py #please modify variable is_multi_turn and file paths as needed
```
Desired Output:

```bash
❯ python -m data.open-image-mapping-resources.scripts.map_openimage_url_to_local
Loading metadata mapping (URL → ImageID)...
Loaded 1,743,042 entries from metadata CSV
Indexing local .jpg images under data/openimage_source_images/input...
Scanning subfolders: 0it [00:00, ?it/s]
Indexed 0 local image files
Mapping input URLs to local files...
Processing JSONL: 257730it [00:01, 196396.25it/s]

 Mapping complete.
  Matched successfully: 0
  URL not found in metadata: 0
  ImageID found but file missing locally: 257,730

Output saved to: data/open-image-mapping-resources/source-info/sft_with_local_source_image_path.jsonl
```

# Filter desired Categories from sft_with_local_source_image_path.jsonl
Verify file paths in code for local execution.
```bash
python filter_editCats_sft.py
```

# Filter train images to delete extra samples and free disk space
Verify file paths in code for local execution.
```bash
python filter_raw_train_images.py
```

# Download Outputs for filtered train samples
Verify file paths in code for local execution.
```bash
python download_edits.py
```

# Add local outputs paths with filtered local map 
- "filtered_sft_with_local_source_image_path" and "filtered_dataset_IO_loca.csv" will be created.

Verify file paths in code for local execution.
```bash
python map_localOutputs_to_filtered_local_sft.py
```

# Note for experiment with picobanana's 257k samples training:

> Transform "data/open-image-mapping-resources/source-info/sft_with_local_source_image_path.jsonl" into csv as "data/open-image-mapping-resources/source-info/dataset_with_local_source_image_path.csv" 

```bash
❯ python -m data.open-image-mapping-resources.scripts.jsonl2csv
CSV file saved at: data/open-image-mapping-resources/source-info/dataset_with_local_source_image_path.csv
```



