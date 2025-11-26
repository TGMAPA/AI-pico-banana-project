# Script for filtering dataset's samples from edit-categories: "rellocate object" and "style-transfer"

import pandas as pd

path = "/home/mapa/Documents/Tec/7S/IA-Avanzada/IA-Avanzada-2/FinalProject/AI-pico-banana-project/dev/data/open-image-mapping-resources/sft_with_local_source_image_path.jsonl"

df_local_map = pd.read_json(path, lines=True)

print("Original size: ", df_local_map.shape)

TARGET_EDIT_TYPES = {
    "Strong artistic style transfer (e.g., Van Gogh/anime/etc.)",
    "Relocate an object (change its position/spatial relation)",
}

df_filtered = df_local_map[df_local_map["edit_type"].isin(TARGET_EDIT_TYPES)]

print("Filtered df size: ", df_filtered.shape)
print(df_filtered["edit_type"].unique())

df_filtered.to_csv("filtered_sft_with_local_source_image_path.csv")