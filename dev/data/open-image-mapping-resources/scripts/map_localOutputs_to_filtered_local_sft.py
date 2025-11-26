import pandas as pd
import os

# -------- CONFIGURAR
CSV_INPUT = "/home/mapa/Documents/Tec/7S/IA-Avanzada/IA-Avanzada-2/FinalProject/AI-pico-banana-project/dev/data/open-image-mapping-resources/source-info/filtered_sft_with_local_source_image_path.csv"
CSV_OUTPUT = "/home/mapa/Documents/Tec/7S/IA-Avanzada/IA-Avanzada-2/FinalProject/AI-pico-banana-project/dev/data/open-image-mapping-resources/source-info/filtered_dataset_IO_local.csv"
OUTPUT_FOLDER = "/home/mapa/Documents/Tec/7S/IA-Avanzada/IA-Avanzada-2/FinalProject/AI-pico-banana-project/dev/data/openimage_source_images/output"   # carpeta local donde están las imágenes

# --------------------------------

# Leer CSV
df = pd.read_csv(CSV_INPUT)

# Función para crear ruta local
def build_local_output_path(path):
    if pd.isna(path):
        return None
    filename = os.path.basename(path)  # 32.png
    return os.path.join(OUTPUT_FOLDER, filename)

# Crear nueva columna
df["local_output_image"] = df["output_image"].apply(build_local_output_path)

# Guardar nuevo CSV
df.to_csv(CSV_OUTPUT, index=False)

print("Nueva columna 'local_output_image' agregada correctamente.")
print(f"Archivo guardado como: {CSV_OUTPUT}")
