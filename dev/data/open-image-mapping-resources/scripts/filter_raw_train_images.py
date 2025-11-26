import os
import csv

# Ruta al CSV que tiene la columna local_input_image
CSV_PATH = r"/home/mapa/Documents/Tec/7S/IA-Avanzada/IA-Avanzada-2/FinalProject/AI-pico-banana-project/dev/data/open-image-mapping-resources/source-info/filtered_sft_with_local_source_image_path.csv"

# Carpetas donde están las imágenes que quieres depurar
IMAGE_DIRS = [
    r"/home/mapa/Documents/Tec/7S/IA-Avanzada/IA-Avanzada-2/FinalProject/AI-pico-banana-project/dev/data/openimage_source_images/input/train_0",
    r"/home/mapa/Documents/Tec/7S/IA-Avanzada/IA-Avanzada-2/FinalProject/AI-pico-banana-project/dev/data/openimage_source_images/input/train_1",
]

DRY_RUN = False 
VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def load_valid_filenames_from_csv(csv_path):
    """
    Lee el CSV y extrae un conjunto de NOMBRES DE ARCHIVO (basename)
    presentes en la columna local_input_image.
    """
    valid_names = set()

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "local_input_image" not in reader.fieldnames:
            raise ValueError("La columna 'local_input_image' no existe en el CSV.")

        for row in reader:
            path = row.get("local_input_image", "")
            if not path:
                continue

            # Nos quedamos con el basename: 00072da456114f3b.jpg
            basename = os.path.basename(path)
            if basename:
                valid_names.add(basename)

    return valid_names


def clean_image_dirs(image_dirs, valid_names, dry_run=True):
    """
    Recorre las carpetas de imágenes y borra cualquier archivo cuyo basename
    NO esté en valid_names.
    """
    total_files = 0
    to_delete = 0

    for base_dir in image_dirs:
        print(f"\nRevisando carpeta: {base_dir}")

        for root, _, files in os.walk(base_dir):
            for fname in files:
                total_files += 1
                ext = os.path.splitext(fname)[1].lower()

                # Solo nos interesan imágenes
                if ext not in VALID_EXTS:
                    continue

                if fname not in valid_names:
                    to_delete += 1
                    full_path = os.path.join(root, fname)

                    try:
                        os.remove(full_path)
                    except Exception as e:
                        print(f"[ERROR] No se pudo borrar {full_path}: {e}")

if __name__ == "__main__":
    print(f"Cargando nombres válidos desde: {CSV_PATH}")
    valid_filenames = load_valid_filenames_from_csv(CSV_PATH)
    print(f"Total de nombres de imagen referenciados en CSV: {len(valid_filenames)}")

    clean_image_dirs(IMAGE_DIRS, valid_filenames, dry_run=DRY_RUN)
