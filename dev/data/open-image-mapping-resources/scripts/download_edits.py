import os
import csv
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_PATH = "https://ml-site.cdn-apple.com/datasets/pico-banana-300k/nb/"
MAX_WORKERS = 32  

def download_one(row, out_dir, session, retries=3):
    output_image_rel = row.get("output_image")
    if not output_image_rel:
        return False

    output_image_rel = output_image_rel.lstrip("/")
    url = BASE_PATH + output_image_rel

    filename = os.path.basename(output_image_rel)
    out_path = os.path.join(out_dir, filename)

    # Si ya existe, no la volvemos a bajar
    if os.path.exists(out_path):
        return False

    # Reintentos básicos
    for _ in range(retries):
        try:
            resp = session.get(url, timeout=20)
            if resp.status_code == 200:
                with open(out_path, "wb") as imgf:
                    imgf.write(resp.content)
                return True
            else:
                return False
        except Exception:
            continue

    return False


def download_output_images(manifest_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # Leer filas del CSV
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader]

    print(f"Total de filas en el CSV: {len(rows)}")

    # Crear una sesión reutilizable
    with requests.Session() as session:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [
                executor.submit(download_one, row, out_dir, session)
                for row in rows
            ]

            for _ in tqdm(as_completed(futures), total=len(futures), desc=f"Downloading into {out_dir}"):
                pass


if __name__ == "__main__":
    download_output_images(
        "../source-info/filtered_sft_with_local_source_image_path.csv",
        "../../openimage_source_images/output"
    )
