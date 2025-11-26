import os
import requests
from tqdm import tqdm

def download_from_url_list(manifest_path, out_dir):
    # Crear carpeta de salida si no existe
    os.makedirs(out_dir, exist_ok=True)

    # Leer todas las líneas (URLs) del .txt
    with open(manifest_path, "r") as f:
        urls = [line.strip() for line in f if line.strip()]

    print(f"Encontradas {len(urls)} URLs en {manifest_path}")

    # Descargar cada URL
    for url in tqdm(urls, desc=f"Downloading into {out_dir}"):
        # Nombre del archivo a partir de la URL (ej: 1.png, 10.png, 100.png)
        filename = url.split("/")[-1]
        out_path = os.path.join(out_dir, filename)

        # Si ya existe, lo saltamos
        if os.path.exists(out_path):
            continue

        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                with open(out_path, "wb") as imgf:
                    imgf.write(r.content)
            else:
                print(f"Error HTTP {r.status_code} para {url}")
        except Exception as e:
            print(f"Error descargando {url}: {e}")


if __name__ == "__main__":
    download_from_url_list("sft.txt", "../../openimage_source_images/ouput")
