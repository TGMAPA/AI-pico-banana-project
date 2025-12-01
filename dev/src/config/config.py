from src.config.libraries import *

# General PATHS and env configuration
IO_DATASET_MAP_LOCAL_PATH = "/home/mapa/Documents/Tec/7S/IA-Avanzada/IA-Avanzada-2/FinalProject/AI-pico-banana-project/dev/data/open-image-mapping-resources/source-info/filtered_dataset_IO_local.csv"

# Image parameters obtained from "dev/data/exploration-scripts/image_explore.ipynb"
N_SAMPLES = 21896 * .25
IMAGE_HEIGHTS_MEDIAN = 768
IMAGE_WIDTHS_MEDIAN = 1024

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')