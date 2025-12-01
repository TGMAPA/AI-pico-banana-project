from src.config.libraries import *

MODEL_NAME = "picobanana_model"

# General PATHS and env configuration
IO_DATASET_MAP_LOCAL_PATH = "/home/mapa/Documents/Tec/7S/IA-Avanzada/IA-Avanzada-2/FinalProject/AI-pico-banana-project/dev/data/open-image-mapping-resources/source-info/filtered_dataset_IO_local.csv"
CHECKPOINTS_DIR_PATH = "src/model/ModelCheckpoints/"
TRAININGLOGS_DIR_PATH = "src/model/TrainingLogs"
MODEL_SERIALIZED_PATH = "src/model/SerializedObjects/" + MODEL_NAME + "_weights.pth"


# Image parameters obtained from "dev/data/exploration-scripts/image_explore.ipynb"
N_SAMPLES = int(21896 * .1)

IMAGE_SCALE_FACTOR = 0.10
# IMAGE_HEIGHTS_MEDIAN = int(768 * IMAGE_SCALE_FACTOR)
# IMAGE_WIDTHS_MEDIAN = int(1024 * IMAGE_SCALE_FACTOR)
IMAGE_HEIGHTS_MEDIAN = 64
IMAGE_WIDTHS_MEDIAN = 64
IMAGE_CHANNELS = 3

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# DDPM Parameters
N_T_STEPS = 1000
BETA_0 = 1e-4
BETA_N = 0.02