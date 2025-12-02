from src.config.libraries import *
import os 

MODEL_NAME = "picobanana_model"

# General PATHS and env configuration
IO_DATASET_MAP_LOCAL_PATH = "/home/picobanana/Documents/project/AI-pico-banana-project/dev/data/open-image-mapping-resources/source-info/dataset_with_local_source_image_path.csv"
CHECKPOINTS_DIR_PATH = "src/model/ModelCheckpoints/"
TRAININGLOGS_DIR_PATH = "src/model/TrainingLogs"

INPUT_IMAGES_CSV_INDEX = -1

# Dir path for model serialized objects storage, verify dir existance
MODEL_SERIALIZED_DIR_PATH = "src/model/SerializedObjects"
os.makedirs(MODEL_SERIALIZED_DIR_PATH, exist_ok=True)

MODEL_SERIALIZED_PATH = MODEL_SERIALIZED_DIR_PATH + "/" + MODEL_NAME + "_weights.pth"


# Image parameters obtained from "dev/data/exploration-scripts/image_explore.ipynb"
#N_SAMPLES = int(21896 * .1)
N_SAMPLES = 257730

# IMAGE_SCALE_FACTOR = 0.10
# IMAGE_HEIGHTS_MEDIAN = int(768 * IMAGE_SCALE_FACTOR)
# IMAGE_WIDTHS_MEDIAN = int(1024 * IMAGE_SCALE_FACTOR)
IMAGE_HEIGHTS_MEDIAN = 80
IMAGE_WIDTHS_MEDIAN = 80
IMAGE_CHANNELS = 3

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# DDPM Parameters
#N_T_STEPS = 1000
N_T_STEPS = 500
BETA_0 = 1e-4
BETA_N = 0.02

# N_Self_attention_heads per UNET Block
N_ATTN_HEADS_ENCODER = 4
N_ATTN_HEADS_MIDDLE = 2
N_ATTN_HEADS_DECODER = 10