from src.config.libraries import *

# Image crop transform measures
IMAGE_HEIGHTS_MEDIAN = 80
IMAGE_WIDTHS_MEDIAN = 80
IMAGE_CHANNELS = 3

# -- Image parameters obtained from "dev/data/exploration-scripts/image_explore.ipynb"
# Numbers of samples used for dataset split
N_SAMPLES = 257730

# Train parameters
BATCH_SIZE = 4
NUM_WORKERS = 24
TRAIN_PROPORTION = 0.95
VAL_PROPORTION = 0.95
N_EPOCHS = 200
LEARNING_RATE = 1e-4
SEED = 42
EARLY_STOPPING_PATIENCE = 200
TRAINER_ACCELERATOR = 'gpu'
TRAINER_PRECISION = "16-mixed"


# DDPM Parameters
N_T_STEPS = 500
BETA_0 = 1e-4
BETA_N = 0.02

VARIABLE_ATTN = False

# Model name
MODEL_NAME = "picobanana_model_"+str(IMAGE_HEIGHTS_MEDIAN)+"_"+str(IMAGE_WIDTHS_MEDIAN)+"_"+str(N_T_STEPS)+"steps_"+str(N_SAMPLES)+"samples_"+"varSelfattn_"+str(VARIABLE_ATTN)

# General PATHS and env configuration
IO_DATASET_MAP_LOCAL_PATH = "data/open-image-mapping-resources/source-info/dataset_with_local_source_image_path.csv"
CHECKPOINTS_DIR_PATH = "src/model/ModelCheckpoints/"
TRAININGLOGS_DIR_PATH = "src/model/TrainingLogs"
os.makedirs(TRAININGLOGS_DIR_PATH, exist_ok = True)
METRICS_PLOTS_OUTPUT_DIR_PATH = "src/model/Metrics_Plots"
os.makedirs(METRICS_PLOTS_OUTPUT_DIR_PATH, exist_ok = True)
METRICS_MODEL_VERSION_TO_PLOT = 30

INPUT_IMAGES_CSV_INDEX = -1

# Dir path for model serialized objects storage, verify dir existance
MODEL_SERIALIZED_DIR_PATH = "src/model/SerializedObjects"
os.makedirs(MODEL_SERIALIZED_DIR_PATH, exist_ok=True)

# Model serialized object "pth" path
MODEL_SERIALIZED_PATH = MODEL_SERIALIZED_DIR_PATH + "/" + MODEL_NAME + "_weights.pth"

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Test Inference Parameters
N_INFERENCES_2_EXEC = 9
OUTPUT_INFERENCES_DIR = "src/"+MODEL_NAME+"_OUTPUT_INFERENCES"
os.makedirs(OUTPUT_INFERENCES_DIR, exist_ok=True)

# N_Self_attention_heads per UNET Block
N_ATTN_HEADS_ENCODER = 4
N_ATTN_HEADS_MIDDLE = 2
N_ATTN_HEADS_DECODER = 10