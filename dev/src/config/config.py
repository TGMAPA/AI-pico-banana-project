# Import libraries and required modules
from src.config.libraries import *


# ====== Configuration file ====== 

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -- General dataset and data (images) params obtained from "dev/data/exploration-scripts/image_explore.ipynb"
# Image crop transform measures
IMAGE_HEIGHTS_MEDIAN = 80
IMAGE_WIDTHS_MEDIAN = 80
IMAGE_CHANNELS = 3

# Numbers of samples used for dataset split
N_SAMPLES = 257730

# -- Model's Training Phase Parameters
BATCH_SIZE = 4
NUM_WORKERS = 24
TRAIN_PROPORTION = 0.95
VAL_PROPORTION = 0.95
N_EPOCHS = 200
LEARNING_RATE = 1e-4
SEED = 42
EARLY_STOPPING_PATIENCE = 40
TRAINER_ACCELERATOR = 'gpu' if DEVICE == 'cuda' else 'cpu'
TRAINER_PRECISION = "16-mixed"


# -- DDPM Parameters
# Time embedding's params
N_T_STEPS = 500
BETA_0 = 1e-4
BETA_N = 0.02

# N_Self_attention_heads per UNET Block
N_ATTN_HEADS_ENCODER = 4
N_ATTN_HEADS_MIDDLE = 2
N_ATTN_HEADS_DECODER = 10
# N_ATTN_HEADS_ENCODER = 4
# N_ATTN_HEADS_MIDDLE = 4
# N_ATTN_HEADS_DECODER = 4


# -- Model's NAME
# Bool: existent variation in n self attentions per unet block 
VARIABLE_ATTN = True

# Model name
MODEL_NAME = "picobanana_model_"+str(IMAGE_HEIGHTS_MEDIAN)+"_"+str(IMAGE_WIDTHS_MEDIAN)+"_"+str(N_T_STEPS)+"steps_"+str(N_SAMPLES)+"samples_"+"varSelfattn_"+str(VARIABLE_ATTN)


# -- Input Data CSV Params
# File from where data input paths will be extracted for model's train phase
IO_DATASET_MAP_LOCAL_PATH = "data/open-image-mapping-resources/source-info/dataset_with_local_source_image_path.csv"
# Csv column from where image input paths will be extraced
INPUT_IMAGES_CSV_INDEX = -1


# -- Lightning paths and params
# Lightning model's checkpoints path
CHECKPOINTS_DIR_PATH = "src/model/ModelCheckpoints/"

# Lightning model's training logs path 
TRAININGLOGS_DIR_PATH = "src/model/TrainingLogs"
os.makedirs(TRAININGLOGS_DIR_PATH, exist_ok = True)

# Ploted metrics dir path
METRICS_PLOTS_OUTPUT_DIR_PATH = "src/model/Metrics_Plots"
os.makedirs(METRICS_PLOTS_OUTPUT_DIR_PATH, exist_ok = True)

# Lightning logger: model's version to plot metrics.csv
METRICS_MODEL_VERSION_TO_PLOT = 30


# -- Model serialization paths
# Dir path for model serialized objects storage and dir existance verification
MODEL_SERIALIZED_DIR_PATH = "src/model/SerializedObjects"
os.makedirs(MODEL_SERIALIZED_DIR_PATH, exist_ok=True)

# Model serialized object "pth" path
MODEL_SERIALIZED_PATH = MODEL_SERIALIZED_DIR_PATH + "/" + MODEL_NAME + "_weights.pth"


# -- Model's Inference Phase Parameters
N_INFERENCES_2_EXEC = 9
OUTPUT_INFERENCES_DIR = "src/"+MODEL_NAME+"_OUTPUT_INFERENCES"
os.makedirs(OUTPUT_INFERENCES_DIR, exist_ok=True)
