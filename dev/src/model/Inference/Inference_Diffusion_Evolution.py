# Plot Inference diffursion evolution as mp4

# == Main for model's inference phase as mp4

# Import libraries and required modules
from src.model.PicoBanana import PicoBanana 
from src.config.config import MODEL_SERIALIZED_PATH, MODEL_NAME, N_INFERENCES_2_EXEC, OUTPUT_INFERENCES_DIR
from src.config.config import BATCH_SIZE, NUM_WORKERS, TRAIN_PROPORTION, VAL_PROPORTION
from src.config.libraries import *


def save_video(images, output_path = os.path.join(OUTPUT_INFERENCES_DIR, MODEL_NAME + "_inference_diffProcess.mp4"), fps=30):
    """
    images: lista de arrays numpy uint8 (H, W, C) en RGB
    """
    height, width, _ = images[0].shape

    # Define video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for img in images:
        # Convert RGB -> BGR for OpenCV
        frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        video.write(frame)

    video.release()
    print("video saved in", os.path.join(OUTPUT_INFERENCES_DIR, MODEL_NAME +"_inference_diffProcess.mp4"))

# Function for model's inference execution
def execute_inference_diffusion_process( 
        batch_size = BATCH_SIZE,
        num_workers = NUM_WORKERS,
        model_serialized_path = MODEL_SERIALIZED_PATH
    ):
    # Create model
    model = PicoBanana(
        batch_size = batch_size,
        num_workers = num_workers,
        train_proportion = TRAIN_PROPORTION,
        val_proportion = VAL_PROPORTION
    )

    # Save model
    model.load_model(model_serialized_path)
    print("Model was correctly loaded from " +model_serialized_path+ " ...")

    # Get Inference as vector
    img_diffusion_vector = model.inference(return_diffusion_vector = True)

    # Save video
    save_video(img_diffusion_vector)	

if __name__ == "__main__":
    execute_inference_diffusion_process()
