# == Main for model's training phase

# Import libraries and required modules
from src.model.PicoBanana import PicoBanana 
from src.config.config import MODEL_SERIALIZED_PATH, BATCH_SIZE, NUM_WORKERS, TRAIN_PROPORTION, VAL_PROPORTION
from src.config.config import N_EPOCHS, LEARNING_RATE
from src.config.libraries import *


# Main file for picobanana model training execution
def main():
    # Create model
    model = PicoBanana(
        batch_size = BATCH_SIZE,
        num_workers = NUM_WORKERS,
        train_proportion = TRAIN_PROPORTION,
        val_proportion = VAL_PROPORTION
    )

    # Start time counter
    start = time.time()

    # Train model
    model.train(
        epochs = N_EPOCHS,
        learning_rate = LEARNING_RATE
    )

    # Stop time counter
    end = time.time()

    # Show execution time
    exec_time_seconds = end - start 
    exec_time_minutes = exec_time_seconds/60 
    exec_time_hrs = exec_time_minutes/60
    print("Executión time  | seconds: ", exec_time_seconds, " | minutes: ", exec_time_minutes, " | hrs: ", exec_time_hrs)

    # Save model
    model.save_model(serialized_object_path_destination = MODEL_SERIALIZED_PATH)
    print("Model was saved in : ", MODEL_SERIALIZED_PATH)

if __name__ == "__main__":
    main()
