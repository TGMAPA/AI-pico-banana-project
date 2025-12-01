# main

from src.model.PicoBanana import PicoBanana 
from src.config.config import DEVICE, MODEL_SERIALIZED_PATH
from src.config.libraries import *


def main():

    # Create model
    model = PicoBanana(
        batch_size = 4,
        num_workers = 32,
        train_proportion = 0.8,
        val_proportion = 0.8 
    )

    start = time.time()

    # Train model
    model.train(
        epochs = 200,
        learning_rate = 1e-4
    )

    end = time.time()

    exec_time_seconds = end - start 
    exec_time_minutes = exec_time_seconds/60 
    exec_time_hrs = exec_time_minutes/60

    print("Executión time  | seconds: ", exec_time_seconds, " | minutes: ", exec_time_minutes, " | hrs: ", exec_time_hrs)

    # Save model
    model.save_model(serialized_object_path_destination = MODEL_SERIALIZED_PATH)
    print("Model was saved in : ", MODEL_SERIALIZED_PATH)

if __name__ == "__main__":
    main()
