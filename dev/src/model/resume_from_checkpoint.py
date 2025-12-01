# main

from src.model.PicoBanana import PicoBanana
from src.config.config import DEVICE, MODEL_SERIALIZED_PATH
from src.config.libraries import *


def main():
     # Create model
    model = PicoBanana(
        batch_size = 8,
        num_workers = 16,
        train_proportion = 0.8,
        val_proportion = 0.8
    )

    #model.load_from_checkpoint("src/model/ModelCheckpoints/best_model.ckpt")

    model.load_model(MODEL_SERIALIZED_PATH)

    print("loaded...")


if __name__ == "__main__":
    main()# main

