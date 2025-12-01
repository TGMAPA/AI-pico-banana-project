# main

from src.model.PicoBanana import PicoBanana 


def main():

    model = PicoBanana(
        batch_size = 16,
        num_workers = 4,
        train_proportion = 0.8,
        val_proportion = 0.8 
    )

    # model.train(
    #     epochs = 2,
    #     learning_rate = 1e-4
    # )

    model.show_batch()

if __name__ == "__main__":
    main()