# main - test inference

# Load libraries
from src.model.PicoBanana import PicoBanana 
from src.config.config import DEVICE, MODEL_SERIALIZED_PATH
from src.config.libraries import *

from tqdm import tqdm


def main():

    # Create model
    model = PicoBanana(
        batch_size = 4,
        num_workers = 4,
        train_proportion = 0.8,
        val_proportion = 0.8 
    )

    # Save model
    model.load_model( MODEL_SERIALIZED_PATH)
    print("Model was correctly loaded ...")

    # Inference
    generated_imgs = []
    for i in tqdm(range(4)):
        xt = model.inference()      # (1, C, 64, 64) en [0,1]
        img = xt[0].cpu().numpy()         # (C, 64, 64)
        img = (img * 255).clip(0, 255).astype(np.uint8)  # [0,1] -> [0,255] uint8
        generated_imgs.append(img)

    fig, axes = plt.subplots(2, 2, figsize=(5, 5))

    for i, ax in enumerate(axes.flat):
        if i >= len(generated_imgs):
            ax.axis("off")
            continue

        img = generated_imgs[i]  # (C, 64, 64)

        if img.shape[0] == 1:
            # Grayscale: (1, H, W) -> (H, W)
            ax.imshow(img[0], cmap="gray")
        else:
            # RGB: (3, H, W) -> (H, W, 3)
            ax.imshow(np.transpose(img, (1, 2, 0)))

        ax.axis('off')

    plt.tight_layout()
    plt.show()

    print("Inference phase finished...")

if __name__ == "__main__":
    main()