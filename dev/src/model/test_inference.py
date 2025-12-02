# main - test inference

# Load libraries
from src.model.PicoBanana import PicoBanana 
from src.config.config import DEVICE, MODEL_SERIALIZED_PATH, MODEL_NAME, N_INFERENCES_2_EXEC, OUTPUT_INFERENCES_DIR
from src.config.config import BATCH_SIZE, NUM_WORKERS, TRAIN_PROPORTION, VAL_PROPORTION
from src.config.libraries import *


def main():

    # Create model
    model = PicoBanana(
        batch_size = BATCH_SIZE,
        num_workers = NUM_WORKERS,
        train_proportion = TRAIN_PROPORTION,
        val_proportion = VAL_PROPORTION
    )

    # Save model
    model.load_model( MODEL_SERIALIZED_PATH)
    print("Model was correctly loaded from " +MODEL_SERIALIZED_PATH+ " ...")

    # Inference
    generated_imgs = []
    for i in tqdm(range(N_INFERENCES_2_EXEC)):
        xt = model.inference()      # (1, C, 64, 64) en [0,1]
        img = xt[0].cpu().numpy()         # (C, 64, 64)
        img = (img * 255).clip(0, 255).astype(np.uint8)  # [0,1] -> [0,255] uint8
        generated_imgs.append(img)

    fig, axes = plt.subplots(3, 3, figsize=(10, 10))

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
    # plt.show()
    
    
    output_path = os.path.join(OUTPUT_INFERENCES_DIR, MODEL_NAME + "inference_"+str(N_INFERENCES_2_EXEC)+"_grid.png")
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print("Inference phase finished, grid saves in "+output_path+ "...")

if __name__ == "__main__":
    main()
