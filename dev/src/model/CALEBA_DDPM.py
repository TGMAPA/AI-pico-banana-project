# Import libraries and required modules
from src.config.libraries import *
from src.model.DataModule.CALEBADataModule import CALEBDataModule
from src.model.DataModule import Transformations
from src.config.config import DEVICE, N_T_STEPS, BETA_0, BETA_N, TRAINER_PRECISION, TRAINER_ACCELERATOR
from src.config.config import  CHECKPOINTS_DIR_PATH, TRAININGLOGS_DIR_PATH, MODEL_NAME, MODEL_SERIALIZED_PATH
from src.config.config import IMAGE_HEIGHTS_MEDIAN, IMAGE_WIDTHS_MEDIAN, IMAGE_CHANNELS, EARLY_STOPPING_PATIENCE
from src.model.Unet.Unet import Unet
from src.model.LightningModule.PicoBananaModel import PicobananaModel
from src.model.DiffusionReversed.DiffusionReversed import DiffusionReversed
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping


# CALEBA 10 model implementation
class CALEBA_DDPM:
    def __init__(
            self,
            batch_size = 32, 
            num_workers = 4
        ):

        self.model = None

        self.learning_rate = None

        # Create DataModule
        self.dm  = CALEBDataModule(
            batch_size=batch_size, 
            num_workers=num_workers, 
            train_transform= Transformations.train_transform, 
            test_transform= Transformations.test_transform
        )

    # Execute model training
    def train(
            self,
            image_channels = IMAGE_CHANNELS,
            epochs = 5,
            learning_rate = 1e-4
        ):

        self.learning_rate = learning_rate
        
        if self.model is None:
            # Unet instance
            unet = Unet(
                image_channels = image_channels
            )

            # Instance picobanana lightning model
            picobananaModel = PicobananaModel(
                model = unet,
                learning_rate = learning_rate,
                num_timesteps = N_T_STEPS
            )

            self.model = picobananaModel

        # Create picobanana's trainer
        trainer = L.Trainer(
            max_epochs = epochs,
            logger = CSVLogger(TRAININGLOGS_DIR_PATH, name = MODEL_NAME),
            callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=EARLY_STOPPING_PATIENCE),
                       ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1, dirpath=CHECKPOINTS_DIR_PATH, filename="best_model_"+MODEL_NAME)],
            accelerator = TRAINER_ACCELERATOR,
            devices=1,
            precision= TRAINER_PRECISION
        )

        # Execute model's training step
        trainer.fit(
            model = self.model,
            train_dataloaders = self.dm.train_dataloader(),
            val_dataloaders = self.dm.val_dataloader()
        )

    # Method for generating new images with trained model
    def inference(self):

        assert self.model is not None, "Training Phase is missing for model's inference mode" 
        
        difusion_reversed = DiffusionReversed(
            N_T_steps = N_T_STEPS,
            beta_0 = BETA_0,
            beta_N = BETA_N
        )

        self.model.eval()

        # Generate noise sample from N(0,1)
        xt = torch.randn(1, IMAGE_CHANNELS, IMAGE_HEIGHTS_MEDIAN, IMAGE_WIDTHS_MEDIAN).to(DEVICE)
        
        with torch.no_grad():
            for t in reversed(range(N_T_STEPS)):
                noise_pred = self.model(xt, torch.as_tensor(t).unsqueeze(0).to(DEVICE))
                xt, x0 = difusion_reversed.reverse_timestep(xt, noise_pred, torch.as_tensor(t).to(DEVICE))

        # Bring result to [0, 1] for visualization
        xt = torch.clamp(xt, -1., 1.).detach().to(DEVICE)
        xt = (xt + 1) / 2.0  # [-1,1] -> [0,1]

        return xt
        
    # Method for loading pretrained picobanana model
    def load_model(self, serialized_object_path = MODEL_SERIALIZED_PATH):
        unet = Unet(
            image_channels = IMAGE_CHANNELS
        )

        # Instance picobanana lightning model
        picobananaModel = PicobananaModel(
            model = unet,
            learning_rate = self.learning_rate,
            num_timesteps = N_T_STEPS
        )
        # Load weights from serialized object
        picobananaModel.load_state_dict(torch.load(serialized_object_path))

        # Assign Loaded model
        self.model = picobananaModel

        self.model.to(DEVICE)

    # Method for loading model form lightning chekpoint (for posterior retraining)
    def load_from_checkpoint(self, checkpoint_path, learning_rate=1e-4):
        # Create base Unet model
        unet = Unet(
            image_channels=IMAGE_CHANNELS
        )

        # Specify learning rate    
        self.learning_rate = learning_rate

        # Load LightningModule from checkpoint
        picobananaModel = PicobananaModel.load_from_checkpoint(
            checkpoint_path,
            model=unet,
            learning_rate=self.learning_rate
        )

        # Set model as actual
        self.model = picobananaModel

        # Save serialized Model as object 
        torch.save(self.model.state_dict(), MODEL_SERIALIZED_PATH)

        # Send Model to proper device
        self.model.to(DEVICE)

        print(f"Model loaded from checkpoint path : {checkpoint_path}")

        return True

    # Method for saving model as serialized object
    def save_model(self, serialized_object_path_destination = MODEL_SERIALIZED_PATH):
        torch.save(self.model.state_dict(), serialized_object_path_destination)
        return True

    # Method for showing a train dataloader's batch
    def show_batch(self):
        # Get traiing batch for visualization
        images = next(iter(self.dm.train_dataloader()))

        # Show n images
        plt.figure(figsize=(10, 8))
        plt.axis("off")
        plt.title("Imagenes de Entrenamiento")

        # Create torch grid
        grid = torchvision.utils.make_grid(
            images[:15],   
            nrow=5,       
            padding=2,
            pad_value=1.0,
            normalize=True
        )

        # Tranform tensors to arrays correct format for plt visualization
        plt.imshow(np.transpose(grid, (1, 2, 0)))
        plt.show()
