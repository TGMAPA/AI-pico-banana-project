from src.config.libraries import *

from src.model.DataModule.PicoBananaDataModule import PicoBananaDataModule
from src.model.DataModule import Transformations
from src.config.config import IO_DATASET_MAP_LOCAL_PATH, DEVICE

from src.model.Unet.Unet import Unet
from src.model.LightningModule.PicoBananaModel import PicobananaModel

from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping



class PicoBanana:
    def __init__(
            self,
            batch_size = 32, 
            num_workers = 4, 
            train_proportion = 0.8, 
            val_proportion = 0.8,
            seed = 42
        ):

        self.model = None

        # Create DataModule
        self.dm  = PicoBananaDataModule(
            annotations_file=IO_DATASET_MAP_LOCAL_PATH, 
            batch_size=batch_size, 
            num_workers=num_workers, 
            train_transform= Transformations.train_transform, 
            test_transform= Transformations.test_transform, 
            train_proportion=train_proportion, 
            val_proportion=val_proportion,
            seed=seed
        )

    # Execute model training
    def train(
            self,
            image_channels = 3,
            epochs = 5,
            learning_rate = 1e-4
        ):

        # Unet instance
        unet = Unet(
            image_channels = image_channels
        )

        # Instance picobanana lightning model
        picobananaModel = PicobananaModel(
            model = unet,
            learning_rate = learning_rate
        )

        self.model = picobananaModel

        # Create picobanana's trainer
        trainer = L.Trainer(
            max_epochs = epochs,
            logger = CSVLogger("../TrainingLogs", name="picobanana_model"),
            callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=3)],
            accelerator = "gpu" if DEVICE == "cuda" else "cpu"
        )

        trainer.fit(
            model = self.model,
            train_dataloaders = self.dm.train_dataloader(),
            val_dataloaders = self.dm.val_dataloader()
        )


    def show_batch(self):
        # Obtener un batch de entrenamiento para mostrar
        images = next(iter(self.dm.train_dataloader()))

        # Mostrar 15 imagenes
        plt.figure(figsize=(10, 8))
        plt.axis("off")
        plt.title("Imagenes de Entrenamiento")

        # Crear torch grid
        grid = torchvision.utils.make_grid(
            images[:15],   
            nrow=5,       
            padding=2,
            pad_value=1.0,
            normalize=True
        )

        # Convertir arreglos al formato correcto para mostrarse
        plt.imshow(np.transpose(grid, (1, 2, 0)))
        plt.show()