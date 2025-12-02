from torch.utils.data import DataLoader

from src.config.libraries import *
from src.config.config import N_SAMPLES

# Data Module para crear dataloaders
class CIFAR10DataModule(L.LightningDataModule):
    def __init__(self, batch_size=64, num_workers=0, train_transform=None, test_transform=None):
        super().__init__()
        # Atributos generales
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Transform Pipelines 
        self.train_transform = train_transform
        self.test_transform = test_transform
        
        # Crear dataloaders
        self.prepare_data()
        self.setup()

    def setup(self, stage=None):

        # Crear datasets independientes con sus transformaciones
        # 50,000 samples
        self.train = datasets.CIFAR10(
                                    root="data",
                                    train=True,
                                    download=True,
                                    transform=self.train_transform,
                                    target_transform=None
                                )
        # 10,000 samples
        self.valid = datasets.CIFAR10(
                                    root="data",
                                    train=False,
                                    download=True,
                                    transform=self.test_transform,
                                    target_transform=None
                                )
        self.test = datasets.CIFAR10(
                                    root="data",
                                    train=False,
                                    download=True,
                                    transform=self.test_transform,
                                    target_transform=None
                                )

    def train_dataloader(self):
        # Dataloader para el conjunto de entrenamiento
        train_loader = DataLoader(
            dataset=self.train,
            batch_size=self.batch_size,
            drop_last=True,
            shuffle=True,
            num_workers=self.num_workers,
        )
        return train_loader

    def val_dataloader(self):
        # Dataloader para el conjunto de validación
        valid_loader = DataLoader(
            dataset=self.valid,
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=self.num_workers,
        )
        return valid_loader

    def test_dataloader(self):
        # Dataloader para el conjunto de prueba
        test_loader = DataLoader(
            dataset=self.test,
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=self.num_workers,
        )
        return test_loader