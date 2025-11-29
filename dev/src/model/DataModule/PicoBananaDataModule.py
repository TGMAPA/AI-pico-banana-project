from torch.utils.data import DataLoader

from src.model.DataModule.PicoBananaDataset import PicoBananaDataset

from src.config.libraries import *
from src.config.config import N_SAMPLES

# Data Module para crear dataloaders
class PicoBananaDataModule(L.LightningDataModule):
    def __init__(self, annotations_file, batch_size=64, num_workers=0, train_transform=None, test_transform=None, train_proportion = 0.8, val_proportion = 0.8, seed = 42):
        super().__init__()
        # Atributos generales
        self.batch_size = batch_size
        self.annotations_file = annotations_file
        self.num_workers = num_workers
        self.seed = seed

        # Transform Pipelines 
        self.train_transform = train_transform
        self.test_transform = test_transform

        # Proporciones de split
        self.val_proportion = val_proportion
        self.train_proportion = train_proportion

        # Crear dataloaders
        self.prepare_data()
        self.setup()

    def setup(self, stage=None):
        # Calcular tamaños para hacer los splits
        train_size = int(N_SAMPLES * self.train_proportion)
        test_size = N_SAMPLES - train_size
        val_size = int(train_size * (1 - self.val_proportion))
        train_size = train_size - val_size

        print(f"Split sizes -> Train: {train_size}, Val: {val_size}, Test: {test_size}")

        # Usar generador de datos con una semilla definida
        generator = torch.Generator().manual_seed(self.seed)

        # Ejecutar el split en términos de indices del datasets
        train_idx, val_idx, test_idx = torch.utils.data.random_split(
            range(N_SAMPLES),
            [train_size, val_size, test_size],
            generator=generator
        )
        
        # Crear datasets independientes con sus transformaciones
        self.train = PicoBananaDataset(self.annotations_file, transform=self.train_transform)
        self.valid = PicoBananaDataset(self.annotations_file, transform=self.test_transform)
        self.test = PicoBananaDataset(self.annotations_file, transform=self.test_transform)

        # Extraer los subconjuntos de acuerdo con los indices calculados en el split
        self.train = torch.utils.data.Subset(self.train, train_idx.indices)
        self.valid = torch.utils.data.Subset(self.valid, val_idx.indices)
        self.test = torch.utils.data.Subset(self.test, test_idx.indices)

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