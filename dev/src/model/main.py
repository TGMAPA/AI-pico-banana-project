# Model - Main file
from src.config.libraries import *
from src.config.config import IO_DATASET_MAP_LOCAL_PATH


from src.model.DataModule.PicoBananaDataModule import PicoBananaDataModule
from src.model.DataModule import Transformations

# Se crea la instancia del modulo de datos con proporción de 80|20 y batch_size de 16
dm : L.LightningDataModule = PicoBananaDataModule(
    annotations_file=IO_DATASET_MAP_LOCAL_PATH, 
    batch_size=32, 
    num_workers=4, 
    train_transform= Transformations.train_transform, 
    test_transform= Transformations.test_transform, 
    train_proportion=0.8, 
    val_proportion=0.8,
    seed=42
)

# Obtener un batch de entrenamiento para mostrar
images = next(iter(dm.train_dataloader()))

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