from torchvision import transforms

from src.config.libraries import *
from src.config.config import N_SAMPLES, IMAGE_HEIGHTS_MEDIAN, IMAGE_WIDTHS_MEDIAN


# Calcular tamaño a recortar para las imagenes 
crop_height = int(np.floor(IMAGE_HEIGHTS_MEDIAN * 0.8))
crop_width = int(np.floor(IMAGE_WIDTHS_MEDIAN * 0.8))

# Transformación de entrenamiento
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_HEIGHTS_MEDIAN, IMAGE_WIDTHS_MEDIAN)),
    transforms.RandomCrop((crop_height, crop_width)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Transformación de prueba
test_transform = transforms.Compose([
    transforms.Resize((IMAGE_HEIGHTS_MEDIAN, IMAGE_WIDTHS_MEDIAN)),
    # transforms.CenterCrop((crop_height, crop_width)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])