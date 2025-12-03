# Import libraries and required modules
from torchvision import transforms
from src.config.libraries import *
from src.config.config import IMAGE_HEIGHTS_MEDIAN, IMAGE_WIDTHS_MEDIAN


# Training transformations
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_HEIGHTS_MEDIAN, IMAGE_WIDTHS_MEDIAN)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Test Transformations
test_transform = transforms.Compose([
    transforms.Resize((IMAGE_HEIGHTS_MEDIAN, IMAGE_WIDTHS_MEDIAN)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])