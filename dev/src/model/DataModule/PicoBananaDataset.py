from torch.utils.data import Dataset

from src.config.config import IO_DATASET_MAP_LOCAL_PATH
from src.config.libraries import *

class PicoBananaDataset(Dataset):
    def __init__(self, annotations_file = IO_DATASET_MAP_LOCAL_PATH, transform=None):
        self.df = pd.read_csv(annotations_file)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Extraer ruta de la imagen de Input
        sample_input_img_path = self.df.iloc[idx, 6]
        sample_input_image = Image.open(sample_input_img_path).convert("RGB")

        # Extraer ruta de la imagen de Output
        #sample_output_img_path = self.df.iloc[idx, 7]
        #sample_ouput_image = Image.open(sample_output_img_path).convert("RGB")

        # APlicar transformacion
        if self.transform:
            sample_input_image = self.transform(sample_input_image)

        return sample_input_image