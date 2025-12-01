from src.config.libraries import *
from src.config.config import DEVICE
from src.model.Unet.components.TimeEmbedding import TimeEmbedding

"""
torch nn.Module con arquitectura Unet + DDPM 
    - Implementar componentes (time embedding.py, downblock.py, midblock.py y upblock.py)
"""

class Unet(nn.Module):
    def __init__(self, time_emebedding_dimension):
        super().__init__()

        self.time_emebedding_dimension = time_emebedding_dimension



    def forward(self, x, time_steps ):

        
        # Time projections
        TimeEmbedder = TimeEmbedding(time_embedding_dimension=self.time_emebedding_dimension)
        time_embedding = TimeEmbedder.forward(time_steps=time_steps)
