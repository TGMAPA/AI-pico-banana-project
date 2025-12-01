# Import requiered modules and libraries
from src.config.libraries import *
from src.config.config import DEVICE

from src.model.Unet.components.TimeEmbedding import TimeEmbedding
from src.model.Unet.Blocks.DownBlock import DownBlock
from src.model.Unet.Blocks.MidBlock import MidBlock
from src.model.Unet.Blocks.UpBlock import UpBlock 

"""
torch nn.Module con arquitectura Unet + DDPM 
    - Implementar componentes (time embedding.py, downblock.py, midblock.py y upblock.py)
"""
class Unet(nn.Module):
    # Class Constructor
    def __init__(self,
                image_channels: int = 1, 
                downBlock_channels: list = [32, 64, 128, 256],
                midBlock_channels: list = [256, 256, 128],
                upBlock_channels: list[int] = [256, 128, 64, 16],
                down_sample: list[bool] = [True, True, False],
                time_embedding_dimension: int = 128,
            ):
        super().__init__()

        # Original image channels
        self.image_channels = image_channels

        # Unet block's channels
        self.downBlock_channels = downBlock_channels
        self.midBlock_channels = midBlock_channels
        self.upBlock_channels = upBlock_channels

        # Down sample boolean array
        self.down_sample = down_sample
        self.time_embedding_dimension = time_embedding_dimension

        # Up sample boolean array
        self.up_sample = list(reversed(self.down_sample)) # [False, True, True]

        # Time embedder
        self.time_embedder = TimeEmbedding(
            time_embedding_dimension=self.time_embedding_dimension,
            fc_out_dimension=self.time_embedding_dimension
        )

        # Initial Convolutional layer 
        # Map the raw image to a higher-dimensional feature space
        # so the U-Net can extract rich spatial representations.
        self.conv_in = nn.Conv2d(self.image_channels, self.downBlock_channels[0], kernel_size=3, padding=1)
        
        # Implementing DownBlock 
        self.encoder = nn.ModuleList([
            DownBlock(
                in_channels = self.downBlock_channels[i],
                out_channels = self.downBlock_channels[i+1],
                time_embedding_dimension = self.time_embedding_dimension,
                down_sample = self.down_sample[i]
            ) 
            for i in range(len(self.downBlock_channels) - 1 )])
        
        # Implementing MidBlock
        self.bottleNeck = nn.ModuleList([
            MidBlock(
                in_channels = self.midBlock_channels[i],
                out_channels = self.midBlock_channels[i+1],
                time_embedding_dimension = self.time_embedding_dimension
            ) 
            for i in range(len(self.midBlock_channels) - 1 )])

        # Implementing UpBlock
        self.decoder = nn.ModuleList([
            UpBlock(
                in_channels = self.upBlock_channels[i],
                out_channels = self.upBlock_channels[i+1],
                time_embedding_dimension = self.time_embedding_dimension,
                up_sample = self.up_sample[i]
            ) 
            for i in range(len(self.upBlock_channels)  - 1 )])
        
        # Output Convolutional Layer for getting the right image output channels
        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, self.upBlock_channels[-1]),
            nn.Conv2d(self.upBlock_channels[-1], self.image_channels, kernel_size=3, padding=1)
        )

    # Execute forward step
    def forward(self, x: torch.Tensor, time_steps ):

        # Time projections
        time_embedding = self.time_embedder(time_steps=time_steps)

        # -- Apply Initial Convolutional Layer
        out = self.conv_in(x)
        
        # -- Apply Encoder with its downblocks layers
        # Encoder's outputs
        down_outs = []

        for downBlock in self.encoder:
            down_outs.append(out)
            out = downBlock(out, time_embedding)
        
        # -- Apply BottleNeck with its midBlock Layers
        for midBlock in self.bottleNeck:
            out = midBlock(out, time_embedding)    

        # -- Apply Decoder with its upBlock Layers  
        for upBlock in self.decoder:
            down_output = down_outs.pop()
            out = upBlock(out, down_output, time_embedding)

        # -- Apply Final Convolutional Later with Noramlization to restore image channels
        out = self.conv_out(out)

        return out