# Resnet Block Component
from src.config.libraries import *

# ResNet Component Implementation
class Resnet(nn.Module):
    def __init__(self, in_channels, out_channels, time_embedding_dimension):
        super().__init__()

        # First Block
        self.conv_block1 = nn.Sequential(
            nn.GroupNorm(1, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )

        # Project ime embedding
        self.time_embedding_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embedding_dimension, out_channels),
        )

        # Second block
        self.conv_block2 = nn.Sequential(
            nn.GroupNorm(1, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )

        # Skip Connection
        if in_channels != out_channels:
            self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip_conv = nn.Identity()
        

    def forward(self, x: torch.Tensor, time_embedding):
        residual = x
        
        # First block: norm -> silu -> conv1 
        h = self.conv_block1(x)

        # --- Project time embedding
        time_embedding = self.time_embedding_mlp(time_embedding)

        # Broadcast 
        time_embedding = time_embedding[:, :, None, None]  # -------- Pending Error
        h = h + time_embedding

        # Second block: norm -> silu -> conv2
        h = self.conv_block2(h)

        # Skip Connection
        skip = self.skip_conv(residual)
        out = h + skip

        return out