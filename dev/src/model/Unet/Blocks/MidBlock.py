# Import requiered modules and libraries
from src.config.libraries import *

from src.model.Unet.components.ResNet import Resnet
from src.model.Unet.components.SelfAttention import SelfAttention


# Middle Block - model's latent space
class MidBlock(nn.Module):
    # Class Constructor
    def __init__(self, in_channels, out_channels, time_embedding_dimension, use_attention = True, down_sample = True, n_heads = 4):
        super().__init__()

        # FirstResnet Block
        self.resnet1 = Resnet(in_channels, out_channels, time_embedding_dimension)

        # Attention Block for model's bottleneck
        self.attn   = SelfAttention(out_channels, num_heads=n_heads) if use_attention else nn.Identity()

        # Second Resnet Block
        self.resnet2 = Resnet(in_channels, out_channels, time_embedding_dimension)

    # Execute forward step
    def forward(self, x: torch.Tensor, time_embedding):
        """
        x:     (Batch, Channels, Height, Width)
        t_emb: (Batch, time_embedding_dimension)

        """

        # Compute first resnet Block
        x = self.resnet1(x, time_embedding)

        # Self-attention over the bottleneck features
        x = self.attn(x)

        # Compute second resnet Block
        x = self.resnet2(x, time_embedding)

        return x