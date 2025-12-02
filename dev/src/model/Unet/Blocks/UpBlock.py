# Import requiered modules and libraries
from src.config.libraries import *
from src.model.Unet.components.ResNet import Resnet
from src.model.Unet.components.SelfAttention import SelfAttention


# Up Block Decoder Block
class UpBlock(nn.Module):
    # Class Constructor
    def __init__(self, in_channels, out_channels, time_embedding_dimension, use_attention = True, up_sample = True, n_heads = 4):
        super().__init__()

        # Up sampling
        self.up_sample_conv = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size = 4, stride = 2, padding = 1) if up_sample else nn.Identity()

        # Resnet Block
        self.resnet = Resnet(in_channels, out_channels, time_embedding_dimension)

        # Attention Block
        self.attn   = SelfAttention(out_channels, num_heads=n_heads) if use_attention else nn.Identity()

    # Execute forward step
    def forward(self, x: torch.Tensor, down_sampling_out,  time_embedding):
        """
        x:     (Batch, Channels, Height, Width)
        down_sampling_out: down sample output for concatenation
        t_emb: (Batch, time_embedding_dimension)
        
        """

        # Apply upsampling 
        x = self.up_sample_conv(x)

        # Concatenate resultant down sample with upsampled x
        x = torch.cat([x, down_sampling_out], dim=1)
        
        # Compute resnet Block
        x = self.resnet(x, time_embedding)
        
        # Compute self attention block
        x = self.attn(x)

        return x