# Import requiered modules and libraries
from src.config.libraries import *
from src.model.Unet.components.ResNet import Resnet
from src.model.Unet.components.SelfAttention import SelfAttention


# Down Block Encoder Block
class DownBlock(nn.Module):
    # Class Constructor
    def __init__(self, in_channels, out_channels, time_embedding_dimension, use_attention = True, down_sample = True, n_heads = 4):
        super().__init__()
        
        # Resnet Block
        self.resnet = Resnet(in_channels, out_channels, time_embedding_dimension)

        # Attention Block
        self.attn   = SelfAttention(out_channels, num_heads=n_heads) if use_attention else nn.Identity()

        '''
        Down Sample Layer with URL standard pattern (kernel_size=4, stride=2, padding=1)
        The downsampling layer keeps the same number of channels because its only job 
        is to reduce spatial resolution (H, W), not to change the feature depth. 
        Channel changes are already handled by the ResNet block before this step, 
        so here we keep in_channels == out_channels.
        '''
        self.down_sample_conv = nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1) if down_sample else nn.Identity()

    # Execute forward step
    def forward(self, x: torch.Tensor, time_embedding):
        """
        x:     (Batch, Channels, Height, Width)
        t_emb: (Batch, time_embedding_dimension)
        
        """
        
        # Compute resnet Block
        x = self.resnet(x, time_embedding)
        
        # Compute self attention block
        x = self.attn(x)

        # Compute down sampling
        x = self.down_sample_conv(x)
        return x