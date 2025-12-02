# Self Attention Component
from src.config.libraries import *


# SelfAttention Component Implementation
class SelfAttention(nn.Module):
    # Class constructor
    def __init__(self, n_channels, n_groups = 8, num_heads = 4):
        super().__init__()
        
        # Number of channels produced by some conv filters
        self.n_channels = n_channels
        
        '''
        num_groups controls how the channels are divided for normalization.
        GroupNorm is used instead of BatchNorm because it is stable even with 
        small batch sizes, which is common in diffusion models.
        '''
        self.n_groups = n_groups

        # Normalize block
        self.normblock = nn.GroupNorm(n_groups, n_channels)

        # Multihead block
        self.multiHeadAttention = nn.MultiheadAttention(embed_dim=self.n_channels, num_heads=num_heads, batch_first=True)

    # Execute forward step
    def forward(self, x: torch.Tensor):
        # Unpack batch dimensions
        batch, channels, height, width = x.shape
        
        # Store original x for final skip proccess
        residual = x

        # Transform to shape (Batch, Channels, Height * Width)
        h = x.reshape(batch, channels, height * width)

        # Apply normalization block
        h = self.normblock(h)

        # Apply transpose for multiheadattention block: (Batch, Channels, H*W) → (Batch, H*W, Channels)
        h = h.transpose(1, 2)

        # Self attention where q = k = v = h
        # We use self-attention here, so the same tensor 'h' is passed as 
        # query, key, and value. This allows every spatial position in the 
        # feature map to attend to all other positions in the same map. 
        # In other words, each pixel can "look at" every other pixel and 
        # decide which ones are important. This helps the model capture 
        # long-range dependencies and global structure in the image.
        attn_out, _ = self.multiHeadAttention(h, h, h)

        # Revert transposed shape into (Batch, Channels, H*W) again
        attn_out = attn_out.transpose(1, 2)

        # Revert reshaped shape into (Batch, Channels, H, W) again
        attn_out  = attn_out.reshape(batch, channels, height, width)

        # Skip
        out = residual + attn_out

        return out