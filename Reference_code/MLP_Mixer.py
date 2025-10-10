import torch
import numpy as np
from torch import nn
from einops.layers.torch import Rearrange

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)

class MixerBlock(nn.Module):
    def __init__(self, dim, num_patches, torken_dim, channel_dim, dropout=0.):
        super().__init__()
        
        self.token_mixer = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n d -> b d n'),
            FeedForward(num_patches, torken_dim, dropout),
            Rearrange('b d n -> b n d')
        )
        
        self.channel_mixer = nn.Sequential(
            nn.LayerNorm(dim),
            FeedForward(dim, channel_dim, dropout)
        )

    def forward(self, x):
        x = x + self.token_mixer(x)
        x = x + self.channel_mixer(x)
        return x
    
class MLPMixer(nn.Module):
    def __init__(self, in_channels, dim, num_classes, patch_size, image_size, depth, token_dim, channel_dim, dropout=0.):
        super().__init__()
        
        assert image_size % patch_size == 0, "Image dimensions must be divisible by the patch size."
        self.num_patches = (image_size // patch_size) ** 2
        
        self.patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size),
            Rearrange('b c h w -> b (h w) c')
        )

        self.mixer_blocks = nn.ModuleList([
            MixerBlock(dim, self.num_patches, token_dim, channel_dim, dropout) 
            for _ in range(depth)
        ])

        self.layer_norm = nn.LayerNorm(dim)

        self.head = nn.Linear(dim, num_classes)
    
    def forward(self, x):
        x = self.patch_embedding(x)
        
        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)
        
        x = self.layer_norm(x)
        x = x.mean(dim=1)
        x = self.head(x)
        
        return x