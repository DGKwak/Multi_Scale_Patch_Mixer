import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

def get_activation(activation):
    if activation == "relu":
        return nn.ReLU()
    elif activation == "gelu":
        return nn.GELU()
    elif activation == "leaky":
        return nn.LeakyReLU()
    else:
        raise ValueError(f"Unsupported activation: {activation}")

class PositionalEmbedding(nn.Module):
    def __init__(self, d_feature, max_len):
        super().__init__()

        position_indices = torch.arange(0, max_len, dtype=torch.float).unsqueeze(0).unsqueeze(0)
        
        # (1, C, max_len)
        self.position = position_indices.expand(1, d_feature, max_len)
        self.position.requires_grad = False
        self.register_buffer('positional_embedding', self.position)

    def forward(self, x):
        B, C, N = x.shape

        return x + self.positional_embedding[:, :, :N]

class MlpBlock(nn.Module):
    def __init__(self,
                 in_features: int,
                 hidden_features: int,
                 out_features: int,
                 activation: str,
                 dropout: float,):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, hidden_features),
            get_activation(activation),
            nn.LayerNorm(hidden_features),
            nn.Linear(hidden_features, out_features),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        res = x
        x = self.mlp(x)
        
        x = x + res
        
        return x

class MixerBlock(nn.Module):
    def __init__(self,
                 patch_dim: int,
                 num_patches: int,
                 dropout: float,
                 activation: str,):
        super().__init__()
        
        self.token_mixer = MlpBlock(in_features=num_patches,
                                    hidden_features=num_patches*2,
                                    out_features=num_patches,
                                    activation=activation,
                                    dropout=dropout)
        
        self.channel_mixer = MlpBlock(in_features=patch_dim,
                                      hidden_features=patch_dim*2,
                                      out_features=patch_dim,
                                      activation=activation,
                                      dropout=dropout)
        
    def forward(self, x):
        B, D, N = x.shape
        
        x = x + self.token_mixer(x)

        x = einops.rearrange(x, 'b d n -> b n d')
        x = x + self.channel_mixer(x)
        x = einops.rearrange(x, 'b n d -> b d n')
        
        return x

class FrequencyTemporalMixer(nn.Module):
    def __init__(self,
                 in_channels: int,
                 patch_dim: int,
                 num_layers: int,
                 dropout: float,
                 patch_size: list=[(224, 1), (1, 224)],
                 activation: str="relu"):
        super().__init__()
        
        self.in_channels = in_channels
        self.patch_dim = patch_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.patch_size = patch_size
        self.activation = activation
        
        self.patch_len = [(224//x[0])*(224//x[1]) for x in self.patch_size]
        
        self.channel_fusion = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels,
                      out_channels=self.in_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(self.in_channels),
        )

        self.patch_embed = nn.ModuleList([
            nn.Conv2d(in_channels=self.in_channels, 
                      out_channels=self.patch_dim, 
                      kernel_size=x, 
                      stride=x)
            for x in self.patch_size
        ])
        
        self.pos_embeddings = nn.ModuleList([
            PositionalEmbedding(d_feature=self.patch_dim, max_len=x)
            for x in self.patch_len
        ])
        
        self.spatio_mixer = nn.ModuleList([
            MixerBlock(patch_dim=self.patch_dim,
                       num_patches=self.patch_len[0],
                       dropout=self.dropout,
                       activation=self.activation)
            for _ in range(self.num_layers)
        ])
        
        self.temporal_mixer = nn.ModuleList([
            MixerBlock(patch_dim=self.patch_dim,
                       num_patches=self.patch_len[1],
                       dropout=self.dropout,
                       activation=self.activation)
            for _ in range(self.num_layers)
        ])
        
        self.head = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, patch_dim//2),
            nn.Linear(patch_dim//2, 6),
        )
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        x = self.channel_fusion(x)

        # Spatio_mixer
        s_x = self.patch_embed[0](x)
        s_x = einops.rearrange(s_x, 'b c h w -> b c (h w)')
        
        s_x = self.pos_embeddings[0](s_x)
        
        # Temporal_mixer
        t_x = self.patch_embed[1](x)
        t_x = einops.rearrange(t_x, 'b c h w -> b c (h w)')
        
        t_x = self.pos_embeddings[1](t_x)
        
        for idx in range(self.num_layers):
            s_x = self.spatio_mixer[idx](s_x)
            t_x = self.temporal_mixer[idx](t_x)
        
        # B, D, N1+N2
        # z = torch.cat([s_x, t_x], dim=2)
        z = (s_x + t_x)

        # GAP
        z = torch.mean(z, dim=2, keepdim=False)
        
        s_x = torch.mean(s_x, dim=2, keepdim=False)
        t_x = torch.mean(t_x, dim=2, keepdim=False)
        
        logits = self.head(z)
        
        return logits, [s_x, t_x]