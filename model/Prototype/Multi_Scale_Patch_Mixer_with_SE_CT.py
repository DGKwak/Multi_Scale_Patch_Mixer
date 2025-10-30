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

class Reweight(nn.Module):
    def __init__(self,
                 num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        
        self.alpha = nn.Parameter(torch.ones(self.num_layers), requires_grad=True)

    def forward(self, output_list):
        weights_alpha = F.softmax(self.alpha[:self.num_layers], dim=0)
        
        z = torch.zeros_like(output_list[0])
        
        for i in range(self.num_layers):
            z = z + output_list[i] * weights_alpha[i]

        return z

class MlpBlock(nn.Module):
    def __init__(self,
                 transpose: int,
                 in_features: int,
                 hidden_features: int,
                 out_features: int,
                 activation: str,
                 dropout: float,
                 residual: bool = True):
        super().__init__()
        
        self.transpose = transpose
        self.residual = residual
        
        self.mlp = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, hidden_features),
            get_activation(activation),
            nn.LayerNorm(hidden_features),
            nn.Linear(hidden_features, out_features),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        x = torch.transpose(x, self.transpose, -1)
        
        res = x
        x = self.mlp(x)
        
        if self.residual:
            x = x + res
        
        x = torch.transpose(x, self.transpose, -1)
        
        return x

class SEBlock(nn.Module):
    def __init__(self, 
                 transpose: int, 
                 in_channels: int,
                 fc_channels: int, 
                 reduction: int = 4):
        super().__init__()

        self.transpose = transpose

        self.squeeze = nn.Conv1d(in_channels, 1, kernel_size=1)
        self.fc = nn.Sequential(
            nn.Linear(fc_channels, fc_channels // reduction),
            nn.ReLU(),
            nn.Linear(fc_channels // reduction, fc_channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = torch.transpose(x, self.transpose, -1)

        z = self.squeeze(x)
        z = z.reshape(z.shape[0], -1)
        z = self.fc(z)

        z = z.unsqueeze(1)
        x = x * z

        x = torch.transpose(x, self.transpose, -1)

        return x

class MultiscaleMixer(nn.Module):
    """
    Multi-Scale Patch Mixer with SE Block 
    
    Architecture:
        Input -> Patch Embedding -> Positional Embedding -> [Channel Mixer -> Inter & Intra Mixer -> Inter & Intra SE -> Token Mixer] x L -> Reweight -> Head
    
    Args:
        in_channels (int): Input data channels
        patch_dim (int): Dimension of patch embedding
        num_layers (int): Number of Mixer layers
        dropout (float): Dropout rate
        patches (list): List of patch sizes [(height, width), ...]
        act (str): Activation function name ('relu', 'gelu', 'leaky')
    """
    def __init__(self,
                 in_channels,
                 patch_dim,
                 num_layers,
                 dropout,
                 patches=[(224, 2), (224, 4)],
                 act='gelu',):
        super().__init__()
        
        self.in_channels = in_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.patches = patches
        self.act = act
        self.num_patches = []

        for x in self.patches:
            self.num_patches.append(224//x[1] * 224//x[0])
        
        self.patch_embedding = nn.ModuleList([
            nn.Conv2d(in_channels=self.in_channels,
                      out_channels=patch_dim,
                      kernel_size=x,
                      stride=x)
            for x in self.patches
        ])
        
        self.positional_embedding = nn.ModuleList([
            PositionalEmbedding(d_feature=patch_dim, max_len=x)
            for x in self.num_patches
        ])
        
        self.token_mixer = nn.ModuleList([
            nn.ModuleList([
                MlpBlock(2, x, x*2, x, self.act, self.dropout)
                for _ in range(self.num_layers)
            ])
            for x in self.num_patches
        ])

        self.inter_mixer = nn.ModuleList([
            nn.ModuleList([
                MlpBlock(2, x, x*2, x, self.act, self.dropout, residual=False)
                for _ in range(self.num_layers)
            ])
            for x in self.num_patches
        ])

        self.intra_mixer = nn.ModuleList([
            nn.ModuleList([
                MlpBlock(1, patch_dim, patch_dim*2, patch_dim, self.act, self.dropout, residual=False)
                for _ in range(self.num_layers)
            ])
            for _ in range(len(self.patches))
        ])

        self.channel_mixer = nn.ModuleList([
            nn.ModuleList([
                MlpBlock(1, patch_dim, patch_dim*2, patch_dim, self.act, self.dropout)
                for _ in range(self.num_layers)
            ])
            for _ in range(len(self.patches))
        ])

        self.inter_SE = nn.ModuleList([
            nn.ModuleList([
                SEBlock(2, patch_dim, x)
                for _ in range(self.num_layers)
            ])
            for x in self.num_patches
        ])

        self.intra_SE = nn.ModuleList([
            nn.ModuleList([
                SEBlock(1, x, patch_dim)
                for _ in range(self.num_layers)
            ])
            for x in self.num_patches
        ])

        self.reweight = Reweight(self.num_layers)
        
        self.a = nn.Parameter(torch.ones(len(self.patches))*0.5, requires_grad=True)
        
        self.head = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, patch_dim//2),
            nn.Linear(patch_dim//2, 6)
        )
        
    def forward(self, x):
        Mixer_output = []
        
        # Apply Multi-Scale Patch
        for p_idx in range(len(self.patches)):
            # Patch Embedding
            z = self.patch_embedding[p_idx](x)
            z = einops.rearrange(z, 'b c h w -> b c (h w)')

            # Positional Embedding
            z = self.positional_embedding[p_idx](z)

            inter_outputs, intra_outputs = [], []
            inter, intra = z, z
            for l_idx in range(self.num_layers):
                # Token Mixer
                inter = self.channel_mixer[p_idx][l_idx](inter)
                intra = self.channel_mixer[p_idx][l_idx](intra)

                # Inter & Intra Mixer
                inter = self.inter_mixer[p_idx][l_idx](inter)
                inter = self.inter_SE[p_idx][l_idx](inter)
                intra = self.intra_mixer[p_idx][l_idx](intra)
                intra = self.intra_SE[p_idx][l_idx](intra)

                # Channel Mixer
                inter = self.token_mixer[p_idx][l_idx](inter)
                intra = self.token_mixer[p_idx][l_idx](intra)

                inter_outputs.append(inter)
                intra_outputs.append(intra)

            # Reweighting
            inter_z = self.reweight(inter_outputs)
            intra_z = self.reweight(intra_outputs)
            
            Mixer_output.append((inter_z, intra_z))
        
        patch_output = []
        for i in range(len(self.patches)):
            patch_output.append(self.a[i] * Mixer_output[i][0] + (1 - self.a[i]) * Mixer_output[i][1])
        
        # Concatenate Multi-Scale Patch
        z = torch.cat(patch_output, dim=2)  # (B, C, N1+N2)

        # GAP
        x = torch.mean(z, dim=2, keepdim=False)

        logit = self.head(x)
        
        return logit, patch_output