import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

# Change log

# - MLP block
# Before : LayerNorm -> Linear -> Activation -> LayerNorm -> Linear -> Dropout
# After : LayerNorm -> Linear -> Activation -> Dropout -> Linear

# - ChannelMixer_F
# MLPblock -> Conv1d

def get_activation(activation):
    if activation == "relu":
        return nn.ReLU()
    elif activation == "gelu":
        return nn.GELU()
    elif activation == "leaky":
        return nn.LeakyReLU()
    else:
        raise ValueError(f"Unsupported activation: {activation}")

def shift_with_padding(x, shift, dim):
    if shift == 0:
        return x
    
    size = x.size(dim)
    abs_shift = abs(shift)

    zeros_size = list(x.size())
    zeros_size[dim] = abs_shift
    zeros = torch.zeros(zeros_size, device=x.device, dtype=x.dtype)

    if shift > 0:
        keep = x.narrow(dim, 0, size - abs_shift)
        shifted = torch.cat((zeros, keep), dim=dim)
    else:
        keep = x.narrow(dim, abs_shift, size - abs_shift)
        shifted = torch.cat((keep, zeros), dim=dim)

    return shifted

def channel_shift(x, shift=[-1, 0, 1], shift_size=3):
    B, D, N = x.shape
    
    x_chunk = torch.chunk(x, shift_size, dim=1)
    shifted_chunks = []

    for chunk, sh in zip(x_chunk, shift):
        # shifted = torch.roll(chunk, shifts=sh, dims=2)
        shifted = shift_with_padding(chunk, sh, dim=2)
        shifted_chunks.append(shifted)
    
    x_shifted = torch.cat(shifted_chunks, dim=1)

    return x_shifted

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
            nn.Dropout(dropout),
            nn.Linear(hidden_features, out_features)
        )
    
    def forward(self, x):
        x = torch.transpose(x, self.transpose, -1)
        
        res = x
        x = self.mlp(x)
        
        if self.residual:
            x = x + res
        
        x = torch.transpose(x, self.transpose, -1)
        
        return x

class ShiftBlock(nn.Module):
    def __init__(self,
                 patch_dim:int, 
                 shift_l:list =[-1, 0, 1],
                 shift_r:list =[1, 0, -1],
                 shift_size:int =3,
                 dropout:float =0.1,
                 act='relu'):
        super().__init__()

        self.shift_l = shift_l
        self.shift_r = shift_r
        self.shift_size = shift_size
        self.dropout = dropout
        self.act = act

        self.channel_mixer_S = MlpBlock(1, patch_dim, patch_dim*2, patch_dim, self.act, self.dropout)

        self.channel_mixer_l = nn.Sequential(
            nn.Linear(patch_dim, patch_dim),
            get_activation(self.act),
            nn.LayerNorm(patch_dim),
        )
        self.channel_mixer_r = nn.Sequential(
            nn.Linear(patch_dim, patch_dim),
            get_activation(self.act),
            nn.LayerNorm(patch_dim),
        )

        self.channel_mixer_F = nn.Conv1d(patch_dim, patch_dim, kernel_size=1, stride=1)

        self.alpha = nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(self, x):
        x = self.channel_mixer_S(x)

        x_l = channel_shift(x, shift=self.shift_l, shift_size=self.shift_size)
        x_r = channel_shift(x, shift=self.shift_r, shift_size=self.shift_size)

        x_l = einops.rearrange(x_l, 'b c n -> b n c')
        x_r = einops.rearrange(x_r, 'b c n -> b n c')
        x_l = self.channel_mixer_l(x_l)
        x_r = self.channel_mixer_r(x_r)
        x_l = einops.rearrange(x_l, 'b n c -> b c n')
        x_r = einops.rearrange(x_r, 'b n c -> b c n')

        z = self.alpha * x_l + (1 - self.alpha) * x_r

        z = self.channel_mixer_F(z) + x

        return z

class MultiscaleMixer(nn.Module):
    """
    Multi-Scale Patch Mixer with Shift Block 
    
    Architecture:
        Input -> Patch Embedding -> Positional Embedding -> [Shift Mixer -> Token Mixer] x L -> Reweight -> Head

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

        self.shift_block = nn.ModuleList([
            nn.ModuleList([
                ShiftBlock(patch_dim, 
                           shift_l=[-1,0,1], 
                           shift_r=[1,0,-1], 
                           shift_size=3, 
                           dropout=self.dropout, 
                           act=self.act)
                for _ in range(self.num_layers)
            ])
            for _ in range(len(self.patches))
        ])

        self.reweight = Reweight(self.num_layers)
        
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

            layer_outputs = []
            for l_idx in range(self.num_layers):
                # Shift Block
                z = self.shift_block[p_idx][l_idx](z)
                
                # Token Mixer
                z = self.token_mixer[p_idx][l_idx](z)

                layer_outputs.append(z)
            
            # Reweighting
            z = self.reweight(layer_outputs)
            
            Mixer_output.append(z)
        
        # Concatenate Multi-Scale Patch
        if len(Mixer_output) > 1:
            z = torch.cat(Mixer_output, dim=2)  # (B, C, N1+N2)

        # GAP
        x = torch.mean(z, dim=2, keepdim=False)

        logit = self.head(x)
        
        return logit, Mixer_output