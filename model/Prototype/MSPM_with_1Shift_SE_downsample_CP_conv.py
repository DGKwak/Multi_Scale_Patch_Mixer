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
        self.position = position_indices.repeat(1, d_feature, 1).clone()
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
                 in_features: int,
                 out_features: int,
                 activation: str,
                 dropout: float,):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(in_features, out_features),
            get_activation(activation),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):        
        x = self.mlp(x)
        
        return x

class ShiftBlock(nn.Module):
    def __init__(self,
                 patch_dim:int, 
                 num_patches:int,
                 shift:list =[-1, 0, 1],
                 shift_size:int =3,
                 dropout:float =0.1,
                 act='relu'):
        super().__init__()

        self.shift = shift
        self.shift_size = shift_size
        self.dropout = dropout
        self.act = act

        self.channel_mixer_S = nn.Sequential(
            nn.LayerNorm(patch_dim),
            MlpBlock(patch_dim, patch_dim, self.act, self.dropout)
        )
        self.channel_projection = nn.Conv1d(num_patches, num_patches, kernel_size=3, stride=1, padding=1)
        
        # SE Block
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(num_patches, num_patches//8),
            get_activation(self.act),
            nn.Linear(num_patches//8, num_patches),
            nn.Sigmoid()
        )

        self.channel_mixer_F = nn.Sequential(
            MlpBlock(patch_dim,  patch_dim*2, self.act, self.dropout),
            nn.Linear(patch_dim*2, patch_dim),
            )

    def forward(self, x):
        B, N, C = x.shape

        res = x
        x = self.channel_mixer_S(x)

        x_shift = channel_shift(x, shift=self.shift, shift_size=self.shift_size)

        x_shift = self.channel_projection(x_shift)

        se = self.squeeze(x_shift)
        se = einops.rearrange(se, 'b n 1 -> b n')
        ex = self.excitation(se)
        ex = einops.rearrange(ex, 'b n -> b n 1')
        z = x_shift * ex

        z = self.channel_mixer_F(z) + res

        return z

class Downsample(nn.Module):
    def __init__(self,
                 in_channels:int):
        super().__init__()

        self.norm = nn.LayerNorm(in_channels*2)
        self.reduction = nn.Conv1d(in_channels*2,
                                   in_channels,
                                   kernel_size=1,
                                   stride=1,)

    def forward(self, x):
        B, C, N = x.shape

        x0 = x[:, :, 0::2]
        x1 = x[:, :, 1::2]
        x = torch.cat([x0, x1], dim=1)  # 채널 방향으로 concat

        x = einops.rearrange(x, 'b c n -> b n c')
        x = self.norm(x)
        x = einops.rearrange(x, 'b n c -> b c n')
        x = self.reduction(x)

        return x

class BasicLayer(nn.Module):
    def __init__(self,
                 patch_dim:int,
                 num_patches:int,
                 num_layers:int,
                 shift:list =[-1, 0, 1],
                 shift_size:int =3,
                 dropout:float =0.1,
                 downsample:bool=False,
                 act='relu'):
        super().__init__()

        self.Shift = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(patch_dim),
                ShiftBlock(patch_dim=patch_dim,
                        num_patches=num_patches,
                        shift=shift,
                        shift_size=shift_size,
                        dropout=dropout,
                        act=act)
            )
            for _ in range(num_layers)
        ])

        self.TokenMixer = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(num_patches),
                MlpBlock(num_patches, num_patches*2, act, dropout),
                nn.Linear(num_patches*2, num_patches)
            )
            for _ in range(num_layers)
        ])

        if downsample:
            self.downsample = Downsample(in_channels=patch_dim)
        else:
            self.downsample = None
    
    def forward(self, x):
        B, C, N = x.shape
        
        results = []
        for shift, token in zip(self.Shift, self.TokenMixer):
            x = einops.rearrange(x, 'b c n -> b n c')
            x = shift(x)

            x = einops.rearrange(x, 'b n c -> b c n')
            x = token(x) + x
            results.append(x)
        
        if self.downsample is not None:
            x = self.downsample(x)
        
        return x, results

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
                 dropout,
                 num_layers=[2, 2, 2, 2],
                 patches=[(224, 2), (224, 4)],
                 stride=[(224, 2), (224, 4)],
                 shift_size=3,
                 shift=[-1,0,1],
                 num_patches=[112, 56],
                 act='gelu',):
        super().__init__()
        
        self.in_channels = in_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.patches = patches
        self.stride = stride
        self.shift_size = shift_size
        self.shift = shift
        self.act = act
        self.num_patches = num_patches
        
        self.patch_embedding = nn.ModuleList([
            nn.Conv2d(in_channels=self.in_channels,
                      out_channels=patch_dim,
                      kernel_size=x,
                      stride=y)
            for (x, y) in zip(self.patches, self.stride)
        ])
        
        self.positional_embedding = nn.ModuleList([
            PositionalEmbedding(d_feature=patch_dim, max_len=x)
            for x in self.num_patches
        ])
        
        self.blocks = nn.ModuleList([
            nn.ModuleList([
                BasicLayer(patch_dim=patch_dim,
                           num_patches=x//(2**idx),
                           num_layers=l,
                           shift=self.shift,
                           shift_size=self.shift_size,
                           dropout=self.dropout,
                           downsample=False if idx==len(self.num_layers)-1 else True,
                           act=self.act)
                for idx, l in enumerate(self.num_layers)
            ])
            for x in self.num_patches
        ])

        self.reweight = Reweight(sum(self.num_layers))
        
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
            for blk in self.blocks[p_idx]:
                z, blk_layer = blk(z)

                layer_outputs.extend(blk_layer)
            
            # Reweighting
            # z = self.reweight(layer_outputs)
            
            Mixer_output.append(z)
        
        # Concatenate Multi-Scale Patch
        z = torch.cat(Mixer_output, dim=2)  # (B, C, N1+N2)

        # GAP
        x = torch.mean(z, dim=2, keepdim=False)

        logit = self.head(x)
        
        return logit, Mixer_output