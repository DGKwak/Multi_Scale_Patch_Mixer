import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.ao.quantization as quantization
import torch.ao.nn.quantized as nnq

from typing import List

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

        res_fp = x + self.positional_embedding[:, :, :N]

        return res_fp

class ShiftBlock(nn.Module):
    def __init__(self,
                 patch_dim:int, 
                 num_patches:int,
                 shift:list =[-1, 0, 1],
                 shift_size:int =3,
                 dropout:float =0.1,
                 act='relu'):
        super().__init__()

        self.ff_add = nnq.FloatFunctional()
        self.ff_mul = nnq.FloatFunctional()
        self.quant_identity = nn.Identity()

        self.shift = shift
        self.shift_size = shift_size
        self.dropout = dropout

        self.channel_mixer_S = nn.Sequential(
            nn.BatchNorm1d(num_patches),
            nn.Linear(patch_dim, patch_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
        )
        self.channel_projection = nn.Sequential(
            nn.Linear(num_patches, num_patches*2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(num_patches*2, num_patches)
        )
        
        # SE Block
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(num_patches, num_patches//8),
            nn.ReLU(),
            nn.Linear(num_patches//8, num_patches),
            nn.Sigmoid()
        )

        self.channel_mixer_F = nn.Sequential(
            nn.Linear(patch_dim, patch_dim*2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(patch_dim*2, patch_dim),
            )

    # Channel Shift Function
    def shift_with_padding(self, x: torch.Tensor, shift: int, dim: int) -> torch.Tensor:        
        if shift == 0:
            return x
        
        x_float = x
        
        size = x_float.size(dim)
        abs_shift = abs(shift)
        
        x_size = list(x_float.size())
        x_size[dim] = abs_shift
        zeros_size_tuple = tuple(x_size)
        
        zeros = torch.zeros(zeros_size_tuple, device=x.device, dtype=torch.float)
        if shift > 0:
            keep = x_float.narrow(dim, 0, size - abs_shift)
            shifted = torch.cat((zeros, keep), dim=dim)
        else:
            keep = x_float.narrow(dim, abs_shift, size - abs_shift)
            shifted = torch.cat((keep, zeros), dim=dim)

        return shifted

    def channel_shift(self, x: torch.Tensor, shift: List[int], shift_size: int) -> torch.Tensor:
        B, D, N = x.shape

        x_chunk = torch.chunk(x, shift_size, dim=1)
        shifted_chunks = []

        for chunk, sh in zip(x_chunk, shift):
            shifted = self.shift_with_padding(chunk, sh, dim=2)
            shifted_chunks.append(shifted)
        
        x_shifted = torch.cat(shifted_chunks, dim=1)

        return x_shifted

    def forward(self, x):
        B, N, C = x.shape

        res = x

        x = self.channel_mixer_S(x)

        x_float = x.float()

        x_shift = self.channel_shift(x_float, shift=self.shift, shift_size=self.shift_size)

        x_shift = self.quant_identity(x_shift)

        x_shift = x_shift.permute(0, 2, 1)  # (B, C, N)
        x_shift = self.channel_projection(x_shift)
        x_shift = x_shift.permute(0, 2, 1)  # (B, N, C)

        se = self.squeeze(x_shift)
        se = se.squeeze()  # (B, N)
        
        if se.dim() == 1:
            se = se.unsqueeze(0)
        
        ex = self.excitation(se)
        ex = ex.unsqueeze(-1)  # (B, N, 1)

        z = self.ff_mul.mul(x_shift, ex)

        z = self.channel_mixer_F(z)
        
        z = self.ff_add.add(z, res)

        return z

class Downsample(nn.Module):
    def __init__(self,
                 in_channels:int,
                 norm:int):
        super().__init__()

        self.norm = nn.BatchNorm1d(norm)
        self.reduction = nn.Conv1d(in_channels,
                                   in_channels,
                                   kernel_size=2,
                                   stride=2,)

    def forward(self, x):
        B, C, N = x.shape

        x = x.permute(0, 2, 1)  # (B, N, C)
        x = self.norm(x)
        x = x.permute(0, 2, 1)  # (B, C, N)
        x = self.reduction(x)

        return x

# Basic Structure of Multi-Scale Patch Shift Mixer
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

        self.ff_add = nnq.FloatFunctional()

        self.Shift = nn.ModuleList([
            nn.Sequential(
                nn.BatchNorm1d(num_patches),
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
                nn.BatchNorm1d(patch_dim),
                nn.Linear(num_patches, num_patches*2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(num_patches*2, num_patches)
            )
            for _ in range(num_layers)
        ])

        if downsample:
            self.downsample = Downsample(in_channels=patch_dim, norm=num_patches)
        else:
            self.downsample = None
    
    def forward(self, x):
        B, C, N = x.shape
        
        results = []
        for shift, token in zip(self.Shift, self.TokenMixer):
            x_shift_in = x.permute(0, 2, 1)  # (B, N, C)
            
            x_shifted = shift(x_shift_in)

            x_tok_in = x_shifted.permute(0, 2, 1)  # (B, C, N)
            x_tok = token(x_tok_in)
            
            x = self.ff_add.add(x_tok, x)

            results.append(x)
        
        if self.downsample is not None:
            x = self.downsample(x)
        
        return x, results

# Multi-Scale Patch Shift Mixer Model
class MultiscaleMixer(nn.Module):
    """
    Multi-Scale Patch Shift Mixer
    
    Architecture:
        Input -> Patch Embedding -> Positional Embedding -> Basic Layer x L -> Reweight -> Head
        Basic Layer: Shift Block -> Token Mixer -> (Downsample)
        Shift Block: Channel Mixer -> Channel Shift -> Channel Projection -> SE Block -> Channel Mixer

    Args:
        in_channels (int): Input data channels
        patch_dim (int): Dimension of patch embedding
        dropout (float): Dropout rate
        num_layers (list): Number of Basic layers [L1, L2, ...]
        patches (list): List of patch sizes [(height, width), ...]
        stride (list): List of patch embedding stride sizes [(height, width), ...]
        shift_size (int): Number of channel groups for shift operation
        shift (list): List of shift values for each channel group
        num_patches (list): List of number of patches for each scale
        act (str): Activation function name ('relu', 'gelu', 'leaky')
    """
    def __init__(self,
                 in_channels:int=3,
                 patch_dim:int=128,
                 dropout:float=0.1,
                 num_layers:list=[2, 2],
                 patches:list=[(224, 2), (224, 4)],
                 stride:list=[(224, 2), (224, 4)],
                 shift_size:int=4,
                 shift:list=[3,-2,2,-3],
                 num_patches:list=[112, 56],
                 act:str='relu'):
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

        self.quant = quantization.QuantStub()
        self.dequant = quantization.DeQuantStub()
        self.quant_flow_in = nn.Identity()
        
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
        
        self.head = nn.Sequential(
            nn.Linear(patch_dim, patch_dim//2),
            nn.Linear(patch_dim//2, 6)
        )
        
    def forward(self, x):
        Mixer_output = []
        zip_layers = zip(self.patch_embedding, self.positional_embedding, self.blocks)
        x = self.quant(x)
        
        # Apply Multi-Scale Patch
        for patch_emb, pos_emb, block in zip_layers:
            # Patch Embedding
            x = x.float()
            z = patch_emb(x)
            z = z.flatten(2)  # (B, C, N)

            # Positional Embedding
            z = pos_emb(z)

            z = self.quant_flow_in(z)

            layer_outputs = []
            for blk in block:
                z, blk_layer = blk(z)

                layer_outputs.extend(blk_layer)
            
            Mixer_output.append(z)
        
        # Concatenate Multi-Scale Patch
        z = torch.cat(Mixer_output, dim=2)  # (B, C, N1+N2)

        z = z.float()

        # GAP
        x = torch.mean(z, dim=2, keepdim=False)

        logit = self.head(x)
        
        return self.dequant(logit)