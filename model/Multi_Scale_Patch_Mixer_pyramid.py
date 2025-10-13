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


class MultiscaleMixer(nn.Module):
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
        
        self.channel_mixer = nn.ModuleList()
        self.inter_mixer = nn.ModuleList()
        self.intra_mixer = nn.ModuleList()
        self.integration_mixer = nn.ModuleList()

        for _ in range(len(self.patches)):
            tmp = nn.ModuleList([
                MlpBlock(1, patch_dim, x*2, x, self.act, self.dropout)
                for x in [1024, 512, 256, 128]
            ])

            self.channel_mixer.append(tmp)
            self.intra_mixer.append(tmp)

        self.channel_mixer = nn.ModuleList([
            MlpBlock(1, patch_dim, patch_dim*2, patch_dim, self.act, self.dropout)
            for _ in range(len(self.patches))])
        self.inter_mixer = nn.ModuleList([
            MlpBlock(2, x, x*2, x, self.act, self.dropout, residual=False)
            for x in self.num_patches])
        self.intra_mixer = nn.ModuleList([
            MlpBlock(1, patch_dim, patch_dim*2, patch_dim, self.act, self.dropout, residual=False)
            for _ in range(len(self.patches))])
        self.mixrep_mixer = nn.ModuleList([
            MlpBlock(2, x, x*2, x, self.act, self.dropout)
            for x in self.num_patches])
        
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
        for idx in range(len(self.patches)):
            # Patch Embedding
            z = self.patch_embedding[idx](x)
            z = einops.rearrange(z, 'b c h w -> b c (h w)')

            # Positional Embedding
            z = self.positional_embedding[idx](z)
            
            inter_outputs, intra_outputs = [], []
            inter, intra = z, z
            for _ in range(self.num_layers):                
                # Inter & Intra Mixer
                inter = self.channel_mixer[idx](inter)
                intra = self.channel_mixer[idx](intra)
                
                inter = self.inter_mixer[idx](inter)
                intra = self.intra_mixer[idx](intra)

                # Channel Mixer
                inter = self.mixrep_mixer[idx](inter)
                intra = self.mixrep_mixer[idx](intra)
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