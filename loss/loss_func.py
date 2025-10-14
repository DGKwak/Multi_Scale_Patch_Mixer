import torch
import torch.nn as nn
import torch.nn.functional as F

class Info_NCELoss(nn.Module):
    def __init__(self, temperature=0.5, device=None):
        super().__init__()
        self.temperature = temperature
        self.device = device

    def forward(self, inter, intra):
        batch_size = inter.shape[0]
        
        z1_norm = F.normalize(inter, dim=1)
        z2_norm = F.normalize(intra, dim=1)
        
        features = torch.cat([z1_norm, z2_norm], dim=0)

        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        labels = torch.arange(2 * batch_size, device=self.device)
        labels = (labels + batch_size) % (2 * batch_size)

        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=self.device)
        similarity_matrix = similarity_matrix.masked_fill(mask, -float('inf'))
        
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss

class Info_NCELoss_test(nn.Module):
    def __init__(self, temperature=0.5, device=None):
        super().__init__()
        self.temperature = temperature
        self.device = device

    def forward(self, output_list):
        batch_size = output_list[0].shape[0]
        
        z1 = torch.mean(output_list[0], dim=2, keepdim=False)
        z2 = torch.mean(output_list[1], dim=2, keepdim=False)

        z1_norm = F.normalize(z1, dim=1)
        z2_norm = F.normalize(z2, dim=1)
        
        features = torch.cat([z1_norm, z2_norm], dim=0)

        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        labels = torch.arange(2 * batch_size, device=self.device)
        labels = (labels + batch_size) % (2 * batch_size)

        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=self.device)
        similarity_matrix = similarity_matrix.masked_fill(mask, -float('inf'))
        
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss

class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.cosine_similarity = nn.CosineSimilarity(dim=1)
    
    def forward(self, s_x, t_x):
        cos_sim = self.cosine_similarity(s_x, t_x)
        loss = 1 - torch.mean(cos_sim)

        return loss