import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class InfoNCELossFG(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        print(f'Use InfoNCELossFG with temperature: {temperature}')

    def forward(self, fg_img_feature, fg_pro_feature, bg_pro_feature):
        # Normalize image features
        fg_img_feature_norm = F.normalize(fg_img_feature, p=2, dim=-1) # [N, D]
        
        # Calculate positive similarities
        # [N, D] * [N, D] -> sum(dim=1) -> [N]
        pos_sim = torch.sum(fg_img_feature_norm * fg_pro_feature, dim=1) # [N]

        # Calculate negative similarities
        # [N, D] @ [N, D, L] -> [N, L] (using einsum for batched matmul)
        neg_sim = torch.einsum('nd,nld->nl', fg_img_feature_norm, bg_pro_feature) # [N, L]

        # Apply temperature
        pos_sim_exp = torch.exp(pos_sim / self.temperature) # [N]
        neg_sim_exp = torch.exp(neg_sim / self.temperature) # [N, L]

        # Sum numerators and denominators
        positive_sum = torch.sum(pos_sim_exp)
        negative_sum = positive_sum + torch.sum(neg_sim_exp)
            
        # The final loss is -log(numerator / denominator)
        loss = -torch.log(positive_sum / negative_sum)

        return loss


class InfoNCELossBG(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        print(f'Use InfoNCELossBG with temperature: {temperature}')

    def forward(self, bg_img_feature, fg_pro_feature, bg_pro_feature):
        # Normalize image features
        bg_img_feature_norm = F.normalize(bg_img_feature, p=2, dim=-1) # [N, D]
        
        # Calculate positive similarities (BG img vs BG text)
        # [N, D] @ [N, D, L] -> [N, L] (using einsum)
        pos_sim = torch.einsum('nd,nld->nl', bg_img_feature_norm, bg_pro_feature) # [N, L]
        
        # Calculate negative similarities (BG img vs FG text)
        # [N, D] * [N, D] -> sum(dim=1) -> [N]
        neg_sim = torch.sum(bg_img_feature_norm * fg_pro_feature, dim=1) # [N]
        
        # Apply temperature
        pos_sim_exp = torch.exp(pos_sim / self.temperature) # [N, L]
        neg_sim_exp = torch.exp(neg_sim / self.temperature) # [N]

        # Sum numerators and denominators
        positive_sum = torch.mean(pos_sim_exp, dim=1).sum()
        negative_sum = positive_sum + torch.sum(neg_sim_exp)

        loss = -torch.log(positive_sum / negative_sum)

        return loss
