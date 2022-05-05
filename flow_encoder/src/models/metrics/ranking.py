import torch
import torch.nn as nn


class RankingLoss(nn.Module):
    def __init__(self, margin: float = 1.0):
        super(RankingLoss, self).__init__()
        self.pdist = nn.PairwiseDistance(p=2)
        self.margin = margin

    def forward(self, r0: torch.Tensor, r1: torch.Tensor, y: torch.Tensor):
        batch_size = r0.shape[0]
        distance = self.pdist(r0.view(batch_size, -1), r1.view(batch_size, -1))
        zero = torch.zeros_like(distance)
        loss = y * distance + (1 - y) * torch.max(zero, self.margin - distance)
        return loss.mean()
