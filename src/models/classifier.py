
import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearClassifierHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class CosineClassifierHead(nn.Module):
    def __init__(self,in_dim: int,num_classes: int,cosine_scale: float = 20.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, in_dim))
        self.cosine_scale = cosine_scale
        nn.init.xavier_uniform_(self.weight)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.normalize(x, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)
        logits = self.cosine_scale * (x @ weight.t())
        return logits