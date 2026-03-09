import torch
import torch.nn as nn

from .backbone import CLIPBackbone
from .classifier import LinearClassifierHead, CosineClassifierHead


class VisionClassifier(nn.Module):
    """
    image -> backbone -> embedding -> classifier head -> logits
    """
    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = head
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        logits = self.head(features)
        return logits

def build_model(num_classes: int,backbone_name: str = "ViT-B-32",pretrained: str = "laion2b_s34b_b79k",freeze_backbone: bool = True,head_type: str = "linear",cosine_scale:float = 20):
    backbone = CLIPBackbone(model_name=backbone_name,pretrained=pretrained,freeze=freeze_backbone)
    embed_dim = backbone.embed_dim
    if head_type == "linear":
        head = LinearClassifierHead(embed_dim, num_classes)
    elif head_type == "cosine":
        head = CosineClassifierHead(embed_dim,num_classes,cosine_scale)
    else:
        raise ValueError(f"Unknown head_type: {head_type}")
    model = VisionClassifier(backbone, head)
    return model
