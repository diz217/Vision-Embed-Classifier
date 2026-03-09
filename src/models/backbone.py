import torch
import torch.nn as nn
import open_clip


class CLIPBackbone(nn.Module):

    def __init__(self,model_name: str = "ViT-B-32", pretrained: str = "laion2b_s34b_b79k",freeze: bool = True):
        super().__init__()
        model, _, _ = open_clip.create_model_and_transforms(model_name=model_name,pretrained=pretrained)
        
        self.visual = model.visual
        self.embed_dim = model.visual.output_dim

        if freeze:
            for p in self.visual.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.visual(x)