from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional, Tuple

from torch.utils.data import Dataset
from torchvision.datasets import OxfordIIITPet

class OxfordPetDataset(Dataset):
    def __init__(self,root: str | Path,split: str,transform: Optional[Callable] = None,target_transform: Optional[Callable] = None,download: bool = False,) -> None:
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform

        self.dataset = OxfordIIITPet(root=str(self.root),split=split,target_types="category",transform=None,target_transform=None,download=download)

        self.class_names = self.dataset.classes
        self.num_classes = len(self.class_names)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[Any, int]:
        image, label = self.dataset[idx]  

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label