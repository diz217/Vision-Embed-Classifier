from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, random_split
from .datasets import OxfordPetDataset
from .transforms import build_eval_transform, build_train_transform

@dataclass
class DataConfig:
    root: str = Path
    image_size: int = 224
    batch_size: int = 64
    num_workers: int = 4
    val_ratio: float = 0.1
    seed: int = 42
    download: bool = False
    pin_memory: bool = True


class OxfordPetDataModule:
    def __init__(self, config: DataConfig) -> None:
        self.config = config

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.class_names = None
        self.num_classes = None

    def setup(self) -> None:
        train_transform = build_train_transform(image_size=self.config.image_size)
        eval_transform = build_eval_transform(image_size=self.config.image_size)

        full_trainval_for_split = OxfordPetDataset(
            root=self.config.root,
            split="trainval",
            transform=None,  # split first, then assign transforms below
            download=self.config.download)

        self.class_names = full_trainval_for_split.class_names
        self.num_classes = full_trainval_for_split.num_classes

        total_size = len(full_trainval_for_split)
        val_size = int(total_size * self.config.val_ratio)
        train_size = total_size - val_size

        generator = torch.Generator().manual_seed(self.config.seed)
        train_subset, val_subset = random_split(
            full_trainval_for_split,
            lengths=[train_size, val_size],
            generator=generator)

        full_trainval_train_transform = OxfordPetDataset(
            root=self.config.root,
            split="trainval",
            transform=train_transform,
            download=False)

        full_trainval_eval_transform = OxfordPetDataset(
            root=self.config.root,
            split="trainval",
            transform=eval_transform,
            download=False)

        self.train_dataset = torch.utils.data.Subset(
            full_trainval_train_transform, train_subset.indices)
        
        self.val_dataset = torch.utils.data.Subset(
            full_trainval_eval_transform, val_subset.indices)
        
        self.test_dataset = OxfordPetDataset(
            root=self.config.root,
            split="test",
            transform=eval_transform,
            download=False)

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("Call setup() before requesting dataloaders.")

        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory)

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("Call setup() before requesting dataloaders.")

        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory)

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise RuntimeError("Call setup() before requesting dataloaders.")

        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory)