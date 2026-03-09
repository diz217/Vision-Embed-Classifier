from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

from .checkpoint import save_checkpoint


@dataclass
class TrainerConfig:
    epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    device: str = "cuda"
    checkpoint_dir: str = "artifacts/checkpoints"
    best_checkpoint_name: str = "best.pt"
    last_checkpoint_name: str = "last.pt"
    log_every_n_steps: int = 20


class Trainer:
    def __init__(self, model: nn.Module, config: TrainerConfig) -> None:
        self.model = model
        self.config = config

        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = AdamW(self.model.parameters(),lr=self.config.learning_rate,weight_decay=self.config.weight_decay)

        self.best_val_acc = float("-inf")
        self.history = {"train_loss": [],"train_acc": [],"val_loss": [], "val_acc": []}

    def train_one_epoch(self, train_loader, epoch: int) -> tuple[float, float]:
        self.model.train()

        running_loss = 0.0
        running_correct = 0
        running_total = 0

        progress_bar = tqdm(train_loader,desc=f"Epoch {epoch + 1}/{self.config.epochs} [Train]",leave=False)

        for step, (images, targets) in enumerate(progress_bar):
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            logits = self.model(images)
            loss = self.criterion(logits, targets)

            loss.backward()
            self.optimizer.step()

            batch_size = targets.size(0)
            running_loss += loss.item() * batch_size
            running_correct += (logits.argmax(dim=1) == targets).sum().item()
            running_total += batch_size

            avg_loss = running_loss / running_total
            avg_acc = running_correct / running_total

            if step % self.config.log_every_n_steps == 0:
                progress_bar.set_postfix(train_loss=f"{avg_loss:.4f}",train_acc=f"{avg_acc:.4f}")

        epoch_loss = running_loss / running_total
        epoch_acc = running_correct / running_total

        return epoch_loss, epoch_acc

    @torch.no_grad()
    def validate_one_epoch(self, val_loader, epoch: int,stage:str) -> tuple[float, float]:
        self.model.eval()

        running_loss = 0.0
        running_correct = 0
        running_total = 0

        progress_bar = tqdm(val_loader,desc=f"Epoch {epoch + 1}/{self.config.epochs} [{stage}]",leave=False)

        for images, targets in progress_bar:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            logits = self.model(images)
            loss = self.criterion(logits, targets)

            batch_size = targets.size(0)
            running_loss += loss.item() * batch_size
            running_correct += (logits.argmax(dim=1) == targets).sum().item()
            running_total += batch_size

            avg_loss = running_loss / running_total
            avg_acc = running_correct / running_total

            progress_bar.set_postfix(val_loss=f"{avg_loss:.4f}",val_acc=f"{avg_acc:.4f}")

        epoch_loss = running_loss / running_total
        epoch_acc = running_correct / running_total

        return epoch_loss, epoch_acc

    def fit(self, train_loader, val_loader) -> dict:
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(self.config.epochs):
            train_loss, train_acc = self.train_one_epoch(train_loader, epoch)
            val_loss, val_acc = self.validate_one_epoch(val_loader, epoch,'Val')

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            print(f"Epoch {epoch + 1}/{self.config.epochs} | " f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

            # save last checkpoint every epoch
            save_checkpoint(
                checkpoint_path=checkpoint_dir / self.config.last_checkpoint_name,
                model=self.model,
                optimizer=self.optimizer,
                epoch=epoch,
                best_val_acc=self.best_val_acc)
            # save best checkpoint
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                save_checkpoint(
                    checkpoint_path=checkpoint_dir / self.config.best_checkpoint_name,
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    best_val_acc=self.best_val_acc)
                print(f"New best checkpoint saved: val_acc={val_acc:.4f}")

        return self.history