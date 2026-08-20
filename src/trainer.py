"""Training loop and trainer class."""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from .models import SurgicalNet
from .data import build_loaders
from .config import DEVICE


class SurgicalTrainer:
    """Trainer for SurgicalNet with early stopping and mixed precision."""

    def __init__(self, train_dir, val_dir, batch_size, num_workers, val_ratio,
                 json_path, input_size=512, patience=5, lr=1e-4, weight_decay=1e-4):
        self.device = DEVICE
        self.patience = patience
        self.counter = 0
        self.mask_weight = 0.3
        self.scaler = GradScaler()

        self.train_loader, self.val_loader, self.class_mapping = build_loaders(
            train_dir, val_dir, batch_size, num_workers, val_ratio,
            json_path, input_size
        )

        self.model = SurgicalNet(num_classes=len(self.class_mapping)).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        for batch in tqdm(self.train_loader, desc="Train"):
            if batch is None:
                continue
            imgs, labels, masks = batch
            imgs = imgs.to(self.device)
            labels = labels.to(self.device)
            masks = masks.to(self.device)

            self.optimizer.zero_grad()
            with autocast():
                out = self.model(imgs)
                cls_loss = self.criterion(out, labels)
                loss = cls_loss + self.mask_weight * masks.mean()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validate"):
                if batch is None:
                    continue
                imgs, labels, _ = batch
                imgs = imgs.to(self.device)
                out = self.model(imgs)
                preds = out.argmax(dim=1).cpu()
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        return correct / total if total > 0 else 0.0

    def run(self, epochs=30, save_path="best_surgical_model_a31.pth"):
        best_acc = 0.0
        for epoch in range(1, epochs + 1):
            self.mask_weight = max(0.1, 0.3 * (1 - epoch / epochs))
            train_loss = self.train_epoch()
            val_acc = self.validate()
            self.scheduler.step()

            print(f"Epoch {epoch}: Loss={train_loss:.4f}, Val Acc={val_acc:.4f}")
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.model.state_dict(), save_path)
                print(f"New best model saved, best Val Acc = {best_acc:.4f}")
                self.counter = 0
            else:
                self.counter += 1
                print(f"  No improvement for {self.counter}/{self.patience} epochs.")
                if self.counter >= self.patience:
                    print(f"Early stopping triggered after epoch {epoch}.")
                    break

        print(f"Training finished. Best Val Acc = {best_acc:.4f}")
        return best_acc
