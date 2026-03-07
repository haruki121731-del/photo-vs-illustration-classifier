#!/usr/bin/env python3
"""
TRAINING PHASE - Phase 2
1,232 images (1000 photos + 232 illustrations)
Target: 99% accuracy
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Setup
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "complete"
CHECKPOINT_DIR = BASE_DIR / "checkpoints_local"
LOG_DIR = BASE_DIR / "logs"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# Logger
class Logger:
    def __init__(self, log_file):
        self.log_file = log_file
    
    def log(self, msg, level="INFO"):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] [{level}] {msg}"
        print(line, flush=True)
        with open(self.log_file, "a") as f:
            f.write(line + "\n")
            f.flush()

logger = Logger(LOG_DIR / "training.log")

# Config
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS = 200
LEARNING_RATE = 0.001

# Model
class GhostModule(nn.Module):
    def __init__(self, inp, oup, ratio=2):
        super().__init__()
        init_ch = int(oup / ratio)
        self.primary = nn.Sequential(
            nn.Conv2d(inp, init_ch, 1, bias=False),
            nn.BatchNorm2d(init_ch),
            nn.ReLU(inplace=True)
        )
        self.cheap = nn.Sequential(
            nn.Conv2d(init_ch, init_ch, 3, 1, 1, groups=init_ch, bias=False),
            nn.BatchNorm2d(init_ch),
            nn.ReLU(inplace=True)
        )
        self.oup = oup
    
    def forward(self, x):
        x1 = self.primary(x)
        x2 = self.cheap(x1)
        return torch.cat([x1, x2], 1)[:, :self.oup]

class TrainingModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            GhostModule(32, 64),
            nn.AvgPool2d(2),
            GhostModule(64, 128),
            nn.AvgPool2d(2),
            GhostModule(128, 256),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

def train():
    logger.log("=" * 80)
    logger.log("PHASE 2: TRAINING")
    logger.log("=" * 80)
    logger.log(f"Device: {DEVICE}")
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.5),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    # Dataset
    full_dataset = datasets.ImageFolder(DATA_DIR, transform=train_transform)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    val_ds.dataset.transform = val_transform
    
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True
    )
    
    logger.log(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
    
    # Model
    model = TrainingModel().to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    logger.log(f"Parameters: {total_params:,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Training loop
    best_acc = 0.0
    best_epoch = 0
    history = []
    
    for epoch in range(EPOCHS):
        epoch_start = time.time()
        
        # Train
        model.train()
        train_loss = train_correct = train_total = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            
            if batch_idx % 10 == 0:
                logger.log(f"Epoch {epoch+1}/{EPOCHS} [{batch_idx}/{len(train_loader)}] "
                          f"Loss: {loss.item():.4f} Acc: {100.*train_correct/train_total:.2f}%")
        
        # Validation
        model.eval()
        val_loss = val_correct = val_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        epoch_time = time.time() - epoch_start
        
        logger.log(f"Epoch {epoch+1}: Train={train_acc:.2f}% Val={val_acc:.2f}% Time={epoch_time:.1f}s")
        
        # Save best
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch + 1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_acc': best_acc,
            }, CHECKPOINT_DIR / "best_model.pth")
            logger.log(f"*** NEW BEST: {best_acc:.2f}% ***")
        
        # Check 99%
        if val_acc >= 99.0:
            logger.log("=" * 80)
            logger.log(f"🎉 99% ACHIEVED AT EPOCH {epoch+1}! 🎉")
            logger.log("=" * 80)
            break
        
        history.append({
            'epoch': epoch + 1,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'best_acc': best_acc
        })
        
        with open(CHECKPOINT_DIR / "history.json", "w") as f:
            json.dump(history, f, indent=2)
        
        scheduler.step()
    
    logger.log("=" * 80)
    logger.log(f"TRAINING COMPLETE - Best: {best_acc:.2f}% (Epoch {best_epoch})")
    logger.log("=" * 80)
    
    return best_acc

if __name__ == "__main__":
    best_acc = train()
    sys.exit(0 if best_acc >= 99.0 else 1)
