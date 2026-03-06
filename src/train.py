"""
トレーニングパイプライン
- 目標: 99%精度達成
- テクニック: 強力なデータ拡張、ラベルスムージング、cosine annealing等
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from model import create_model


# 再現性のためのシード設定
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CutMix:
    """CutMixデータ拡張"""
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
    
    def __call__(self, batch):
        images, labels = batch
        batch_size = images.size(0)
        
        # ランダムに別の画像を選択
        indices = torch.randperm(batch_size)
        shuffled_images = images[indices]
        shuffled_labels = labels[indices]
        
        # CutMix領域を計算
        lam = np.random.beta(self.alpha, self.alpha)
        cx = np.random.randint(images.size(2))
        cy = np.random.randint(images.size(3))
        cut_ratio = np.sqrt(1 - lam)
        cut_w = int(images.size(2) * cut_ratio)
        cut_h = int(images.size(3) * cut_ratio)
        
        x1 = np.clip(cx - cut_w // 2, 0, images.size(2))
        y1 = np.clip(cy - cut_h // 2, 0, images.size(3))
        x2 = np.clip(cx + cut_w // 2, 0, images.size(2))
        y2 = np.clip(cy + cut_h // 2, 0, images.size(3))
        
        # 画像を混ぜる
        images[:, :, x1:x2, y1:y2] = shuffled_images[:, :, x1:x2, y1:y2]
        
        # ラベルの比率を調整
        lam = 1 - ((x2 - x1) * (y2 - y1) / (images.size(2) * images.size(3)))
        
        return images, labels, shuffled_labels, lam


class LabelSmoothingCrossEntropy(nn.Module):
    """ラベルスムージング付き交差エントロピー損失"""
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, pred, target):
        log_probs = torch.nn.functional.log_softmax(pred, dim=-1)
        n_classes = pred.size(-1)
        
        # 正解ラベルに対する確率 (1 - smoothing)
        # その他のラベルに対する確率 (smoothing / n_classes)
        one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
        smoothed = one_hot * (1 - self.smoothing) + self.smoothing / n_classes
        
        loss = (-smoothed * log_probs).sum(dim=-1).mean()
        return loss


class FocalLoss(nn.Module):
    """Focal Loss（クラス不均衡時に有効）"""
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class Mixup:
    """Mixupデータ拡張"""
    def __init__(self, alpha: float = 0.4):
        self.alpha = alpha
    
    def __call__(self, batch):
        images, labels = batch
        batch_size = images.size(0)
        
        lam = np.random.beta(self.alpha, self.alpha)
        indices = torch.randperm(batch_size)
        
        mixed_images = lam * images + (1 - lam) * images[indices]
        labels_a = labels
        labels_b = labels[indices]
        
        return mixed_images, labels_a, labels_b, lam


def get_train_transforms(image_size: int = 224):
    """訓練用のデータ拡張"""
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.33)),
    ])


def get_val_transforms(image_size: int = 224):
    """検証用のデータ拡張"""
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),  # 256 for 224
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class Trainer:
    """トレーニングクラス"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        config: Dict
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        
        # 損失関数
        if config.get('use_focal_loss', False):
            self.criterion = FocalLoss(alpha=1.0, gamma=2.0)
        elif config.get('label_smoothing', 0) > 0:
            self.criterion = LabelSmoothingCrossEntropy(smoothing=config['label_smoothing'])
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # オプティマイザ
        self.optimizer = self._create_optimizer()
        
        # スケジューラ
        self.scheduler = self._create_scheduler()
        
        # Mixed Precision Training
        self.use_amp = config.get('use_amp', True)
        self.scaler = GradScaler() if self.use_amp else None
        
        # 早期終了
        self.early_stopping_patience = config.get('early_stopping_patience', 10)
        self.best_val_acc = 0.0
        self.patience_counter = 0
        
        # チェックポイント
        self.checkpoint_dir = Path(config.get('checkpoint_dir', './checkpoints'))
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        
        # メトリクス記録
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': []
        }
    
    def _create_optimizer(self):
        """オプティマイザを作成"""
        optimizer_name = self.config.get('optimizer', 'adamw').lower()
        lr = self.config.get('learning_rate', 1e-3)
        weight_decay = self.config.get('weight_decay', 1e-4)
        
        # Layer-wise Learning Rate Decay
        parameters = []
        
        if optimizer_name == 'adamw':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=(0.9, 0.999)
            )
        elif optimizer_name == 'sgd':
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=weight_decay,
                nesterov=True
            )
        elif optimizer_name == 'adam':
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        return optimizer
    
    def _create_scheduler(self):
        """学習率スケジューラを作成"""
        scheduler_name = self.config.get('scheduler', 'cosine').lower()
        epochs = self.config.get('epochs', 100)
        
        if scheduler_name == 'cosine':
            scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=epochs,
                eta_min=self.config.get('min_lr', 1e-6)
            )
        elif scheduler_name == 'onecycle':
            scheduler = OneCycleLR(
                self.optimizer,
                max_lr=self.config.get('learning_rate', 1e-3),
                epochs=epochs,
                steps_per_epoch=len(self.train_loader),
                pct_start=0.3
            )
        else:
            scheduler = None
        
        return scheduler
    
    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """1エポックの訓練"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        use_cutmix = self.config.get('use_cutmix', True) and epoch > 5
        use_mixup = self.config.get('use_mixup', False) and epoch > 5
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # CutMix / Mixup
            if use_cutmix and np.random.rand() < 0.5:
                cutmix = CutMix(alpha=1.0)
                images, labels_a, labels_b, lam = cutmix((images, labels))
                use_aux_labels = True
            elif use_mixup and np.random.rand() < 0.3:
                mixup = Mixup(alpha=0.4)
                images, labels_a, labels_b, lam = mixup((images, labels))
                use_aux_labels = True
            else:
                labels_a = labels
                labels_b = None
                lam = 1.0
                use_aux_labels = False
            
            self.optimizer.zero_grad()
            
            # Mixed Precision Training
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    if use_aux_labels:
                        loss = lam * self.criterion(outputs, labels_a) + \
                               (1 - lam) * self.criterion(outputs, labels_b)
                    else:
                        loss = self.criterion(outputs, labels)
                
                self.scaler.scale(loss).backward()
                
                # Gradient Clipping
                if self.config.get('grad_clip', 0) > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config['grad_clip']
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                if use_aux_labels:
                    loss = lam * self.criterion(outputs, labels_a) + \
                           (1 - lam) * self.criterion(outputs, labels_b)
                else:
                    loss = self.criterion(outputs, labels)
                
                loss.backward()
                
                if self.config.get('grad_clip', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['grad_clip']
                    )
                
                self.optimizer.step()
            
            # メトリクス
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels_a).sum().item()
            
            # 進捗表示
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    @torch.no_grad()
    def validate(self, use_tta: bool = False) -> Tuple[float, float, Dict]:
        """検証"""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        for images, labels in tqdm(self.val_loader, desc="Validation"):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Test Time Augmentation
            if use_tta:
                outputs = self._tta_predict(images)
            else:
                outputs = self.model(images)
            
            loss = nn.CrossEntropyLoss()(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(all_labels, all_preds) * 100
        
        # 詳細メトリクス
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary'
        )
        
        cm = confusion_matrix(all_labels, all_preds)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision * 100,
            'recall': recall * 100,
            'f1': f1 * 100,
            'confusion_matrix': cm.tolist()
        }
        
        return avg_loss, accuracy, metrics
    
    def _tta_predict(self, images: torch.Tensor, n_augmentations: int = 5) -> torch.Tensor:
        """Test Time Augmentation"""
        outputs = []
        
        # 元の画像
        outputs.append(self.model(images))
        
        # 水平フリップ
        outputs.append(self.model(torch.flip(images, dims=[3])))
        
        # 追加の augmentations
        for _ in range(n_augmentations - 2):
            # 軽いスケール変換
            scale = np.random.uniform(0.95, 1.05)
            scaled = torch.nn.functional.interpolate(
                images, scale_factor=scale, mode='bilinear', align_corners=False
            )
            # パディングまたはクロップで元のサイズに
            if scale > 1:
                h, w = scaled.shape[2:]
                start_h = (h - images.size(2)) // 2
                start_w = (w - images.size(3)) // 2
                scaled = scaled[:, :, start_h:start_h+images.size(2), start_w:start_w+images.size(3)]
            else:
                pad_h = images.size(2) - scaled.size(2)
                pad_w = images.size(3) - scaled.size(3)
                scaled = torch.nn.functional.pad(scaled, [pad_w//2, pad_w//2, pad_h//2, pad_h//2])
            
            outputs.append(self.model(scaled))
        
        return torch.stack(outputs).mean(dim=0)
    
    def train(self, epochs: int):
        """トレーニング実行"""
        print(f"\n{'='*60}")
        print(f"Starting Training: {epochs} epochs")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters())/1e6:.2f}M")
        print(f"{'='*60}\n")
        
        for epoch in range(1, epochs + 1):
            start_time = time.time()
            
            # 訓練
            train_loss, train_acc = self.train_epoch(epoch)
            
            # 検証
            val_loss, val_acc, metrics = self.validate()
            
            # 学習率の記録
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # スケジューラ更新
            if self.scheduler is not None and not isinstance(self.scheduler, OneCycleLR):
                self.scheduler.step()
            
            # 時間計算
            epoch_time = time.time() - start_time
            
            # 結果表示
            print(f"\nEpoch {epoch}/{epochs} - {epoch_time:.1f}s")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            print(f"  Precision: {metrics['precision']:.2f}% | Recall: {metrics['recall']:.2f}%")
            print(f"  F1: {metrics['f1']:.2f}% | LR: {current_lr:.6f}")
            
            # 履歴記録
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rate'].append(current_lr)
            
            # ベストモデルの保存
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.patience_counter = 0
                self.save_checkpoint('best_model.pth')
                print(f"  ✓ New best model saved! ({val_acc:.2f}%)")
            else:
                self.patience_counter += 1
            
            # 早期終了
            if self.patience_counter >= self.early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                break
            
            # 定期的なチェックポイント
            if epoch % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pth')
        
        # 最終モデル保存
        self.save_checkpoint('final_model.pth')
        self.save_history()
        
        print(f"\n{'='*60}")
        print(f"Training completed! Best validation accuracy: {self.best_val_acc:.2f}%")
        print(f"{'='*60}")
        
        return self.history
    
    def save_checkpoint(self, filename: str):
        """チェックポイントを保存"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'config': self.config
        }
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, self.checkpoint_dir / filename)
    
    def load_checkpoint(self, filename: str):
        """チェックポイントを読み込み"""
        checkpoint = torch.load(self.checkpoint_dir / filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_acc = checkpoint.get('best_val_acc', 0)
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    def save_history(self):
        """訓練履歴を保存"""
        with open(self.checkpoint_dir / 'training_history.json', 'w') as f:
            json.dump(self.history, f, indent=2)


def train_model(
    data_dir: str,
    model_name: str = 'photo_classifier',
    image_size: int = 224,
    batch_size: int = 64,
    epochs: int = 100,
    learning_rate: float = 1e-3,
    **kwargs
):
    """
    モデルを訓練するメイン関数
    
    Args:
        data_dir: データディレクトリ（train/val/testのサブディレクトリを含む）
        model_name: モデル名
        image_size: 入力画像サイズ
        batch_size: バッチサイズ
        epochs: エポック数
        learning_rate: 初期学習率
        **kwargs: 追加の設定
    """
    set_seed(kwargs.get('seed', 42))
    
    # デバイス
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # データセット
    train_transform = get_train_transforms(image_size)
    val_transform = get_val_transforms(image_size)
    
    train_dataset = ImageFolder(os.path.join(data_dir, 'train'), transform=train_transform)
    val_dataset = ImageFolder(os.path.join(data_dir, 'val'), transform=val_transform)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=kwargs.get('num_workers', 4),
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=kwargs.get('num_workers', 4),
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Classes: {train_dataset.classes}")
    
    # モデル
    model = create_model(model_name, num_classes=2, **kwargs.get('model_config', {}))
    
    # 設定
    config = {
        'model_name': model_name,
        'image_size': image_size,
        'batch_size': batch_size,
        'epochs': epochs,
        'learning_rate': learning_rate,
        'optimizer': kwargs.get('optimizer', 'adamw'),
        'weight_decay': kwargs.get('weight_decay', 1e-4),
        'scheduler': kwargs.get('scheduler', 'cosine'),
        'use_amp': kwargs.get('use_amp', True),
        'use_cutmix': kwargs.get('use_cutmix', True),
        'use_mixup': kwargs.get('use_mixup', False),
        'label_smoothing': kwargs.get('label_smoothing', 0.1),
        'grad_clip': kwargs.get('grad_clip', 1.0),
        'early_stopping_patience': kwargs.get('early_stopping_patience', 15),
        'checkpoint_dir': kwargs.get('checkpoint_dir', './checkpoints'),
    }
    
    # トレーナー
    trainer = Trainer(model, train_loader, val_loader, device, config)
    
    # 訓練
    history = trainer.train(epochs)
    
    return trainer, history


if __name__ == "__main__":
    print("Training module ready!")
    print("\nExample usage:")
    print("""
    trainer, history = train_model(
        data_dir='./data/processed',
        model_name='photo_classifier',
        image_size=224,
        batch_size=64,
        epochs=100,
        learning_rate=1e-3,
        label_smoothing=0.1,
        use_cutmix=True,
        weight_decay=1e-4
    )
    """)
