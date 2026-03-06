"""
分散学習・マルチGPU対応トレーニング
- DataParallel / DistributedDataParallel
- 混合精度トレーニング（AMP）
- 勾配累積（大きなバッチサイズを模擬）
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import time


class DistributedTrainer:
    """分散学習トレーナー"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        config: dict,
        local_rank: int = 0,
        world_size: int = 1
    ):
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = device
        self.config = config
        
        # モデルをデバイスに配置
        self.model = model.to(device)
        
        # DDPラッパー（マルチGPU時）
        if world_size > 1:
            self.model = DDP(
                self.model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=False
            )
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # オプティマイザ
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # 混合精度
        self.use_amp = config.get('use_amp', True) and torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_amp else None
        
        # 勾配累積
        self.gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)
        
        self.best_val_acc = 0.0
        self.is_main_process = (local_rank == 0)
    
    def _create_optimizer(self):
        """オプティマイザ作成"""
        lr = self.config.get('learning_rate', 1e-3)
        weight_decay = self.config.get('weight_decay', 1e-4)
        
        # Layer-wise Learning Rate Decay
        param_groups = []
        
        # バックボーン（低いLR）
        backbone_params = []
        # 分類ヘッド（高いLR）
        head_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if 'classifier' in name or 'fc' in name or 'head' in name:
                head_params.append(param)
            else:
                backbone_params.append(param)
        
        param_groups.append({'params': backbone_params, 'lr': lr * 0.1})
        param_groups.append({'params': head_params, 'lr': lr})
        
        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
        
        return optimizer
    
    def _create_scheduler(self):
        """スケジューラ作成"""
        scheduler_name = self.config.get('scheduler', 'cosine')
        epochs = self.config.get('epochs', 100)
        
        if scheduler_name == 'cosine':
            from torch.optim.lr_scheduler import CosineAnnealingLR
            return CosineAnnealingLR(
                self.optimizer,
                T_max=epochs,
                eta_min=self.config.get('min_lr', 1e-6)
            )
        elif scheduler_name == 'onecycle':
            from torch.optim.lr_scheduler import OneCycleLR
            return OneCycleLR(
                self.optimizer,
                max_lr=self.config.get('learning_rate', 1e-3),
                epochs=epochs,
                steps_per_epoch=len(self.train_loader) // self.gradient_accumulation_steps,
                pct_start=0.3
            )
        return None
    
    def train_epoch(self, epoch: int):
        """1エポックの訓練"""
        self.model.train()
        
        if isinstance(self.model, DDP):
            self.train_loader.sampler.set_epoch(epoch)
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        self.optimizer.zero_grad()
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}", 
                   disable=not self.is_main_process)
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Mixed Precision
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = nn.functional.cross_entropy(outputs, labels)
                    loss = loss / self.gradient_accumulation_steps
                
                self.scaler.scale(loss).backward()
                
                # 勾配累積
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.get('grad_clip', 1.0)
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                outputs = self.model(images)
                loss = nn.functional.cross_entropy(outputs, labels)
                loss = loss / self.gradient_accumulation_steps
                
                loss.backward()
                
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.get('grad_clip', 1.0)
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            # メトリクス
            total_loss += loss.item() * self.gradient_accumulation_steps
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # 進捗表示
            if self.is_main_process:
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })
        
        # 全プロセスで集約
        if self.world_size > 1:
            metrics = torch.tensor([total_loss, correct, total], device=self.device)
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
            total_loss, correct, total = metrics.tolist()
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    @torch.no_grad()
    def validate(self):
        """検証"""
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in self.val_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = nn.functional.cross_entropy(outputs, labels)
            else:
                outputs = self.model(images)
                loss = nn.functional.cross_entropy(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        # 全プロセスで集約
        if self.world_size > 1:
            metrics = torch.tensor([total_loss, correct, total], device=self.device)
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
            total_loss, correct, total = metrics.tolist()
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def train(self, epochs: int):
        """トレーニング実行"""
        if self.is_main_process:
            print(f"\n{'='*60}")
            print(f"Distributed Training: {epochs} epochs")
            print(f"World size: {self.world_size}")
            print(f"Gradient accumulation: {self.gradient_accumulation_steps}")
            print(f"Effective batch size: {self.config.get('batch_size', 64) * self.world_size * self.gradient_accumulation_steps}")
            print(f"{'='*60}\n")
        
        for epoch in range(1, epochs + 1):
            start_time = time.time()
            
            # 訓練
            train_loss, train_acc = self.train_epoch(epoch)
            
            # 検証
            val_loss, val_acc = self.validate()
            
            # スケジューラ更新
            if self.scheduler is not None:
                self.scheduler.step()
            
            # ログ（メインプロセスのみ）
            if self.is_main_process:
                epoch_time = time.time() - start_time
                lr = self.optimizer.param_groups[0]['lr']
                
                print(f"Epoch {epoch}/{epochs} - {epoch_time:.1f}s")
                print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
                print(f"  Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}%")
                print(f"  LR: {lr:.6f}")
                
                # ベストモデル保存
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.save_checkpoint('best_model.pth')
                    print(f"  ✓ New best model saved!")
        
        if self.is_main_process:
            print(f"\nTraining completed! Best val accuracy: {self.best_val_acc:.2f}%")
    
    def save_checkpoint(self, filename: str):
        """チェックポイント保存"""
        checkpoint = {
            'model_state_dict': self.model.module.state_dict() if isinstance(self.model, DDP) else self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'config': self.config
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, filename)


def setup_distributed():
    """分散学習のセットアップ"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(local_rank)
        
        return rank, local_rank, world_size
    else:
        return 0, 0, 1


def cleanup_distributed():
    """分散学習のクリーンアップ"""
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    print("Distributed Training module ready!")
    print("\nUsage:")
    print("  torchrun --nproc_per_node=4 distributed_training.py")
