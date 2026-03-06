#!/usr/bin/env python3
"""
最終モデルトレーニングスクリプト
- 99%精度達成を目指す本格トレーニング
- 自己改善ループ + 高度なアーキテクチャ
"""

import os
import sys
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

# モジュールインポート
sys.path.insert(0, 'src')
from advanced_models import UltraLightClassifier, NanoClassifier, ModelFactory
from train import Trainer
from auto_optimizer import GeneticNAS, SearchableModel, ModelConfig


def get_transforms(image_size: int = 224, is_training: bool = True):
    """データ拡張"""
    if is_training:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, 
                                  saturation=0.3, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.33)),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
        ])


def train_ultra_light_model(data_dir: str, output_dir: str, epochs: int = 150):
    """
    UltraLightモデルを本格訓練
    目標: 99%精度 + <500KB
    """
    print("="*70)
    print("TRAINING ULTRA-LIGHT MODEL FOR 99% ACCURACY")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # データセット
    train_transform = get_transforms(224, is_training=True)
    val_transform = get_transforms(224, is_training=False)
    
    train_dataset = ImageFolder(os.path.join(data_dir, 'train'), 
                                transform=train_transform)
    val_dataset = ImageFolder(os.path.join(data_dir, 'val'), 
                             transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=64, 
                             shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=64,
                           shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # モデル選択
    configs = [
        {'name': 'UltraLight_0.5', 'width': 0.5},
        {'name': 'UltraLight_0.65', 'width': 0.65},
        {'name': 'UltraLight_0.75', 'width': 0.75},
    ]
    
    best_model = None
    best_acc = 0.0
    results = []
    
    for cfg in configs:
        print(f"\n{'='*70}")
        print(f"Training {cfg['name']}")
        print(f"{'='*70}")
        
        model = UltraLightClassifier(num_classes=2, width_mult=cfg['width'])
        params = model.count_parameters()
        print(f"Parameters: {params:,} ({params*2/1024:.1f}KB in FP16)")
        
        # トレーニング設定
        config = {
            'learning_rate': 1e-3,
            'weight_decay': 1e-4,
            'optimizer': 'adamw',
            'scheduler': 'cosine',
            'use_amp': True,
            'label_smoothing': 0.1,
            'grad_clip': 1.0,
            'early_stopping_patience': 20,
            'checkpoint_dir': os.path.join(output_dir, cfg['name']),
        }
        
        trainer = Trainer(model, train_loader, val_loader, device, config)
        history = trainer.train(epochs)
        
        # 結果記録
        results.append({
            'name': cfg['name'],
            'params': params,
            'best_acc': trainer.best_val_acc,
            'width': cfg['width']
        })
        
        # ベストモデル更新
        if trainer.best_val_acc > best_acc:
            best_acc = trainer.best_val_acc
            best_model = cfg['name']
        
        # 99%達成チェック
        if trainer.best_val_acc >= 99.0:
            print(f"\n🎉 {cfg['name']} achieved 99%+ accuracy!")
            print(f"   Best: {trainer.best_val_acc:.2f}%")
            print(f"   Size: {params*2/1024:.1f}KB (FP16)")
    
    # 結果サマリー
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")
    for r in results:
        status = "✓ 99%" if r['best_acc'] >= 99.0 else ""
        print(f"{r['name']:20s}: {r['best_acc']:.2f}% | {r['params']:,} params {status}")
    
    print(f"\nBest model: {best_model} ({best_acc:.2f}%)")
    
    # 結果保存
    with open(os.path.join(output_dir, 'final_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    return best_model


def run_nas_and_train(data_dir: str, output_dir: str):
    """
    NASで最適なアーキテクチャを探索し、訓練
    """
    from torch.utils.data import Subset
    
    print("="*70)
    print("NAS + FULL TRAINING PIPELINE")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # クイックNAS用の小さいデータセット
    train_transform = get_transforms(224, is_training=True)
    val_transform = get_transforms(224, is_training=False)
    
    train_dataset = ImageFolder(os.path.join(data_dir, 'train'), 
                                transform=train_transform)
    val_dataset = ImageFolder(os.path.join(data_dir, 'val'), 
                             transform=val_transform)
    
    # NAS用にサブセット（高速化のため）
    nas_train_size = min(5000, len(train_dataset))
    nas_val_size = min(1000, len(val_dataset))
    
    train_indices = torch.randperm(len(train_dataset))[:nas_train_size]
    val_indices = torch.randperm(len(val_dataset))[:nas_val_size]
    
    nas_train_dataset = Subset(train_dataset, train_indices)
    nas_val_dataset = Subset(val_dataset, val_indices)
    
    nas_train_loader = DataLoader(nas_train_dataset, batch_size=32, 
                                 shuffle=True, num_workers=2)
    nas_val_loader = DataLoader(nas_val_dataset, batch_size=32,
                               shuffle=False, num_workers=2)
    
    # NAS実行
    print("\n[Phase 1] Running NAS...")
    nas = GeneticNAS(population_size=10, generations=5)
    best_config = nas.search(nas_train_loader, nas_val_loader, device,
                            output_dir=os.path.join(output_dir, 'nas'))
    
    print(f"\nBest config found: {best_config}")
    
    # フルトレーニング
    print("\n[Phase 2] Full training with best architecture...")
    
    train_loader = DataLoader(train_dataset, batch_size=64, 
                             shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=64,
                           shuffle=False, num_workers=4, pin_memory=True)
    
    model = SearchableModel(best_config, num_classes=2)
    print(f"Model parameters: {model.count_parameters():,}")
    
    config = {
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'optimizer': 'adamw',
        'scheduler': 'cosine',
        'use_amp': True,
        'label_smoothing': 0.1,
        'early_stopping_patience': 25,
        'checkpoint_dir': output_dir,
    }
    
    trainer = Trainer(model, train_loader, val_loader, device, config)
    trainer.train(epochs=150)
    
    print(f"\n{'='*70}")
    print(f"Final Best Accuracy: {trainer.best_val_acc:.2f}%")
    print(f"{'='*70}")
    
    return trainer.best_val_acc


def main():
    parser = argparse.ArgumentParser(description='Train Final Model')
    parser.add_argument('--data-dir', type=str, default='./data/processed',
                       help='Data directory')
    parser.add_argument('--output-dir', type=str, default='./final_training',
                       help='Output directory')
    parser.add_argument('--mode', type=str, default='ultra_light',
                       choices=['ultra_light', 'nas', 'all'],
                       help='Training mode')
    parser.add_argument('--epochs', type=int, default=150,
                       help='Number of epochs')
    
    args = parser.parse_args()
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    if args.mode == 'ultra_light':
        train_ultra_light_model(args.data_dir, args.output_dir, args.epochs)
    elif args.mode == 'nas':
        run_nas_and_train(args.data_dir, args.output_dir)
    elif args.mode == 'all':
        # 両方実行
        train_ultra_light_model(args.data_dir, 
                               os.path.join(args.output_dir, 'ultra_light'),
                               args.epochs)
        run_nas_and_train(args.data_dir,
                         os.path.join(args.output_dir, 'nas'))


if __name__ == '__main__':
    main()
