#!/usr/bin/env python3
"""
完全自動化パイプライン
- データ収集 → 前処理 → 本格トレーニング → 評価
- 99%精度達成まで自動実行
"""

import os
import sys
import time
import json
import shutil
from pathlib import Path
from datetime import datetime
import subprocess

sys.path.insert(0, 'src')


class FullPipeline:
    """完全自動化パイプライン"""
    
    def __init__(self, target_images=10000, output_dir='./full_pipeline'):
        self.target_images = target_images
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_file = self.output_dir / 'pipeline.log'
        self.results = {
            'start_time': datetime.now().isoformat(),
            'phases': {}
        }
        
    def log(self, message):
        """ログ出力"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        
        with open(self.log_file, 'a') as f:
            f.write(log_message + '\n')
    
    def phase1_collect_data(self):
        """Phase 1: 大規模データ収集"""
        self.log("="*70)
        self.log("PHASE 1: DATA COLLECTION")
        self.log("="*70)
        
        start_time = time.time()
        
        # ディレクトリ準備
        raw_dir = Path('data/massive_raw')
        raw_dir.mkdir(parents=True, exist_ok=True)
        
        # イラスト収集（Safebooru）
        self.log("Collecting illustrations from Safebooru...")
        from data_collector import SafebooruCollector
        
        collector = SafebooruCollector(delay=0.2)
        
        # 多様なタグで収集
        tags_config = [
            ('1girl', 2000),
            ('landscape', 2000),
            ('portrait', 2000),
            ('animal', 1500),
            ('architecture', 1500),
            ('fantasy', 1000),
        ]
        
        total_collected = 0
        for tag, count in tags_config:
            self.log(f"  Tag '{tag}': targeting {count} images")
            collected = collector.collect_images(
                output_dir=str(raw_dir / 'illustrations'),
                tags_list=[tag],
                max_images=count,
                min_width=224,
                min_height=224
            )
            success = len([c for c in collected if c.downloaded])
            total_collected += success
            self.log(f"    Collected: {success} images")
        
        # 写真収集（CIFAR-10等を使用）
        self.log("Collecting photos from CIFAR-10...")
        try:
            import torchvision
            photo_dir = raw_dir / 'photos'
            photo_dir.mkdir(exist_ok=True)
            
            # CIFAR-10から写真に近いクラスを取得
            dataset = torchvision.datasets.CIFAR10(
                root=str(raw_dir / 'temp'),
                train=True,
                download=True
            )
            
            photo_classes = [0, 1, 8, 9]  # airplane, automobile, ship, truck
            count = 0
            for i, (img, label) in enumerate(dataset):
                if label in photo_classes and count < 5000:
                    img.save(photo_dir / f'cifar_photo_{count:05d}.png')
                    count += 1
                    if count % 500 == 0:
                        self.log(f"    Downloaded: {count} photos")
            
            self.log(f"  Total photos from CIFAR-10: {count}")
            
            # 一時ファイル削除
            shutil.rmtree(raw_dir / 'temp', ignore_errors=True)
            
        except Exception as e:
            self.log(f"  Error: {e}")
        
        elapsed = time.time() - start_time
        
        # 統計
        illust_files = len(list((raw_dir / 'illustrations').glob('*')))
        photo_files = len(list((raw_dir / 'photos').glob('*')))
        
        self.results['phases']['data_collection'] = {
            'duration_minutes': elapsed / 60,
            'illustrations': illust_files,
            'photos': photo_files,
            'total': illust_files + photo_files
        }
        
        self.log(f"Data collection completed: {illust_files} illustrations, {photo_files} photos")
        
        return raw_dir
    
    def phase2_prepare_dataset(self, raw_dir):
        """Phase 2: データセット準備"""
        self.log("\n" + "="*70)
        self.log("PHASE 2: DATASET PREPARATION")
        self.log("="*70)
        
        start_time = time.time()
        
        processed_dir = Path('data/processed_massive')
        
        from massive_data_pipeline import build_balanced_dataset
        
        build_balanced_dataset(
            raw_dir=str(raw_dir),
            output_dir=str(processed_dir),
            max_per_class=min(self.target_images, 5000)
        )
        
        # 統計
        stats = {}
        for split in ['train', 'val', 'test']:
            stats[split] = {}
            for cls in ['photo', 'illustration']:
                files = list((processed_dir / split / cls).glob('*'))
                stats[split][cls] = len(files)
        
        elapsed = time.time() - start_time
        
        self.results['phases']['dataset_preparation'] = {
            'duration_minutes': elapsed / 60,
            'stats': stats
        }
        
        self.log(f"Dataset prepared: {stats}")
        
        return processed_dir
    
    def phase3_training(self, data_dir):
        """Phase 3: 本格トレーニング"""
        self.log("\n" + "="*70)
        self.log("PHASE 3: FULL TRAINING")
        self.log("="*70)
        
        start_time = time.time()
        
        training_dir = self.output_dir / 'training'
        training_dir.mkdir(exist_ok=True)
        
        # UltraLightモデルで本格トレーニング
        self.log("Starting UltraLight 0.75x training (150 epochs)...")
        
        import torch
        from torch.utils.data import DataLoader
        from torchvision.datasets import ImageFolder
        import torchvision.transforms as transforms
        from advanced_models import UltraLightClassifier
        from train import Trainer
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.log(f"Using device: {device}")
        
        # データセット
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.3)
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        train_dataset = ImageFolder(str(data_dir / 'train'), transform=train_transform)
        val_dataset = ImageFolder(str(data_dir / 'val'), transform=val_transform)
        
        train_loader = DataLoader(train_dataset, batch_size=64,
                                 shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=64,
                               shuffle=False, num_workers=4, pin_memory=True)
        
        self.log(f"Train samples: {len(train_dataset)}")
        self.log(f"Val samples: {len(val_dataset)}")
        
        # モデル作成
        model = UltraLightClassifier(num_classes=2, width_mult=0.75)
        params = model.count_parameters()
        self.log(f"Model parameters: {params:,} ({params*2/1024:.1f}KB in FP16)")
        
        # トレーニング設定（99%達成向け）
        config = {
            'learning_rate': 1e-3,
            'weight_decay': 1e-4,
            'optimizer': 'adamw',
            'scheduler': 'cosine',
            'use_amp': torch.cuda.is_available(),
            'use_cutmix': True,
            'label_smoothing': 0.1,
            'grad_clip': 1.0,
            'early_stopping_patience': 25,
            'checkpoint_dir': str(training_dir),
        }
        
        trainer = Trainer(model, train_loader, val_loader, device, config)
        history = trainer.train(epochs=150)
        
        elapsed = time.time() - start_time
        
        self.results['phases']['training'] = {
            'duration_hours': elapsed / 3600,
            'best_accuracy': float(trainer.best_val_acc),
            'parameters': params,
            'model_size_kb_fp16': params * 2 / 1024
        }
        
        self.log(f"Training completed!")
        self.log(f"Best validation accuracy: {trainer.best_val_acc:.2f}%")
        
        # 99%達成チェック
        if trainer.best_val_acc >= 99.0:
            self.log("🎉 SUCCESS! 99% accuracy achieved!")
        else:
            self.log(f"⚠️  Reached {trainer.best_val_acc:.2f}%. Need more data or tuning.")
        
        return training_dir / 'best_model.pth'
    
    def phase4_evaluation(self, model_path, data_dir):
        """Phase 4: 最終評価"""
        self.log("\n" + "="*70)
        self.log("PHASE 4: FINAL EVALUATION")
        self.log("="*70)
        
        from evaluate import evaluate_model
        
        results = evaluate_model(
            model_path=str(model_path),
            data_dir=str(data_dir),
            model_name='UltraLightClassifier',
            image_size=224,
            batch_size=64,
            use_tta=True,
            output_dir=str(self.output_dir / 'evaluation')
        )
        
        self.results['phases']['evaluation'] = results
        
        return results
    
    def run(self):
        """完全パイプライン実行"""
        self.log("\n" + "="*70)
        self.log("FULL PIPELINE STARTING")
        self.log("Target: 99% accuracy with <500KB model")
        self.log("="*70)
        
        total_start = time.time()
        
        try:
            # Phase 1: データ収集
            raw_dir = self.phase1_collect_data()
            
            # Phase 2: データセット準備
            data_dir = self.phase2_prepare_dataset(raw_dir)
            
            # Phase 3: トレーニング
            model_path = self.phase3_training(data_dir)
            
            # Phase 4: 評価
            eval_results = self.phase4_evaluation(model_path, data_dir)
            
        except Exception as e:
            self.log(f"ERROR: {e}")
            import traceback
            self.log(traceback.format_exc())
        
        total_elapsed = time.time() - total_start
        self.results['total_duration_hours'] = total_elapsed / 3600
        self.results['end_time'] = datetime.now().isoformat()
        
        # 結果保存
        with open(self.output_dir / 'pipeline_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        self.log("\n" + "="*70)
        self.log("PIPELINE COMPLETED")
        self.log("="*70)
        self.log(f"Total time: {total_elapsed/3600:.1f} hours")
        self.log(f"Results saved to: {self.output_dir}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--target-images', type=int, default=10000,
                       help='Target images per class')
    parser.add_argument('--output-dir', type=str, default='./full_pipeline',
                       help='Output directory')
    
    args = parser.parse_args()
    
    pipeline = FullPipeline(
        target_images=args.target_images,
        output_dir=args.output_dir
    )
    pipeline.run()
