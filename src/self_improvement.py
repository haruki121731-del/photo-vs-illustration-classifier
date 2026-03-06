"""
自己改善システム
NAS → Pruning → Knowledge Distillation の自動パイプライン
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

from auto_optimizer import (
    GeneticNAS, ModelPruner, KnowledgeDistillation, 
    SearchableModel, ModelConfig
)


class SelfImprovementPipeline:
    """
    自己改善パイプライン
    
    Phase 1: NAS (Neural Architecture Search)
    Phase 2: Full Training of Best Architecture
    Phase 3: Pruning (Model Compression)
    Phase 4: Knowledge Distillation
    Phase 5: Final Evaluation
    """
    
    def __init__(
        self,
        data_dir: str,
        output_dir: str = './self_improvement',
        image_size: int = 224,
        batch_size: int = 64
    ):
        self.data_dir = data_dir
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.image_size = image_size
        self.batch_size = batch_size
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # データローダー準備
        self.train_loader, self.val_loader, self.test_loader = self._prepare_data()
        
        # 結果記録
        self.results = {
            'start_time': datetime.now().isoformat(),
            'phases': {}
        }
    
    def _prepare_data(self):
        """データローダーを準備"""
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(self.image_size, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize(int(self.image_size * 1.14)),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        train_dataset = ImageFolder(os.path.join(self.data_dir, 'train'), transform=train_transform)
        val_dataset = ImageFolder(os.path.join(self.data_dir, 'val'), transform=val_transform)
        test_dataset = ImageFolder(os.path.join(self.data_dir, 'test'), transform=val_transform)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, 
                                 shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size,
                               shuffle=False, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size,
                                shuffle=False, num_workers=4, pin_memory=True)
        
        print(f"Data loaded: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
        
        return train_loader, val_loader, test_loader
    
    def phase1_nas(self, population_size: int = 15, generations: int = 5) -> ModelConfig:
        """
        Phase 1: Neural Architecture Search
        最適なアーキテクチャを探索
        """
        print("\n" + "="*70)
        print("PHASE 1: Neural Architecture Search")
        print("="*70)
        
        start_time = time.time()
        
        nas = GeneticNAS(
            population_size=population_size,
            generations=generations,
            mutation_rate=0.25,
            crossover_rate=0.5,
            elite_ratio=0.2
        )
        
        best_config = nas.search(
            self.train_loader,
            self.val_loader,
            self.device,
            output_dir=self.output_dir / 'nas_results'
        )
        
        elapsed = time.time() - start_time
        
        # 結果記録
        self.results['phases']['nas'] = {
            'duration_seconds': elapsed,
            'best_config': best_config.to_dict(),
            'best_fitness': nas.best_fitness
        }
        
        # ベストモデルを保存
        best_model = SearchableModel(best_config, num_classes=2)
        torch.save(best_model.state_dict(), self.output_dir / 'phase1_nas_model.pth')
        
        print(f"\nNAS completed in {elapsed/60:.1f} minutes")
        print(f"Best config: {best_config}")
        
        return best_config
    
    def phase2_full_training(self, config: ModelConfig, epochs: int = 100) -> nn.Module:
        """
        Phase 2: Full Training
        ベストアーキテクチャを本格訓練
        """
        print("\n" + "="*70)
        print("PHASE 2: Full Training")
        print("="*70)
        
        start_time = time.time()
        
        model = SearchableModel(config, num_classes=2).to(self.device)
        print(f"Model parameters: {model.count_parameters():,}")
        
        # オプティマイザとスケジューラ
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2
        )
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        best_acc = 0.0
        best_model = None
        patience_counter = 0
        max_patience = 15
        
        for epoch in range(epochs):
            # Train
            model.train()
            train_loss = 0.0
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                train_loss += loss.item()
            
            scheduler.step()
            
            # Validation
            model.eval()
            correct = 0
            total = 0
            val_loss = 0.0
            with torch.no_grad():
                for images, labels in self.val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
            
            acc = 100. * correct / total
            
            if acc > best_acc:
                best_acc = acc
                best_model = copy.deepcopy(model)
                patience_counter = 0
                # ベストモデルを保存
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'accuracy': acc,
                    'config': config.to_dict()
                }, self.output_dir / 'phase2_best_model.pth')
            else:
                patience_counter += 1
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs}: Train Loss={train_loss/len(self.train_loader):.4f}, "
                      f"Val Acc={acc:.2f}%, Best={best_acc:.2f}%")
            
            # 早期終了
            if patience_counter >= max_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        elapsed = time.time() - start_time
        
        # Test evaluation
        test_acc = self._evaluate_model(best_model)
        
        self.results['phases']['full_training'] = {
            'duration_seconds': elapsed,
            'best_val_accuracy': float(best_acc),
            'test_accuracy': float(test_acc),
            'epochs_trained': epoch + 1,
            'parameters': model.count_parameters()
        }
        
        print(f"\nFull training completed in {elapsed/60:.1f} minutes")
        print(f"Best validation accuracy: {best_acc:.2f}%")
        print(f"Test accuracy: {test_acc:.2f}%")
        
        return best_model
    
    def phase3_pruning(self, model: nn.Module, target_ratio: float = 0.3) -> nn.Module:
        """
        Phase 3: Model Pruning
        モデルを剪定して軽量化
        """
        print("\n" + "="*70)
        print("PHASE 3: Model Pruning")
        print("="*70)
        
        start_time = time.time()
        
        original_params = sum(p.numel() for p in model.parameters())
        print(f"Original parameters: {original_params:,}")
        
        # 剪定
        pruner = ModelPruner(model, pruning_ratio=target_ratio)
        pruned_model = pruner.prune_channels()
        
        # Fine-tuning
        print("Fine-tuning pruned model...")
        optimizer = torch.optim.AdamW(pruned_model.parameters(), lr=0.0005, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        pruned_model.to(self.device)
        best_acc = 0.0
        best_pruned = None
        
        for epoch in range(20):
            pruned_model.train()
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = pruned_model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            # Validation
            acc = self._evaluate_model(pruned_model, use_test=False)
            if acc > best_acc:
                best_acc = acc
                best_pruned = copy.deepcopy(pruned_model)
        
        pruned_params = sum(p.numel() for p in best_pruned.parameters())
        reduction = (1 - pruned_params / original_params) * 100
        
        # テスト評価
        test_acc = self._evaluate_model(best_pruned)
        
        elapsed = time.time() - start_time
        
        self.results['phases']['pruning'] = {
            'duration_seconds': elapsed,
            'original_parameters': original_params,
            'pruned_parameters': pruned_params,
            'reduction_percent': float(reduction),
            'test_accuracy': float(test_acc)
        }
        
        torch.save(best_pruned.state_dict(), self.output_dir / 'phase3_pruned_model.pth')
        
        print(f"\nPruning completed in {elapsed/60:.1f} minutes")
        print(f"Parameters: {original_params:,} → {pruned_params:,} ({reduction:.1f}% reduction)")
        print(f"Test accuracy: {test_acc:.2f}%")
        
        return best_pruned
    
    def phase4_knowledge_distillation(self, teacher_model: nn.Module) -> nn.Module:
        """
        Phase 4: Knowledge Distillation
        より小さな生徒モデルに知識を転移
        """
        print("\n" + "="*70)
        print("PHASE 4: Knowledge Distillation")
        print("="*70)
        
        start_time = time.time()
        
        # 小さな生徒モデルを作成
        student_config = ModelConfig(width_mult=0.5, num_blocks=5, dropout=0.2)
        student_model = SearchableModel(student_config, num_classes=2)
        
        teacher_params = sum(p.numel() for p in teacher_model.parameters())
        student_params = sum(p.numel() for p in student_model.parameters())
        
        print(f"Teacher parameters: {teacher_params:,}")
        print(f"Student parameters: {student_params:,}")
        print(f"Compression ratio: {teacher_params / student_params:.2f}x")
        
        # 知識蒸留
        kd = KnowledgeDistillation(teacher_model, temperature=4.0)
        distilled_model = kd.train_student(
            student_model,
            self.train_loader,
            self.val_loader,
            self.device,
            epochs=50,
            lr=0.001
        )
        
        # 評価
        test_acc = self._evaluate_model(distilled_model)
        
        elapsed = time.time() - start_time
        
        self.results['phases']['knowledge_distillation'] = {
            'duration_seconds': elapsed,
            'teacher_parameters': teacher_params,
            'student_parameters': student_params,
            'compression_ratio': float(teacher_params / student_params),
            'test_accuracy': float(test_acc)
        }
        
        torch.save(distilled_model.state_dict(), self.output_dir / 'phase4_distilled_model.pth')
        
        print(f"\nKnowledge distillation completed in {elapsed/60:.1f} minutes")
        print(f"Test accuracy: {test_acc:.2f}%")
        
        return distilled_model
    
    def phase5_final_evaluation(self, models: dict):
        """
        Phase 5: Final Evaluation
        全モデルを最終評価
        """
        print("\n" + "="*70)
        print("PHASE 5: Final Evaluation")
        print("="*70)
        
        results = {}
        
        for name, model in models.items():
            if model is None:
                continue
            
            acc = self._evaluate_model(model)
            params = sum(p.numel() for p in model.parameters())
            size_mb = params * 4 / 1024 / 1024  # FP32
            
            results[name] = {
                'accuracy': float(acc),
                'parameters': params,
                'size_mb': float(size_mb)
            }
            
            print(f"{name:20s}: {acc:5.2f}% | {params:8,} params | {size_mb:.2f} MB")
        
        # ベストモデルを選択（99%を超えた中で最小のモデル）
        candidates = {k: v for k, v in results.items() if v['accuracy'] >= 99.0}
        
        if candidates:
            best_model_name = min(candidates, key=lambda x: candidates[x]['parameters'])
            best_model_info = candidates[best_model_name]
            print(f"\n✓ Best model achieving >99%: {best_model_name}")
        else:
            best_model_name = max(results, key=lambda x: results[x]['accuracy'])
            best_model_info = results[best_model_name]
            print(f"\n⚠ No model achieved 99%. Best: {best_model_name} ({best_model_info['accuracy']:.2f}%)")
        
        self.results['phases']['final_evaluation'] = results
        self.results['best_model'] = {
            'name': best_model_name,
            **best_model_info
        }
        
        return results
    
    def _evaluate_model(self, model: nn.Module, use_test: bool = True) -> float:
        """モデルを評価"""
        model.eval()
        correct = 0
        total = 0
        
        loader = self.test_loader if use_test else self.val_loader
        
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        return 100. * correct / total
    
    def run(self, skip_nas: bool = False, use_pretrained: str = None):
        """
        完全なパイプラインを実行
        """
        print("\n" + "="*70)
        print("SELF-IMPROVEMENT PIPELINE STARTING")
        print("="*70)
        print(f"Output directory: {self.output_dir}")
        print(f"Device: {self.device}")
        
        total_start = time.time()
        
        models = {}
        
        # Phase 1: NAS
        if not skip_nas:
            best_config = self.phase1_nas(population_size=15, generations=5)
        elif use_pretrained:
            # 既存のモデルを読み込み
            checkpoint = torch.load(use_pretrained, map_location=self.device)
            best_config = ModelConfig.from_dict(checkpoint.get('config', {}))
            print(f"Loaded config from {use_pretrained}")
        else:
            best_config = ModelConfig()
        
        # Phase 2: Full Training
        trained_model = self.phase2_full_training(best_config, epochs=100)
        models['original'] = trained_model
        
        # Phase 3: Pruning
        pruned_model = self.phase3_pruning(trained_model, target_ratio=0.3)
        models['pruned'] = pruned_model
        
        # Phase 4: Knowledge Distillation
        distilled_model = self.phase4_knowledge_distillation(trained_model)
        models['distilled'] = distilled_model
        
        # Phase 5: Final Evaluation
        self.phase5_final_evaluation(models)
        
        total_elapsed = time.time() - total_start
        self.results['total_duration_seconds'] = total_elapsed
        self.results['end_time'] = datetime.now().isoformat()
        
        # 結果を保存
        with open(self.output_dir / 'self_improvement_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print("\n" + "="*70)
        print("SELF-IMPROVEMENT PIPELINE COMPLETED")
        print("="*70)
        print(f"Total time: {total_elapsed/3600:.1f} hours")
        print(f"Results saved to: {self.output_dir / 'self_improvement_results.json'}")
        
        return self.results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Self-Improvement Pipeline')
    parser.add_argument('--data-dir', type=str, default='./data/processed', help='Data directory')
    parser.add_argument('--output-dir', type=str, default='./self_improvement', help='Output directory')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--skip-nas', action='store_true', help='Skip NAS phase')
    parser.add_argument('--use-pretrained', type=str, help='Use pretrained model path')
    
    args = parser.parse_args()
    
    pipeline = SelfImprovementPipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size
    )
    
    results = pipeline.run(skip_nas=args.skip_nas, use_pretrained=args.use_pretrained)
