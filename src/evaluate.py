"""
モデル評価スクリプト
- 精度、速度、モデルサイズを評価
- 99%精度達成確認
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt

from model import create_model


class ModelEvaluator:
    """モデル評価クラス"""
    
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model.to(device)
        self.device = device
        self.model.eval()
    
    @torch.no_grad()
    def evaluate_accuracy(self, test_loader: DataLoader, use_tta: bool = False) -> Dict:
        """精度を評価"""
        all_preds = []
        all_labels = []
        all_probs = []
        
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            if use_tta:
                outputs = self._tta_predict(images)
            else:
                outputs = self.model(images)
            
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # メトリクス計算
        accuracy = accuracy_score(all_labels, all_preds) * 100
        precision = precision_score(all_labels, all_preds, average='binary') * 100
        recall = recall_score(all_labels, all_preds, average='binary') * 100
        f1 = f1_score(all_labels, all_preds, average='binary') * 100
        
        # AUC-ROC
        try:
            auc = roc_auc_score(all_labels, all_probs[:, 1]) * 100
        except:
            auc = 0.0
        
        # 混同行列
        cm = confusion_matrix(all_labels, all_preds)
        
        # クラス別精度
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) * 100  # True Negative Rate
        sensitivity = tp / (tp + fn) * 100  # True Positive Rate
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_roc': auc,
            'specificity': specificity,
            'sensitivity': sensitivity,
            'confusion_matrix': cm.tolist(),
            'predictions': all_preds.tolist(),
            'labels': all_labels.tolist(),
            'probabilities': all_probs.tolist()
        }
    
    def _tta_predict(self, images: torch.Tensor, n_augmentations: int = 5) -> torch.Tensor:
        """Test Time Augmentation"""
        outputs = []
        outputs.append(self.model(images))
        outputs.append(self.model(torch.flip(images, dims=[3])))
        
        for _ in range(n_augmentations - 2):
            scale = np.random.uniform(0.95, 1.05)
            scaled = torch.nn.functional.interpolate(
                images, scale_factor=scale, mode='bilinear', align_corners=False
            )
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
    
    @torch.no_grad()
    def measure_inference_speed(self, input_size: Tuple[int, int, int] = (3, 224, 224), 
                                n_iterations: int = 100) -> Dict:
        """推論速度を測定"""
        dummy_input = torch.randn(1, *input_size).to(self.device)
        
        # Warmup
        for _ in range(10):
            _ = self.model(dummy_input)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # 測定
        times = []
        for _ in tqdm(range(n_iterations), desc="Measuring speed"):
            start = time.time()
            _ = self.model(dummy_input)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times.append(time.time() - start)
        
        times = np.array(times)
        
        return {
            'mean_ms': np.mean(times) * 1000,
            'std_ms': np.std(times) * 1000,
            'min_ms': np.min(times) * 1000,
            'max_ms': np.max(times) * 1000,
            'fps': 1.0 / np.mean(times)
        }
    
    @torch.no_grad()
    def measure_memory_usage(self, input_size: Tuple[int, int, int] = (3, 224, 224)) -> Dict:
        """メモリ使用量を測定"""
        if not torch.cuda.is_available():
            return {'note': 'CUDA not available'}
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        dummy_input = torch.randn(1, *input_size).to(self.device)
        _ = self.model(dummy_input)
        
        memory_allocated = torch.cuda.memory_allocated() / 1024**2  # MB
        memory_reserved = torch.cuda.memory_reserved() / 1024**2  # MB
        max_memory_allocated = torch.cuda.max_memory_allocated() / 1024**2  # MB
        
        return {
            'allocated_mb': memory_allocated,
            'reserved_mb': memory_reserved,
            'peak_mb': max_memory_allocated
        }
    
    def count_parameters(self) -> Dict:
        """パラメータ数をカウント"""
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'total': total,
            'trainable': trainable,
            'total_mb': total * 4 / 1024**2,  # FP32で計算
            'trainable_mb': trainable * 4 / 1024**2
        }
    
    def get_model_size(self) -> float:
        """保存時のモデルサイズを推定（MB）"""
        param_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.model.buffers())
        size_mb = (param_size + buffer_size) / 1024**2
        return size_mb
    
    def plot_confusion_matrix(self, cm: List, save_path: str = None):
        """混同行列をプロット"""
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        
        classes = ['Photo', 'Illustration']
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        
        # 値を表示
        thresh = np.array(cm).max() / 2.
        for i in range(len(cm)):
            for j in range(len(cm[i])):
                plt.text(j, i, format(cm[i][j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i][j] > thresh else "black")
        
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_roc_curve(self, labels: List, probs: List, save_path: str = None):
        """ROC曲線をプロット"""
        fpr, tpr, _ = roc_curve(labels, np.array(probs)[:, 1])
        auc = roc_auc_score(labels, np.array(probs)[:, 1])
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def evaluate_model(
    model_path: str,
    data_dir: str,
    model_name: str = 'photo_classifier',
    image_size: int = 224,
    batch_size: int = 64,
    use_tta: bool = False,
    output_dir: str = './evaluation_results'
):
    """
    モデルを評価するメイン関数
    
    Args:
        model_path: モデルチェックポイントのパス
        data_dir: テストデータディレクトリ
        model_name: モデル名
        image_size: 入力画像サイズ
        batch_size: バッチサイズ
        use_tta: Test Time Augmentationを使用
        output_dir: 結果出力ディレクトリ
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # モデル読み込み
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    
    # 設定を復元
    config = checkpoint.get('config', {})
    model_config = config.get('model_config', {})
    
    model = create_model(model_name, num_classes=2, **model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 評価クラス
    evaluator = ModelEvaluator(model, device)
    
    # データセット
    test_transform = transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    test_dataset = ImageFolder(os.path.join(data_dir, 'test'), transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"\nTest samples: {len(test_dataset)}")
    print(f"Classes: {test_dataset.classes}")
    
    # 評価実行
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    # 1. 精度評価
    print("\n[1/4] Evaluating accuracy...")
    metrics = evaluator.evaluate_accuracy(test_loader, use_tta=use_tta)
    
    print(f"\nAccuracy Metrics:")
    print(f"  Overall Accuracy:  {metrics['accuracy']:.2f}% {'✓ 99%達成!' if metrics['accuracy'] >= 99 else ''}")
    print(f"  Precision:         {metrics['precision']:.2f}%")
    print(f"  Recall:            {metrics['recall']:.2f}%")
    print(f"  F1-Score:          {metrics['f1_score']:.2f}%")
    print(f"  AUC-ROC:           {metrics['auc_roc']:.2f}%")
    print(f"  Specificity (TNR): {metrics['specificity']:.2f}%")
    print(f"  Sensitivity (TPR): {metrics['sensitivity']:.2f}%")
    
    # 混同行列
    cm = metrics['confusion_matrix']
    print(f"\nConfusion Matrix:")
    print(f"                 Predicted")
    print(f"                 Photo  Illustration")
    print(f"  Actual Photo     {cm[0][0]:4d}     {cm[0][1]:4d}")
    print(f"  Illustration     {cm[1][0]:4d}     {cm[1][1]:4d}")
    
    # 2. モデル統計
    print("\n[2/4] Model statistics...")
    params = evaluator.count_parameters()
    model_size = evaluator.get_model_size()
    
    print(f"\nModel Statistics:")
    print(f"  Total parameters:   {params['total']:,} ({params['total']/1e6:.2f}M)")
    print(f"  Trainable params:   {params['trainable']:,} ({params['trainable']/1e6:.2f}M)")
    print(f"  Model size (FP32):  {model_size:.2f} MB")
    print(f"  Model size (FP16):  {model_size/2:.2f} MB")
    
    # 3. 推論速度
    print("\n[3/4] Measuring inference speed...")
    speed = evaluator.measure_inference_speed((3, image_size, image_size), n_iterations=100)
    
    print(f"\nInference Speed (single image):")
    print(f"  Mean latency:  {speed['mean_ms']:.2f} ± {speed['std_ms']:.2f} ms")
    print(f"  Min/Max:       {speed['min_ms']:.2f} / {speed['max_ms']:.2f} ms")
    print(f"  Throughput:    {speed['fps']:.1f} FPS")
    
    # 4. メモリ使用量
    print("\n[4/4] Measuring memory usage...")
    memory = evaluator.measure_memory_usage((3, image_size, image_size))
    
    if 'note' not in memory:
        print(f"\nMemory Usage:")
        print(f"  Allocated:  {memory['allocated_mb']:.2f} MB")
        print(f"  Peak:       {memory['peak_mb']:.2f} MB")
    else:
        print(f"  {memory['note']}")
    
    # 結果保存
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # JSON結果
    results = {
        'accuracy_metrics': {
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score'],
            'auc_roc': metrics['auc_roc'],
            'specificity': metrics['specificity'],
            'sensitivity': metrics['sensitivity'],
        },
        'confusion_matrix': cm,
        'model_statistics': {
            'total_parameters': params['total'],
            'trainable_parameters': params['trainable'],
            'model_size_mb': model_size,
        },
        'inference_speed': speed,
        'memory_usage': memory,
    }
    
    with open(output_path / 'evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # プロット保存
    evaluator.plot_confusion_matrix(cm, save_path=output_path / 'confusion_matrix.png')
    evaluator.plot_roc_curve(metrics['labels'], metrics['probabilities'], 
                            save_path=output_path / 'roc_curve.png')
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}")
    
    # 99%達成判定
    if metrics['accuracy'] >= 99:
        print("\n🎉 SUCCESS: 99% accuracy achieved!")
    else:
        print(f"\n⚠️  Need improvement: Current accuracy is {metrics['accuracy']:.2f}%")
        print("   Suggestions:")
        print("   - Collect more diverse training data")
        print("   - Try data augmentation")
        print("   - Increase model capacity slightly")
        print("   - Use ensemble of multiple models")
    
    return results


if __name__ == "__main__":
    print("Evaluation module ready!")
    print("\nExample usage:")
    print("""
    results = evaluate_model(
        model_path='./checkpoints/best_model.pth',
        data_dir='./data/processed',
        model_name='photo_classifier',
        image_size=224,
        batch_size=64,
        use_tta=True
    )
    """)
