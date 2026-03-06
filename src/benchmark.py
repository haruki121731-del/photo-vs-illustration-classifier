"""
ベンチマーク・性能評価システム
- 複数モデルの比較評価
- 速度 vs 精度トレードオフ分析
- ハードウェア別性能測定
"""

import os
import json
import time
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from advanced_models import ModelFactory, benchmark_model
from model import create_model as create_basic_model


@dataclass
class BenchmarkResult:
    """ベンチマーク結果"""
    model_name: str
    parameters: int
    size_kb_fp32: float
    size_kb_fp16: float
    
    # 速度メトリクス
    latency_cpu_ms: float
    latency_gpu_ms: Optional[float]
    throughput_cpu_fps: float
    throughput_gpu_fps: Optional[float]
    
    # 精度メトリクス
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    
    # 効率スコア
    efficiency_score: float  # 独自指標: 精度 * スループット / サイズ
    
    def to_dict(self) -> Dict:
        return asdict(self)


class ModelBenchmark:
    """モデルベンチマーククラス"""
    
    def __init__(self, test_loader: DataLoader, device: str = 'auto'):
        self.test_loader = test_loader
        self.device = torch.device(device if device != 'auto' else 
                                  ('cuda' if torch.cuda.is_available() else 'cpu'))
        print(f"Benchmark device: {self.device}")
    
    def evaluate_accuracy(self, model: nn.Module) -> Dict[str, float]:
        """精度を評価"""
        model.eval()
        model.to(self.device)
        
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc="Evaluating"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = model(images)
                _, predicted = outputs.max(1)
                
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = 100. * correct / total
        
        # Precision, Recall, F1
        from sklearn.metrics import precision_score, recall_score, f1_score
        precision = precision_score(all_labels, all_preds, average='binary') * 100
        recall = recall_score(all_labels, all_preds, average='binary') * 100
        f1 = f1_score(all_labels, all_preds, average='binary') * 100
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    def measure_speed(self, model: nn.Module, n_iterations: int = 100) -> Dict[str, float]:
        """推論速度を測定"""
        model.eval()
        
        dummy_input = torch.randn(1, 3, 224, 224)
        
        # CPU測定
        model_cpu = model.to('cpu')
        dummy_cpu = dummy_input.to('cpu')
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = model_cpu(dummy_cpu)
        
        # CPUベンチマーク
        cpu_times = []
        with torch.no_grad():
            for _ in range(n_iterations):
                start = time.time()
                _ = model_cpu(dummy_cpu)
                cpu_times.append(time.time() - start)
        
        cpu_times = np.array(cpu_times)
        latency_cpu = cpu_times.mean() * 1000  # ms
        throughput_cpu = 1.0 / cpu_times.mean()
        
        # GPU測定（利用可能な場合）
        latency_gpu = None
        throughput_gpu = None
        
        if torch.cuda.is_available():
            model_gpu = model.to('cuda')
            dummy_gpu = dummy_input.to('cuda')
            torch.cuda.synchronize()
            
            # Warmup
            for _ in range(10):
                with torch.no_grad():
                    _ = model_gpu(dummy_gpu)
                    torch.cuda.synchronize()
            
            # GPUベンチマーク
            gpu_times = []
            with torch.no_grad():
                for _ in range(n_iterations):
                    torch.cuda.synchronize()
                    start = time.time()
                    _ = model_gpu(dummy_gpu)
                    torch.cuda.synchronize()
                    gpu_times.append(time.time() - start)
            
            gpu_times = np.array(gpu_times)
            latency_gpu = gpu_times.mean() * 1000  # ms
            throughput_gpu = 1.0 / gpu_times.mean()
        
        return {
            'latency_cpu_ms': latency_cpu,
            'latency_gpu_ms': latency_gpu,
            'throughput_cpu_fps': throughput_cpu,
            'throughput_gpu_fps': throughput_gpu
        }
    
    def run_full_benchmark(self, model: nn.Module, model_name: str) -> BenchmarkResult:
        """完全なベンチマークを実行"""
        print(f"\nBenchmarking {model_name}...")
        print("-" * 60)
        
        # モデル情報
        params = sum(p.numel() for p in model.parameters())
        size_kb_fp32 = params * 4 / 1024
        size_kb_fp16 = params * 2 / 1024
        
        print(f"Parameters: {params:,}")
        print(f"Size (FP32): {size_kb_fp32:.1f} KB")
        print(f"Size (FP16): {size_kb_fp16:.1f} KB")
        
        # 速度測定
        print("\nMeasuring speed...")
        speed_metrics = self.measure_speed(model)
        print(f"CPU Latency: {speed_metrics['latency_cpu_ms']:.2f} ms")
        print(f"CPU Throughput: {speed_metrics['throughput_cpu_fps']:.1f} FPS")
        if speed_metrics['latency_gpu_ms']:
            print(f"GPU Latency: {speed_metrics['latency_gpu_ms']:.2f} ms")
            print(f"GPU Throughput: {speed_metrics['throughput_gpu_fps']:.1f} FPS")
        
        # 精度測定
        print("\nEvaluating accuracy...")
        accuracy_metrics = self.evaluate_accuracy(model)
        print(f"Accuracy: {accuracy_metrics['accuracy']:.2f}%")
        print(f"Precision: {accuracy_metrics['precision']:.2f}%")
        print(f"Recall: {accuracy_metrics['recall']:.2f}%")
        print(f"F1-Score: {accuracy_metrics['f1_score']:.2f}%")
        
        # 効率スコア計算
        # 効率 = 精度 * スループット / サイズ（対数スケール）
        efficiency = (accuracy_metrics['accuracy'] * 
                     speed_metrics['throughput_cpu_fps'] / 
                     np.log1p(size_kb_fp16))
        
        print(f"\nEfficiency Score: {efficiency:.2f}")
        
        return BenchmarkResult(
            model_name=model_name,
            parameters=params,
            size_kb_fp32=size_kb_fp32,
            size_kb_fp16=size_kb_fp16,
            latency_cpu_ms=speed_metrics['latency_cpu_ms'],
            latency_gpu_ms=speed_metrics['latency_gpu_ms'],
            throughput_cpu_fps=speed_metrics['throughput_cpu_fps'],
            throughput_gpu_fps=speed_metrics['throughput_gpu_fps'],
            accuracy=accuracy_metrics['accuracy'],
            precision=accuracy_metrics['precision'],
            recall=accuracy_metrics['recall'],
            f1_score=accuracy_metrics['f1_score'],
            efficiency_score=efficiency
        )


class BenchmarkSuite:
    """複数モデルのベンチマークスイート"""
    
    def __init__(self, test_loader: DataLoader, output_dir: str = './benchmark_results'):
        self.test_loader = test_loader
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.benchmarker = ModelBenchmark(test_loader)
    
    def run_comparison(self, models: List[tuple]) -> List[BenchmarkResult]:
        """
        複数モデルを比較
        
        Args:
            models: [(name, model_instance), ...]
        """
        results = []
        
        print("="*70)
        print("MODEL COMPARISON BENCHMARK")
        print("="*70)
        
        for name, model in models:
            try:
                result = self.benchmarker.run_full_benchmark(model, name)
                results.append(result)
            except Exception as e:
                print(f"Error benchmarking {name}: {e}")
        
        # 結果を保存
        self._save_results(results)
        
        # サマリーを表示
        self._print_summary(results)
        
        return results
    
    def _save_results(self, results: List[BenchmarkResult]):
        """結果を保存"""
        data = [r.to_dict() for r in results]
        
        with open(self.output_dir / 'benchmark_results.json', 'w') as f:
            json.dump(data, f, indent=2)
        
        # CSVでも保存
        import csv
        if results:
            with open(self.output_dir / 'benchmark_results.csv', 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=results[0].to_dict().keys())
                writer.writeheader()
                for r in results:
                    writer.writerow(r.to_dict())
        
        print(f"\nResults saved to {self.output_dir}")
    
    def _print_summary(self, results: List[BenchmarkResult]):
        """サマリーを表示"""
        print("\n" + "="*70)
        print("BENCHMARK SUMMARY")
        print("="*70)
        
        # テーブルヘッダー
        print(f"{'Model':<20} {'Params':<10} {'Size(KB)':<10} {'Acc(%)':<8} {'CPU(ms)':<10} {'Score':<10}")
        print("-"*70)
        
        # 結果をソート（効率スコア順）
        sorted_results = sorted(results, key=lambda x: x.efficiency_score, reverse=True)
        
        for r in sorted_results:
            print(f"{r.model_name:<20} {r.parameters:<10,} {r.size_kb_fp16:<10.1f} "
                  f"{r.accuracy:<8.2f} {r.latency_cpu_ms:<10.2f} {r.efficiency_score:<10.2f}")
        
        # 99%達成モデルを強調
        high_accuracy = [r for r in results if r.accuracy >= 99.0]
        if high_accuracy:
            print("\n" + "="*70)
            print("🎉 MODELS ACHIEVING 99%+ ACCURACY:")
            print("="*70)
            for r in sorted(high_accuracy, key=lambda x: x.parameters):
                print(f"  ✓ {r.model_name}: {r.accuracy:.2f}% | {r.size_kb_fp16:.1f} KB | {r.latency_cpu_ms:.2f} ms")
        
        # 最も効率的なモデル
        best = sorted_results[0]
        print(f"\n⭐ MOST EFFICIENT: {best.model_name}")
        print(f"   Efficiency Score: {best.efficiency_score:.2f}")
        print("="*70)


def compare_all_models(test_loader: DataLoader, checkpoint_paths: Optional[Dict] = None):
    """
    全モデルを比較
    """
    models = []
    
    # 基本モデル
    print("Loading models...")
    
    # 1. Basic Models
    models.append(('PhotoClassifier (0.75x)', 
                   create_basic_model('photo_classifier', width_mult=0.75)))
    models.append(('TinyClassifier', 
                   create_basic_model('tiny')))
    
    # 2. Advanced Models
    models.append(('UltraLight (0.5x)', 
                   ModelFactory.create_model('ultra_light', width_mult=0.5)))
    models.append(('UltraLight (0.75x)', 
                   ModelFactory.create_model('ultra_light', width_mult=0.75)))
    models.append(('Nano', 
                   ModelFactory.create_model('nano')))
    
    # 3. チェックポイントから読み込み（提供されている場合）
    if checkpoint_paths:
        for name, path in checkpoint_paths.items():
            if os.path.exists(path):
                checkpoint = torch.load(path, map_location='cpu')
                # モデルを再構築して重みを読み込み
                # （実装はチェックポイントの形式による）
                pass
    
    # ベンチマーク実行
    suite = BenchmarkSuite(test_loader)
    results = suite.run_comparison(models)
    
    return results


if __name__ == "__main__":
    print("Benchmark module ready!")
    print("\nUsage:")
    print("  from benchmark import ModelBenchmark, BenchmarkSuite")
    print("  benchmarker = ModelBenchmark(test_loader)")
    print("  result = benchmarker.run_full_benchmark(model, 'model_name')")
