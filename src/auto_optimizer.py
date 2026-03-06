"""
自動モデル最適化システム（NAS + Pruning + Knowledge Distillation）
目標: 99%精度 + 最小パラメータ数を自動探索
"""

import os
import json
import copy
import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import itertools

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm


@dataclass
class ModelConfig:
    """モデル設定の遺伝子表現"""
    width_mult: float = 0.75
    num_blocks: int = 6
    expand_ratio: int = 6
    se_reduction: int = 4
    dropout: float = 0.2
    activation: str = 'relu6'  # relu6, swish, mish
    use_se: bool = True
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict):
        return cls(**d)
    
    def mutate(self, mutation_rate: float = 0.2) -> 'ModelConfig':
        """突然変異"""
        child = copy.deepcopy(self)
        
        if random.random() < mutation_rate:
            child.width_mult = random.choice([0.5, 0.65, 0.75, 0.85, 1.0])
        if random.random() < mutation_rate:
            child.num_blocks = random.randint(4, 8)
        if random.random() < mutation_rate:
            child.expand_ratio = random.choice([4, 6, 8])
        if random.random() < mutation_rate:
            child.se_reduction = random.choice([2, 4, 8])
        if random.random() < mutation_rate:
            child.dropout = random.uniform(0.1, 0.5)
        if random.random() < mutation_rate:
            child.activation = random.choice(['relu6', 'swish', 'mish'])
        
        return child
    
    def crossover(self, other: 'ModelConfig') -> 'ModelConfig':
        """交叉"""
        child = copy.deepcopy(self)
        
        # ランダムに親の遺伝子を選択
        if random.random() < 0.5:
            child.width_mult = other.width_mult
        if random.random() < 0.5:
            child.num_blocks = other.num_blocks
        if random.random() < 0.5:
            child.expand_ratio = other.expand_ratio
        if random.random() < 0.5:
            child.se_reduction = other.se_reduction
        if random.random() < 0.5:
            child.dropout = other.dropout
        if random.random() < 0.5:
            child.activation = other.activation
        
        return child


class SearchableBlock(nn.Module):
    """探索可能な基本ブロック"""
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, 
                 expand_ratio: int = 6, use_se: bool = True, 
                 se_reduction: int = 4, activation: str = 'relu6'):
        super().__init__()
        
        hidden_dim = in_ch * expand_ratio
        
        # Activation
        if activation == 'swish':
            act = nn.SiLU(inplace=True)
        elif activation == 'mish':
            act = nn.Mish(inplace=True)
        else:
            act = nn.ReLU6(inplace=True)
        
        layers = []
        
        # Expansion
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_ch, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                act
            ])
        
        # Depthwise
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, 
                     groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            act
        ])
        
        # SE Block
        if use_se:
            layers.append(SEBlock(hidden_dim, se_reduction))
        
        # Projection
        layers.extend([
            nn.Conv2d(hidden_dim, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch)
        ])
        
        self.conv = nn.Sequential(*layers)
        self.use_residual = stride == 1 and in_ch == out_ch
    
    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Hardsigmoid(inplace=True)
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SearchableModel(nn.Module):
    """探索可能なモデル"""
    def __init__(self, config: ModelConfig, num_classes: int = 2):
        super().__init__()
        self.config = config
        
        # First conv
        input_channel = int(32 * config.width_mult)
        self.first_conv = nn.Sequential(
            nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel),
            self._get_activation(config.activation)
        )
        
        # Searchable blocks
        layers = []
        cfgs = self._generate_configs(config.num_blocks)
        
        for t, c, n, s in cfgs:
            output_channel = int(c * config.width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                layers.append(SearchableBlock(
                    input_channel, output_channel, stride,
                    expand_ratio=t,
                    use_se=config.use_se,
                    se_reduction=config.se_reduction,
                    activation=config.activation
                ))
                input_channel = output_channel
        
        self.features = nn.Sequential(*layers)
        
        # Last conv
        self.last_channel = int(128 * config.width_mult)
        self.last_conv = nn.Sequential(
            nn.Conv2d(input_channel, self.last_channel, 1, bias=False),
            nn.BatchNorm2d(self.last_channel),
            self._get_activation(config.activation)
        )
        
        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(self.last_channel, num_classes)
        )
        
        self._initialize_weights()
    
    def _get_activation(self, name: str):
        if name == 'swish':
            return nn.SiLU(inplace=True)
        elif name == 'mish':
            return nn.Mish(inplace=True)
        return nn.ReLU6(inplace=True)
    
    def _generate_configs(self, num_blocks: int):
        """ブロック設定を生成"""
        # [expand_ratio, out_channels, num_layers, stride]
        base_configs = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 2, 2],
            [6, 96, 2, 1],
            [6, 160, 1, 2],
        ]
        
        # num_blocksに応じて調整
        if num_blocks < 6:
            return base_configs[:num_blocks]
        elif num_blocks > 6:
            # 追加ブロックを挿入
            extra = num_blocks - 6
            configs = base_configs.copy()
            for i in range(extra):
                configs.insert(-1, [6, 96, 1, 1])
            return configs
        return base_configs
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = self.first_conv(x)
        x = self.features(x)
        x = self.last_conv(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class GeneticNAS:
    """遺伝的アルゴリズムによるNAS"""
    
    def __init__(
        self,
        population_size: int = 20,
        generations: int = 10,
        mutation_rate: float = 0.2,
        crossover_rate: float = 0.5,
        elite_ratio: float = 0.2
    ):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = int(population_size * elite_ratio)
        
        self.history = []
        self.best_config = None
        self.best_fitness = 0.0
    
    def initialize_population(self) -> List[ModelConfig]:
        """初期集団を生成"""
        population = []
        
        # デフォルト設定から開始
        population.append(ModelConfig())
        
        # ランダムに生成
        for _ in range(self.population_size - 1):
            config = ModelConfig()
            config = config.mutate(mutation_rate=0.5)
            population.append(config)
        
        return population
    
    def evaluate_fitness(
        self, 
        config: ModelConfig, 
        train_loader,
        val_loader,
        device: torch.device,
        quick_eval: bool = True
    ) -> float:
        """
        適応度を評価
        
        Fitness = accuracy * 0.7 + (1 - params_ratio) * 0.3
        params_ratio = params / 1_000_000  # 目標は1Mパラメータ以下
        """
        model = SearchableModel(config, num_classes=2).to(device)
        params = model.count_parameters()
        
        # パラメータ数ペナルティ（1M超えると厳しいペナルティ）
        params_ratio = params / 1_000_000
        if params_ratio > 1.0:
            return 0.0  # 1M超えは採用しない
        
        # クイック評価（少量のエポックで評価）
        if quick_eval:
            accuracy = self._quick_train(model, train_loader, val_loader, device, epochs=3)
        else:
            accuracy = self._full_train(model, train_loader, val_loader, device, epochs=20)
        
        # 適応度計算
        # 精度が99%以上ならパラメータ数を優先、未満なら精度を優先
        if accuracy >= 99.0:
            fitness = 100 + (1 - params_ratio) * 100  # ボーナス
        else:
            fitness = accuracy * 0.7 + (1 - params_ratio) * 30
        
        return fitness
    
    def _quick_train(
        self, 
        model: nn.Module, 
        train_loader, 
        val_loader, 
        device: torch.device,
        epochs: int = 3
    ) -> float:
        """クイック訓練"""
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        best_acc = 0.0
        
        for epoch in range(epochs):
            model.train()
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            # 検証
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
            
            acc = 100. * correct / total
            best_acc = max(best_acc, acc)
        
        return best_acc
    
    def _full_train(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        device: torch.device,
        epochs: int = 20
    ) -> float:
        """フル訓練（より正確な評価）"""
        return self._quick_train(model, train_loader, val_loader, device, epochs)
    
    def select_parents(self, population: List[ModelConfig], 
                      fitnesses: List[float]) -> List[ModelConfig]:
        """親選択（トーナメント選択）"""
        parents = []
        
        # エリート選択
        elite_indices = np.argsort(fitnesses)[-self.elite_size:]
        for idx in elite_indices:
            parents.append(population[idx])
        
        # トーナメント選択
        while len(parents) < self.population_size:
            tournament = random.sample(list(zip(population, fitnesses)), 3)
            winner = max(tournament, key=lambda x: x[1])[0]
            parents.append(winner)
        
        return parents
    
    def create_next_generation(self, parents: List[ModelConfig]) -> List[ModelConfig]:
        """次世代を作成"""
        next_gen = []
        
        # エリートはそのまま
        next_gen.extend(parents[:self.elite_size])
        
        # 交叉と突然変異
        while len(next_gen) < self.population_size:
            parent1, parent2 = random.sample(parents, 2)
            
            if random.random() < self.crossover_rate:
                child = parent1.crossover(parent2)
            else:
                child = copy.deepcopy(parent1)
            
            child = child.mutate(self.mutation_rate)
            next_gen.append(child)
        
        return next_gen
    
    def search(
        self,
        train_loader,
        val_loader,
        device: torch.device,
        output_dir: str = './nas_results'
    ) -> ModelConfig:
        """
        NAS探索を実行
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        print("="*60)
        print("Genetic NAS Starting")
        print("="*60)
        print(f"Population: {self.population_size}, Generations: {self.generations}")
        print(f"Target: >99% accuracy, <1M parameters")
        print("="*60)
        
        # 初期集団
        population = self.initialize_population()
        
        for generation in range(self.generations):
            print(f"\n{'='*60}")
            print(f"Generation {generation + 1}/{self.generations}")
            print(f"{'='*60}")
            
            # 適応度評価
            fitnesses = []
            for i, config in enumerate(population):
                print(f"\nEvaluating config {i+1}/{len(population)}...")
                fitness = self.evaluate_fitness(
                    config, train_loader, val_loader, device, quick_eval=True
                )
                fitnesses.append(fitness)
                
                # モデル情報を表示
                model = SearchableModel(config)
                params = model.count_parameters()
                print(f"  Params: {params:,}, Fitness: {fitness:.2f}")
                
                # ベストを更新
                if fitness > self.best_fitness:
                    self.best_fitness = fitness
                    self.best_config = copy.deepcopy(config)
                    print(f"  ✓ New best!")
            
            # 統計
            avg_fitness = np.mean(fitnesses)
            max_fitness = np.max(fitnesses)
            
            print(f"\nGeneration {generation + 1} Summary:")
            print(f"  Average fitness: {avg_fitness:.2f}")
            print(f"  Max fitness: {max_fitness:.2f}")
            print(f"  Best overall: {self.best_fitness:.2f}")
            
            # 履歴記録
            self.history.append({
                'generation': generation + 1,
                'avg_fitness': float(avg_fitness),
                'max_fitness': float(max_fitness),
                'best_fitness': float(self.best_fitness)
            })
            
            # 親選択
            parents = self.select_parents(population, fitnesses)
            
            # 次世代作成
            population = self.create_next_generation(parents)
            
            # 中間結果保存
            self._save_checkpoint(output_path / f'gen_{generation+1}.json')
        
        # 最終結果保存
        self._save_checkpoint(output_path / 'best_config.json')
        
        print(f"\n{'='*60}")
        print("NAS Search Completed!")
        print(f"Best fitness: {self.best_fitness:.2f}")
        print(f"{'='*60}")
        
        return self.best_config
    
    def _save_checkpoint(self, path: Path):
        """チェックポイント保存"""
        data = {
            'best_config': self.best_config.to_dict() if self.best_config else None,
            'best_fitness': self.best_fitness,
            'history': self.history
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)


class ModelPruner:
    """モデル剪定（Pruning）システム"""
    
    def __init__(self, model: nn.Module, pruning_ratio: float = 0.3):
        self.model = model
        self.pruning_ratio = pruning_ratio
    
    def prune_channels(self) -> nn.Module:
        """チャネル剪定（L1ノルムベース）"""
        pruned_model = copy.deepcopy(self.model)
        
        for name, module in pruned_model.named_modules():
            if isinstance(module, nn.Conv2d) and module.groups == 1:
                # L1ノルムで重要度計算
                weight = module.weight.data
                importance = weight.abs().sum(dim=(1, 2, 3))
                
                # 重要度でソート
                num_channels = len(importance)
                num_keep = int(num_channels * (1 - self.pruning_ratio))
                
                if num_keep < num_channels:
                    keep_indices = torch.topk(importance, num_keep).indices
                    
                    # チャネルを削減
                    module.out_channels = num_keep
                    module.weight = nn.Parameter(weight[keep_indices])
                    if module.bias is not None:
                        module.bias = nn.Parameter(module.bias.data[keep_indices])
        
        return pruned_model
    
    def prune_structured(self) -> nn.Module:
        """構造化剪定"""
        # 実装省略（複雑なため）
        return self.model


class KnowledgeDistillation:
    """知識蒸留"""
    
    def __init__(self, teacher_model: nn.Module, temperature: float = 4.0):
        self.teacher_model = teacher_model
        self.temperature = temperature
        self.teacher_model.eval()
    
    def distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        alpha: float = 0.5
    ) -> torch.Tensor:
        """
        知識蒸留損失
        
        L = alpha * CE(student, labels) + (1-alpha) * KL(student, teacher)
        """
        # ハードラベル損失
        hard_loss = nn.functional.cross_entropy(student_logits, labels)
        
        # ソフトラベル損失（KLダイバージェンス）
        soft_student = nn.functional.log_softmax(student_logits / self.temperature, dim=1)
        soft_teacher = nn.functional.softmax(teacher_logits / self.temperature, dim=1)
        soft_loss = nn.functional.kl_div(soft_student, soft_teacher, reduction='batchmean')
        soft_loss *= (self.temperature ** 2)
        
        return alpha * hard_loss + (1 - alpha) * soft_loss
    
    def train_student(
        self,
        student_model: nn.Module,
        train_loader,
        val_loader,
        device: torch.device,
        epochs: int = 50,
        lr: float = 0.001
    ) -> nn.Module:
        """生徒モデルを訓練"""
        optimizer = torch.optim.AdamW(student_model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
        
        best_acc = 0.0
        best_model = None
        
        for epoch in range(epochs):
            student_model.train()
            total_loss = 0.0
            
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                
                # 教師モデルの出力
                with torch.no_grad():
                    teacher_logits = self.teacher_model(images)
                
                # 生徒モデルの出力
                student_logits = student_model(images)
                
                # 蒸留損失
                loss = self.distillation_loss(student_logits, teacher_logits, labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            scheduler.step()
            
            # 検証
            student_model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = student_model(images)
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
            
            acc = 100. * correct / total
            if acc > best_acc:
                best_acc = acc
                best_model = copy.deepcopy(student_model)
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}, Acc: {acc:.2f}%")
        
        return best_model


if __name__ == "__main__":
    print("Auto Optimizer module ready!")
    print("\nFeatures:")
    print("  - GeneticNAS: Neural Architecture Search with genetic algorithm")
    print("  - ModelPruner: Channel pruning for model compression")
    print("  - KnowledgeDistillation: Transfer knowledge from teacher to student")
