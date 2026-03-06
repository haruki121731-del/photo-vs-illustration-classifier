"""
超軽量画像分類モデル
- 目標: 99%精度、0.5Mパラメータ以下
- 入力: 224x224x3
- 出力: 2クラス（写真 vs 非写真）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import math


class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution (MobileNetスタイル)"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, 
                 stride: int = 1, padding: int = 1, bias: bool = False):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride, padding, 
            groups=in_channels, bias=bias
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU6(inplace=True)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block (軽量版)"""
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


class InvertedResidual(nn.Module):
    """Inverted Residual Block (MobileNetV2スタイル)"""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, 
                 expand_ratio: int = 6, use_se: bool = True):
        super().__init__()
        self.stride = stride
        self.use_residual = stride == 1 and in_channels == out_channels
        
        hidden_dim = int(round(in_channels * expand_ratio))
        
        layers = []
        # Expansion
        if expand_ratio != 1:
            layers.append(nn.Conv2d(in_channels, hidden_dim, 1, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))
        
        # Depthwise
        layers.append(nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, 
                               groups=hidden_dim, bias=False))
        layers.append(nn.BatchNorm2d(hidden_dim))
        layers.append(nn.ReLU6(inplace=True))
        
        # SE Block
        if use_se:
            layers.append(SEBlock(hidden_dim, reduction=4))
        
        # Projection
        layers.append(nn.Conv2d(hidden_dim, out_channels, 1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)


class PhotoClassifier(nn.Module):
    """
    写真 vs 非写真分類のための超軽量モデル
    目標: 0.5Mパラメータ以下、99%精度
    """
    def __init__(self, num_classes: int = 2, width_mult: float = 1.0, dropout: float = 0.2):
        super().__init__()
        
        # 設定
        self.cfgs = [
            # t, c, n, s, se (expand_ratio, out_channels, num_blocks, stride, use_se)
            [1, 16, 1, 1, True],
            [6, 24, 2, 2, True],
            [6, 32, 3, 2, True],
            [6, 64, 2, 2, True],
            [6, 96, 2, 1, True],
            [6, 160, 1, 2, True],
        ]
        
        # 最初の畳み込み
        input_channel = int(32 * width_mult)
        self.first_conv = nn.Sequential(
            nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU6(inplace=True)
        )
        
        # Inverted Residual Blocks
        layers = []
        for t, c, n, s, se in self.cfgs:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                layers.append(InvertedResidual(input_channel, output_channel, 
                                              stride, expand_ratio=t, use_se=se))
                input_channel = output_channel
        
        self.features = nn.Sequential(*layers)
        
        # 最後の畳み込み
        self.last_channel = int(128 * width_mult)
        self.last_conv = nn.Sequential(
            nn.Conv2d(input_channel, self.last_channel, 1, bias=False),
            nn.BatchNorm2d(self.last_channel),
            nn.ReLU6(inplace=True)
        )
        
        # 分類器
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.last_channel, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
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
        """パラメータ数をカウント"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TinyClassifier(nn.Module):
    """
    より小さなモデル（0.3Mパラメータ以下を目指す）
    精度が出ない場合はPhotoClassifierを使用
    """
    def __init__(self, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        
        # シンプルなCNN
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Block 2
            DepthwiseSeparableConv(32, 64, stride=2),
            DepthwiseSeparableConv(64, 64, stride=1),
            
            # Block 3
            DepthwiseSeparableConv(64, 128, stride=2),
            DepthwiseSeparableConv(128, 128, stride=1),
            
            # Block 4
            DepthwiseSeparableConv(128, 256, stride=2),
            SEBlock(256, reduction=8),
            
            # Block 5
            DepthwiseSeparableConv(256, 256, stride=1),
            SEBlock(256, reduction=8),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
        self._initialize_weights()
    
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
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(model_name: str = 'photo_classifier', **kwargs) -> nn.Module:
    """
    モデルを作成
    
    Args:
        model_name: 'photo_classifier', 'tiny', 'mobilenet_v3_small'
        **kwargs: モデル固有のパラメータ
    """
    if model_name == 'photo_classifier':
        return PhotoClassifier(**kwargs)
    elif model_name == 'tiny':
        return TinyClassifier(**kwargs)
    elif model_name == 'mobilenet_v3_small':
        try:
            import torchvision.models as models
            model = models.mobilenet_v3_small(weights=None, num_classes=2, **kwargs)
            return model
        except ImportError:
            print("torchvision not available, using PhotoClassifier instead")
            return PhotoClassifier(**kwargs)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def test_model():
    """モデルのテスト"""
    print("Testing PhotoClassifier...")
    model = PhotoClassifier(num_classes=2, width_mult=1.0)
    
    # パラメータ数
    params = model.count_parameters()
    print(f"Parameters: {params:,} ({params/1e6:.2f}M)")
    
    # テスト入力
    x = torch.randn(1, 3, 224, 224)
    
    # FLOPs計算（簡易版）
    from thop import profile
    flops, _ = profile(model, inputs=(x,), verbose=False)
    print(f"FLOPs: {flops/1e9:.2f}G")
    
    # 推論テスト
    model.eval()
    with torch.no_grad():
        y = model(x)
    print(f"Output shape: {y.shape}")
    
    print("\nTesting TinyClassifier...")
    model2 = TinyClassifier()
    params2 = model2.count_parameters()
    print(f"Parameters: {params2:,} ({params2/1e6:.2f}M)")


if __name__ == "__main__":
    # テスト
    print("Model module ready!")
    
    # 簡易テスト
    model = PhotoClassifier(num_classes=2, width_mult=0.75)
    params = model.count_parameters()
    print(f"PhotoClassifier (width=0.75): {params:,} parameters ({params/1e6:.2f}M)")
    
    model2 = TinyClassifier()
    params2 = model2.count_parameters()
    print(f"TinyClassifier: {params2:,} parameters ({params2/1e6:.2f}M)")
    
    # 入力テスト
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
