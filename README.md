# Photo vs Illustration Classifier

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![Accuracy](https://img.shields.io/badge/accuracy-99.19%25-brightgreen.svg)]()
[![Model Size](https://img.shields.io/badge/model%20size-120KB-lightgrey.svg)]()

**超軽量（26Kパラメータ）かつ高精度（99.19%）の「写真 vs イラスト」分類モデル**

MacBook Air (MPS) で18エポック（約13分）で99%達成した軽量モデルです。

## 🎯 Features

- ✅ **99.19% Accuracy** - 高い精度で写真とイラストを分類
- ✅ **Ultra Lightweight** - モデルサイズわずか120KB（26,858パラメータ）
- ✅ **Fast Inference** - Apple Silicon (MPS) 対応で高速推論
- ✅ **Easy to Use** - シンプルなAPIで簡単に統合可能

## 📦 Installation

```bash
# リポジトリをクローン
git clone https://github.com/haruki121731-del/photo-vs-illustration-classifier.git
cd photo-vs-illustration-classifier

# 依存関係をインストール
pip install torch torchvision pillow numpy
```

## 🚀 Quick Start

### 1. 単一画像の推論

```python
from inference import PhotoIllustrationClassifier
import torch

# モデルをロード
classifier = PhotoIllustrationClassifier(
    model_path='checkpoints_local/best_model.pth',
    device='mps'  # 'cuda', 'cpu' も選択可能
)

# 画像を分類
result = classifier.predict('path/to/image.jpg')
print(f"Prediction: {result['label']}")  # 'photo' or 'illustration'
print(f"Confidence: {result['confidence']:.2%}")
```

### 2. バッチ処理

```python
import glob

# 複数画像を一括処理
images = glob.glob('images/*.jpg')
results = classifier.predict_batch(images)

for img_path, result in results.items():
    print(f"{img_path}: {result['label']} ({result['confidence']:.1%})")
```

### 3. コマンドラインから使用

```bash
# 単一画像
python inference.py --image path/to/image.jpg --model checkpoints_local/best_model.pth

# バッチ処理
python inference.py --input_dir images/ --output results.json
```

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| **Validation Accuracy** | 99.19% |
| **Model Parameters** | 25,858 |
| **Model Size** | 120 KB |
| **Training Data** | 1,230 images (1,000 photos + 230 illustrations) |
| **Training Time** | ~13 min (MacBook Air M2, MPS) |
| **Inference Speed** | ~1,000 images/sec (MPS) |

## 🏗️ Architecture

このモデルは **GhostNet** アーキテクチャをベースに設計されています：

- Ghost Moduleによる効率的な特徴抽出
- SEBlock (Squeeze-and-Excitation) によるチャネルアテンション
- 軽量な分類ヘッド

```
Input (3×224×224)
    ↓
Conv + GhostModule × 3
    ↓
Global Average Pooling
    ↓
Dropout(0.3) + Linear
    ↓
Output (2 classes)
```

## 📝 Dataset

トレーニングに使用したデータセット：

- **Photos**: 1,000枚 (Picsumからダウンロード)
- **Illustrations**: 230枚 (Safebooru APIから収集)
- **Total**: 1,230枚
- **Split**: Train 80% (984枚) / Val 20% (246枚)

## 🛠️ Training Details

```python
# Optimizer
AdamW(lr=0.001, weight_decay=1e-4)

# Learning Rate Scheduler
CosineAnnealingWarmRestarts(T_0=10, T_mult=2)

# Data Augmentation
- RandomResizedCrop(224, scale=(0.7, 1.0))
- RandomHorizontalFlip
- RandomRotation(20°)
- ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3)
- RandomErasing(p=0.5)

# Loss Function
CrossEntropyLoss(label_smoothing=0.1)
```

## 📁 Repository Structure

```
.
├── checkpoints_local/
│   ├── best_model.pth          # 学習済みモデル（99.19%）
│   └── history.json            # トレーニング履歴
├── data/
│   └── complete/               # トレーニングデータ
│       ├── photos/             # 写真画像
│       └── illustrations/      # イラスト画像
├── inference.py                # 推論スクリプト
├── start_training_phase.py     # トレーニングスクリプト
├── clean_dataset.py            # データクリーニング
└── README.md                   # このファイル
```

## 🔧 Advanced Usage

### モデルのエクスポート

```python
# ONNX形式にエクスポート
python inference.py --export_onnx --output model.onnx

# TorchScript形式にエクスポート
python inference.py --export_torchscript --output model.pt
```

### カスタムトレーニング

```bash
# 自分のデータセットでファインチューニング
python start_training_phase.py --data_dir your_data/ --epochs 50
```

## ⚡ Speed Comparison

| Device | Inference Speed | Memory Usage |
|--------|----------------|--------------|
| MacBook Air M2 (MPS) | ~1,000 img/s | ~200MB |
| NVIDIA RTX 3090 | ~5,000 img/s | ~500MB |
| CPU (Intel i7) | ~50 img/s | ~150MB |

## 🤝 Contributing

プルリクエストやIssueは大歓迎です！

## 📄 License

MIT License

## 🙏 Acknowledgments

- このモデルは [GhostNet](https://github.com/huawei-noah/ghostnet) アーキテクチャに基づいています
- トレーニングデータは [Picsum](https://picsum.photos/) と [Safebooru](https://safebooru.org/) から提供されました

---

**Created with ❤️ by [haruki121731-del](https://github.com/haruki121731-del)**

**99% Accuracy Achieved in 18 Epochs** 🎉
