# 📸 写真 vs イラスト分類モデル

超軽量な深層学習モデルで、画像が「写真」か「イラスト（アニメ・漫画・絵画）」かを99%の精度で判別します。

## 🎯 目標

- **精度**: 99%以上
- **モデルサイズ**: 0.5MB以下（FP16）
- **推論速度**: 1ms以下（GPU）
- **入力サイズ**: 224×224×3

## 🏗️ プロジェクト構造

```
photo_classifier/
├── main.py                 # メイン実行スクリプト
├── requirements.txt        # 依存ライブラリ
├── README.md              # このファイル
├── configs/               # 設定ファイル
│   └── default.yaml
├── src/                   # ソースコード
│   ├── model.py          # モデル定義
│   ├── train.py          # トレーニング
│   ├── evaluate.py       # 評価
│   ├── data_collector.py # データ収集
│   └── download_datasets.py
├── data/                  # データディレクトリ
│   ├── raw/              # 生データ
│   └── processed/        # 前処理済みデータ
└── checkpoints/          # モデルチェックポイント
```

## 🚀 クイックスタート

### 1. インストール

```bash
pip install -r requirements.txt
```

### 2. データ準備

#### オプションA: ミニデータセット（テスト用）
```bash
python src/download_datasets.py --type mini
```

#### オプションB: フルデータセット
```bash
# アニメ顔画像と写真をダウンロード
python src/download_datasets.py --type all --n-samples 10000
```

#### オプションC: 自分のデータを使用
```bash
# データを配置
mkdir -p data/raw/photos data/raw/illustrations
# 自分の画像をコピー

# データセット分割
python main.py prepare \
  --photo-dir ./data/raw/photos \
  --illust-dir ./data/raw/illustrations \
  --output-dir ./data/processed
```

### 3. モデル訓練

```bash
python main.py train \
  --data-dir ./data/processed \
  --model-name photo_classifier \
  --epochs 100 \
  --batch-size 64 \
  --learning-rate 0.001
```

### 4. モデル評価

```bash
python main.py evaluate \
  --model-path ./checkpoints/best_model.pth \
  --data-dir ./data/processed \
  --use-tta
```

### 5. 推論

```bash
python main.py predict \
  --model-path ./checkpoints/best_model.pth \
  --image-path ./test_image.jpg
```

## 🧠 モデルアーキテクチャ

### PhotoClassifier（推奨）
- **ベース**: MobileNetV3スタイルの軽量CNN
- **パラメータ**: ~300K（width_mult=0.75）
- **特徴**: 
  - Inverted Residual Blocks
  - Squeeze-and-Excitation Blocks
  - Depthwise Separable Convolutions

### TinyClassifier
- **パラメータ**: ~200K
- **用途**: 超軽量モデルが必要な場合

## 📊 99%精度達成のためのテクニック

### データ拡張
- ✅ RandomResizedCrop（scale=0.08-1.0）
- ✅ ColorJitter（brightness=0.3, contrast=0.3, saturation=0.3）
- ✅ CutMix（alpha=1.0）
- ✅ RandomErasing
- ✅ Horizontal/Vertical Flip

### 学習戦略
- ✅ Label Smoothing（0.1）
- ✅ Cosine Annealing LR
- ✅ AdamW Optimizer
- ✅ Weight Decay（1e-4）
- ✅ Gradient Clipping
- ✅ Mixed Precision Training

### 評価時
- ✅ Test Time Augmentation（TTA）

## 📈 期待される結果

| メトリクス | 目標値 | 実測値 |
|-----------|--------|--------|
| Accuracy  | 99%+   | ??.??% |
| Precision | 99%+   | ??.??% |
| Recall    | 99%+   | ??.??% |
| F1-Score  | 99%+   | ??.??% |
| モデルサイズ | <0.5MB | ??.??MB |
| 推論時間 | <1ms | ??.??ms |

## 🔧 高度な使用法

### カスタム設定で訓練

```python
from src.train import train_model

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
```

### モデルエクスポート

```bash
# ONNX形式
python main.py export \
  --model-path ./checkpoints/best_model.pth \
  --format onnx

# TorchScript形式
python main.py export \
  --model-path ./checkpoints/best_model.pth \
  --format torchscript

# 量子化モデル
python main.py export \
  --model-path ./checkpoints/best_model.pth \
  --format quantized
```

## 📝 注意事項

### データ収集時
- Safebooru APIにはレート制限があります（delayパラメータで調整）
- Unsplash APIキーが必要です（環境変数`UNSPLASH_ACCESS_KEY`）
- ダウンロードした画像は商用利用可能なライセンスを確認してください

### 訓練時
- GPUを推奨（CPUでも動作しますが遅い）
- ミニデータセットでは精度が出ない場合があります
- 99%達成には十分なデータ多様性が必要です

## 🐛 トラブルシューティング

### CUDA out of memory
```bash
# バッチサイズを減らす
python main.py train --batch-size 32
```

### 精度が出ない
```bash
# より強力なデータ拡張を使用
python main.py train --use-cutmix --label-smoothing 0.1 --epochs 150
```

### データセットが見つからない
```bash
# ミニデータセットを作成
python src/download_datasets.py --type mini
```

## 📚 参考文献

- MobileNetV3: [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244)
- CutMix: [CutMix: Regularization Strategy to Train Strong Classifiers](https://arxiv.org/abs/1905.04899)
- Label Smoothing: [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567)

## 📄 ライセンス

MIT License

## 🙏 謝辞

- Anime Face Dataset: [Hugging Face](https://huggingface.co/datasets/huggan/anime-faces)
- ImageNet: [ImageNet.org](https://www.image-net.org/)
- Safebooru: [safebooru.donmai.us](https://safebooru.donmai.us)
