#!/bin/bash
# クイックスタートスクリプト

set -e

echo "=============================================="
echo "  Photo vs Illustration Classifier"
echo "  Quick Start Script"
echo "=============================================="

# 1. 依存関係インストール
echo ""
echo "[1/5] Installing dependencies..."
pip install -q -r requirements.txt

# 2. ミニデータセット作成（素早いテスト用）
echo ""
echo "[2/5] Creating mini dataset..."
python src/download_datasets.py --type mini

# 3. モデル訓練（少ないエポックでテスト）
echo ""
echo "[3/5] Training model (10 epochs for quick test)..."
python main.py train \
    --data-dir ./data/mini \
    --model-name tiny \
    --epochs 10 \
    --batch-size 16 \
    --learning-rate 0.001 \
    --patience 5

# 4. モデル評価
echo ""
echo "[4/5] Evaluating model..."
python main.py evaluate \
    --model-path ./checkpoints/best_model.pth \
    --data-dir ./data/mini

# 5. サンプル推論
echo ""
echo "[5/5] Testing inference..."
# テスト用画像を作成
python -c "
from PIL import Image
import numpy as np
# 写真風画像
img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
img.save('test_photo.jpg')
print('Created test_photo.jpg')
"

python main.py predict \
    --model-path ./checkpoints/best_model.pth \
    --image-path ./test_photo.jpg

echo ""
echo "=============================================="
echo "  Quick Start Completed!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "  1. For full training with real data:"
echo "     python src/download_datasets.py --type all"
echo "     python main.py train --data-dir ./data/processed --epochs 100"
echo ""
echo "  2. To evaluate on test set:"
echo "     python main.py evaluate --model-path ./checkpoints/best_model.pth --data-dir ./data/processed --use-tta"
echo ""
echo "  3. To predict on new images:"
echo "     python main.py predict --model-path ./checkpoints/best_model.pth --image-path YOUR_IMAGE.jpg"
echo ""
