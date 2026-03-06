#!/bin/bash
# 大規模トレーニング開始スクリプト

set -e

echo "=============================================="
echo "  MASSIVE TRAINING FOR 99% ACCURACY"
echo "=============================================="

# 設定
DATA_DIR="./data/massive_dataset"
PROCESSED_DIR="./data/processed_massive"
OUTPUT_DIR="./massive_training"
TARGET_PHOTOS=10000
TARGET_ILLUSTRATIONS=10000

echo ""
echo "Target: ${TARGET_PHOTOS} photos + ${TARGET_ILLUSTRATIONS} illustrations"
echo "Expected training time: 6-12 hours (depends on hardware)"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! \$REPLY =~ ^[Yy]$ ]]; then
    exit 1
fi

# Step 1: データ収集
echo ""
echo "[Step 1/4] Collecting massive dataset..."
python3 -c "
import sys
sys.path.insert(0, 'src')
from massive_data_pipeline import DatasetBuilder
import asyncio

builder = DatasetBuilder(output_dir='${DATA_DIR}')
asyncio.run(builder.build_dataset(
    target_photos=${TARGET_PHOTOS},
    target_illustrations=${TARGET_ILLUSTRATIONS}
))
"

# Step 2: データセット準備
echo ""
echo "[Step 2/4] Preparing balanced dataset..."
python3 -c "
from massive_data_pipeline import build_balanced_dataset
build_balanced_dataset(
    raw_dir='${DATA_DIR}/raw',
    output_dir='${PROCESSED_DIR}',
    max_per_class=10000
)
"

# Step 3: 自己改善パイプライン（NAS + Training + Pruning + Distillation）
echo ""
echo "[Step 3/4] Starting self-improvement pipeline..."
python3 src/self_improvement.py \
    --data-dir "${PROCESSED_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --batch-size 64

# Step 4: 最終評価
echo ""
echo "[Step 4/4] Final evaluation..."
python3 main.py evaluate \
    --model-path "${OUTPUT_DIR}/phase2_best_model.pth" \
    --data-dir "${PROCESSED_DIR}" \
    --use-tta

echo ""
echo "=============================================="
echo "  TRAINING COMPLETED!"
echo "=============================================="
echo "Results saved to: ${OUTPUT_DIR}"
echo ""
echo "Best model: ${OUTPUT_DIR}/phase2_best_model.pth"
echo ""

# 結果表示
if [ -f "${OUTPUT_DIR}/self_improvement_results.json" ]; then
    echo "Final Results:"
    cat "${OUTPUT_DIR}/self_improvement_results.json" | grep -A 5 "best_model"
fi
