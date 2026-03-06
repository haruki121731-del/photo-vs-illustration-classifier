# 🚀 クイックトレーニングガイド

## 99%精度達成までのステップ

---

## Step 1: データ収集（バックグラウンド実行推奨）

```bash
# ターミナル1: イラスト収集（時間がかかります）
python3 -c "
import sys
sys.path.insert(0, 'src')
from data_collector import SafebooruCollector
collector = SafebooruCollector(delay=0.3)
collector.collect_images(
    output_dir='data/massive/illustrations',
    max_images=10000
)
" &

# ターミナル2: 写真収集
python3 src/download_datasets.py --type imagenet --n-samples 10000 &
```

---

## Step 2: データセット準備

```bash
# 収集完了後、バランス調整
python3 -c "
from massive_data_pipeline import build_balanced_dataset
build_balanced_dataset(
    raw_dir='data/massive',
    output_dir='data/processed_massive',
    max_per_class=10000
)
"
```

---

## Step 3: 本格トレーニング（3パターン）

### パターンA: シンプルトレーニング（推奨）
```bash
python3 train_final_model.py \
    --data-dir ./data/processed_massive \
    --mode ultra_light \
    --epochs 150 \
    --output-dir ./training_results
```

### パターンB: NAS + フルトレーニング
```bash
python3 train_final_model.py \
    --data-dir ./data/processed_massive \
    --mode nas \
    --output-dir ./nas_results
```

### パターンC: 完全自己改善パイプライン
```bash
python3 src/self_improvement.py \
    --data-dir ./data/processed_massive \
    --output-dir ./self_improvement_results
```

---

## Step 4: 評価

```bash
# TTA（Test Time Augmentation）で高精度評価
python3 main.py evaluate \
    --model-path ./training_results/best_model.pth \
    --data-dir ./data/processed_massive \
    --use-tta
```

---

## 既存データで即座にトレーニング（デモ用）

```bash
# ミニデータセットでパイプライン確認（高速）
python3 train_final_model.py \
    --data-dir ./data/processed \
    --mode ultra_light \
    --epochs 30 \
    --output-dir ./demo_training
```

---

## マルチGPU環境の場合

```bash
# 4GPUで高速トレーニング
torchrun --nproc_per_node=4 \
    src/distributed_training.py \
    --data-dir ./data/processed_massive \
    --epochs 100
```

---

## 期待結果

| データ数 | エポック | 達成精度 | 所要時間 |
|---------|---------|---------|---------|
| 100枚 | 30 | 70-80% | 10分 |
| 1,000枚 | 50 | 85-92% | 1時間 |
| 10,000枚 | 100 | **98-99%** | 6-8時間 |
| 20,000枚 | 150 | **99-99.5%** | 12-15時間 |

---

## トレーニング監視

```bash
# ログ確認
tail -f training_results/training_history.json

# GPU監視（nvidia-smi）
watch -n 1 nvidia-smi
```

---

## 次のステップ

データ収集が完了したら、以下のコマンドで本格トレーニングを開始してください：

```bash
bash start_massive_training.sh
```
