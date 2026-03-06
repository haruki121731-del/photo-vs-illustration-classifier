#!/usr/bin/env python3
"""
写真 vs 非写真 分類モデル
メイン実行スクリプト

使い方:
  # データ収集
  python main.py collect --output-dir ./data/raw
  
  # データセット準備
  python main.py prepare --photo-dir ./data/raw/photos --illust-dir ./data/raw/illustrations --output-dir ./data/processed
  
  # モデル訓練
  python main.py train --data-dir ./data/processed --epochs 100 --batch-size 64
  
  # モデル評価
  python main.py evaluate --model-path ./checkpoints/best_model.pth --data-dir ./data/processed
  
  # 推論
  python main.py predict --model-path ./checkpoints/best_model.pth --image-path ./test.jpg
"""

import os
import sys
import argparse
import json
from pathlib import Path

# srcディレクトリをパスに追加
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_collector import SafebooruCollector, UnsplashCollector, create_balanced_dataset
from model import create_model, PhotoClassifier, TinyClassifier
from train import train_model
from evaluate import evaluate_model

import torch
import torchvision.transforms as transforms
from PIL import Image


def cmd_collect(args):
    """データ収集コマンド"""
    print("="*60)
    print("DATA COLLECTION")
    print("="*60)
    
    # 写真データ収集
    print("\n[1/2] Collecting photos...")
    if args.unsplash_key:
        unsplash = UnsplashCollector(access_key=args.unsplash_key)
        photos = unsplash.collect_photos(
            output_dir=os.path.join(args.output_dir, 'photos'),
            max_images=args.max_photos
        )
        print(f"Collected {len(photos)} photos from Unsplash")
    else:
        print("Skipping Unsplash collection (no API key provided)")
        print("Alternative: Use ImageNet subset or other photo datasets")
        print(f"Please place photo images in: {args.output_dir}/photos/")
    
    # イラストデータ収集
    print("\n[2/2] Collecting illustrations...")
    safebooru = SafebooruCollector(delay=args.delay)
    illustrations = safebooru.collect_images(
        output_dir=os.path.join(args.output_dir, 'illustrations'),
        max_images=args.max_illustrations,
        min_width=args.min_image_size,
        min_height=args.min_image_size
    )
    print(f"Collected {len(illustrations)} illustrations from Safebooru")
    
    print("\n" + "="*60)
    print("Data collection completed!")
    print(f"Output directory: {args.output_dir}")
    print("="*60)


def cmd_prepare(args):
    """データセット準備コマンド"""
    print("="*60)
    print("DATASET PREPARATION")
    print("="*60)
    
    create_balanced_dataset(
        photo_dir=args.photo_dir,
        illustration_dir=args.illust_dir,
        output_dir=args.output_dir,
        max_per_class=args.max_per_class,
        val_split=args.val_split,
        test_split=args.test_split
    )
    
    print("\n" + "="*60)
    print("Dataset preparation completed!")
    print("="*60)


def cmd_train(args):
    """モデル訓練コマンド"""
    print("="*60)
    print("MODEL TRAINING")
    print("="*60)
    
    trainer, history = train_model(
        data_dir=args.data_dir,
        model_name=args.model_name,
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        optimizer=args.optimizer,
        weight_decay=args.weight_decay,
        scheduler=args.scheduler,
        use_amp=not args.no_amp,
        use_cutmix=args.use_cutmix,
        label_smoothing=args.label_smoothing,
        grad_clip=args.grad_clip,
        early_stopping_patience=args.patience,
        checkpoint_dir=args.checkpoint_dir,
        seed=args.seed
    )
    
    print("\n" + "="*60)
    print("Training completed!")
    print("="*60)


def cmd_evaluate(args):
    """モデル評価コマンド"""
    print("="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    results = evaluate_model(
        model_path=args.model_path,
        data_dir=args.data_dir,
        model_name=args.model_name,
        image_size=args.image_size,
        batch_size=args.batch_size,
        use_tta=args.use_tta,
        output_dir=args.output_dir
    )
    
    return results


def cmd_predict(args):
    """推論コマンド"""
    print("="*60)
    print("INFERENCE")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # モデル読み込み
    checkpoint = torch.load(args.model_path, map_location=device)
    config = checkpoint.get('config', {})
    model_config = config.get('model_config', {})
    
    model = create_model(args.model_name, num_classes=2, **model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # 画像前処理
    image_size = args.image_size
    transform = transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # 画像読み込み
    image = Image.open(args.image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # 推論
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        pred = output.argmax(dim=1).item()
        confidence = probs[0][pred].item()
    
    classes = ['Photo', 'Illustration']
    result = classes[pred]
    
    print(f"\nImage: {args.image_path}")
    print(f"Prediction: {result}")
    print(f"Confidence: {confidence*100:.2f}%")
    print(f"Probabilities:")
    print(f"  Photo:        {probs[0][0]*100:.2f}%")
    print(f"  Illustration: {probs[0][1]*100:.2f}%")
    
    if args.output:
        result_data = {
            'image_path': args.image_path,
            'prediction': result,
            'confidence': confidence,
            'probabilities': {
                'photo': probs[0][0].item(),
                'illustration': probs[0][1].item()
            }
        }
        with open(args.output, 'w') as f:
            json.dump(result_data, f, indent=2)
        print(f"\nResult saved to: {args.output}")
    
    print("\n" + "="*60)


def cmd_export(args):
    """モデルエクスポートコマンド"""
    print("="*60)
    print("MODEL EXPORT")
    print("="*60)
    
    device = torch.device('cpu')  # エクスポートはCPUで
    
    # モデル読み込み
    checkpoint = torch.load(args.model_path, map_location=device)
    config = checkpoint.get('config', {})
    model_config = config.get('model_config', {})
    
    model = create_model(args.model_name, num_classes=2, **model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # ONNXエクスポート
    if args.format in ['onnx', 'all']:
        dummy_input = torch.randn(1, 3, args.image_size, args.image_size)
        onnx_path = args.output_path.replace('.pth', '.onnx')
        
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
            opset_version=11
        )
        print(f"Exported to ONNX: {onnx_path}")
    
    # TorchScriptエクスポート
    if args.format in ['torchscript', 'all']:
        dummy_input = torch.randn(1, 3, args.image_size, args.image_size)
        traced_model = torch.jit.trace(model, dummy_input)
        ts_path = args.output_path.replace('.pth', '.pt')
        traced_model.save(ts_path)
        print(f"Exported to TorchScript: {ts_path}")
    
    # 量子化モデル
    if args.format in ['quantized', 'all']:
        model.eval()
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        torch.quantization.prepare(model, inplace=True)
        
        # キャリブレーション（ダミー入力で）
        with torch.no_grad():
            for _ in range(100):
                dummy_input = torch.randn(1, 3, args.image_size, args.image_size)
                model(dummy_input)
        
        torch.quantization.convert(model, inplace=True)
        q_path = args.output_path.replace('.pth', '_quantized.pth')
        torch.save(model.state_dict(), q_path)
        print(f"Exported quantized model: {q_path}")
    
    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(
        description='Photo vs Illustration Classifier',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Collect data
  python main.py collect --output-dir ./data/raw --max-photos 5000 --max-illustrations 5000
  
  # Prepare dataset
  python main.py prepare --photo-dir ./data/raw/photos --illust-dir ./data/raw/illustrations --output-dir ./data/processed
  
  # Train model
  python main.py train --data-dir ./data/processed --epochs 100 --batch-size 64 --learning-rate 0.001
  
  # Evaluate model
  python main.py evaluate --model-path ./checkpoints/best_model.pth --data-dir ./data/processed --use-tta
  
  # Predict single image
  python main.py predict --model-path ./checkpoints/best_model.pth --image-path ./test.jpg
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # collect コマンド
    collect_parser = subparsers.add_parser('collect', help='Collect training data')
    collect_parser.add_argument('--output-dir', type=str, default='./data/raw', help='Output directory')
    collect_parser.add_argument('--max-photos', type=int, default=5000, help='Max photos to collect')
    collect_parser.add_argument('--max-illustrations', type=int, default=5000, help='Max illustrations to collect')
    collect_parser.add_argument('--unsplash-key', type=str, default=os.getenv('UNSPLASH_ACCESS_KEY'), help='Unsplash API key')
    collect_parser.add_argument('--delay', type=float, default=0.5, help='Request delay')
    collect_parser.add_argument('--min-image-size', type=int, default=224, help='Minimum image size')
    collect_parser.set_defaults(func=cmd_collect)
    
    # prepare コマンド
    prepare_parser = subparsers.add_parser('prepare', help='Prepare dataset')
    prepare_parser.add_argument('--photo-dir', type=str, required=True, help='Photo directory')
    prepare_parser.add_argument('--illust-dir', type=str, required=True, help='Illustration directory')
    prepare_parser.add_argument('--output-dir', type=str, default='./data/processed', help='Output directory')
    prepare_parser.add_argument('--max-per-class', type=int, default=10000, help='Max images per class')
    prepare_parser.add_argument('--val-split', type=float, default=0.1, help='Validation split ratio')
    prepare_parser.add_argument('--test-split', type=float, default=0.1, help='Test split ratio')
    prepare_parser.set_defaults(func=cmd_prepare)
    
    # train コマンド
    train_parser = subparsers.add_parser('train', help='Train model')
    train_parser.add_argument('--data-dir', type=str, default='./data/processed', help='Data directory')
    train_parser.add_argument('--model-name', type=str, default='photo_classifier', choices=['photo_classifier', 'tiny', 'mobilenet_v3_small'], help='Model architecture')
    train_parser.add_argument('--image-size', type=int, default=224, help='Input image size')
    train_parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    train_parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    train_parser.add_argument('--learning-rate', type=float, default=1e-3, help='Learning rate')
    train_parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'adamw', 'sgd'], help='Optimizer')
    train_parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    train_parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'onecycle', 'none'], help='LR scheduler')
    train_parser.add_argument('--no-amp', action='store_true', help='Disable automatic mixed precision')
    train_parser.add_argument('--use-cutmix', action='store_true', default=True, help='Use CutMix augmentation')
    train_parser.add_argument('--label-smoothing', type=float, default=0.1, help='Label smoothing factor')
    train_parser.add_argument('--grad-clip', type=float, default=1.0, help='Gradient clipping threshold')
    train_parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    train_parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints', help='Checkpoint directory')
    train_parser.add_argument('--seed', type=int, default=42, help='Random seed')
    train_parser.set_defaults(func=cmd_train)
    
    # evaluate コマンド
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate model')
    eval_parser.add_argument('--model-path', type=str, required=True, help='Model checkpoint path')
    eval_parser.add_argument('--data-dir', type=str, default='./data/processed', help='Data directory')
    eval_parser.add_argument('--model-name', type=str, default='photo_classifier', help='Model name')
    eval_parser.add_argument('--image-size', type=int, default=224, help='Input image size')
    eval_parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    eval_parser.add_argument('--use-tta', action='store_true', help='Use Test Time Augmentation')
    eval_parser.add_argument('--output-dir', type=str, default='./evaluation_results', help='Output directory')
    eval_parser.set_defaults(func=cmd_evaluate)
    
    # predict コマンド
    predict_parser = subparsers.add_parser('predict', help='Predict single image')
    predict_parser.add_argument('--model-path', type=str, required=True, help='Model checkpoint path')
    predict_parser.add_argument('--image-path', type=str, required=True, help='Image path')
    predict_parser.add_argument('--model-name', type=str, default='photo_classifier', help='Model name')
    predict_parser.add_argument('--image-size', type=int, default=224, help='Input image size')
    predict_parser.add_argument('--output', type=str, help='Output JSON path')
    predict_parser.set_defaults(func=cmd_predict)
    
    # export コマンド
    export_parser = subparsers.add_parser('export', help='Export model')
    export_parser.add_argument('--model-path', type=str, required=True, help='Model checkpoint path')
    export_parser.add_argument('--output-path', type=str, default='./exported_model.pth', help='Output path')
    export_parser.add_argument('--model-name', type=str, default='photo_classifier', help='Model name')
    export_parser.add_argument('--image-size', type=int, default=224, help='Input image size')
    export_parser.add_argument('--format', type=str, default='all', choices=['onnx', 'torchscript', 'quantized', 'all'], help='Export format')
    export_parser.set_defaults(func=cmd_export)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    args.func(args)


if __name__ == '__main__':
    main()
