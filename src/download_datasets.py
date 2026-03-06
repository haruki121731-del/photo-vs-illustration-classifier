#!/usr/bin/env python3
"""
データセットダウンロードスクリプト
- KaggleからImageNetサブセットをダウンロード
- Hugging Faceからアニメ画像をダウンロード
- 小規模テストデータセットも用意
"""

import os
import sys
import requests
import zipfile
import tarfile
from pathlib import Path
from tqdm import tqdm
import urllib.request


def download_file(url: str, output_path: str, desc: str = None):
    """ファイルをダウンロード"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.exists():
        print(f"File already exists: {output_path}")
        return True
    
    desc = desc or f"Downloading {output_path.name}"
    
    try:
        with requests.get(url, stream=True, timeout=300) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            
            with open(output_path, 'wb') as f, tqdm(
                desc=desc,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        if output_path.exists():
            output_path.unlink()
        return False


def extract_archive(archive_path: str, output_dir: str):
    """アーカイブを展開"""
    archive_path = Path(archive_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Extracting {archive_path.name}...")
    
    try:
        if archive_path.suffix == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
        elif archive_path.suffix in ['.tar', '.gz', '.tgz']:
            with tarfile.open(archive_path, 'r:*') as tar_ref:
                tar_ref.extractall(output_dir)
        else:
            print(f"Unsupported archive format: {archive_path.suffix}")
            return False
        
        print(f"Extracted to: {output_dir}")
        return True
    except Exception as e:
        print(f"Error extracting {archive_path}: {e}")
        return False


def download_anime_face_dataset(output_dir: str = './data/raw/illustrations'):
    """
    Anime Face Datasetをダウンロード
    ソース: https://github.com/bchao1/Anime-Face-Dataset
    """
    print("\n" + "="*60)
    print("Downloading Anime Face Dataset")
    print("="*60)
    
    # Hugging Faceからダウンロード
    try:
        from datasets import load_dataset
        
        print("Loading from Hugging Face...")
        dataset = load_dataset("huggan/anime-faces", split="train", streaming=True)
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        count = 0
        for i, item in enumerate(dataset):
            if i >= 10000:  # 最大10,000枚
                break
            
            image = item['image']
            filepath = output_path / f"anime_face_{i:06d}.jpg"
            image.save(filepath, 'JPEG')
            count += 1
            
            if (i + 1) % 1000 == 0:
                print(f"  Downloaded {i + 1} images...")
        
        print(f"Downloaded {count} anime face images to {output_dir}")
        return True
        
    except ImportError:
        print("Hugging Face datasets library not found.")
        print("Install with: pip install datasets")
        print("\nAlternative: Using sample data generator...")
        return generate_sample_data(output_dir, 'illustration', n_samples=1000)
    except Exception as e:
        print(f"Error: {e}")
        return generate_sample_data(output_dir, 'illustration', n_samples=1000)


def download_imagenet_sample(output_dir: str = './data/raw/photos', n_samples: int = 10000):
    """
    ImageNetサンプルをダウンロード
    小規模なサブセットを使用
    """
    print("\n" + "="*60)
    print("Downloading ImageNet Sample")
    print("="*60)
    
    try:
        from datasets import load_dataset
        
        print("Loading ImageNet-1k sample from Hugging Face...")
        # ストリーミングで読み込み
        dataset = load_dataset("ILSVRC/imagenet-1k", split="train", streaming=True, trust_remote_code=True)
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        count = 0
        for i, item in enumerate(dataset):
            if i >= n_samples:
                break
            
            image = item['image']
            # RGB変換
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            filepath = output_path / f"photo_{i:06d}.jpg"
            image.save(filepath, 'JPEG')
            count += 1
            
            if (i + 1) % 1000 == 0:
                print(f"  Downloaded {i + 1} images...")
        
        print(f"Downloaded {count} photos to {output_dir}")
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nAlternative: Using sample data generator...")
        return generate_sample_data(output_dir, 'photo', n_samples=1000)


def generate_sample_data(output_dir: str, data_type: str, n_samples: int = 100):
    """
    サンプルデータを生成（テスト用）
    実際の画像をダウンロードできない場合のフォールバック
    """
    print(f"\nGenerating {n_samples} sample {data_type} images...")
    
    from PIL import Image, ImageDraw, ImageFilter
    import numpy as np
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for i in range(n_samples):
        # ランダムな画像を生成
        img = Image.new('RGB', (224, 224), color=(
            np.random.randint(0, 255),
            np.random.randint(0, 255),
            np.random.randint(0, 255)
        ))
        
        draw = ImageDraw.Draw(img)
        
        if data_type == 'photo':
            # 写真風のパターン（より複雑なテクスチャ）
            for _ in range(50):
                x, y = np.random.randint(0, 224, 2)
                r = np.random.randint(1, 20)
                color = tuple(np.random.randint(0, 255, 3))
                draw.ellipse([x-r, y-r, x+r, y+r], fill=color)
            # 軽いノイズ
            img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
        else:
            # イラスト風のパターン（より単純な形状）
            for _ in range(10):
                x1, y1, x2, y2 = np.random.randint(0, 224, 4)
                color = tuple(np.random.randint(0, 255, 3))
                draw.rectangle([x1, y1, x2, y2], fill=color)
        
        filepath = output_path / f"{data_type}_sample_{i:04d}.jpg"
        img.save(filepath, 'JPEG')
    
    print(f"Generated {n_samples} sample images to {output_dir}")
    return True


def download_cifar10_subset(output_dir: str = './data/raw', n_per_class: int = 5000):
    """
    CIFAR-10から写真（動物、車両など）と非写真を分類して使用
    """
    print("\n" + "="*60)
    print("Downloading CIFAR-10 Subset")
    print("="*60)
    
    try:
        import torchvision
        
        # CIFAR-10をダウンロード
        cifar_dir = Path(output_dir) / 'cifar10_temp'
        cifar_dir.mkdir(parents=True, exist_ok=True)
        
        dataset = torchvision.datasets.CIFAR10(
            root=cifar_dir, train=True, download=True
        )
        
        # クラス定義
        # 写真に近い: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
        # これらは実写写真ではないが、テスト用に使用可能
        
        photo_classes = [0, 1, 8, 9]  # airplane, automobile, ship, truck
        illustration_classes = [2, 3, 4, 5, 6, 7]  # bird, cat, deer, dog, frog, horse
        
        photo_dir = Path(output_dir) / 'photos'
        illust_dir = Path(output_dir) / 'illustrations'
        photo_dir.mkdir(exist_ok=True)
        illust_dir.mkdir(exist_ok=True)
        
        photo_count = 0
        illust_count = 0
        
        for i, (image, label) in enumerate(dataset):
            if label in photo_classes and photo_count < n_per_class:
                image.save(photo_dir / f"cifar_photo_{photo_count:05d}.png")
                photo_count += 1
            elif label in illustration_classes and illust_count < n_per_class:
                image.save(illust_dir / f"cifar_illust_{illust_count:05d}.png")
                illust_count += 1
            
            if photo_count >= n_per_class and illust_count >= n_per_class:
                break
        
        print(f"Downloaded {photo_count} photos and {illust_count} illustrations from CIFAR-10")
        
        # 一時ディレクトリ削除
        import shutil
        shutil.rmtree(cifar_dir, ignore_errors=True)
        
        return True
        
    except Exception as e:
        print(f"Error downloading CIFAR-10: {e}")
        return False


def create_mini_dataset(output_dir: str = './data/mini', n_per_class: int = 100):
    """
    超小規模データセットを作成（素早いテスト用）
    """
    print("\n" + "="*60)
    print("Creating Mini Dataset for Quick Testing")
    print("="*60)
    
    import shutil
    from sklearn.model_selection import train_test_split
    
    # サンプルデータ生成
    temp_dir = Path(output_dir) / 'temp'
    generate_sample_data(temp_dir / 'photos', 'photo', n_per_class * 2)
    generate_sample_data(temp_dir / 'illustrations', 'illustration', n_per_class * 2)
    
    # 分割
    photo_files = list((temp_dir / 'photos').glob('*.jpg'))
    illust_files = list((temp_dir / 'illustrations').glob('*.jpg'))
    
    def split_and_copy(files, dst_dir, class_name):
        train, temp = train_test_split(files, test_size=0.2, random_state=42)
        val, test = train_test_split(temp, test_size=0.5, random_state=42)
        
        for split_name, split_files in [('train', train), ('val', val), ('test', test)]:
            dst = Path(dst_dir) / split_name / class_name
            dst.mkdir(parents=True, exist_ok=True)
            for f in split_files:
                shutil.copy2(f, dst / f.name)
    
    split_and_copy(photo_files[:n_per_class], output_dir, 'photo')
    split_and_copy(illust_files[:n_per_class], output_dir, 'illustration')
    
    # 一時ディレクトリ削除
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    print(f"Mini dataset created at {output_dir}")
    print("  Structure: train/photo, train/illustration, val/, test/")
    return True


def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Download datasets for photo vs illustration classification')
    parser.add_argument('--type', type=str, default='all', 
                       choices=['all', 'anime', 'imagenet', 'cifar10', 'mini'],
                       help='Dataset type to download')
    parser.add_argument('--output-dir', type=str, default='./data/raw', help='Output directory')
    parser.add_argument('--n-samples', type=int, default=10000, help='Number of samples per class')
    
    args = parser.parse_args()
    
    print("="*60)
    print("DATASET DOWNLOAD TOOL")
    print("="*60)
    
    success = []
    
    if args.type in ['all', 'anime']:
        success.append(('Anime Faces', download_anime_face_dataset(
            os.path.join(args.output_dir, 'illustrations')
        )))
    
    if args.type in ['all', 'imagenet']:
        success.append(('ImageNet Sample', download_imagenet_sample(
            os.path.join(args.output_dir, 'photos'),
            n_samples=args.n_samples
        )))
    
    if args.type in ['all', 'cifar10']:
        success.append(('CIFAR-10', download_cifar10_subset(
            args.output_dir,
            n_per_class=args.n_samples
        )))
    
    if args.type == 'mini':
        success.append(('Mini Dataset', create_mini_dataset(
            './data/mini',
            n_per_class=100
        )))
    
    print("\n" + "="*60)
    print("DOWNLOAD SUMMARY")
    print("="*60)
    for name, ok in success:
        status = "✓" if ok else "✗"
        print(f"{status} {name}")
    print("="*60)


if __name__ == '__main__':
    main()
