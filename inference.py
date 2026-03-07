#!/usr/bin/env python3
"""
Photo vs Illustration Classifier - Inference Script
Easy-to-use inference API for the trained model
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Union, List, Dict

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np


class PhotoIllustrationClassifier:
    """
    写真 vs イラスト 分類器
    
    Usage:
        classifier = PhotoIllustrationClassifier()
        result = classifier.predict('image.jpg')
        print(result['label'])  # 'photo' or 'illustration'
    """
    
    def __init__(self, model_path: str = 'checkpoints_local/best_model.pth', 
                 device: str = None):
        """
        Args:
            model_path: 学習済みモデルのパス
            device: 'mps', 'cuda', 'cpu', or None (auto)
        """
        self.device = self._get_device(device)
        self.model = self._load_model(model_path)
        self.transform = self._get_transform()
        
        self.classes = ['photo', 'illustration']
        
    def _get_device(self, device: str = None) -> torch.device:
        """自動で最適なデバイスを選択"""
        if device:
            return torch.device(device)
        if torch.backends.mps.is_available():
            return torch.device('mps')
        elif torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')
    
    def _load_model(self, model_path: str) -> torch.nn.Module:
        """モデルをロード"""
        from start_training_phase import TrainingModel
        
        model = TrainingModel(num_classes=2)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        return model
    
    def _get_transform(self):
        """前処理パイプライン"""
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225]),
        ])
    
    def predict(self, image_path: Union[str, Path]) -> Dict:
        """
        単一画像を分類
        
        Args:
            image_path: 画像ファイルのパス
            
        Returns:
            {
                'label': 'photo' or 'illustration',
                'confidence': float (0.0 ~ 1.0),
                'probabilities': {'photo': float, 'illustration': float}
            }
        """
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = F.softmax(output, dim=1)
            
        prob_photo = probabilities[0][0].item()
        prob_illust = probabilities[0][1].item()
        
        pred_class = probabilities.argmax(dim=1).item()
        confidence = probabilities.max(dim=1)[0].item()
        
        return {
            'label': self.classes[pred_class],
            'confidence': confidence,
            'probabilities': {
                'photo': prob_photo,
                'illustration': prob_illust
            }
        }
    
    def predict_batch(self, image_paths: List[Union[str, Path]]) -> Dict[str, Dict]:
        """
        複数画像をバッチ処理
        
        Args:
            image_paths: 画像ファイルパスのリスト
            
        Returns:
            {image_path: result_dict, ...}
        """
        results = {}
        for path in image_paths:
            try:
                results[str(path)] = self.predict(path)
            except Exception as e:
                results[str(path)] = {'error': str(e)}
        return results


def main():
    parser = argparse.ArgumentParser(description='Photo vs Illustration Classifier')
    parser.add_argument('--image', type=str, help='Path to single image')
    parser.add_argument('--input_dir', type=str, help='Directory containing images')
    parser.add_argument('--model', type=str, default='checkpoints_local/best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default=None,
                       help='Device: mps, cuda, cpu (auto if not specified)')
    parser.add_argument('--output', type=str, help='Output JSON file for batch results')
    
    args = parser.parse_args()
    
    if not args.image and not args.input_dir:
        print("Error: Please specify --image or --input_dir")
        parser.print_help()
        sys.exit(1)
    
    # 分類器を初期化
    print(f"Loading model from {args.model}...")
    classifier = PhotoIllustrationClassifier(args.model, args.device)
    print(f"Using device: {classifier.device}")
    
    if args.image:
        # 単一画像
        result = classifier.predict(args.image)
        print(f"\nImage: {args.image}")
        print(f"Prediction: {result['label'].upper()}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Probabilities:")
        print(f"  Photo:        {result['probabilities']['photo']:.2%}")
        print(f"  Illustration: {result['probabilities']['illustration']:.2%}")
        
    elif args.input_dir:
        # バッチ処理
        import glob
        
        image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp']:
            image_paths.extend(glob.glob(os.path.join(args.input_dir, ext)))
        
        print(f"\nProcessing {len(image_paths)} images...")
        results = classifier.predict_batch(image_paths)
        
        # 統計
        photos = sum(1 for r in results.values() if 'label' in r and r['label'] == 'photo')
        illustrations = sum(1 for r in results.values() if 'label' in r and r['label'] == 'illustration')
        
        print(f"\nResults:")
        print(f"  Photos: {photos}")
        print(f"  Illustrations: {illustrations}")
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nSaved to {args.output}")


if __name__ == '__main__':
    main()
