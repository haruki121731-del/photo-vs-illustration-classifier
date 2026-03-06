"""
データ収集モジュール
- 写真データ: ImageNetサブセット、Unsplashなどから収集
- 非写真データ: Safebooru、Danbooru、Hugging Faceから収集
"""

import os
import requests
import json
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
from tqdm import tqdm
import hashlib
import time
from urllib.parse import urlparse


@dataclass
class ImageMetadata:
    """画像メタデータ"""
    url: str
    source: str  # 'photo' or 'illustration'
    category: str  # 詳細カテゴリ
    filename: Optional[str] = None
    downloaded: bool = False


class SafebooruCollector:
    """Safebooruからアニメ/イラスト画像を収集"""
    
    BASE_URL = "https://safebooru.donmai.us"
    
    def __init__(self, delay: float = 0.5):
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; DataCollector/1.0)'
        })
    
    def search_posts(self, tags: str = "", limit: int = 100, page: int = 1) -> List[Dict]:
        """投稿を検索"""
        url = f"{self.BASE_URL}/posts.json"
        params = {
            'tags': tags,
            'limit': min(limit, 200),  # API制限
            'page': page
        }
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching posts: {e}")
            return []
    
    def collect_images(self, 
                      output_dir: str,
                      tags_list: List[str] = None,
                      max_images: int = 5000,
                      min_width: int = 224,
                      min_height: int = 224) -> List[ImageMetadata]:
        """
        イラスト画像を収集
        
        Args:
            output_dir: 保存先ディレクトリ
            tags_list: 収集するタグのリスト
            max_images: 最大収集枚数
            min_width: 最小画像幅
            min_height: 最小画像高さ
        """
        if tags_list is None:
            # 多様なイラストを収集するタグ
            tags_list = [
                "landscape",
                "portrait", 
                "1girl",
                "1boy",
                "animal",
                "building",
                "nature",
                "fantasy",
                "sci-fi",
            ]
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        collected = []
        target_per_tag = max_images // len(tags_list)
        
        for tags in tags_list:
            print(f"\nCollecting images with tags: {tags}")
            page = 1
            tag_count = 0
            
            while tag_count < target_per_tag:
                posts = self.search_posts(tags=tags, limit=100, page=page)
                
                if not posts:
                    break
                
                for post in posts:
                    if tag_count >= target_per_tag:
                        break
                    
                    # 画像URLを取得
                    file_url = post.get('file_url') or post.get('large_file_url')
                    if not file_url:
                        continue
                    
                    # サイズチェック
                    width = post.get('image_width', 0)
                    height = post.get('image_height', 0)
                    if width < min_width or height < min_height:
                        continue
                    
                    # メタデータ作成
                    metadata = ImageMetadata(
                        url=file_url,
                        source='illustration',
                        category=f"anime_{tags}",
                        filename=f"illust_{tags}_{post['id']}.jpg"
                    )
                    
                    # ダウンロード
                    filepath = output_path / metadata.filename
                    if self._download_image(file_url, filepath):
                        metadata.downloaded = True
                        collected.append(metadata)
                        tag_count += 1
                    
                    time.sleep(self.delay)
                
                page += 1
                if page > 50:  # ページ上限
                    break
            
            print(f"  Collected {tag_count} images for tag '{tags}'")
        
        return collected
    
    def _download_image(self, url: str, filepath: Path) -> bool:
        """画像をダウンロード"""
        try:
            if filepath.exists():
                return True
            
            response = self.session.get(url, timeout=30, stream=True)
            response.raise_for_status()
            
            filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            return True
        except Exception as e:
            # print(f"Error downloading {url}: {e}")
            return False


class HuggingFaceDatasetCollector:
    """Hugging Faceからデータセットをダウンロード"""
    
    def __init__(self):
        self.session = requests.Session()
    
    def download_anime_faces(self, output_dir: str, max_images: int = 10000) -> List[str]:
        """
        Anime Face Datasetをダウンロード
        ソース: https://www.kaggle.com/datasets/splcher/animefacedataset
        またはHugging Face経由
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Hugging Faceからanime facesを取得
        # 注: 実際の実装ではdatasetsライブラリを使用
        try:
            from datasets import load_dataset
            
            print("Loading anime face dataset from Hugging Face...")
            # anime face datasetの例
            dataset = load_dataset("huggan/anime-faces", split="train", streaming=True)
            
            downloaded = []
            for i, item in enumerate(dataset):
                if i >= max_images:
                    break
                
                # 画像を保存
                image = item['image']
                filepath = output_path / f"anime_face_{i:06d}.jpg"
                image.save(filepath, 'JPEG')
                downloaded.append(str(filepath))
                
                if (i + 1) % 100 == 0:
                    print(f"  Downloaded {i + 1}/{max_images} images")
            
            return downloaded
            
        except ImportError:
            print("Hugging Face datasets library not installed. Install with: pip install datasets")
            return []
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return []


class UnsplashCollector:
    """Unsplash APIから写真を収集（APIキー必要）"""
    
    BASE_URL = "https://api.unsplash.com"
    
    def __init__(self, access_key: Optional[str] = None):
        self.access_key = access_key or os.getenv('UNSPLASH_ACCESS_KEY')
        self.session = requests.Session()
        if self.access_key:
            self.session.headers.update({
                'Authorization': f'Client-ID {self.access_key}'
            })
    
    def search_photos(self, query: str, per_page: int = 30, page: int = 1) -> Dict:
        """写真を検索"""
        if not self.access_key:
            print("Unsplash API key not provided")
            return {}
        
        url = f"{self.BASE_URL}/search/photos"
        params = {
            'query': query,
            'per_page': per_page,
            'page': page,
            'orientation': 'all'
        }
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error searching photos: {e}")
            return {}
    
    def collect_photos(self,
                      output_dir: str,
                      queries: List[str] = None,
                      max_images: int = 5000) -> List[ImageMetadata]:
        """写真を収集"""
        if not self.access_key:
            print("Skipping Unsplash collection (no API key)")
            return []
        
        if queries is None:
            queries = [
                "nature landscape",
                "portrait people",
                "architecture building",
                "animal wildlife",
                "city urban",
                "food",
                "technology",
                "travel",
            ]
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        collected = []
        target_per_query = max_images // len(queries)
        
        for query in queries:
            print(f"\nCollecting photos for: {query}")
            page = 1
            query_count = 0
            
            while query_count < target_per_query:
                data = self.search_photos(query, per_page=30, page=page)
                results = data.get('results', [])
                
                if not results:
                    break
                
                for photo in results:
                    if query_count >= target_per_query:
                        break
                    
                    # 画像URL
                    urls = photo.get('urls', {})
                    image_url = urls.get('regular') or urls.get('small')
                    
                    if not image_url:
                        continue
                    
                    # ファイル名
                    photo_id = photo['id']
                    filename = f"photo_{query.replace(' ', '_')}_{photo_id}.jpg"
                    
                    metadata = ImageMetadata(
                        url=image_url,
                        source='photo',
                        category=query,
                        filename=filename
                    )
                    
                    # ダウンロード
                    filepath = output_path / filename
                    if self._download_image(image_url, filepath):
                        metadata.downloaded = True
                        collected.append(metadata)
                        query_count += 1
                    
                    time.sleep(0.3)  # APIレート制限対策
                
                page += 1
                if page > 50:
                    break
            
            print(f"  Collected {query_count} photos for '{query}'")
        
        return collected
    
    def _download_image(self, url: str, filepath: Path) -> bool:
        """画像をダウンロード"""
        try:
            if filepath.exists():
                return True
            
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            return True
        except Exception as e:
            return False


def create_balanced_dataset(photo_dir: str, 
                           illustration_dir: str,
                           output_dir: str,
                           max_per_class: int = 10000,
                           val_split: float = 0.1,
                           test_split: float = 0.1):
    """
    バランスの取れたデータセットを作成
    
    Args:
        photo_dir: 写真画像のディレクトリ
        illustration_dir: イラスト画像のディレクトリ
        output_dir: 出力ディレクトリ
        max_per_class: クラスあたりの最大画像数
        val_split: 検証セットの割合
        test_split: テストセットの割合
    """
    import shutil
    from sklearn.model_selection import train_test_split
    
    photo_dir = Path(photo_dir)
    illustration_dir = Path(illustration_dir)
    output_dir = Path(output_dir)
    
    # 画像ファイルを収集
    photo_files = list(photo_dir.glob("*.jpg")) + list(photo_dir.glob("*.png"))
    illustration_files = list(illustration_dir.glob("*.jpg")) + list(illustration_dir.glob("*.png"))
    
    # バランスを取る
    photo_files = photo_files[:max_per_class]
    illustration_files = illustration_files[:max_per_class]
    
    print(f"Photo images: {len(photo_files)}")
    print(f"Illustration images: {len(illustration_files)}")
    
    # 分割
    def split_files(files, val_split, test_split):
        train_files, temp_files = train_test_split(files, test_size=val_split + test_split, random_state=42)
        val_files, test_files = train_test_split(temp_files, test_size=test_split / (val_split + test_split), random_state=42)
        return train_files, val_files, test_files
    
    photo_train, photo_val, photo_test = split_files(photo_files, val_split, test_split)
    illust_train, illust_val, illust_test = split_files(illustration_files, val_split, test_split)
    
    # コピー関数
    def copy_files(files, dst_dir, label):
        dst_path = Path(dst_dir) / label
        dst_path.mkdir(parents=True, exist_ok=True)
        for f in tqdm(files, desc=f"Copying {label}"):
            shutil.copy2(f, dst_path / f.name)
    
    # コピー実行
    for split_name, photo_split, illust_split in [
        ('train', photo_train, illust_train),
        ('val', photo_val, illust_val),
        ('test', photo_test, illust_test)
    ]:
        copy_files(photo_split, output_dir / split_name, 'photo')
        copy_files(illust_split, output_dir / split_name, 'illustration')
    
    print(f"\nDataset created at: {output_dir}")
    print(f"  Train: {len(photo_train) + len(illust_train)} images")
    print(f"  Val: {len(photo_val) + len(illust_val)} images")
    print(f"  Test: {len(photo_test) + len(illust_test)} images")


if __name__ == "__main__":
    # テスト用
    print("Data collector module ready!")
    print("\nUsage:")
    print("1. SafebooruCollector: Collect anime illustrations from Safebooru")
    print("2. HuggingFaceDatasetCollector: Download from Hugging Face")
    print("3. UnsplashCollector: Collect photos from Unsplash (API key required)")
