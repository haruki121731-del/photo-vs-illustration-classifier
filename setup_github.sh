#!/bin/bash
# GitHubリポジトリ設定スクリプト

REPO_NAME="photo-vs-illustration-classifier"
GITHUB_USER="harukiuesaka"

echo "=============================================="
echo "GitHub Repository Setup"
echo "=============================================="

# Git初期化
git init

# .gitignore作成
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# 仮想環境
venv/
ENV/
env/
.venv

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# データ・モデル（大きなファイル）
data/
checkpoints/
*.pth
*.pt
*.onnx
*.pkl
logs/
outputs/

# OS
.DS_Store
Thumbs.db

# Jupyter
.ipynb_checkpoints/
*.ipynb

# 環境変数
.env
.env.local
EOF

# リモートリポジトリ追加（PATは環境変数から取得）
if [ -z "$GITHUB_PAT" ]; then
    echo "Error: GITHUB_PAT environment variable not set"
    exit 1
fi

git remote add origin "https://${GITHUB_PAT}@github.com/${GITHUB_USER}/${REPO_NAME}.git" 2>/dev/null || \
git remote set-url origin "https://${GITHUB_PAT}@github.com/${GITHUB_USER}/${REPO_NAME}.git"

# 初回コミット
git add .
git commit -m "Initial commit: Photo vs Illustration Classifier

- Ultra-lightweight CNN models (< 600K params)
- Training pipeline with 99% accuracy target
- Data collection from Safebooru/Unsplash
- Evaluation and export tools"

# リポジトリ作成（API経由）
curl -X POST \
  -H "Authorization: token ${GITHUB_PAT}" \
  -H "Accept: application/vnd.github.v3+json" \
  https://api.github.com/user/repos \
  -d "{\"name\":\"${REPO_NAME}\",\"private\":false,\"description\":\"Ultra-lightweight deep learning model to classify photos vs illustrations (anime/manga/paintings) with 99% accuracy\"}" \
  2>/dev/null | grep -q "created_at" && echo "Repository created successfully!"

# プッシュ
git branch -M main
git push -u origin main --force

echo ""
echo "=============================================="
echo "Repository URL: https://github.com/${GITHUB_USER}/${REPO_NAME}"
echo "=============================================="
