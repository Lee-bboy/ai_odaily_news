#!/bin/bash

echo "股票消息情感分析系统 - 快速安装脚本"
echo "=========================================="

# 检查Python版本
echo "检查Python版本..."
python3 --version

# 检查pip
echo "检查pip..."
python3 -m pip --version

# 升级pip
echo "升级pip..."
python3 -m pip install --upgrade pip

# 安装核心依赖
echo "安装核心依赖包..."

# 安装PyTorch (CPU版本，适用于macOS)
echo "安装PyTorch..."
python3 -m pip install torch torchvision torchaudio

# 安装其他依赖
echo "安装transformers..."
python3 -m pip install transformers

echo "安装数据处理包..."
python3 -m pip install pandas numpy scikit-learn

echo "安装中文分词..."
python3 -m pip install jieba

echo "安装数据库连接..."
python3 -m pip install pymysql sqlalchemy

echo "安装其他工具..."
python3 -m pip install tqdm matplotlib seaborn configparser

# 验证安装
echo "验证安装结果..."
python3 -c "
try:
    import torch
    print(f'✓ PyTorch: {torch.__version__}')
except ImportError:
    print('✗ PyTorch: 未安装')

try:
    import transformers
    print(f'✓ Transformers: {transformers.__version__}')
except ImportError:
    print('✗ Transformers: 未安装')

try:
    import pandas
    print(f'✓ Pandas: {pandas.__version__}')
except ImportError:
    print('✗ Pandas: 未安装')

try:
    import jieba
    print(f'✓ Jieba: {jieba.__version__}')
except ImportError:
    print('✗ Jieba: 未安装')

try:
    import pymysql
    print(f'✓ PyMySQL: {pymysql.__version__}')
except ImportError:
    print('✗ PyMySQL: 未安装')
"

echo ""
echo "安装完成！"
echo "现在可以运行: python3 enhanced_train_model.py"
