#!/bin/bash

echo "股票消息情感分析系统 - macOS安装脚本"
echo "======================================"

# 检查操作系统
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "此脚本仅适用于macOS系统"
    exit 1
fi

echo "检测到macOS系统"

# 检查Python版本
echo "检查Python版本..."
python3 --version

# 检查pip
echo "检查pip..."
python3 -m pip --version

# 升级pip
echo "升级pip..."
python3 -m pip install --upgrade pip

# 安装依赖包
echo "开始安装依赖包..."

echo "1. 安装PyTorch (CPU版本)..."
python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

echo "2. 安装Transformers..."
python3 -m pip install transformers

echo "3. 安装数据处理包..."
python3 -m pip install pandas numpy scikit-learn

echo "4. 安装中文分词..."
python3 -m pip install jieba

echo "5. 安装数据库连接..."
python3 -m pip install pymysql sqlalchemy

echo "6. 安装其他工具..."
python3 -m pip install tqdm matplotlib seaborn configparser

# 验证安装
echo ""
echo "验证安装结果..."
echo "=================="

python3 -c "
import sys
print(f'Python版本: {sys.version}')

packages = [
    ('torch', 'PyTorch'),
    ('transformers', 'Transformers'),
    ('pandas', 'Pandas'),
    ('numpy', 'NumPy'),
    ('sklearn', 'Scikit-learn'),
    ('jieba', 'Jieba'),
    ('pymysql', 'PyMySQL'),
    ('sqlalchemy', 'SQLAlchemy')
]

for module, name in packages:
    try:
        if module == 'sklearn':
            import sklearn
            version = sklearn.__version__
        else:
            mod = __import__(module)
            version = getattr(mod, '__version__', 'unknown')
        print(f'✓ {name}: {version}')
    except ImportError:
        print(f'✗ {name}: 未安装')
    except Exception as e:
        print(f'⚠️  {name}: 检查版本时发生错误')
"

echo ""
echo "安装完成！"
echo "=========="
echo "现在可以运行以下命令："
echo "1. 测试配置: python3 test_new_structure.py"
echo "2. 开始训练: python3 enhanced_train_model.py"
echo "3. 批量预测: python3 predict_batch.py"
