#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
依赖包安装脚本
自动安装股票消息情感分析系统所需的所有Python包
"""

import subprocess
import sys
import os
import platform

def install_package(package_name, version=None):
    """安装单个包"""
    try:
        if version:
            package_spec = f"{package_name}{version}"
        else:
            package_spec = package_name
            
        print(f"正在安装 {package_spec}...")
        
        # 使用pip安装
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", package_spec
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✓ {package_spec} 安装成功")
            return True
        else:
            print(f"✗ {package_spec} 安装失败: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"✗ 安装 {package_spec} 时发生错误: {e}")
        return False

def install_from_requirements():
    """从requirements.txt安装依赖"""
    print("从requirements.txt安装依赖...")
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ 所有依赖安装成功")
            return True
        else:
            print(f"✗ 依赖安装失败: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"✗ 安装依赖时发生错误: {e}")
        return False

def check_python_version():
    """检查Python版本"""
    print("检查Python版本...")
    
    version = sys.version_info
    print(f"当前Python版本: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print("⚠️  警告: 建议使用Python 3.7或更高版本")
        return False
    else:
        print("✓ Python版本符合要求")
        return True

def check_pip():
    """检查pip是否可用"""
    print("检查pip...")
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "pip", "--version"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✓ pip可用: {result.stdout.strip()}")
            return True
        else:
            print("✗ pip不可用")
            return False
            
    except Exception as e:
        print(f"✗ 检查pip时发生错误: {e}")
        return False

def upgrade_pip():
    """升级pip到最新版本"""
    print("升级pip...")
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "--upgrade", "pip"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ pip升级成功")
            return True
        else:
            print(f"⚠️  pip升级失败: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"⚠️  升级pip时发生错误: {e}")
        return False

def install_core_dependencies():
    """安装核心依赖包"""
    print("\n安装核心依赖包...")
    
    core_packages = [
        "torch",
        "transformers", 
        "pandas",
        "numpy",
        "scikit-learn",
        "jieba",
        "pymysql",
        "sqlalchemy",
        "tqdm",
        "matplotlib",
        "seaborn"
    ]
    
    success_count = 0
    total_count = len(core_packages)
    
    for package in core_packages:
        if install_package(package):
            success_count += 1
    
    print(f"\n核心依赖安装结果: {success_count}/{total_count} 成功")
    return success_count == total_count

def install_pytorch_with_cuda():
    """安装支持CUDA的PyTorch（如果可用）"""
    print("\n检查CUDA支持...")
    
    try:
        # 检查是否有NVIDIA GPU
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ 检测到NVIDIA GPU，安装CUDA版本的PyTorch")
            
            # 安装CUDA版本的PyTorch
            cuda_result = subprocess.run([
                sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio", 
                "--index-url", "https://download.pytorch.org/whl/cu118"
            ], capture_output=True, text=True)
            
            if cuda_result.returncode == 0:
                print("✓ CUDA版本PyTorch安装成功")
                return True
            else:
                print(f"⚠️  CUDA版本PyTorch安装失败，将安装CPU版本")
                return False
        else:
            print("未检测到NVIDIA GPU，安装CPU版本的PyTorch")
            return False
            
    except Exception as e:
        print(f"检查CUDA时发生错误: {e}")
        print("将安装CPU版本的PyTorch")
        return False

def verify_installation():
    """验证安装结果"""
    print("\n验证安装结果...")
    
    packages_to_check = [
        "torch", "transformers", "pandas", "numpy", 
        "sklearn", "jieba", "pymysql", "sqlalchemy"
    ]
    
    success_count = 0
    total_count = len(packages_to_check)
    
    for package in packages_to_check:
        try:
            if package == "sklearn":
                import sklearn
                version = sklearn.__version__
            else:
                module = __import__(package)
                version = getattr(module, '__version__', 'unknown')
            
            print(f"✓ {package}: {version}")
            success_count += 1
            
        except ImportError:
            print(f"✗ {package}: 未安装")
        except Exception as e:
            print(f"⚠️  {package}: 检查版本时发生错误: {e}")
            success_count += 1  # 如果能导入，认为安装成功
    
    print(f"\n验证结果: {success_count}/{total_count} 包可用")
    return success_count == total_count

def main():
    """主安装函数"""
    print("股票消息情感分析系统 - 依赖安装程序")
    print("=" * 50)
    
    # 检查Python版本
    if not check_python_version():
        print("建议升级Python版本后重试")
    
    # 检查pip
    if not check_pip():
        print("pip不可用，请先安装pip")
        return
    
    # 升级pip
    upgrade_pip()
    
    # 尝试从requirements.txt安装
    print("\n尝试从requirements.txt安装所有依赖...")
    if install_from_requirements():
        print("✓ 所有依赖安装完成")
    else:
        print("⚠️  从requirements.txt安装失败，尝试逐个安装...")
        
        # 逐个安装核心依赖
        if not install_core_dependencies():
            print("⚠️  部分依赖安装失败")
    
    # 验证安装结果
    if verify_installation():
        print("\n🎉 依赖安装完成！现在可以运行训练程序了")
        print("\n下一步操作:")
        print("1. 运行 python test_new_structure.py 测试配置")
        print("2. 运行 python enhanced_train_model.py 开始训练")
    else:
        print("\n⚠️  部分依赖安装失败，请检查错误信息")
        print("\n手动安装建议:")
        print("pip install torch transformers pandas numpy scikit-learn jieba pymysql sqlalchemy tqdm matplotlib seaborn")

if __name__ == "__main__":
    main()
