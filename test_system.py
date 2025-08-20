#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
股票消息情感分析系统测试脚本
"""

import sys
import os
import logging

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """测试模块导入"""
    print("测试模块导入...")
    
    try:
        import torch
        print(f"✓ PyTorch: {torch.__version__}")
    except ImportError as e:
        print(f"✗ PyTorch导入失败: {e}")
        return False
    
    try:
        import transformers
        print(f"✓ Transformers: {transformers.__version__}")
    except ImportError as e:
        print(f"✗ Transformers导入失败: {e}")
        return False
    
    try:
        import pandas
        print(f"✓ Pandas: {pandas.__version__}")
    except ImportError as e:
        print(f"✗ Pandas导入失败: {e}")
        return False
    
    try:
        import jieba
        print(f"✓ Jieba: {jieba.__version__}")
    except ImportError as e:
        print(f"✗ Jieba导入失败: {e}")
        return False
    
    try:
        import pymysql
        print(f"✓ PyMySQL: {pymysql.__version__}")
    except ImportError as e:
        print(f"✗ PyMySQL导入失败: {e}")
        return False
    
    try:
        import sqlalchemy
        print(f"✓ SQLAlchemy: {sqlalchemy.__version__}")
    except ImportError as e:
        print(f"✗ SQLAlchemy导入失败: {e}")
        return False
    
    return True

def test_custom_modules():
    """测试自定义模块"""
    print("\n测试自定义模块...")
    
    try:
        from database import DatabaseManager
        print("✓ DatabaseManager导入成功")
    except ImportError as e:
        print(f"✗ DatabaseManager导入失败: {e}")
        return False
    
    try:
        from data_processor import TextProcessor, DatasetBuilder
        print("✓ TextProcessor和DatasetBuilder导入成功")
    except ImportError as e:
        print(f"✗ 数据处理器导入失败: {e}")
        return False
    
    try:
        from model import SentimentClassifier, SentimentModel
        print("✓ 模型模块导入成功")
    except ImportError as e:
        print(f"✗ 模型模块导入失败: {e}")
        return False
    
    try:
        from trainer import SentimentTrainer
        print("✓ 训练器导入成功")
    except ImportError as e:
        print(f"✗ 训练器导入失败: {e}")
        return False
    
    try:
        from utils import create_sample_data, analyze_sentiment_distribution
        print("✓ 工具函数导入成功")
    except ImportError as e:
        print(f"✗ 工具函数导入失败: {e}")
        return False
    
    return True

def test_text_processing():
    """测试文本处理功能"""
    print("\n测试文本处理功能...")
    
    try:
        from data_processor import TextProcessor
        
        processor = TextProcessor()
        
        # 测试文本清洗
        test_text = "<p>某公司业绩增长50%！股价上涨</p>"
        cleaned = processor.clean_text(test_text)
        print(f"✓ 文本清洗: {test_text} -> {cleaned}")
        
        # 测试分词
        segmented = processor.segment_text(cleaned)
        print(f"✓ 分词结果: {segmented}")
        
        # 测试特征提取
        features = processor.extract_features(cleaned)
        print(f"✓ 特征提取: {features}")
        
        return True
        
    except Exception as e:
        print(f"✗ 文本处理测试失败: {e}")
        return False

def test_data_analysis():
    """测试数据分析功能"""
    print("\n测试数据分析功能...")
    
    try:
        from utils import create_sample_data, analyze_sentiment_distribution
        
        # 创建示例数据
        sample_data = create_sample_data()
        print(f"✓ 示例数据创建成功，共{len(sample_data)}条")
        
        # 分析情感分布
        distribution = analyze_sentiment_distribution(sample_data)
        print(f"✓ 情感分布分析成功: {distribution['counts']}")
        
        return True
        
    except Exception as e:
        print(f"✗ 数据分析测试失败: {e}")
        return False

def test_model_creation():
    """测试模型创建"""
    print("\n测试模型创建...")
    
    try:
        from model import SentimentModel
        
        # 创建模型实例
        model = SentimentModel('hfl/chinese-bert-wwm-ext', 3)
        print("✓ 模型实例创建成功")
        
        # 获取模型信息
        info = model.get_model_info()
        print(f"✓ 模型信息: {info}")
        
        return True
        
    except Exception as e:
        print(f"✗ 模型创建测试失败: {e}")
        return False

def test_configuration():
    """测试配置文件"""
    print("\n测试配置文件...")
    
    try:
        import configparser
        
        if os.path.exists('config.ini'):
            config = configparser.ConfigParser()
            config.read('config.ini', encoding='utf-8')
            
            # 检查必要的配置节
            required_sections = ['DATABASE', 'MODEL', 'DATA']
            for section in required_sections:
                if section in config:
                    print(f"✓ 配置节 {section} 存在")
                else:
                    print(f"✗ 配置节 {section} 缺失")
                    return False
            
            print("✓ 配置文件格式正确")
            return True
        else:
            print("✗ 配置文件不存在")
            return False
            
    except Exception as e:
        print(f"✗ 配置文件测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("股票消息情感分析系统 - 系统测试")
    print("="*50)
    
    tests = [
        ("模块导入测试", test_imports),
        ("自定义模块测试", test_custom_modules),
        ("文本处理测试", test_text_processing),
        ("数据分析测试", test_data_analysis),
        ("模型创建测试", test_model_creation),
        ("配置文件测试", test_configuration)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            if test_func():
                passed += 1
                print(f"✓ {test_name} 通过")
            else:
                print(f"✗ {test_name} 失败")
        except Exception as e:
            print(f"✗ {test_name} 异常: {e}")
    
    print("\n" + "="*50)
    print(f"测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！系统可以正常使用")
        print("\n下一步:")
        print("1. 配置数据库连接")
        print("2. 运行 python demo.py 查看演示")
        print("3. 运行 python train_model.py 开始训练")
    else:
        print("⚠️  部分测试失败，请检查错误信息")
        print("\n常见问题:")
        print("1. 依赖包未正确安装 - 运行 python setup.py")
        print("2. 配置文件缺失 - 检查 config.ini")
        print("3. 模块导入错误 - 检查文件路径")

if __name__ == "__main__":
    main()
