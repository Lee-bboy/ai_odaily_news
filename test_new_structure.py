#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试新表结构的配置和功能
"""

import sys
import os
import logging

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database import DatabaseManager
from data_processor import TextProcessor, DatasetBuilder

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_database_connection():
    """测试数据库连接"""
    print("=" * 50)
    print("测试数据库连接")
    print("=" * 50)
    
    try:
        db_manager = DatabaseManager()
        
        if db_manager.connect():
            print("✓ 数据库连接成功")
            
            # 测试获取统计信息
            stats = db_manager.get_statistics()
            if stats:
                print(f"✓ 数据库统计信息获取成功:")
                print(f"  总数据量: {stats.get('total', 0)}")
                print(f"  已标注: {stats.get('labeled', 0)}")
                print(f"  未标注: {stats.get('unlabeled', 0)}")
                
                if 'sentiment_distribution' in stats:
                    print(f"  情感分布: {stats['sentiment_distribution']}")
                
                if 'confidence_stats' in stats:
                    conf_stats = stats['confidence_stats']
                    print(f"  置信度统计: 平均={conf_stats['average']:.2f}, 最小={conf_stats['min']:.2f}, 最大={conf_stats['max']:.2f}")
            else:
                print("⚠️  数据库为空或表结构不正确")
            
            db_manager.disconnect()
            return True
        else:
            print("✗ 数据库连接失败")
            return False
            
    except Exception as e:
        print(f"✗ 数据库连接测试失败: {e}")
        return False

def test_data_processing():
    """测试数据处理功能"""
    print("\n" + "=" * 50)
    print("测试数据处理功能")
    print("=" * 50)
    
    try:
        # 创建示例数据
        import pandas as pd
        
        sample_data = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'title': [
                'BTC突破50000美元大关',
                '某公司业绩不及预期',
                '市场震荡调整',
                '科技股表现强劲',
                '行业政策收紧'
            ],
            'description': [
                '比特币价格突破50000美元重要关口，市场情绪乐观',
                '公司发布财报显示业绩下滑，股价可能承压',
                '今日市场出现震荡调整，投资者观望情绪浓厚',
                '科技板块今日表现强劲，多只股票涨停',
                '监管部门发布新政策，行业面临挑战'
            ],
            'sentiment': ['positive', 'negative', 'neutral', 'positive', 'negative']
        })
        
        print("✓ 示例数据创建成功")
        print(f"  数据条数: {len(sample_data)}")
        
        # 测试文本处理器
        text_processor = TextProcessor()
        
        print("\n测试标题和描述组合:")
        for i, row in sample_data.iterrows():
            title = row['title']
            description = row['description']
            combined = text_processor.combine_title_description(title, description)
            print(f"  {i+1}. 标题: {title}")
            print(f"     描述: {description}")
            print(f"     组合: {combined}")
            print()
        
        # 测试数据集构建
        dataset_builder = DatasetBuilder(text_processor)
        texts, labels = dataset_builder.prepare_dataset(
            sample_data, 'title', 'description', 'sentiment'
        )
        
        print(f"✓ 数据集构建成功:")
        print(f"  处理后的文本数量: {len(texts)}")
        print(f"  标签数量: {len(labels)}")
        
        # 测试数据集分割
        train_data, val_data, test_data = dataset_builder.split_dataset(texts, labels)
        train_texts, train_labels = train_data
        val_texts, val_labels = val_data
        test_texts, test_labels = test_data
        
        print(f"  训练集: {len(train_texts)} 样本")
        print(f"  验证集: {len(val_texts)} 样本")
        print(f"  测试集: {len(test_texts)} 样本")
        
        return True
        
    except Exception as e:
        print(f"✗ 数据处理测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_configuration():
    """测试配置文件"""
    print("\n" + "=" * 50)
    print("测试配置文件")
    print("=" * 50)
    
    try:
        import configparser
        
        if os.path.exists('config.ini'):
            config = configparser.ConfigParser()
            config.read('config.ini', encoding='utf-8')
            
            print("✓ 配置文件加载成功")
            
            # 检查数据库配置
            if 'DATABASE' in config:
                print("✓ 数据库配置节存在")
                db_config = config['DATABASE']
                print(f"  主机: {db_config.get('host', 'N/A')}")
                print(f"  端口: {db_config.get('port', 'N/A')}")
                print(f"  数据库: {db_config.get('database', 'N/A')}")
            else:
                print("✗ 数据库配置节缺失")
            
            # 检查数据字段配置
            if 'DATA' in config:
                print("✓ 数据字段配置节存在")
                data_config = config['DATA']
                print(f"  表名: {data_config.get('table_name', 'N/A')}")
                print(f"  标题字段: {data_config.get('title_column', 'N/A')}")
                print(f"  描述字段: {data_config.get('description_column', 'N/A')}")
                print(f"  标签字段: {data_config.get('label_column', 'N/A')}")
                print(f"  置信度字段: {data_config.get('confidence_column', 'N/A')}")
            else:
                print("✗ 数据字段配置节缺失")
            
            # 检查模型配置
            if 'MODEL' in config:
                print("✓ 模型配置节存在")
                model_config = config['MODEL']
                print(f"  模型名称: {model_config.get('model_name', 'N/A')}")
                print(f"  最大长度: {model_config.get('max_length', 'N/A')}")
                print(f"  批次大小: {model_config.get('batch_size', 'N/A')}")
            else:
                print("✗ 模型配置节缺失")
            
            # 检查微调配置
            if 'FINETUNE' in config:
                print("✓ 微调配置节存在")
                finetune_config = config['FINETUNE']
                print(f"  领域适应: {finetune_config.get('domain_adaptation', 'N/A')}")
                print(f"  词汇扩展: {finetune_config.get('vocab_expansion', 'N/A')}")
            else:
                print("⚠️  微调配置节缺失（可选）")
            
            return True
        else:
            print("✗ 配置文件不存在")
            return False
            
    except Exception as e:
        print(f"✗ 配置文件测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("新表结构配置测试")
    print("=" * 60)
    
    tests = [
        ("配置文件测试", test_configuration),
        ("数据库连接测试", test_database_connection),
        ("数据处理测试", test_data_processing)
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
    
    print("\n" + "=" * 60)
    print(f"测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！新表结构配置正确")
        print("\n下一步操作:")
        print("1. 确保数据库中有 src_odaily_news_info 表")
        print("2. 运行 python train_model.py 开始训练")
        print("3. 运行 python predict_batch.py 进行批量预测")
    else:
        print("⚠️  部分测试失败，请检查配置")
        print("\n常见问题:")
        print("1. 配置文件 config.ini 是否正确")
        print("2. 数据库连接信息是否正确")
        print("3. 数据库表结构是否存在")

if __name__ == "__main__":
    main()
