#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
股票消息情感分析系统演示程序
"""

import sys
import os
import pandas as pd
import logging

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import create_sample_data, analyze_sentiment_distribution, plot_sentiment_distribution
from data_processor import TextProcessor
from model import SentimentModel

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def demo_text_processing():
    """演示文本预处理功能"""
    print("\n" + "="*50)
    print("文本预处理演示")
    print("="*50)
    
    # 创建文本处理器
    processor = TextProcessor()
    
    # 示例文本
    sample_texts = [
        "某公司发布业绩预告，预计净利润同比增长50%，股价有望上涨！",
        "市场分析师看好该股票，给予买入评级，目标价上调至15元。",
        "公司发布公告称业绩不及预期，股价可能承压下跌。",
        "市场震荡调整，投资者观望情绪浓厚，成交量萎缩。"
    ]
    
    print("原始文本:")
    for i, text in enumerate(sample_texts, 1):
        print(f"{i}. {text}")
    
    print("\n预处理后的文本:")
    for i, text in enumerate(sample_texts, 1):
        processed = processor.process_text(text)
        print(f"{i}. {processed}")
    
    print("\n情感特征提取:")
    for i, text in enumerate(sample_texts, 1):
        features = processor.extract_features(text)
        print(f"{i}. 正面词: {features['positive']}, 负面词: {features['negative']}, 中性词: {features['neutral']}")

def demo_sentiment_analysis():
    """演示情感分析功能"""
    print("\n" + "="*50)
    print("情感分析演示")
    print("="*50)
    
    # 创建示例数据
    sample_data = create_sample_data()
    print(f"示例数据数量: {len(sample_data)}")
    
    # 分析情感分布
    distribution = analyze_sentiment_distribution(sample_data)
    print("\n情感分布统计:")
    for sentiment, count in distribution['counts'].items():
        percentage = distribution['percentages'][sentiment]
        print(f"  {sentiment}: {count} 条 ({percentage:.1f}%)")
    
    # 绘制情感分布图
    try:
        plot_sentiment_distribution(sample_data, save_path='demo_sentiment_distribution.png')
        print("\n情感分布图已保存为: demo_sentiment_distribution.png")
    except Exception as e:
        print(f"绘图失败: {e}")

def demo_model_inference():
    """演示模型推理功能"""
    print("\n" + "="*50)
    print("模型推理演示")
    print("="*50)
    
    # 检查是否有训练好的模型
    model_path = 'models/best_model'
    if not os.path.exists(model_path):
        print("未找到训练好的模型，请先运行 train_model.py 训练模型")
        return
    
    try:
        # 初始化模型
        sentiment_model = SentimentModel('hfl/chinese-bert-wwm-ext', 3)
        
        # 加载模型
        if not sentiment_model.load_model(model_path):
            print("模型加载失败")
            return
        
        print("模型加载成功!")
        
        # 测试文本
        test_texts = [
            "公司业绩大幅增长，股价涨停",
            "行业政策收紧，公司面临挑战",
            "市场消息面平静，等待数据公布"
        ]
        
        print("\n测试文本情感分析结果:")
        for i, text in enumerate(test_texts, 1):
            result = sentiment_model.predict(text, return_probs=True)
            if result:
                print(f"{i}. 文本: {text}")
                print(f"   预测情感: {result['predicted_label']}")
                print(f"   置信度: {result['confidence']:.4f}")
                if 'probabilities' in result:
                    print(f"   各类别概率:")
                    for label, prob in result['probabilities'].items():
                        print(f"     {label}: {prob:.4f}")
                print()
        
    except Exception as e:
        print(f"模型推理演示失败: {e}")

def demo_database_operations():
    """演示数据库操作功能"""
    print("\n" + "="*50)
    print("数据库操作演示")
    print("="*50)
    
    try:
        from database import DatabaseManager
        
        # 创建数据库管理器
        db_manager = DatabaseManager()
        
        # 尝试连接数据库
        if db_manager.connect():
            print("数据库连接成功!")
            
            # 获取统计信息
            stats = db_manager.get_statistics()
            if stats:
                print(f"数据库统计信息:")
                print(f"  总数据量: {stats.get('total', 0)}")
                print(f"  已标注: {stats.get('labeled', 0)}")
                print(f"  未标注: {stats.get('unlabeled', 0)}")
                
                if 'sentiment_distribution' in stats:
                    print(f"  情感分布: {stats['sentiment_distribution']}")
            else:
                print("数据库为空或表结构不正确")
            
            # 关闭连接
            db_manager.disconnect()
        else:
            print("数据库连接失败，请检查配置")
            
    except Exception as e:
        print(f"数据库操作演示失败: {e}")
        print("请确保:")
        print("1. 已安装 pymysql 和 sqlalchemy")
        print("2. config.ini 中的数据库配置正确")
        print("3. MySQL服务正在运行")

def main():
    """主演示函数"""
    print("股票消息情感分析系统演示")
    print("="*60)
    
    try:
        # 演示文本预处理
        demo_text_processing()
        
        # 演示情感分析
        demo_sentiment_analysis()
        
        # 演示模型推理
        demo_model_inference()
        
        # 演示数据库操作
        demo_database_operations()
        
        print("\n" + "="*60)
        print("演示完成!")
        print("\n下一步操作:")
        print("1. 配置数据库连接信息 (config.ini)")
        print("2. 运行 train_model.py 训练模型")
        print("3. 运行 predict_batch.py 进行批量预测")
        print("4. 查看 README.md 了解详细使用方法")
        
    except Exception as e:
        logger.error(f"演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
