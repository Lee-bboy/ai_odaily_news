#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import configparser
import logging
import torch
from transformers import AutoTokenizer
import os
import sys
import json
from tqdm import tqdm

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database import DatabaseManager
from data_processor import TextProcessor
from model import SentimentModel

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('prediction.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_label_mapping(mapping_file: str) -> dict:
    """加载标签映射"""
    try:
        with open(mapping_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(f"标签映射文件 {mapping_file} 不存在，使用默认映射")
        return {0: 'negative', 1: 'neutral', 2: 'positive'}
    except Exception as e:
        logger.error(f"加载标签映射失败: {e}")
        return {0: 'negative', 1: 'neutral', 2: 'positive'}

def main():
    """主预测函数"""
    try:
        # 加载配置
        config = configparser.ConfigParser()
        config.read('config.ini', encoding='utf-8')
        
        logger.info("开始批量预测和标注...")
        
        # 连接数据库
        db_manager = DatabaseManager()
        if not db_manager.connect():
            logger.error("数据库连接失败，退出预测")
            return
        
        # 获取数据库统计信息
        stats = db_manager.get_statistics()
        logger.info(f"数据库统计信息: {stats}")
        
        # 获取未标注数据
        logger.info("获取未标注数据...")
        unlabeled_data = db_manager.get_unlabeled_data()
        
        if len(unlabeled_data) == 0:
            logger.info("没有未标注的数据，所有数据都已标注完成")
            return
        
        logger.info(f"获取到 {len(unlabeled_data)} 条未标注数据")
        
        # 加载标签映射
        label_mapping_file = 'models/label_mapping.json'
        label_mapping = load_label_mapping(label_mapping_file)
        num_labels = len(label_mapping)
        logger.info(f"标签映射: {label_mapping}")
        
        # 设置设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {device}")
        
        # 初始化模型
        model_name = config['MODEL']['model_name']
        sentiment_model = SentimentModel(model_name, num_labels, device)
        
        # 加载训练好的模型
        model_path = 'models/best_model'  # 优先使用最佳模型
        if not os.path.exists(model_path):
            model_path = 'models/final_model'  # 如果没有最佳模型，使用最终模型
        
        if not os.path.exists(model_path):
            logger.error(f"模型路径不存在: {model_path}")
            logger.error("请先运行 train_model.py 训练模型")
            return
        
        logger.info(f"加载模型: {model_path}")
        if not sentiment_model.load_model(model_path):
            logger.error("模型加载失败")
            return
        
        # 设置标签映射
        sentiment_model.label_mapping = label_mapping
        
        # 文本预处理
        text_processor = TextProcessor()
        
        # 批量预测
        logger.info("开始批量预测...")
        batch_size = int(config['MODEL']['batch_size'])
        
        # 准备文本数据 - 组合标题和描述
        texts = []
        ids = []
        
        for _, row in unlabeled_data.iterrows():
            title = row[config['DATA']['title_column']]
            description = row[config['DATA']['description_column']]
            
            if pd.isna(title) and pd.isna(description):
                continue
            
            # 组合标题和描述
            combined_text = text_processor.combine_title_description(title, description)
            if combined_text:
                texts.append(combined_text)
                ids.append(row[config['DATA']['id_column']])
        
        if len(texts) == 0:
            logger.warning("没有有效的文本数据")
            return
        
        logger.info(f"开始预测 {len(texts)} 条文本...")
        
        # 批量预测
        predictions = sentiment_model.batch_predict(texts, batch_size)
        
        # 更新数据库
        logger.info("更新数据库标注结果...")
        updates = []
        
        for i, (prediction, id_value) in enumerate(zip(predictions, ids)):
            sentiment = prediction['predicted_label']
            confidence = prediction['confidence']
            
            updates.append({
                'id': id_value,
                'sentiment': sentiment,
                'confidence': confidence
            })
            
            # 每100条记录一次进度
            if (i + 1) % 100 == 0:
                logger.info(f"已处理 {i + 1}/{len(predictions)} 条记录")
        
        # 批量更新数据库
        if updates:
            db_manager.batch_update_sentiment(updates)
            logger.info(f"成功更新 {len(updates)} 条记录的标注结果")
        
        # 输出预测统计
        sentiment_counts = {}
        confidence_sum = 0
        for update in updates:
            sentiment = update['sentiment']
            confidence = update['confidence']
            
            sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
            confidence_sum += confidence
        
        logger.info("预测结果统计:")
        for sentiment, count in sentiment_counts.items():
            logger.info(f"  {sentiment}: {count} 条")
        
        if len(updates) > 0:
            avg_confidence = confidence_sum / len(updates)
            logger.info(f"  平均置信度: {avg_confidence:.4f}")
        
        # 获取更新后的统计信息
        updated_stats = db_manager.get_statistics()
        logger.info(f"更新后的数据库统计: {updated_stats}")
        
        # 关闭数据库连接
        db_manager.disconnect()
        
        logger.info("批量预测和标注完成!")
        
    except Exception as e:
        logger.error(f"预测过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
