#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import configparser
import logging
import torch
from transformers import AutoTokenizer
import os
import sys

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database import DatabaseManager
from data_processor import TextProcessor, DatasetBuilder
from model import SentimentClassifier
from trainer import SentimentTrainer

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """主训练函数"""
    try:
        # 加载配置
        config = configparser.ConfigParser()
        config.read('config.ini', encoding='utf-8')
        
        logger.info("开始股票消息情感分析模型训练...")
        
        # 连接数据库
        db_manager = DatabaseManager()
        if not db_manager.connect():
            logger.error("数据库连接失败，退出训练")
            return
        
        # 获取已标注的训练数据
        logger.info("获取训练数据...")
        labeled_data = db_manager.get_labeled_data()
        
        if len(labeled_data) == 0:
            logger.error("没有找到已标注的训练数据，请先标注一些数据")
            return
        
        logger.info(f"获取到 {len(labeled_data)} 条已标注数据")
        
        # 数据预处理
        logger.info("开始数据预处理...")
        text_processor = TextProcessor()
        dataset_builder = DatasetBuilder(text_processor)
        
        # 准备数据集 - 使用标题和描述字段
        texts, labels = dataset_builder.prepare_dataset(
            labeled_data, 
            config['DATA']['title_column'], 
            config['DATA']['description_column'],
            config['DATA']['label_column']
        )
        
        if len(texts) == 0:
            logger.error("数据预处理失败，没有有效的训练样本")
            return
        
        logger.info(f"预处理完成，有效样本数: {len(texts)}")
        
        # 分割数据集
        train_data, val_data, test_data = dataset_builder.split_dataset(
            texts, labels,
            test_size=0.2,
            val_size=0.1,
            random_state=int(config['MODEL']['random_seed'])
        )
        
        train_texts, train_labels = train_data
        val_texts, val_labels = val_data
        test_texts, test_labels = test_data
        
        logger.info(f"数据集分割完成:")
        logger.info(f"  训练集: {len(train_texts)} 样本")
        logger.info(f"  验证集: {len(val_texts)} 样本")
        logger.info(f"  测试集: {len(test_texts)} 样本")
        
        # 获取标签映射
        label_mapping = dataset_builder.get_label_mapping()
        num_labels = len(label_mapping)
        logger.info(f"标签映射: {label_mapping}")
        
        # 设置设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {device}")
        
        # 加载分词器
        model_name = config['MODEL']['model_name']
        logger.info(f"加载分词器: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # 初始化模型
        logger.info("初始化模型...")
        model = SentimentClassifier(model_name, num_labels)
        model.to(device)
        
        # 创建训练器
        trainer = SentimentTrainer(model, tokenizer, device, config)
        
        # 开始训练
        logger.info("开始训练...")
        training_results = trainer.train(
            train_texts, train_labels,
            val_texts, val_labels,
            save_dir='models'
        )
        
        # 输出训练结果
        logger.info("训练完成!")
        logger.info(f"最佳验证准确率: {training_results['best_val_accuracy']:.4f}")
        logger.info(f"最终验证准确率: {training_results['final_val_accuracy']:.4f}")
        logger.info(f"最终F1分数: {training_results['final_metrics']['f1_macro']:.4f}")
        
        # 在测试集上评估
        logger.info("在测试集上评估模型...")
        test_loader = trainer.create_data_loader(test_texts, test_labels)
        test_loss, test_acc, test_metrics = trainer.evaluate(test_loader)
        
        logger.info(f"测试集结果:")
        logger.info(f"  Loss: {test_loss:.4f}")
        logger.info(f"  Accuracy: {test_acc:.4f}")
        logger.info(f"  F1 (Macro): {test_metrics['f1_macro']:.4f}")
        
        # 保存标签映射
        import json
        label_mapping_file = 'models/label_mapping.json'
        os.makedirs('models', exist_ok=True)
        with open(label_mapping_file, 'w', encoding='utf-8') as f:
            json.dump(label_mapping, f, ensure_ascii=False, indent=2)
        logger.info(f"标签映射已保存到: {label_mapping_file}")
        
        # 关闭数据库连接
        db_manager.disconnect()
        
        logger.info("训练流程完成!")
        
    except Exception as e:
        logger.error(f"训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
