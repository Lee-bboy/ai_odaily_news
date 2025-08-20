#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
增强版股票消息情感分析模型训练程序
支持金融词汇扩展和领域适应微调
"""

import configparser
import logging
import torch
from transformers import AutoTokenizer, AutoConfig
import os
import sys
import json

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database import DatabaseManager
from data_processor import TextProcessor, DatasetBuilder
from model import SentimentClassifier
from enhanced_trainer import EnhancedSentimentTrainer
from financial_vocab import create_financial_tokenizer, FinancialVocabulary

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_training.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def setup_financial_vocabulary(config):
    """设置金融词汇扩展"""
    vocab_expansion = config.getboolean('FINETUNE', 'vocab_expansion', fallback=False)
    
    if vocab_expansion:
        logger.info("启用金融词汇扩展...")
        
        # 创建词汇管理器
        financial_vocab = FinancialVocabulary()
        
        # 打印词汇表摘要
        financial_vocab.print_vocabulary_summary()
        
        # 导出词汇表
        os.makedirs('vocabulary', exist_ok=True)
        financial_vocab.export_vocabulary('vocabulary/financial_vocabulary.json')
        
        return financial_vocab
    else:
        logger.info("未启用词汇扩展")
        return None

def create_enhanced_tokenizer(model_name: str, config, financial_vocab=None):
    """创建增强的分词器"""
    logger.info(f"加载基础分词器: {model_name}")
    
    # 加载基础分词器
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 应用金融词汇扩展
    if financial_vocab and config.getboolean('FINETUNE', 'vocab_expansion', fallback=False):
        logger.info("应用金融词汇扩展...")
        tokenizer = create_financial_tokenizer(tokenizer, vocab_expansion=True)
        
        # 保存扩展后的分词器
        os.makedirs('vocabulary', exist_ok=True)
        tokenizer.save_pretrained('vocabulary/enhanced_tokenizer')
        logger.info("扩展后的分词器已保存到: vocabulary/enhanced_tokenizer")
    
    return tokenizer

def create_enhanced_model(model_name: str, num_labels: int, tokenizer, config):
    """创建增强的模型"""
    logger.info("创建增强的情感分析模型...")
    
    # 创建模型
    model = SentimentClassifier(model_name, num_labels)
    
    # 如果启用了词汇扩展，调整模型嵌入层大小
    if config.getboolean('FINETUNE', 'vocab_expansion', fallback=False):
        original_vocab_size = model.bert.embeddings.word_embeddings.num_embeddings
        new_vocab_size = len(tokenizer)
        
        if new_vocab_size > original_vocab_size:
            logger.info(f"调整模型嵌入层大小: {original_vocab_size} -> {new_vocab_size}")
            model.bert.resize_token_embeddings(new_vocab_size)
    
    return model

def main():
    """主训练函数"""
    try:
        # 加载配置
        config = configparser.ConfigParser()
        config.read('config.ini', encoding='utf-8')
        
        logger.info("开始增强版股票消息情感分析模型训练...")
        
        # 设置金融词汇扩展
        financial_vocab = setup_financial_vocabulary(config)
        
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
        
        # 创建增强的分词器
        model_name = config['MODEL']['model_name']
        tokenizer = create_enhanced_tokenizer(model_name, config, financial_vocab)
        
        # 创建增强的模型
        model = create_enhanced_model(model_name, num_labels, tokenizer, config)
        model.to(device)
        
        # 创建增强的训练器
        trainer = EnhancedSentimentTrainer(model, tokenizer, device, config)
        
        # 开始训练
        logger.info("开始增强微调训练...")
        training_results = trainer.train(
            train_texts, train_labels,
            val_texts, val_labels,
            save_dir='enhanced_models'
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
        
        # 保存标签映射和训练配置
        save_dir = 'enhanced_models'
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存标签映射
        label_mapping_file = os.path.join(save_dir, 'label_mapping.json')
        with open(label_mapping_file, 'w', encoding='utf-8') as f:
            json.dump(label_mapping, f, ensure_ascii=False, indent=2)
        logger.info(f"标签映射已保存到: {label_mapping_file}")
        
        # 保存训练配置
        config_file = os.path.join(save_dir, 'training_config.json')
        config_dict = {
            'model_name': model_name,
            'num_labels': num_labels,
            'vocab_expansion': config.getboolean('FINETUNE', 'vocab_expansion', fallback=False),
            'domain_adaptation': config.getboolean('FINETUNE', 'domain_adaptation', fallback=False),
            'freeze_bert_layers': config.getint('FINETUNE', 'freeze_bert_layers', fallback=0),
            'learning_rate': config['MODEL']['learning_rate'],
            'batch_size': config['MODEL']['batch_size'],
            'epochs': config['MODEL']['epochs'],
            'max_length': config['MODEL']['max_length']
        }
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, ensure_ascii=False, indent=2)
        logger.info(f"训练配置已保存到: {config_file}")
        
        # 保存词汇扩展信息
        if financial_vocab:
            vocab_info_file = os.path.join(save_dir, 'vocabulary_info.json')
            vocab_info = {
                'total_terms': len(financial_vocab._flatten_terms()),
                'categories': list(financial_vocab.all_terms.keys()),
                'stock_terms_count': sum(len(terms) for terms in financial_vocab.stock_terms.values()),
                'crypto_terms_count': sum(len(terms) for terms in financial_vocab.crypto_terms.values()),
                'financial_indicators_count': sum(len(terms) for terms in financial_vocab.financial_indicators.values()),
                'market_sentiment_count': sum(len(terms) for terms in financial_vocab.market_sentiment.values())
            }
            with open(vocab_info_file, 'w', encoding='utf-8') as f:
                json.dump(vocab_info, f, ensure_ascii=False, indent=2)
            logger.info(f"词汇扩展信息已保存到: {vocab_info_file}")
        
        # 关闭数据库连接
        db_manager.disconnect()
        
        logger.info("增强版训练流程完成!")
        logger.info(f"模型和分词器已保存到: {save_dir}")
        logger.info("现在可以使用增强的模型进行预测了!")
        
    except Exception as e:
        logger.error(f"训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
