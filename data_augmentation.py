#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据增强模块
用于解决训练数据样本数量不足的问题
"""

import random
import jieba
import pandas as pd
from typing import List, Tuple, Dict
import logging

logger = logging.getLogger(__name__)

class DataAugmentor:
    """数据增强器"""
    
    def __init__(self):
        # 同义词词典（金融领域）
        self.synonyms = {
            '上涨': ['上涨', '上升', '走高', '攀升', '上扬', '走高'],
            '下跌': ['下跌', '下降', '走低', '下挫', '回落', '走低'],
            '利好': ['利好', '好消息', '积极消息', '正面消息', '好消息'],
            '利空': ['利空', '坏消息', '负面消息', '不利消息', '坏消息'],
            '增长': ['增长', '增加', '上升', '提高', '提升', '增加'],
            '减少': ['减少', '下降', '降低', '减少', '下降', '降低'],
            '政策': ['政策', '措施', '办法', '举措', '政策'],
            '市场': ['市场', '股市', '资本市场', '金融市场', '市场'],
            '经济': ['经济', '经济形势', '经济状况', '经济环境', '经济'],
            '投资': ['投资', '投资机会', '投资选择', '投资决策', '投资']
        }
        
        # 金融相关词汇
        self.financial_terms = [
            '股票', '债券', '基金', '期货', '期权', '外汇',
            '指数', '板块', '行业', '公司', '企业', '机构',
            '投资者', '分析师', '专家', '研究员', '基金经理',
            '央行', '证监会', '银保监会', '财政部', '发改委'
        ]
    
    def augment_text(self, text: str, method: str = 'synonym_replacement') -> str:
        """增强单个文本"""
        if method == 'synonym_replacement':
            return self._synonym_replacement(text)
        elif method == 'word_insertion':
            return self._word_insertion(text)
        elif method == 'word_deletion':
            return self._word_deletion(text)
        elif method == 'sentence_restructure':
            return self._sentence_restructure(text)
        else:
            return text
    
    def _synonym_replacement(self, text: str) -> str:
        """同义词替换"""
        words = list(jieba.cut(text))
        augmented_words = []
        
        for word in words:
            if word in self.synonyms:
                # 随机选择同义词
                synonym = random.choice(self.synonyms[word])
                augmented_words.append(synonym)
            else:
                augmented_words.append(word)
        
        return ''.join(augmented_words)
    
    def _word_insertion(self, text: str) -> str:
        """词汇插入"""
        words = list(jieba.cut(text))
        
        if len(words) < 3:
            return text
        
        # 随机插入金融相关词汇
        insert_word = random.choice(self.financial_terms)
        insert_pos = random.randint(0, len(words))
        
        words.insert(insert_pos, insert_word)
        return ''.join(words)
    
    def _word_deletion(self, text: str) -> str:
        """词汇删除"""
        words = list(jieba.cut(text))
        
        if len(words) < 4:
            return text
        
        # 随机删除一个词
        delete_pos = random.randint(0, len(words) - 1)
        words.pop(delete_pos)
        
        return ''.join(words)
    
    def _sentence_restructure(self, text: str) -> str:
        """句子重构"""
        # 简单的句子重构：添加修饰词
        modifiers = ['据悉', '据了解', '据报道', '有消息称', '业内人士表示']
        
        if random.random() < 0.5:
            modifier = random.choice(modifiers)
            return f"{modifier}，{text}"
        else:
            return text
    
    def augment_dataset(self, df: pd.DataFrame, title_col: str, description_col: str, 
                       label_col: str, target_samples_per_class: int = 10) -> pd.DataFrame:
        """增强整个数据集"""
        logger.info(f"开始数据增强，目标每类样本数: {target_samples_per_class}")
        
        # 统计每个标签的样本数量
        label_counts = df[label_col].value_counts()
        logger.info(f"原始数据标签分布:\n{label_counts}")
        
        augmented_data = []
        
        for label in df[label_col].unique():
            current_count = label_counts[label]
            needed_samples = max(0, target_samples_per_class - current_count)
            
            logger.info(f"标签 '{label}': 当前 {current_count} 样本，需要增加 {needed_samples} 样本")
            
            if needed_samples > 0:
                # 获取该类别的原始样本
                class_samples = df[df[label_col] == label]
                
                # 生成增强样本
                for i in range(needed_samples):
                    # 随机选择一个原始样本
                    original_sample = class_samples.sample(n=1).iloc[0]
                    
                    # 随机选择增强方法
                    methods = ['synonym_replacement', 'word_insertion', 'word_deletion', 'sentence_restructure']
                    method = random.choice(methods)
                    
                    # 增强标题和描述
                    augmented_title = self.augment_text(original_sample[title_col], method)
                    augmented_description = self.augment_text(original_sample[description_col], method)
                    
                    # 创建增强样本
                    augmented_sample = original_sample.copy()
                    augmented_sample[title_col] = augmented_title
                    augmented_sample[description_col] = augmented_description
                    
                    # 添加标识
                    augmented_sample['is_augmented'] = True
                    augmented_sample['augmentation_method'] = method
                    
                    augmented_data.append(augmented_sample)
        
        # 合并原始数据和增强数据
        if augmented_data:
            augmented_df = pd.DataFrame(augmented_data)
            result_df = pd.concat([df, augmented_df], ignore_index=True)
            
            logger.info(f"数据增强完成，新增 {len(augmented_data)} 个样本")
            logger.info(f"最终数据标签分布:\n{result_df[label_col].value_counts()}")
            
            return result_df
        else:
            logger.info("无需数据增强")
            return df
    
    def create_balanced_dataset(self, df: pd.DataFrame, title_col: str, description_col: str, 
                               label_col: str, samples_per_class: int = 20) -> pd.DataFrame:
        """创建平衡的数据集"""
        logger.info(f"创建平衡数据集，每类 {samples_per_class} 个样本")
        
        balanced_data = []
        
        for label in df[label_col].unique():
            class_samples = df[df[label_col] == label]
            
            if len(class_samples) >= samples_per_class:
                # 如果样本足够，随机选择
                selected_samples = class_samples.sample(n=samples_per_class)
            else:
                # 如果样本不足，先使用所有样本，再通过增强补充
                selected_samples = class_samples.copy()
                needed_samples = samples_per_class - len(class_samples)
                
                # 增强样本
                for i in range(needed_samples):
                    original_sample = class_samples.sample(n=1).iloc[0]
                    method = random.choice(['synonym_replacement', 'word_insertion', 'sentence_restructure'])
                    
                    augmented_title = self.augment_text(original_sample[title_col], method)
                    augmented_description = self.augment_text(original_sample[description_col], method)
                    
                    augmented_sample = original_sample.copy()
                    augmented_sample[title_col] = augmented_title
                    augmented_sample[description_col] = augmented_description
                    augmented_sample['is_augmented'] = True
                    augmented_sample['augmentation_method'] = method
                    
                    selected_samples = pd.concat([selected_samples, pd.DataFrame([augmented_sample])], ignore_index=True)
            
            balanced_data.append(selected_samples)
        
        result_df = pd.concat(balanced_data, ignore_index=True)
        
        logger.info(f"平衡数据集创建完成，总样本数: {len(result_df)}")
        logger.info(f"标签分布:\n{result_df[label_col].value_counts()}")
        
        return result_df

def demo_data_augmentation():
    """演示数据增强功能"""
    print("数据增强功能演示")
    print("=" * 50)
    
    # 创建示例数据
    sample_data = pd.DataFrame({
        'title': ['股票大涨', '市场下跌', '政策利好'],
        'description': [
            '今日股市表现强劲，主要指数上涨超过2%',
            '受外部因素影响，市场出现调整',
            '政府出台新政策，支持经济发展'
        ],
        'sentiment': ['positive', 'negative', 'positive']
    })
    
    print("原始数据:")
    print(sample_data)
    print()
    
    # 创建数据增强器
    augmentor = DataAugmentor()
    
    # 演示文本增强
    print("文本增强示例:")
    test_text = "股票市场今日表现强劲"
    print(f"原文: {test_text}")
    
    methods = ['synonym_replacement', 'word_insertion', 'word_deletion', 'sentence_restructure']
    for method in methods:
        augmented = augmentor.augment_text(test_text, method)
        print(f"{method}: {augmented}")
    
    print()
    
    # 演示数据集增强
    print("数据集增强:")
    augmented_df = augmentor.augment_dataset(sample_data, 'title', 'description', 'sentiment', target_samples_per_class=5)
    print(f"增强后数据行数: {len(augmented_df)}")
    
    print("\n增强后的数据:")
    for _, row in augmented_df.iterrows():
        is_aug = "✓" if row.get('is_augmented', False) else " "
        method = row.get('augmentation_method', 'original')
        print(f"{is_aug} [{method}] 标题: {row['title']}")
        print(f"    描述: {row['description']}")
        print(f"    情感: {row['sentiment']}")
        print()
    
    # 演示平衡数据集创建
    print("创建平衡数据集:")
    balanced_df = augmentor.create_balanced_dataset(sample_data, 'title', 'description', 'sentiment', samples_per_class=8)
    print(f"平衡数据集行数: {len(balanced_df)}")
    print(f"标签分布:\n{balanced_df['sentiment'].value_counts()}")

if __name__ == "__main__":
    demo_data_augmentation()
