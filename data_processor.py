import jieba
import re
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import logging

logger = logging.getLogger(__name__)

class TextProcessor:
    def __init__(self):
        # 金融领域专业词汇
        self.financial_terms = [
            '涨停', '跌停', '涨幅', '跌幅', '股价', '股票', '股市', '大盘', '指数',
            '成交量', '成交额', '市盈率', '市净率', '净资产', '净利润', '营收',
            '业绩', '财报', '分红', '送股', '配股', '增发', '回购', '减持',
            '并购', '重组', '上市', '退市', 'IPO', '科创板', '创业板', '主板',
            '基金', '债券', '期货', '期权', '外汇', '黄金', '原油', '大宗商品'
        ]
        
        # 添加金融词汇到jieba词典
        for term in self.financial_terms:
            jieba.add_word(term)
        
        # 情感词典
        self.positive_words = [
            '上涨', '增长', '利好', '突破', '创新高', '强势', '看好', '推荐',
            '买入', '增持', '业绩向好', '盈利', '收益', '分红', '送股',
            '并购', '重组', '上市', '获批', '通过', '成功', '突破', '领先'
        ]
        
        self.negative_words = [
            '下跌', '下降', '利空', '跌破', '创新低', '弱势', '看空', '减持',
            '卖出', '亏损', '业绩下滑', '亏损', '负债', '退市', '违规',
            '处罚', '风险', '警告', '失败', '延迟', '取消', '暂停'
        ]
        
        self.neutral_words = [
            '持平', '稳定', '维持', '不变', '调整', '震荡', '盘整', '观望',
            '中性', '平衡', '正常', '常规', '标准', '一般', '普通'
        ]
    
    def clean_text(self, text: str) -> str:
        if pd.isna(text) or text == '':
            return ''
        
        text = str(text)
        
        # 移除HTML标签
        text = re.sub(r'<[^>]+>', '', text)
        
        # 移除URL
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # 移除特殊字符和标点符号（保留中文标点）
        text = re.sub(r'[^\u4e00-\u9fa5\u3000-\u303f\uff00-\uffefa-zA-Z0-9\s]', '', text)
        
        # 移除多余空格
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def segment_text(self, text: str) -> str:
        if not text:
            return ''
        
        words = jieba.cut(text)
        return ' '.join(words)
    
    def extract_features(self, text: str) -> Dict[str, int]:
        if not text:
            return {'positive': 0, 'negative': 0, 'neutral': 0}
        
        text_lower = text.lower()
        
        positive_count = sum(1 for word in self.positive_words if word in text_lower)
        negative_count = sum(1 for word in self.negative_words if word in text_lower)
        neutral_count = sum(1 for word in self.neutral_words if word in text_lower)
        
        return {
            'positive': positive_count,
            'negative': negative_count,
            'neutral': neutral_count
        }
    
    def process_text(self, text: str) -> str:
        cleaned_text = self.clean_text(text)
        segmented_text = self.segment_text(cleaned_text)
        return segmented_text
    
    def combine_title_description(self, title: str, description: str, max_length: int = 512) -> str:
        """组合标题和描述，用于情感分析"""
        if pd.isna(title):
            title = ""
        if pd.isna(description):
            description = ""
        
        # 清理文本
        clean_title = self.clean_text(title)
        clean_description = self.clean_text(description)
        
        # 组合标题和描述
        combined_text = f"{clean_title} {clean_description}".strip()
        
        # 如果组合后太长，优先保留标题和描述的开头部分
        if len(combined_text) > max_length:
            # 标题通常更重要，保留完整标题
            if len(clean_title) <= max_length // 2:
                remaining_length = max_length - len(clean_title) - 1
                combined_text = f"{clean_title} {clean_description[:remaining_length]}"
            else:
                # 如果标题太长，只保留标题
                combined_text = clean_title[:max_length]
        
        return combined_text

class DatasetBuilder:
    def __init__(self, text_processor: TextProcessor):
        self.text_processor = text_processor
        self.label_encoder = LabelEncoder()
        
    def prepare_dataset(self, df: pd.DataFrame, title_col: str, description_col: str, label_col: str) -> Tuple[List[str], List[int]]:
        """准备训练数据集，支持标题和描述的组合"""
        texts = []
        labels = []
        
        for _, row in df.iterrows():
            title = row[title_col]
            description = row[description_col]
            label = row[label_col]
            
            if pd.isna(label) or label == '':
                continue
            
            # 组合标题和描述
            combined_text = self.text_processor.combine_title_description(title, description)
            
            if combined_text:
                texts.append(combined_text)
                labels.append(label)
        
        # 编码标签
        if labels:
            encoded_labels = self.label_encoder.fit_transform(labels)
            return texts, encoded_labels.tolist()
        
        return [], []
    
    def split_dataset(self, texts: List[str], labels: List[int], 
                     test_size: float = 0.2, val_size: float = 0.1, 
                     random_state: int = 42) -> Tuple:
        # 首先分割出测试集
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            texts, labels, test_size=test_size, random_state=random_state, stratify=labels
        )
        
        # 从训练集中分割出验证集
        val_size_adjusted = val_size / (1 - test_size)
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            train_texts, train_labels, test_size=val_size_adjusted, 
            random_state=random_state, stratify=train_labels
        )
        
        return (train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels)
    
    def get_label_mapping(self) -> Dict[int, str]:
        if hasattr(self.label_encoder, 'classes_'):
            return {i: label for i, label in enumerate(self.label_encoder.classes_)}
        return {}
    
    def get_class_weights(self, labels: List[int]) -> Dict[int, float]:
        from collections import Counter
        
        label_counts = Counter(labels)
        total = len(labels)
        
        weights = {}
        for label, count in label_counts.items():
            weights[label] = total / (len(label_counts) * count)
        
        return weights

class DataLoader:
    def __init__(self, max_length: int = 512):
        self.max_length = max_length
    
    def create_attention_mask(self, input_ids: List[int]) -> List[int]:
        return [1 if token_id != 0 else 0 for token_id in input_ids]
    
    def pad_sequence(self, sequence: List[int], max_length: int, pad_token_id: int = 0) -> List[int]:
        if len(sequence) > max_length:
            return sequence[:max_length]
        else:
            return sequence + [pad_token_id] * (max_length - len(sequence))
    
    def prepare_batch(self, texts: List[str], tokenizer, labels: List[int] = None) -> Dict:
        encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        batch = {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask']
        }
        
        if labels:
            batch['labels'] = torch.tensor(labels)
        
        return batch
