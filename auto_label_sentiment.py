#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
智能情感标注脚本
自动标注现有数据库中的新闻数据情感
"""

import pandas as pd
import numpy as np
import jieba
import re
import logging
from database import DatabaseManager
import configparser
from typing import Dict, List, Tuple
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentLabeler:
    """智能情感标注器"""
    
    def __init__(self):
        # 加载情感词典
        self.positive_words = self._load_positive_words()
        self.negative_words = self._load_negative_words()
        self.neutral_words = self._load_neutral_words()
        
        # 金融特定词汇
        self.financial_positive = self._load_financial_positive()
        self.financial_negative = self._load_financial_negative()
        
        # 情感强度词汇
        self.intensity_words = self._load_intensity_words()
        
        # 否定词
        self.negation_words = ['不', '没', '无', '非', '未', '别', '莫', '勿', '毋', '弗', '否', '反', '逆']
        
        # 转折词
        self.turn_words = ['但是', '然而', '不过', '可是', '只是', '却', '反而', '反倒', '反而', '反倒']
        
    def _load_positive_words(self) -> set:
        """加载正面情感词汇"""
        words = {
            '上涨', '上升', '走高', '攀升', '上扬', '增长', '增加', '提高', '提升', '改善',
            '利好', '好消息', '积极', '正面', '乐观', '强劲', '繁荣', '兴旺', '发达',
            '成功', '胜利', '突破', '创新', '领先', '优秀', '卓越', '杰出', '突出',
            '稳定', '安全', '可靠', '信任', '信心', '希望', '机遇', '机会', '前景',
            '盈利', '收益', '回报', '利润', '收入', '销售额', '市场份额', '竞争力'
        }
        return words
    
    def _load_negative_words(self) -> set:
        """加载负面情感词汇"""
        words = {
            '下跌', '下降', '走低', '下挫', '回落', '减少', '降低', '恶化', '衰退', '萎缩',
            '利空', '坏消息', '消极', '负面', '悲观', '疲软', '低迷', '萧条', '困难',
            '失败', '损失', '亏损', '债务', '风险', '危机', '问题', '困难', '挑战',
            '不稳定', '不安全', '不可靠', '不信任', '失望', '担忧', '恐惧', '恐慌',
            '亏损', '损失', '债务', '破产', '倒闭', '裁员', '失业', '经济危机'
        }
        return words
    
    def _load_neutral_words(self) -> set:
        """加载中性情感词汇"""
        words = {
            '平稳', '稳定', '维持', '保持', '观察', '分析', '研究', '调查', '统计',
            '发布', '公布', '宣布', '通知', '报告', '数据', '指标', '趋势', '变化',
            '政策', '规定', '制度', '标准', '要求', '建议', '意见', '看法', '观点',
            '会议', '讨论', '协商', '合作', '交流', '沟通', '联系', '关系', '影响'
        }
        return words
    
    def _load_financial_positive(self) -> set:
        """加载金融正面词汇"""
        words = {
            '牛市', '涨停', '大涨', '暴涨', '飙升', '突破', '新高', '历史新高',
            '业绩', '盈利', '净利润', '营收', '销售额', '市场份额', '行业龙头',
            '政策支持', '减税降费', '降息', '降准', '流动性', '资金面', '投资机会',
            '并购', '重组', '上市', 'IPO', '融资', '投资', '扩张', '发展'
        }
        return words
    
    def _load_financial_negative(self) -> set:
        """加载金融负面词汇"""
        words = {
            '熊市', '跌停', '大跌', '暴跌', '跳水', '破位', '新低', '历史新低',
            '业绩下滑', '亏损', '净利润下降', '营收下降', '销售额下降', '市场份额下降',
            '政策收紧', '加息', '加准', '流动性紧张', '资金面紧张', '投资风险',
            '退市', '破产', '债务违约', '信用风险', '市场风险', '系统性风险'
        }
        return words
    
    def _load_intensity_words(self) -> Dict[str, float]:
        """加载情感强度词汇"""
        intensity = {
            # 极强正面
            '暴涨': 2.0, '飙升': 2.0, '突破': 1.8, '历史新高': 1.8, '涨停': 1.8,
            '极大利好': 2.0, '重大突破': 1.8, '显著改善': 1.6, '大幅增长': 1.6,
            
            # 强正面
            '大涨': 1.5, '上涨': 1.2, '增长': 1.2, '改善': 1.2, '利好': 1.3,
            '积极': 1.2, '乐观': 1.2, '强劲': 1.3, '优秀': 1.2, '成功': 1.3,
            
            # 中等正面
            '小幅上涨': 0.8, '微涨': 0.6, '稳定': 0.5, '维持': 0.5, '平稳': 0.5,
            
            # 极强负面
            '暴跌': -2.0, '跳水': -2.0, '破位': -1.8, '历史新低': -1.8, '跌停': -1.8,
            '极大利空': -2.0, '重大危机': -1.8, '显著恶化': -1.6, '大幅下降': -1.6,
            
            # 强负面
            '大跌': -1.5, '下跌': -1.2, '下降': -1.2, '恶化': -1.2, '利空': -1.3,
            '消极': -1.2, '悲观': -1.2, '疲软': -1.3, '失败': -1.3, '亏损': -1.3,
            
            # 中等负面
            '小幅下跌': -0.8, '微跌': -0.6, '调整': -0.5, '波动': -0.3
        }
        return intensity
    
    def analyze_sentiment(self, title: str, description: str) -> Tuple[str, float]:
        """分析文本情感"""
        if pd.isna(title):
            title = ""
        if pd.isna(description):
            description = ""
        
        # 组合标题和描述
        combined_text = f"{title} {description}".strip()
        if not combined_text:
            return 'neutral', 0.0
        
        # 分词
        words = list(jieba.cut(combined_text))
        
        # 计算情感分数
        sentiment_score = 0.0
        word_count = len(words)
        
        # 检查否定词和转折词
        has_negation = any(word in combined_text for word in self.negation_words)
        has_turn = any(word in combined_text for word in self.turn_words)
        
        # 分析每个词的情感
        for word in words:
            word_score = 0.0
            
            # 检查情感强度词汇
            if word in self.intensity_words:
                word_score = self.intensity_words[word]
            # 检查正面词汇
            elif word in self.positive_words or word in self.financial_positive:
                word_score = 1.0
            # 检查负面词汇
            elif word in self.negative_words or word in self.financial_negative:
                word_score = -1.0
            # 检查中性词汇
            elif word in self.neutral_words:
                word_score = 0.0
            
            sentiment_score += word_score
        
        # 处理否定词
        if has_negation:
            sentiment_score = -sentiment_score * 0.8
        
        # 处理转折词
        if has_turn:
            sentiment_score = sentiment_score * 0.6
        
        # 标准化分数
        if word_count > 0:
            normalized_score = sentiment_score / word_count
        else:
            normalized_score = 0.0
        
        # 确定情感标签
        if normalized_score > 0.1:
            sentiment = 'positive'
        elif normalized_score < -0.1:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        # 计算置信度
        confidence = min(abs(normalized_score) * 2, 0.95)
        confidence = max(confidence, 0.5)  # 最低置信度0.5
        
        return sentiment, confidence
    
    def batch_analyze(self, data: List[Dict]) -> List[Dict]:
        """批量分析情感"""
        results = []
        
        for i, item in enumerate(data):
            title = item.get('title', '')
            description = item.get('description', '')
            
            sentiment, confidence = self.analyze_sentiment(title, description)
            
            result = item.copy()
            result['predicted_sentiment'] = sentiment
            result['confidence'] = confidence
            
            results.append(result)
            
            if (i + 1) % 100 == 0:
                logger.info(f"已处理 {i + 1}/{len(data)} 条数据")
        
        return results

def auto_label_database_data():
    """自动标注数据库中的数据"""
    
    try:
        logger.info("开始自动标注数据库数据...")
        
        # 连接数据库
        db_manager = DatabaseManager()
        
        # 获取未标注的数据
        logger.info("获取未标注数据...")
        unlabeled_data = db_manager.get_unlabeled_data()
        
        if not unlabeled_data:
            logger.info("没有找到未标注的数据")
            return
        
        logger.info(f"找到 {len(unlabeled_data)} 条未标注数据")
        
        # 创建标注器
        labeler = SentimentLabeler()
        
        # 批量分析情感
        logger.info("开始情感分析...")
        labeled_results = labeler.batch_analyze(unlabeled_data)
        
        # 统计标注结果
        sentiment_counts = {}
        for result in labeled_results:
            sentiment = result['predicted_sentiment']
            sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
        
        logger.info("标注结果统计:")
        for sentiment, count in sentiment_counts.items():
            logger.info(f"  {sentiment}: {count}")
        
        # 更新数据库
        logger.info("开始更新数据库...")
        
        updates = []
        for result in labeled_results:
            updates.append({
                'id': result['id'],
                'sentiment': result['predicted_sentiment'],
                'confidence': result['confidence']
            })
        
        # 批量更新
        db_manager.batch_update_sentiment(updates)
        
        logger.info(f"✅ 成功标注 {len(updates)} 条数据")
        
        # 验证结果
        labeled_data = db_manager.get_labeled_data()
        logger.info(f"数据库中现有已标注数据: {len(labeled_data)} 条")
        
        return True
        
    except Exception as e:
        logger.error(f"自动标注失败: {e}")
        return False

def preview_labeling_results(limit: int = 10):
    """预览标注结果"""
    
    try:
        logger.info(f"预览前 {limit} 条标注结果...")
        
        # 连接数据库
        db_manager = DatabaseManager()
        
        # 获取已标注数据
        labeled_data = db_manager.get_labeled_data()
        
        if not labeled_data:
            logger.info("没有找到已标注数据")
            return
        
        # 显示前几条结果
        for i, row in enumerate(labeled_data.head(limit).itertuples()):
            print(f"\n--- 第 {i+1} 条 ---")
            print(f"ID: {getattr(row, db_manager.id_column)}")
            print(f"标题: {getattr(row, db_manager.title_column)}")
            print(f"描述: {getattr(row, db_manager.description_column)}")
            print(f"情感: {getattr(row, db_manager.label_column)}")
            print(f"置信度: {getattr(row, db_manager.confidence_column)}")
        
        # 统计信息
        sentiment_counts = labeled_data[db_manager.label_column].value_counts()
        print(f"\n情感分布:")
        for sentiment, count in sentiment_counts.items():
            print(f"  {sentiment}: {count}")
        
        avg_confidence = labeled_data[db_manager.confidence_column].mean()
        print(f"平均置信度: {avg_confidence:.3f}")
        
    except Exception as e:
        logger.error(f"预览失败: {e}")

def main():
    """主函数"""
    print("股票消息情感分析系统 - 智能情感标注")
    print("=" * 60)
    
    print("\n选择操作:")
    print("1. 自动标注数据库中的所有未标注数据")
    print("2. 预览已标注数据的结果")
    print("3. 退出")
    
    while True:
        choice = input("\n请输入选择 (1-3): ").strip()
        
        if choice == '1':
            print("\n开始自动标注...")
            success = auto_label_database_data()
            
            if success:
                print("\n🎉 自动标注完成！")
                print("现在可以运行训练脚本:")
                print("python3 enhanced_train_model.py")
            else:
                print("\n❌ 自动标注失败")
            break
            
        elif choice == '2':
            print("\n预览标注结果...")
            preview_labeling_results()
            break
            
        elif choice == '3':
            print("退出程序")
            break
            
        else:
            print("无效选择，请重新输入")

if __name__ == "__main__":
    main()
