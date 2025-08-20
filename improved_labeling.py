#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
改进的情感标注脚本
解决标注倾斜问题，提高标注质量
"""

import pandas as pd
import jieba
import re
import logging
from database import DatabaseManager
import configparser
from typing import Dict, List, Tuple
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImprovedSentimentLabeler:
    """改进的情感标注器"""
    
    def __init__(self):
        # 加载改进的情感词典
        self.positive_words = self._load_improved_positive_words()
        self.negative_words = self._load_improved_negative_words()
        self.neutral_words = self._load_improved_neutral_words()
        
        # 情感强度词汇
        self.intensity_words = self._load_intensity_words()
        
        # 否定词和转折词
        self.negation_words = ['不', '没', '无', '非', '未', '别', '莫', '勿', '毋', '弗', '否', '反', '逆', '难', '少', '低']
        self.turn_words = ['但是', '然而', '不过', '可是', '只是', '却', '反而', '反倒', '反而', '反倒', '虽然', '尽管']
        
        # 金融特定模式
        self.financial_patterns = self._load_financial_patterns()
        
    def _load_improved_positive_words(self) -> set:
        """加载改进的正面词汇"""
        words = {
            # 基础正面词汇
            '上涨', '上升', '走高', '攀升', '上扬', '增长', '增加', '提高', '提升', '改善',
            '利好', '好消息', '积极', '正面', '乐观', '强劲', '繁荣', '兴旺', '发达',
            '成功', '胜利', '突破', '创新', '领先', '优秀', '卓越', '杰出', '突出',
            '稳定', '安全', '可靠', '信任', '信心', '希望', '机遇', '机会', '前景',
            '盈利', '收益', '回报', '利润', '收入', '销售额', '市场份额', '竞争力',
            
            # 金融特定正面词汇
            '牛市', '涨停', '大涨', '暴涨', '飙升', '突破', '新高', '历史新高', '创纪录',
            '业绩', '盈利', '净利润', '营收', '销售额', '市场份额', '行业龙头', '龙头股',
            '政策支持', '减税降费', '降息', '降准', '流动性', '资金面', '投资机会',
            '并购', '重组', '上市', 'IPO', '融资', '投资', '扩张', '发展', '增长',
            
            # 市场情绪词汇
            '看好', '乐观', '信心', '热情', '追捧', '抢购', '火爆', '热销', '畅销',
            '领先', '第一', '最佳', '优质', '优秀', '卓越', '突出', '显著', '明显',
            '大幅', '显著', '明显', '强劲', '有力', '有效', '成功', '顺利', '圆满'
        }
        return words
    
    def _load_improved_negative_words(self) -> set:
        """加载改进的负面词汇"""
        words = {
            # 基础负面词汇
            '下跌', '下降', '走低', '下挫', '回落', '减少', '降低', '恶化', '衰退', '萎缩',
            '利空', '坏消息', '消极', '负面', '悲观', '疲软', '低迷', '萧条', '困难',
            '失败', '损失', '亏损', '债务', '风险', '危机', '问题', '困难', '挑战',
            '不稳定', '不安全', '不可靠', '不信任', '失望', '担忧', '恐惧', '恐慌',
            '亏损', '损失', '债务', '破产', '倒闭', '裁员', '失业', '经济危机',
            
            # 金融特定负面词汇
            '熊市', '跌停', '大跌', '暴跌', '跳水', '破位', '新低', '历史新低', '创新低',
            '业绩下滑', '亏损', '净利润下降', '营收下降', '销售额下降', '市场份额下降',
            '政策收紧', '加息', '加准', '流动性紧张', '资金面紧张', '投资风险',
            '退市', '破产', '债务违约', '信用风险', '市场风险', '系统性风险',
            
            # 市场情绪词汇
            '看空', '悲观', '担忧', '恐慌', '抛售', '清仓', '割肉', '止损', '套牢',
            '落后', '垫底', '最差', '劣质', '糟糕', '恶劣', '严重', '重大', '巨大',
            '大幅', '显著', '明显', '疲软', '无力', '无效', '失败', '困难', '挫折'
        }
        return words
    
    def _load_improved_neutral_words(self) -> set:
        """加载改进的中性词汇"""
        words = {
            # 基础中性词汇
            '平稳', '稳定', '维持', '保持', '观察', '分析', '研究', '调查', '统计',
            '发布', '公布', '宣布', '通知', '报告', '数据', '指标', '趋势', '变化',
            '政策', '规定', '制度', '标准', '要求', '建议', '意见', '看法', '观点',
            '会议', '讨论', '协商', '合作', '交流', '沟通', '联系', '关系', '影响',
            
            # 时间词汇
            '今日', '昨日', '本周', '本月', '今年', '近期', '未来', '预期', '预计',
            '即将', '将要', '计划', '安排', '准备', '考虑', '研究', '探讨', '讨论',
            
            # 程度词汇
            '一般', '普通', '正常', '常规', '标准', '平均', '中等', '适中', '适度'
        }
        return words
    
    def _load_intensity_words(self) -> Dict[str, float]:
        """加载情感强度词汇"""
        intensity = {
            # 极强正面
            '暴涨': 3.0, '飙升': 3.0, '突破': 2.5, '历史新高': 2.5, '涨停': 2.5,
            '极大利好': 3.0, '重大突破': 2.5, '显著改善': 2.0, '大幅增长': 2.0,
            '创纪录': 2.5, '前所未有': 2.5, '史无前例': 2.5, '里程碑': 2.0,
            
            # 强正面
            '大涨': 2.0, '上涨': 1.5, '增长': 1.5, '改善': 1.5, '利好': 1.8,
            '积极': 1.5, '乐观': 1.5, '强劲': 1.8, '优秀': 1.5, '成功': 1.8,
            '看好': 1.8, '信心': 1.5, '热情': 1.5, '追捧': 1.8,
            
            # 中等正面
            '小幅上涨': 1.0, '微涨': 0.8, '稳定': 0.5, '维持': 0.5, '平稳': 0.5,
            
            # 极强负面
            '暴跌': -3.0, '跳水': -3.0, '破位': -2.5, '历史新低': -2.5, '跌停': -2.5,
            '极大利空': -3.0, '重大危机': -2.5, '显著恶化': -2.0, '大幅下降': -2.0,
            '创新低': -2.5, '前所未有': -2.5, '史无前例': -2.5, '灾难性': -2.5,
            
            # 强负面
            '大跌': -2.0, '下跌': -1.5, '下降': -1.5, '恶化': -1.5, '利空': -1.8,
            '消极': -1.5, '悲观': -1.5, '疲软': -1.8, '失败': -1.8, '亏损': -1.8,
            '看空': -1.8, '担忧': -1.5, '恐慌': -1.8, '抛售': -1.8,
            
            # 中等负面
            '小幅下跌': -1.0, '微跌': -0.8, '调整': -0.5, '波动': -0.3
        }
        return intensity
    
    def _load_financial_patterns(self) -> Dict[str, float]:
        """加载金融模式匹配"""
        patterns = {
            # 正面模式
            r'上涨\s*\d+%': 2.0,  # 上涨X%
            r'增长\s*\d+%': 1.8,  # 增长X%
            r'突破\s*\d+': 2.0,   # 突破X
            r'创\s*新高': 2.5,     # 创新高
            r'利好': 1.8,          # 利好
            r'支持': 1.5,          # 支持
            r'促进': 1.5,          # 促进
            
            # 负面模式
            r'下跌\s*\d+%': -2.0, # 下跌X%
            r'下降\s*\d+%': -1.8, # 下降X%
            r'跌破\s*\d+': -2.0,  # 跌破X
            r'创\s*新低': -2.5,   # 创新低
            r'利空': -1.8,         # 利空
            r'收紧': -1.5,         # 收紧
            r'限制': -1.5,         # 限制
        }
        return patterns
    
    def analyze_sentiment_improved(self, title: str, description: str) -> Tuple[str, float]:
        """改进的情感分析"""
        if pd.isna(title):
            title = ""
        if pd.isna(description):
            description = ""
        
        # 组合标题和描述
        combined_text = f"{title} {description}".strip()
        if not combined_text:
            return 'neutral', 0.5
        
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
            elif word in self.positive_words:
                word_score = 1.0
            # 检查负面词汇
            elif word in self.negative_words:
                word_score = -1.0
            # 检查中性词汇
            elif word in self.neutral_words:
                word_score = 0.0
            
            sentiment_score += word_score
        
        # 模式匹配
        for pattern, score in self.financial_patterns.items():
            if re.search(pattern, combined_text):
                sentiment_score += score
        
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
        
        # 调整阈值，降低neutral的比例
        if normalized_score > 0.05:  # 降低正面阈值
            sentiment = 'positive'
        elif normalized_score < -0.05:  # 降低负面阈值
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        # 计算置信度
        confidence = min(abs(normalized_score) * 3, 0.95)
        confidence = max(confidence, 0.6)  # 提高最低置信度
        
        return sentiment, confidence
    
    def batch_analyze_improved(self, data: List[Dict]) -> List[Dict]:
        """批量分析情感（改进版）"""
        results = []
        
        for i, item in enumerate(data):
            title = item.get('title', '')
            description = item.get('description', '')
            
            sentiment, confidence = self.analyze_sentiment_improved(title, description)
            
            result = item.copy()
            result['predicted_sentiment'] = sentiment
            result['confidence'] = confidence
            
            results.append(result)
            
            if (i + 1) % 100 == 0:
                logger.info(f"已处理 {i + 1}/{len(data)} 条数据")
        
        return results

def rebalance_sentiment_labels():
    """重新平衡情感标签"""
    
    try:
        logger.info("开始重新平衡情感标签...")
        
        # 连接数据库
        db_manager = DatabaseManager()
        
        # 获取已标注数据
        labeled_data = db_manager.get_labeled_data()
        
        if labeled_data.empty:
            logger.info("没有找到已标注数据")
            return False
        
        logger.info(f"找到 {len(labeled_data)} 条已标注数据")
        
        # 创建改进的标注器
        labeler = ImprovedSentimentLabeler()
        
        # 重新分析情感
        logger.info("重新分析情感...")
        
        # 转换为字典列表
        data_list = []
        for _, row in labeled_data.iterrows():
            data_list.append({
                'id': getattr(row, db_manager.id_column),
                'title': getattr(row, db_manager.title_column),
                'description': getattr(row, db_manager.description_column)
            })
        
        # 重新标注
        relabeled_results = labeler.batch_analyze_improved(data_list)
        
        # 统计标注结果
        sentiment_counts = {}
        for result in relabeled_results:
            sentiment = result['predicted_sentiment']
            sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
        
        logger.info("重新标注结果统计:")
        for sentiment, count in sentiment_counts.items():
            logger.info(f"  {sentiment}: {count}")
        
        # 更新数据库
        logger.info("开始更新数据库...")
        
        updates = []
        for result in relabeled_results:
            updates.append({
                'id': result['id'],
                'sentiment': result['predicted_sentiment'],
                'confidence': result['confidence']
            })
        
        # 批量更新
        db_manager.batch_update_sentiment(updates)
        
        logger.info(f"✅ 成功重新标注 {len(updates)} 条数据")
        
        return True
        
    except Exception as e:
        logger.error(f"重新平衡情感标签失败: {e}")
        return False

def main():
    """主函数"""
    print("改进的情感标注系统")
    print("=" * 60)
    
    print("\n选择操作:")
    print("1. 重新平衡现有标注数据")
    print("2. 退出")
    
    while True:
        choice = input("\n请输入选择 (1-2): ").strip()
        
        if choice == '1':
            print("\n开始重新平衡情感标签...")
            success = rebalance_sentiment_labels()
            
            if success:
                print("\n🎉 情感标签重新平衡完成！")
                print("现在可以运行训练脚本:")
                print("python3 enhanced_train_model.py")
            else:
                print("\n❌ 重新平衡失败")
            break
            
        elif choice == '2':
            print("退出程序")
            break
            
        else:
            print("无效选择，请重新输入")

if __name__ == "__main__":
    main()
