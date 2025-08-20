#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
智能情感标注脚本 - 简化版
自动标注现有数据库中的新闻数据情感
"""

import pandas as pd
import jieba
import logging
from database import DatabaseManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sentiment_dictionary():
    """创建情感词典"""
    
    # 正面词汇
    positive_words = {
        '上涨', '上升', '走高', '攀升', '上扬', '增长', '增加', '提高', '提升', '改善',
        '利好', '好消息', '积极', '正面', '乐观', '强劲', '繁荣', '兴旺', '发达',
        '成功', '胜利', '突破', '创新', '领先', '优秀', '卓越', '杰出', '突出',
        '稳定', '安全', '可靠', '信任', '信心', '希望', '机遇', '机会', '前景',
        '盈利', '收益', '回报', '利润', '收入', '销售额', '市场份额', '竞争力',
        '牛市', '涨停', '大涨', '暴涨', '飙升', '突破', '新高', '历史新高',
        '业绩', '盈利', '净利润', '营收', '销售额', '市场份额', '行业龙头',
        '政策支持', '减税降费', '降息', '降准', '流动性', '资金面', '投资机会',
        '并购', '重组', '上市', 'IPO', '融资', '投资', '扩张', '发展'
    }
    
    # 负面词汇
    negative_words = {
        '下跌', '下降', '走低', '下挫', '回落', '减少', '降低', '恶化', '衰退', '萎缩',
        '利空', '坏消息', '消极', '负面', '悲观', '疲软', '低迷', '萧条', '困难',
        '失败', '损失', '亏损', '债务', '风险', '危机', '问题', '困难', '挑战',
        '不稳定', '不安全', '不可靠', '不信任', '失望', '担忧', '恐惧', '恐慌',
        '亏损', '损失', '债务', '破产', '倒闭', '裁员', '失业', '经济危机',
        '熊市', '跌停', '大跌', '暴跌', '跳水', '破位', '新低', '历史新低',
        '业绩下滑', '亏损', '净利润下降', '营收下降', '销售额下降', '市场份额下降',
        '政策收紧', '加息', '加准', '流动性紧张', '资金面紧张', '投资风险',
        '退市', '破产', '债务违约', '信用风险', '市场风险', '系统性风险'
    }
    
    # 中性词汇
    neutral_words = {
        '平稳', '稳定', '维持', '保持', '观察', '分析', '研究', '调查', '统计',
        '发布', '公布', '宣布', '通知', '报告', '数据', '指标', '趋势', '变化',
        '政策', '规定', '制度', '标准', '要求', '建议', '意见', '看法', '观点',
        '会议', '讨论', '协商', '合作', '交流', '沟通', '联系', '关系', '影响'
    }
    
    return positive_words, negative_words, neutral_words

def analyze_sentiment(title, description, positive_words, negative_words, neutral_words):
    """分析单个文本的情感"""
    
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
    positive_count = sum(1 for word in words if word in positive_words)
    negative_count = sum(1 for word in words if word in negative_words)
    neutral_count = sum(1 for word in words if word in neutral_words)
    
    # 计算置信度
    total_words = len(words)
    if total_words == 0:
        return 'neutral', 0.5
    
    # 情感强度
    positive_score = positive_count / total_words
    negative_score = negative_count / total_words
    neutral_score = neutral_count / total_words
    
    # 确定情感标签
    if positive_score > negative_score and positive_score > 0.1:
        sentiment = 'positive'
        confidence = min(positive_score * 3, 0.95)
    elif negative_score > positive_score and negative_score > 0.1:
        sentiment = 'negative'
        confidence = min(negative_score * 3, 0.95)
    else:
        sentiment = 'neutral'
        confidence = min(max(neutral_score, 0.3), 0.8)
    
    # 确保最低置信度
    confidence = max(confidence, 0.5)
    
    return sentiment, confidence

def auto_label_data():
    """自动标注数据"""
    
    try:
        logger.info("开始自动标注数据库数据...")
        
        # 连接数据库
        db_manager = DatabaseManager()
        
        # 获取未标注的数据
        logger.info("获取未标注数据...")
        unlabeled_data = db_manager.get_unlabeled_data()
        
        if unlabeled_data.empty:
            logger.info("没有找到未标注的数据")
            return False
        
        logger.info(f"找到 {len(unlabeled_data)} 条未标注数据")
        
        # 创建情感词典
        positive_words, negative_words, neutral_words = create_sentiment_dictionary()
        
        # 批量分析情感
        logger.info("开始情感分析...")
        
        updates = []
        for i, row in enumerate(unlabeled_data.itertuples()):
            title = getattr(row, db_manager.title_column)
            description = getattr(row, db_manager.description_column)
            
            sentiment, confidence = analyze_sentiment(
                title, description, positive_words, negative_words, neutral_words
            )
            
            updates.append({
                'id': getattr(row, db_manager.id_column),
                'sentiment': sentiment,
                'confidence': confidence
            })
            
            if (i + 1) % 100 == 0:
                logger.info(f"已处理 {i + 1}/{len(unlabeled_data)} 条数据")
        
        # 统计标注结果
        sentiment_counts = {}
        for update in updates:
            sentiment = update['sentiment']
            sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
        
        logger.info("标注结果统计:")
        for sentiment, count in sentiment_counts.items():
            logger.info(f"  {sentiment}: {count}")
        
        # 更新数据库
        logger.info("开始更新数据库...")
        db_manager.batch_update_sentiment(updates)
        
        logger.info(f"✅ 成功标注 {len(updates)} 条数据")
        
        # 验证结果
        labeled_data = db_manager.get_labeled_data()
        logger.info(f"数据库中现有已标注数据: {len(labeled_data)} 条")
        
        return True
        
    except Exception as e:
        logger.error(f"自动标注失败: {e}")
        return False

def show_statistics():
    """显示数据库统计信息"""
    
    try:
        logger.info("显示数据库统计信息...")
        
        # 连接数据库
        db_manager = DatabaseManager()
        
        # 获取统计信息
        stats = db_manager.get_statistics()
        
        print("\n数据库统计信息:")
        print(f"总数据量: {stats['total']}")
        print(f"已标注数据: {stats['labeled']}")
        print(f"未标注数据: {stats['unlabeled']}")
        
        if stats['labeled'] > 0:
            # 显示情感分布
            labeled_data = db_manager.get_labeled_data()
            sentiment_counts = labeled_data[db_manager.label_column].value_counts()
            
            print(f"\n情感分布:")
            for sentiment, count in sentiment_counts.items():
                print(f"  {sentiment}: {count}")
            
            # 显示置信度统计
            avg_confidence = labeled_data[db_manager.confidence_column].mean()
            print(f"平均置信度: {avg_confidence:.3f}")
        
    except Exception as e:
        logger.error(f"获取统计信息失败: {e}")

def main():
    """主函数"""
    print("股票消息情感分析系统 - 智能情感标注")
    print("=" * 60)
    
    print("\n选择操作:")
    print("1. 自动标注数据库中的所有未标注数据")
    print("2. 显示数据库统计信息")
    print("3. 退出")
    
    while True:
        choice = input("\n请输入选择 (1-3): ").strip()
        
        if choice == '1':
            print("\n开始自动标注...")
            success = auto_label_data()
            
            if success:
                print("\n🎉 自动标注完成！")
                print("现在可以运行训练脚本:")
                print("python3 enhanced_train_model.py")
            else:
                print("\n❌ 自动标注失败")
            break
            
        elif choice == '2':
            print("\n显示统计信息...")
            show_statistics()
            break
            
        elif choice == '3':
            print("退出程序")
            break
            
        else:
            print("无效选择，请重新输入")

if __name__ == "__main__":
    main()
