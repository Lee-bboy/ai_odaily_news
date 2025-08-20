#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Tuple
import logging
import json
import os

logger = logging.getLogger(__name__)

def create_sample_data() -> pd.DataFrame:
    sample_data = [
        {'id': 1, 'content': '某公司发布业绩预告，预计净利润同比增长50%，股价有望上涨', 'sentiment': 'positive'},
        {'id': 2, 'content': '市场分析师看好该股票，给予买入评级，目标价上调', 'sentiment': 'positive'},
        {'id': 3, 'content': '公司发布公告称业绩不及预期，股价可能承压', 'sentiment': 'negative'},
        {'id': 4, 'content': '市场震荡调整，投资者观望情绪浓厚', 'sentiment': 'neutral'},
        {'id': 5, 'content': '公司获得重要合同，预计将显著提升营收', 'sentiment': 'positive'},
        {'id': 6, 'content': '行业政策收紧，相关公司面临挑战', 'sentiment': 'negative'},
        {'id': 7, 'content': '市场维持稳定，成交量略有下降', 'sentiment': 'neutral'},
        {'id': 8, 'content': '公司回购股份，显示管理层信心', 'sentiment': 'positive'},
        {'id': 9, 'content': '业绩下滑明显，投资者担忧情绪上升', 'sentiment': 'negative'},
        {'id': 10, 'content': '市场消息面平静，等待重要数据公布', 'sentiment': 'neutral'}
    ]
    return pd.DataFrame(sample_data)

def analyze_sentiment_distribution(data: pd.DataFrame, sentiment_col: str = 'sentiment') -> Dict:
    sentiment_counts = data[sentiment_col].value_counts()
    total = len(data)
    
    distribution = {
        'counts': sentiment_counts.to_dict(),
        'percentages': (sentiment_counts / total * 100).to_dict(),
        'total': total
    }
    return distribution

def plot_sentiment_distribution(data: pd.DataFrame, sentiment_col: str = 'sentiment', save_path: str = None):
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    sentiment_counts = data[sentiment_col].value_counts()
    plt.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%')
    plt.title('情感分布')
    
    plt.subplot(2, 2, 2)
    sentiment_counts.plot(kind='bar')
    plt.title('情感分布统计')
    plt.xlabel('情感类别')
    plt.ylabel('数量')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"图表已保存到: {save_path}")
    else:
        plt.show()
    
    plt.close()

def generate_training_report(training_results: Dict, save_path: str = None) -> str:
    report = []
    report.append("=" * 50)
    report.append("股票消息情感分析模型训练报告")
    report.append("=" * 50)
    report.append("")
    
    report.append("训练结果摘要:")
    report.append(f"  最佳验证准确率: {training_results['best_val_accuracy']:.4f}")
    report.append(f"  最终验证准确率: {training_results['final_val_accuracy']:.4f}")
    report.append(f"  最终F1分数 (Macro): {training_results['final_metrics']['f1_macro']:.4f}")
    report.append("")
    
    report_text = "\n".join(report)
    
    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        logger.info(f"训练报告已保存到: {save_path}")
    
    return report_text

def create_sample_database_schema() -> str:
    sql = """
CREATE TABLE IF NOT EXISTS stock_news (
    id INT AUTO_INCREMENT PRIMARY KEY,
    title VARCHAR(500) COMMENT '新闻标题',
    content TEXT COMMENT '新闻内容',
    source VARCHAR(100) COMMENT '新闻来源',
    publish_time DATETIME COMMENT '发布时间',
    stock_code VARCHAR(20) COMMENT '相关股票代码',
    stock_name VARCHAR(100) COMMENT '相关股票名称',
    sentiment VARCHAR(20) COMMENT '情感标签: positive/negative/neutral',
    confidence DECIMAL(5,4) COMMENT '预测置信度',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    INDEX idx_sentiment (sentiment),
    INDEX idx_stock_code (stock_code),
    INDEX idx_publish_time (publish_time)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='股票新闻情感分析数据表';
"""
    return sql
