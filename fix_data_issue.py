#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速修复数据问题的脚本
解决样本数量不足导致的训练问题
"""

import pandas as pd
import numpy as np
from data_augmentation import DataAugmentor
from data_processor import TextProcessor, DatasetBuilder
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_data():
    """创建足够的样本数据用于测试"""
    print("创建样本数据...")
    
    # 基础样本
    base_data = [
        # positive samples
        ('股票大涨', '今日股市表现强劲，主要指数上涨超过2%', 'positive'),
        ('政策利好', '政府出台新政策，支持经济发展', 'positive'),
        ('经济数据向好', '最新经济数据显示增长态势良好', 'positive'),
        ('央行降息', '央行宣布降息，释放流动性', 'positive'),
        ('投资机会', '市场出现新的投资机会', 'positive'),
        ('企业盈利', '多家企业发布盈利报告，业绩超预期', 'positive'),
        ('市场信心', '投资者信心增强，市场情绪乐观', 'positive'),
        ('技术突破', '新技术取得重大突破，推动行业发展', 'positive'),
        
        # negative samples
        ('市场下跌', '受外部因素影响，市场出现调整', 'negative'),
        ('经济下行', '经济数据显示下行压力增大', 'negative'),
        ('政策收紧', '监管政策收紧，市场流动性减少', 'negative'),
        ('企业亏损', '多家企业发布亏损报告，业绩不及预期', 'negative'),
        ('市场恐慌', '投资者恐慌情绪蔓延，市场波动加剧', 'negative'),
        ('风险事件', '突发风险事件影响市场稳定', 'negative'),
        ('监管处罚', '多家机构因违规被监管处罚', 'negative'),
        ('市场低迷', '市场交投清淡，成交量萎缩', 'negative'),
        
        # neutral samples
        ('市场观察', '市场表现平稳，投资者观望情绪浓厚', 'neutral'),
        ('政策解读', '专家解读最新政策对市场的影响', 'neutral'),
        ('数据发布', '统计局发布最新经济数据', 'neutral'),
        ('会议召开', '重要经济会议即将召开', 'neutral'),
        ('行业分析', '分析师发布行业研究报告', 'neutral'),
        ('市场动态', '市场出现新的变化和趋势', 'neutral'),
        ('政策预期', '市场对即将出台的政策有所预期', 'neutral'),
        ('技术发展', '新技术在行业中的应用情况', 'neutral')
    ]
    
    # 创建DataFrame
    df = pd.DataFrame(base_data, columns=['title', 'description', 'sentiment'])
    
    print(f"创建了 {len(df)} 个基础样本")
    print("标签分布:")
    print(df['sentiment'].value_counts())
    
    return df

def enhance_dataset(df, target_samples_per_class=15):
    """增强数据集"""
    print(f"\n开始数据增强，目标每类 {target_samples_per_class} 个样本...")
    
    augmentor = DataAugmentor()
    enhanced_df = augmentor.create_balanced_dataset(
        df, 'title', 'description', 'sentiment', 
        samples_per_class=target_samples_per_class
    )
    
    return enhanced_df

def test_data_processing(enhanced_df):
    """测试数据处理"""
    print("\n测试数据处理...")
    
    processor = TextProcessor()
    builder = DatasetBuilder(processor)
    
    # 准备数据集
    texts, labels = builder.prepare_dataset(
        enhanced_df, 'title', 'description', 'sentiment'
    )
    
    print(f"处理后的文本数量: {len(texts)}")
    print(f"标签数量: {len(labels)}")
    
    if len(texts) > 0:
        try:
            # 测试数据集分割
            train_data, val_data, test_data = builder.split_dataset(texts, labels)
            print("✓ 数据集分割成功")
            print(f"  训练集: {len(train_data[0])} 样本")
            print(f"  验证集: {len(val_data[0])} 样本")
            print(f"  测试集: {len(test_data[0])} 样本")
            
            return True
        except Exception as e:
            print(f"✗ 数据集分割失败: {e}")
            return False
    
    return False

def save_enhanced_data(enhanced_df, filename='enhanced_sample_data.csv'):
    """保存增强后的数据"""
    enhanced_df.to_csv(filename, index=False, encoding='utf-8')
    print(f"\n增强后的数据已保存到: {filename}")
    
    # 显示统计信息
    print("\n最终数据统计:")
    print(f"总样本数: {len(enhanced_df)}")
    print("标签分布:")
    print(enhanced_df['sentiment'].value_counts())
    
    # 显示增强样本统计
    if 'is_augmented' in enhanced_df.columns:
        augmented_count = enhanced_df['is_augmented'].sum()
        original_count = len(enhanced_df) - augmented_count
        print(f"\n原始样本: {original_count}")
        print(f"增强样本: {augmented_count}")
        
        if augmented_count > 0:
            print("\n增强方法统计:")
            method_counts = enhanced_df[enhanced_df['is_augmented']]['augmentation_method'].value_counts()
            print(method_counts)

def main():
    """主函数"""
    print("股票消息情感分析系统 - 数据问题修复脚本")
    print("=" * 60)
    
    try:
        # 1. 创建基础样本数据
        base_df = create_sample_data()
        
        # 2. 增强数据集
        enhanced_df = enhance_dataset(base_df, target_samples_per_class=20)
        
        # 3. 测试数据处理
        success = test_data_processing(enhanced_df)
        
        if success:
            print("\n🎉 数据处理测试成功！")
            
            # 4. 保存增强后的数据
            save_enhanced_data(enhanced_df)
            
            print("\n现在可以使用以下命令进行训练:")
            print("python3 enhanced_train_model.py")
            
        else:
            print("\n⚠️  数据处理测试失败，请检查代码")
            
    except Exception as e:
        print(f"\n❌ 脚本执行失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
