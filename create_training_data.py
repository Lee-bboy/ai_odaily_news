#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
创建训练数据脚本
生成示例数据并插入到数据库中进行训练
"""

import pandas as pd
import numpy as np
from database import DatabaseManager
import logging
import configparser
from datetime import datetime, timedelta
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_news_data():
    """创建示例新闻数据"""
    
    # 基础新闻模板
    news_templates = [
        # Positive news
        {
            'title': '股票市场表现强劲，主要指数上涨超过2%',
            'description': '今日股市表现强劲，主要指数上涨超过2%，投资者信心增强，市场情绪乐观。多家机构分析师表示，市场基本面良好，有望继续上涨。',
            'sentiment': 'positive'
        },
        {
            'title': '央行宣布降息，释放流动性支持经济',
            'description': '央行今日宣布降息，释放流动性支持经济发展。专家认为此举将有效降低企业融资成本，提振市场信心，对股市形成利好。',
            'sentiment': 'positive'
        },
        {
            'title': '政策利好频出，支持实体经济发展',
            'description': '近期政策利好频出，政府出台多项措施支持实体经济发展。包括减税降费、优化营商环境等政策，预计将显著改善企业经营状况。',
            'sentiment': 'positive'
        },
        {
            'title': '企业盈利超预期，业绩表现亮眼',
            'description': '多家上市公司发布盈利报告，业绩表现超预期。分析师认为，企业基本面改善，盈利能力增强，为股价上涨提供支撑。',
            'sentiment': 'positive'
        },
        {
            'title': '新技术突破推动行业发展',
            'description': '新技术取得重大突破，在多个行业得到应用。专家表示，技术创新将推动产业升级，创造新的投资机会和增长点。',
            'sentiment': 'positive'
        },
        
        # Negative news
        {
            'title': '市场出现调整，主要指数下跌',
            'description': '受外部因素影响，市场出现调整，主要指数下跌。投资者担忧情绪上升，市场波动加剧，建议谨慎操作。',
            'sentiment': 'negative'
        },
        {
            'title': '经济数据显示下行压力',
            'description': '最新经济数据显示下行压力增大，多个指标不及预期。专家认为需要更多政策支持来稳定经济增长。',
            'sentiment': 'negative'
        },
        {
            'title': '监管政策收紧影响市场',
            'description': '监管政策收紧，市场流动性减少，对部分行业造成影响。投资者需要关注政策变化对投资的影响。',
            'sentiment': 'negative'
        },
        {
            'title': '企业业绩不及预期',
            'description': '多家企业发布业绩报告，表现不及预期。分析师下调了相关企业的评级和目标价。',
            'sentiment': 'negative'
        },
        {
            'title': '风险事件影响市场稳定',
            'description': '突发风险事件影响市场稳定，投资者恐慌情绪蔓延。建议保持冷静，理性分析市场情况。',
            'sentiment': 'negative'
        },
        
        # Neutral news
        {
            'title': '市场表现平稳，投资者观望',
            'description': '市场表现平稳，投资者观望情绪浓厚。成交量相对较低，市场缺乏明确方向。',
            'sentiment': 'neutral'
        },
        {
            'title': '政策解读：新政策对市场影响分析',
            'description': '专家解读最新政策对市场的影响，认为政策总体中性，对市场影响有限。投资者需要关注后续政策细节。',
            'sentiment': 'neutral'
        },
        {
            'title': '经济数据发布，市场反应平淡',
            'description': '统计局发布最新经济数据，市场反应相对平淡。数据基本符合预期，未对市场造成明显影响。',
            'sentiment': 'neutral'
        },
        {
            'title': '重要会议即将召开',
            'description': '重要经济会议即将召开，市场预期会议将讨论经济政策。投资者关注会议可能释放的政策信号。',
            'sentiment': 'neutral'
        },
        {
            'title': '行业分析报告发布',
            'description': '多家机构发布行业分析报告，对行业前景持谨慎乐观态度。报告认为行业将保持稳定发展。',
            'sentiment': 'neutral'
        }
    ]
    
    # 生成更多变体
    additional_news = []
    
    # 金融相关词汇变体
    financial_terms = {
        '股票': ['股市', 'A股', '港股', '美股', '创业板', '科创板'],
        '市场': ['资本市场', '金融市场', '证券市场', '期货市场'],
        '政策': ['货币政策', '财政政策', '产业政策', '监管政策'],
        '经济': ['宏观经济', '微观经济', '区域经济', '数字经济'],
        '企业': ['上市公司', '民营企业', '国有企业', '外资企业']
    }
    
    # 情感词汇变体
    sentiment_words = {
        'positive': ['利好', '积极', '正面', '乐观', '强劲', '增长', '提升', '改善'],
        'negative': ['利空', '负面', '悲观', '下跌', '下降', '恶化', '风险', '担忧'],
        'neutral': ['平稳', '中性', '观望', '谨慎', '稳定', '维持', '观察', '分析']
    }
    
    # 生成变体新闻
    for template in news_templates:
        # 添加原始模板
        additional_news.append(template.copy())
        
        # 生成2-3个变体
        for _ in range(random.randint(2, 3)):
            variant = template.copy()
            
            # 替换部分词汇
            for old_term, new_terms in financial_terms.items():
                if old_term in variant['title'] or old_term in variant['description']:
                    new_term = random.choice(new_terms)
                    variant['title'] = variant['title'].replace(old_term, new_term)
                    variant['description'] = variant['description'].replace(old_term, new_term)
            
            # 添加情感词汇变体
            sentiment = variant['sentiment']
            if sentiment in sentiment_words:
                sentiment_word = random.choice(sentiment_words[sentiment])
                if sentiment_word not in variant['title'] and sentiment_word not in variant['description']:
                    variant['title'] = variant['title'] + f'，{sentiment_word}'
            
            additional_news.append(variant)
    
    return additional_news

def insert_training_data_to_db():
    """将训练数据插入数据库"""
    
    try:
        # 读取配置
        config = configparser.ConfigParser()
        config.read('config.ini', encoding='utf-8')
        
        # 连接数据库
        db_manager = DatabaseManager()
        
        # 创建示例数据
        sample_news = create_sample_news_data()
        
        logger.info(f"生成了 {len(sample_news)} 条示例新闻数据")
        
        # 显示数据分布
        sentiment_counts = {}
        for news in sample_news:
            sentiment = news['sentiment']
            sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
        
        logger.info("数据分布:")
        for sentiment, count in sentiment_counts.items():
            logger.info(f"  {sentiment}: {count}")
        
        # 插入数据到数据库
        logger.info("开始插入数据到数据库...")
        
        inserted_count = 0
        for i, news in enumerate(sample_news):
            try:
                # 生成唯一ID（使用时间戳+随机数）
                news_id = int(datetime.now().timestamp() * 1000) + i + random.randint(1000, 9999)
                
                # 生成发布时间（最近30天内）
                days_ago = random.randint(0, 30)
                publish_time = int((datetime.now() - timedelta(days=days_ago)).timestamp())
                
                # 准备插入数据
                insert_data = {
                    'id': news_id,
                    'title': news['title'],
                    'description': news['description'],
                    'sentiment': news['sentiment'],
                    'confidence': round(random.uniform(0.8, 0.95), 2),  # 随机置信度
                    'publish_timestamp': publish_time,
                    'is_important': random.choice([0, 1]),
                    'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                # 插入数据
                sql = f"""
                    INSERT INTO {db_manager.table_name} 
                    (id, title, description, sentiment, confidence, publish_timestamp, is_important, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                    title = VALUES(title),
                    description = VALUES(description),
                    sentiment = VALUES(sentiment),
                    confidence = VALUES(confidence),
                    updated_at = CURRENT_TIMESTAMP
                """
                
                with db_manager.connection.cursor() as cursor:
                    cursor.execute(sql, (
                        insert_data['id'],
                        insert_data['title'],
                        insert_data['description'],
                        insert_data['sentiment'],
                        insert_data['confidence'],
                        insert_data['publish_timestamp'],
                        insert_data['is_important'],
                        insert_data['created_at']
                    ))
                
                inserted_count += 1
                
                if (i + 1) % 10 == 0:
                    logger.info(f"已插入 {i + 1}/{len(sample_news)} 条数据")
                
            except Exception as e:
                logger.error(f"插入第 {i+1} 条数据失败: {e}")
                continue
        
        # 提交事务
        db_manager.connection.commit()
        
        logger.info(f"✅ 成功插入 {inserted_count} 条训练数据到数据库")
        
        # 验证插入结果
        labeled_data = db_manager.get_labeled_data()
        logger.info(f"数据库中现有已标注数据: {len(labeled_data)} 条")
        
        return True
        
    except Exception as e:
        logger.error(f"插入训练数据失败: {e}")
        return False

def main():
    """主函数"""
    print("股票消息情感分析系统 - 创建训练数据")
    print("=" * 60)
    
    try:
        success = insert_training_data_to_db()
        
        if success:
            print("\n🎉 训练数据创建成功！")
            print("现在可以运行训练脚本:")
            print("python3 enhanced_train_model.py")
            
            print("\n或者先检查数据:")
            print("python3 test_improved.py")
        else:
            print("\n❌ 训练数据创建失败")
            
    except Exception as e:
        print(f"\n❌ 脚本执行失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
