#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
金融词汇扩展演示脚本
展示如何为美股和币圈情感分析添加专业术语
"""

import sys
import os
import logging

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from financial_vocab import FinancialVocabulary, create_financial_tokenizer
from transformers import AutoTokenizer

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def demo_vocabulary_management():
    """演示词汇管理功能"""
    print("\n" + "="*60)
    print("金融词汇管理演示")
    print("="*60)
    
    # 创建词汇管理器
    vocab = FinancialVocabulary()
    
    # 打印词汇表摘要
    vocab.print_vocabulary_summary()
    
    # 搜索特定术语
    print("\n搜索包含 'BTC' 的术语:")
    btc_terms = vocab.search_terms('BTC')
    for term in btc_terms:
        category = vocab.get_term_category(term)
        print(f"  {term} -> {category}")
    
    # 搜索美股相关术语
    print("\n搜索包含 'NASDAQ' 的术语:")
    nasdaq_terms = vocab.search_terms('NASDAQ')
    for term in nasdaq_terms:
        category = vocab.get_term_category(term)
        print(f"  {term} -> {category}")
    
    # 搜索金融指标
    print("\n搜索包含 'PE' 的术语:")
    pe_terms = vocab.search_terms('PE')
    for term in pe_terms:
        category = vocab.get_term_category(term)
        print(f"  {term} -> {category}")

def demo_tokenizer_expansion():
    """演示分词器扩展功能"""
    print("\n" + "="*60)
    print("分词器扩展演示")
    print("="*60)
    
    # 加载基础分词器
    model_name = 'hfl/chinese-bert-wwm-ext'
    print(f"加载基础分词器: {model_name}")
    
    try:
        base_tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"基础词汇表大小: {len(base_tokenizer)}")
        
        # 应用金融词汇扩展
        print("\n应用金融词汇扩展...")
        enhanced_tokenizer = create_financial_tokenizer(base_tokenizer, vocab_expansion=True)
        print(f"扩展后词汇表大小: {len(enhanced_tokenizer)}")
        
        # 测试新词汇的编码
        test_terms = [
            '[BTC]', '[ETH]', '[NASDAQ]', '[S&P500]', '[PE_RATIO]',
            '[FOMO]', '[HODL]', '[BULL_MARKET]', '[BEAR_MARKET]'
        ]
        
        print("\n测试新词汇编码:")
        for term in test_terms:
            if term in enhanced_tokenizer.get_vocab():
                token_id = enhanced_tokenizer.convert_tokens_to_ids(term)
                print(f"  {term} -> ID: {token_id}")
            else:
                print(f"  {term} -> 未找到")
        
        # 保存扩展后的分词器
        os.makedirs('demo_vocabulary', exist_ok=True)
        enhanced_tokenizer.save_pretrained('demo_vocabulary/enhanced_tokenizer')
        print(f"\n扩展后的分词器已保存到: demo_vocabulary/enhanced_tokenizer")
        
    except Exception as e:
        print(f"分词器扩展演示失败: {e}")

def demo_financial_text_processing():
    """演示金融文本处理"""
    print("\n" + "="*60)
    print("金融文本处理演示")
    print("="*60)
    
    # 示例金融文本
    financial_texts = [
        "BTC突破50000美元大关，市场情绪乐观，FOMO情绪高涨",
        "NASDAQ指数创历史新高，科技股表现强劲，BULL市场持续",
        "某公司PE_RATIO过高，投资者担忧估值泡沫",
        "ETH2.0升级成功，质押收益提升，DEFI生态繁荣",
        "市场出现恐慌情绪，VIX指数飙升，投资者寻求避险"
    ]
    
    try:
        # 加载扩展后的分词器
        enhanced_tokenizer = AutoTokenizer.from_pretrained('demo_vocabulary/enhanced_tokenizer')
        
        print("使用扩展词汇表处理金融文本:")
        for i, text in enumerate(financial_texts, 1):
            print(f"\n{i}. 原文: {text}")
            
            # 分词
            tokens = enhanced_tokenizer.tokenize(text)
            print(f"   分词结果: {tokens}")
            
            # 编码
            encoding = enhanced_tokenizer(
                text,
                truncation=True,
                padding=True,
                max_length=128,
                return_tensors='pt'
            )
            
            print(f"   输入ID长度: {len(encoding['input_ids'][0])}")
            
            # 检查是否包含特殊词汇
            special_tokens = []
            for token in tokens:
                if token.startswith('[') and token.endswith(']'):
                    special_tokens.append(token)
            
            if special_tokens:
                print(f"   识别到的特殊词汇: {special_tokens}")
            else:
                print("   未识别到特殊词汇")
                
    except Exception as e:
        print(f"金融文本处理演示失败: {e}")

def demo_vocabulary_export():
    """演示词汇表导出功能"""
    print("\n" + "="*60)
    print("词汇表导出演示")
    print("="*60)
    
    vocab = FinancialVocabulary()
    
    # 导出完整词汇表
    os.makedirs('demo_vocabulary', exist_ok=True)
    vocab.export_vocabulary('demo_vocabulary/complete_vocabulary.json')
    
    # 导出分类词汇表
    export_data = {
        'stock_terms': vocab.get_stock_specific_tokens(),
        'crypto_terms': vocab.get_crypto_specific_tokens(),
        'financial_indicators': vocab.get_financial_indicator_tokens()
    }
    
    with open('demo_vocabulary/categorized_vocabulary.json', 'w', encoding='utf-8') as f:
        import json
        json.dump(export_data, f, ensure_ascii=False, indent=2)
    
    print("词汇表导出完成:")
    print("  - 完整词汇表: demo_vocabulary/complete_vocabulary.json")
    print("  - 分类词汇表: demo_vocabulary/categorized_vocabulary.json")

def main():
    """主演示函数"""
    print("金融词汇扩展系统演示")
    print("="*60)
    
    try:
        # 词汇管理演示
        demo_vocabulary_management()
        
        # 分词器扩展演示
        demo_tokenizer_expansion()
        
        # 金融文本处理演示
        demo_financial_text_processing()
        
        # 词汇表导出演示
        demo_vocabulary_export()
        
        print("\n" + "="*60)
        print("演示完成!")
        print("\n生成的文件:")
        print("  - demo_vocabulary/enhanced_tokenizer/ - 扩展后的分词器")
        print("  - demo_vocabulary/complete_vocabulary.json - 完整词汇表")
        print("  - demo_vocabulary/categorized_vocabulary.json - 分类词汇表")
        
        print("\n下一步操作:")
        print("1. 运行 python enhanced_train_model.py 开始增强训练")
        print("2. 使用扩展后的模型进行金融文本情感分析")
        print("3. 根据需要添加更多专业术语到词汇表")
        
    except Exception as e:
        logger.error(f"演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
