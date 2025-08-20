#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
金融领域词汇扩展模块
专门为美股和币圈情感分析添加专业术语
"""

import json
import os
from typing import Dict, List, Set, Tuple
import logging

logger = logging.getLogger(__name__)

class FinancialVocabulary:
    def __init__(self):
        """初始化金融词汇管理器"""
        self.stock_terms = self._load_stock_terms()
        self.crypto_terms = self._load_crypto_terms()
        self.financial_indicators = self._load_financial_indicators()
        self.market_sentiment = self._load_market_sentiment()
        
        # 合并所有词汇
        self.all_terms = {
            'stock': self.stock_terms,
            'crypto': self.crypto_terms,
            'indicators': self.financial_indicators,
            'sentiment': self.market_sentiment
        }
    
    def _load_stock_terms(self) -> Dict[str, List[str]]:
        """加载美股相关术语"""
        return {
            'exchanges': [
                'NASDAQ', 'NYSE', 'AMEX', 'OTC', 'TSX', 'LSE', 'TSE', 'HKEX'
            ],
            'indices': [
                'S&P500', 'SPX', 'DOW', 'DJI', 'NASDAQ100', 'NDX', 'RUSSELL2000', 'RUT',
                'VIX', 'VIX指数', '恐慌指数', '波动率指数'
            ],
            'trading_terms': [
                'BULL', 'BEAR', 'SHORT', 'LONG', 'OPTION', 'CALL', 'PUT',
                'STRIKE', 'EXPIRY', 'VOLATILITY', 'IV', 'DELTA', 'GAMMA', 'THETA',
                'MARGIN', 'LEVERAGE', 'SHORT_SELLING', 'COVER', 'SQUEEZE'
            ],
            'market_conditions': [
                'BULL_MARKET', 'BEAR_MARKET', 'SIDEWAYS', 'CONSOLIDATION',
                'BREAKOUT', 'BREAKDOWN', 'SUPPORT', 'RESISTANCE', 'TREND',
                'REVERSAL', 'CORRECTION', 'CRASH', 'RALLY', 'BOUNCE'
            ],
            'volume_analysis': [
                'VOLUME', 'AVERAGE_VOLUME', 'VOLUME_SPIKE', 'LOW_VOLUME',
                'HIGH_VOLUME', 'VOLUME_CONFIRMATION', 'VOLUME_DIVERGENCE'
            ],
            'technical_analysis': [
                'MA', 'SMA', 'EMA', 'RSI', 'MACD', 'BOLLINGER_BANDS',
                'STOCHASTIC', 'WILLIAMS_R', 'ADX', 'ATR', 'FIBONACCI'
            ]
        }
    
    def _load_crypto_terms(self) -> Dict[str, List[str]]:
        """加载币圈相关术语"""
        return {
            'major_coins': [
                'BTC', 'ETH', 'BNB', 'ADA', 'SOL', 'DOT', 'AVAX', 'MATIC',
                'LINK', 'UNI', 'ATOM', 'FTM', 'NEAR', 'ALGO', 'VET'
            ],
            'defi_tokens': [
                'DEFI', 'YIELD_FARMING', 'LIQUIDITY_POOL', 'AMM', 'DEX',
                'CEX', 'STAKING', 'VALIDATOR', 'DELEGATOR', 'REWARDS',
                'APY', 'APR', 'TVL', 'TOTAL_VALUE_LOCKED'
            ],
            'nft_terms': [
                'NFT', 'PFP', 'GENERATIVE', 'RARITY', 'TRAITS', 'FLOOR_PRICE',
                'MINT', 'MINTING', 'REVEAL', 'WHITELIST', 'PRESALE'
            ],
            'blockchain_tech': [
                'BLOCKCHAIN', 'SMART_CONTRACT', 'GAS', 'GAS_LIMIT', 'GAS_PRICE',
                'CONSENSUS', 'PROOF_OF_WORK', 'PROOF_OF_STAKE', 'DELEGATED_POS',
                'SHARDING', 'LAYER2', 'ROLLUP', 'SIDECHAIN', 'BRIDGE'
            ],
            'crypto_trading': [
                'HODL', 'FOMO', 'FUD', 'MOON', 'LAMBO', 'REKT', 'WAGMI',
                'NGMI', 'APE_IN', 'DIAMOND_HANDS', 'PAPER_HANDS',
                'PUMP', 'DUMP', 'PUMP_AND_DUMP', 'WHALE', 'WHALE_WATCHING'
            ],
            'mining_staking': [
                'MINING', 'MINER', 'HASHRATE', 'DIFFICULTY', 'BLOCK_REWARD',
                'STAKING', 'VALIDATOR', 'DELEGATOR', 'SLASHING', 'COMMISSION'
            ]
        }
    
    def _load_financial_indicators(self) -> Dict[str, List[str]]:
        """加载金融指标术语"""
        return {
            'valuation_metrics': [
                'PE_RATIO', 'PB_RATIO', 'PS_RATIO', 'EV_EBITDA', 'ROE', 'ROA',
                'ROIC', 'DEBT_RATIO', 'CURRENT_RATIO', 'QUICK_RATIO'
            ],
            'financial_statements': [
                'REVENUE', 'SALES', 'GROSS_PROFIT', 'OPERATING_INCOME', 'EBIT',
                'EBITDA', 'NET_INCOME', 'EPS', 'DILUTED_EPS', 'BOOK_VALUE'
            ],
            'cash_flow': [
                'OPERATING_CASH_FLOW', 'INVESTING_CASH_FLOW', 'FINANCING_CASH_FLOW',
                'FREE_CASH_FLOW', 'FCF', 'CASH_FROM_OPERATIONS'
            ],
            'dividends': [
                'DIVIDEND', 'DIVIDEND_YIELD', 'DIVIDEND_PAYOUT_RATIO',
                'DIVIDEND_GROWTH', 'DIVIDEND_SAFETY'
            ],
            'growth_metrics': [
                'REVENUE_GROWTH', 'EPS_GROWTH', 'BOOK_VALUE_GROWTH',
                'COMPOUND_ANNUAL_GROWTH_RATE', 'CAGR'
            ]
        }
    
    def _load_market_sentiment(self) -> Dict[str, List[str]]:
        """加载市场情绪术语"""
        return {
            'positive_sentiment': [
                'BULLISH', 'OPTIMISTIC', 'CONFIDENT', 'EXCITED', 'HOPEFUL',
                'MOONSHOT', 'TO_THE_MOON', 'ROCKET', 'EXPLOSIVE', 'BREAKTHROUGH'
            ],
            'negative_sentiment': [
                'BEARISH', 'PESSIMISTIC', 'WORRIED', 'FEARFUL', 'PANIC',
                'CRASH', 'DUMP', 'REKT', 'DISASTER', 'CATASTROPHE'
            ],
            'neutral_sentiment': [
                'NEUTRAL', 'SIDEWAYS', 'CONSOLIDATION', 'RANGE_BOUND',
                'WAIT_AND_SEE', 'UNCERTAIN', 'MIXED_SIGNALS'
            ],
            'market_emotions': [
                'FOMO', 'FUD', 'GREED', 'FEAR', 'HOPE', 'DESPAIR',
                'EXCITEMENT', 'ANXIETY', 'CONFIDENCE', 'DOUBT'
            ]
        }
    
    def get_all_tokens(self) -> Dict[str, List[str]]:
        """获取所有特殊词汇"""
        return {
            'additional_special_tokens': self._flatten_terms()
        }
    
    def _flatten_terms(self) -> List[str]:
        """将嵌套的词汇结构展平"""
        flat_terms = []
        
        for category, subcategories in self.all_terms.items():
            if isinstance(subcategories, dict):
                for subcategory, terms in subcategories.items():
                    flat_terms.extend(terms)
            else:
                flat_terms.extend(subcategories)
        
        # 添加方括号包围
        return [f'[{term}]' for term in flat_terms]
    
    def get_stock_specific_tokens(self) -> List[str]:
        """获取美股特定词汇"""
        stock_tokens = []
        for subcategory, terms in self.stock_terms.items():
            stock_tokens.extend(terms)
        return [f'[{term}]' for term in stock_tokens]
    
    def get_crypto_specific_tokens(self) -> List[str]:
        """获取币圈特定词汇"""
        crypto_tokens = []
        for subcategory, terms in self.crypto_terms.items():
            crypto_tokens.extend(terms)
        return [f'[{term}]' for term in crypto_tokens]
    
    def get_financial_indicator_tokens(self) -> List[str]:
        """获取金融指标词汇"""
        indicator_tokens = []
        for subcategory, terms in self.financial_indicators.items():
            indicator_tokens.extend(terms)
        return [f'[{term}]' for term in indicator_tokens]
    
    def search_terms(self, query: str) -> List[str]:
        """搜索包含特定关键词的术语"""
        query = query.upper()
        matching_terms = []
        
        for category, subcategories in self.all_terms.items():
            if isinstance(subcategories, dict):
                for subcategory, terms in subcategories.items():
                    for term in terms:
                        if query in term.upper():
                            matching_terms.append(term)
            else:
                for term in subcategories:
                    if query in term.upper():
                        matching_terms.append(term)
        
        return matching_terms
    
    def get_term_category(self, term: str) -> str:
        """获取术语所属的类别"""
        term = term.upper()
        
        for category, subcategories in self.all_terms.items():
            if isinstance(subcategories, dict):
                for subcategory, terms in subcategories.items():
                    if term in [t.upper() for t in terms]:
                        return f"{category}.{subcategory}"
            else:
                if term in [t.upper() for t in subcategories]:
                    return category
        
        return "unknown"
    
    def export_vocabulary(self, file_path: str):
        """导出词汇表到JSON文件"""
        export_data = {
            'metadata': {
                'total_terms': len(self._flatten_terms()),
                'categories': list(self.all_terms.keys()),
                'description': '金融领域专业术语词汇表，包含美股、币圈、金融指标等'
            },
            'vocabulary': self.all_terms,
            'flat_tokens': self._flatten_terms()
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"词汇表已导出到: {file_path}")
    
    def print_vocabulary_summary(self):
        """打印词汇表摘要"""
        print("=" * 60)
        print("金融领域词汇表摘要")
        print("=" * 60)
        
        total_terms = 0
        for category, subcategories in self.all_terms.items():
            if isinstance(subcategories, dict):
                category_count = sum(len(terms) for terms in subcategories.values())
                print(f"{category}: {category_count} 个术语")
                total_terms += category_count
            else:
                print(f"{category}: {len(subcategories)} 个术语")
                total_terms += len(subcategories)
        
        print("-" * 60)
        print(f"总计: {total_terms} 个术语")
        print("=" * 60)

def create_financial_tokenizer(base_tokenizer, vocab_expansion: bool = True):
    """创建扩展的金融领域分词器"""
    if not vocab_expansion:
        return base_tokenizer
    
    financial_vocab = FinancialVocabulary()
    special_tokens = financial_vocab.get_all_tokens()
    
    # 添加特殊词汇到分词器
    num_added = base_tokenizer.add_special_tokens(special_tokens)
    
    if num_added > 0:
        logger.info(f"成功添加 {num_added} 个金融领域特殊词汇")
        logger.info(f"词汇表大小从 {len(base_tokenizer) - num_added} 扩展到 {len(base_tokenizer)}")
    
    return base_tokenizer

if __name__ == "__main__":
    # 测试词汇表
    vocab = FinancialVocabulary()
    vocab.print_vocabulary_summary()
    
    # 搜索示例
    print("\n搜索包含 'BTC' 的术语:")
    btc_terms = vocab.search_terms('BTC')
    for term in btc_terms:
        category = vocab.get_term_category(term)
        print(f"  {term} -> {category}")
    
    # 导出词汇表
    vocab.export_vocabulary('financial_vocabulary.json')
