#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import (
    get_linear_schedule_with_warmup, 
    AutoTokenizer,
    AutoModel,
    AutoConfig
)
from torch.optim import AdamW
from sklearn.metrics import classification_report, accuracy_score, f1_score
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import logging
import os
import json

logger = logging.getLogger(__name__)

class EnhancedSentimentTrainer:
    def __init__(self, model, tokenizer, device, config):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.config = config
        
        # 训练参数
        self.learning_rate = float(config['MODEL']['learning_rate'])
        self.epochs = int(config['MODEL']['epochs'])
        self.batch_size = int(config['MODEL']['batch_size'])
        self.max_length = int(config['MODEL']['max_length'])
        
        # 微调参数
        self.domain_adaptation = config.getboolean('FINETUNE', 'domain_adaptation', fallback=False)
        self.vocab_expansion = config.getboolean('FINETUNE', 'vocab_expansion', fallback=False)
        self.freeze_bert_layers = config.getint('FINETUNE', 'freeze_bert_layers', fallback=0)
        self.gradient_accumulation_steps = config.getint('FINETUNE', 'gradient_accumulation_steps', fallback=1)
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
    def setup_model_for_finetuning(self):
        """设置模型进行微调"""
        logger.info("设置模型进行微调...")
        
        # 冻结部分BERT层（可选）
        if self.freeze_bert_layers > 0:
            self._freeze_bert_layers()
        
        # 领域适应微调设置
        if self.domain_adaptation:
            self._setup_domain_adaptation()
        
        # 词汇扩展设置
        if self.vocab_expansion:
            self._setup_vocab_expansion()
        
        # 设置不同的学习率
        self._setup_differential_learning_rates()
        
        logger.info("微调设置完成")
    
    def _freeze_bert_layers(self):
        """冻结BERT的前几层"""
        logger.info(f"冻结BERT前 {self.freeze_bert_layers} 层")
        
        for name, param in self.model.bert.named_parameters():
            if 'layer.' in name:
                layer_num = int(name.split('.')[2])
                if layer_num < self.freeze_bert_layers:
                    param.requires_grad = False
                    logger.info(f"冻结层: {name}")
    
    def _setup_domain_adaptation(self):
        """设置领域适应微调"""
        logger.info("设置领域适应微调")
        
        # 为金融领域相关层设置更高的学习率
        for name, param in self.model.bert.named_parameters():
            if any(keyword in name for keyword in ['pooler', 'classifier']):
                param.requires_grad = True
                logger.info(f"领域适应层: {name}")
    
    def _setup_vocab_expansion(self):
        """设置词汇扩展"""
        logger.info("设置词汇扩展")
        
        # 扩展词汇表
        special_tokens = self._get_financial_special_tokens()
        num_added = self.tokenizer.add_special_tokens(special_tokens)
        
        if num_added > 0:
            # 调整模型嵌入层大小
            self.model.bert.resize_token_embeddings(len(self.tokenizer))
            logger.info(f"添加了 {num_added} 个特殊词汇")
    
    def _get_financial_special_tokens(self) -> Dict[str, str]:
        """获取金融领域特殊词汇"""
        return {
            'additional_special_tokens': [
                # 美股相关
                '[STOCK]', '[NASDAQ]', '[NYSE]', '[S&P500]', '[DOW]',
                '[BULL]', '[BEAR]', '[SHORT]', '[LONG]', '[OPTION]',
                '[CALL]', '[PUT]', '[STRIKE]', '[EXPIRY]', '[VOLATILITY]',
                
                # 币圈相关
                '[CRYPTO]', '[BTC]', '[ETH]', '[ALTCOIN]', '[DEFI]',
                '[NFT]', '[MINING]', '[STAKING]', '[YIELD]', '[LIQUIDITY]',
                '[GAS]', '[BLOCKCHAIN]', '[SMART_CONTRACT]', '[DAO]',
                
                # 金融指标
                '[PE_RATIO]', '[PB_RATIO]', '[ROE]', '[ROA]', '[DEBT_RATIO]',
                '[CASH_FLOW]', '[REVENUE]', '[PROFIT]', '[LOSS]', '[DIVIDEND]',
                
                # 市场情绪
                '[FOMO]', '[FUD]', '[MOON]', '[DUMP]', '[PUMP]',
                '[HODL]', '[DIAMOND_HANDS]', '[PAPER_HANDS]'
            ]
        }
    
    def _setup_differential_learning_rates(self):
        """设置不同层的学习率"""
        logger.info("设置分层学习率")
        
        # 不同层组的学习率
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.bert.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': 0.01,
                'lr': self.learning_rate * 0.1  # BERT层使用较低学习率
            },
            {
                'params': [p for n, p in self.model.bert.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
                'lr': self.learning_rate * 0.1
            },
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if 'classifier' in n],
                'weight_decay': 0.01,
                'lr': self.learning_rate  # 分类头使用原始学习率
            }
        ]
        
        self.optimizer_grouped_parameters = optimizer_grouped_parameters
    
    def create_data_loader(self, texts: List[str], labels: List[int], 
                          batch_size: int = None) -> DataLoader:
        """创建数据加载器"""
        if batch_size is None:
            batch_size = self.batch_size
            
        # 使用扩展后的词汇表进行编码
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        dataset = TensorDataset(
            encodings['input_ids'],
            encodings['attention_mask'],
            torch.tensor(labels)
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True
        )
        
        return dataloader
    
    def train_epoch(self, train_loader: DataLoader, optimizer, scheduler) -> Tuple[float, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        progress_bar = tqdm(train_loader, desc="训练中")
        
        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch[0].to(self.device)
            attention_mask = batch[1].to(self.device)
            labels = batch[2].to(self.device)
            
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs[0]
            
            # 梯度累积
            if self.gradient_accumulation_steps > 1:
                loss = loss / self.gradient_accumulation_steps
                loss.backward()
                
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            # 统计
            total_loss += loss.item() * self.gradient_accumulation_steps
            logits = outputs[1]
            preds = torch.argmax(logits, dim=-1)
            correct_predictions += (preds == labels).sum().item()
            total_predictions += labels.size(0)
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{correct_predictions/total_predictions:.4f}'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct_predictions / total_predictions
        
        return avg_loss, accuracy
    
    def evaluate(self, val_loader: DataLoader) -> Tuple[float, float, Dict]:
        """评估模型"""
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device)
                labels = batch[2].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs[0]
                logits = outputs[1]
                
                total_loss += loss.item()
                preds = torch.argmax(logits, dim=-1)
                correct_predictions += (preds == labels).sum().item()
                total_predictions += labels.size(0)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct_predictions / total_predictions
        
        metrics = {
            'accuracy': accuracy,
            'f1_macro': f1_score(all_labels, all_preds, average='macro'),
            'f1_weighted': f1_score(all_labels, all_preds, average='weighted'),
            'classification_report': classification_report(all_labels, all_preds, output_dict=True)
        }
        
        return avg_loss, accuracy, metrics
    
    def train(self, train_texts: List[str], train_labels: List[int],
              val_texts: List[str], val_labels: List[int],
              save_dir: str = 'models') -> Dict:
        """完整训练流程"""
        logger.info("开始增强微调训练...")
        
        # 设置微调参数
        self.setup_model_for_finetuning()
        
        os.makedirs(save_dir, exist_ok=True)
        
        train_loader = self.create_data_loader(train_texts, train_labels)
        val_loader = self.create_data_loader(val_texts, val_labels)
        
        # 使用分层学习率
        if hasattr(self, 'optimizer_grouped_parameters'):
            optimizer = AdamW(self.optimizer_grouped_parameters)
        else:
            optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        
        total_steps = len(train_loader) * self.epochs // self.gradient_accumulation_steps
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=total_steps * 0.1,  # 10% warmup
            num_training_steps=total_steps
        )
        
        best_val_accuracy = 0
        
        for epoch in range(self.epochs):
            logger.info(f"Epoch {epoch + 1}/{self.epochs}")
            
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, scheduler)
            val_loss, val_acc, val_metrics = self.evaluate(val_loader)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            logger.info(f"训练 - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
            logger.info(f"验证 - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
            logger.info(f"F1 (Macro): {val_metrics['f1_macro']:.4f}")
            
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                best_model_path = os.path.join(save_dir, 'best_model')
                self.model.save_pretrained(best_model_path)
                self.tokenizer.save_pretrained(best_model_path)
                logger.info(f"保存最佳模型到: {best_model_path}")
        
        final_model_path = os.path.join(save_dir, 'final_model')
        self.model.save_pretrained(final_model_path)
        self.tokenizer.save_pretrained(final_model_path)
        logger.info(f"保存最终模型到: {final_model_path}")
        
        self.plot_training_curves(save_dir)
        
        return {
            'best_val_accuracy': best_val_accuracy,
            'final_val_accuracy': val_acc,
            'final_metrics': val_metrics,
            'train_history': {
                'losses': self.train_losses,
                'accuracies': self.train_accuracies,
                'val_losses': self.val_losses,
                'val_accuracies': self.val_accuracies
            }
        }
    
    def plot_training_curves(self, save_dir: str):
        """绘制训练曲线"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        ax1.plot(self.train_losses, label='训练损失')
        ax1.plot(self.val_losses, label='验证损失')
        ax1.set_title('训练和验证损失')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(self.train_accuracies, label='训练准确率')
        ax2.plot(self.val_accuracies, label='验证准确率')
        ax2.set_title('训练和验证准确率')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("训练曲线已保存")
