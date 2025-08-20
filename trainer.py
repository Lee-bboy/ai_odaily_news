import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import classification_report, accuracy_score, f1_score
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import logging
import os

logger = logging.getLogger(__name__)

class SentimentTrainer:
    def __init__(self, model, tokenizer, device, config):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.config = config
        
        self.learning_rate = float(config['MODEL']['learning_rate'])
        self.epochs = int(config['MODEL']['epochs'])
        self.batch_size = int(config['MODEL']['batch_size'])
        self.max_length = int(config['MODEL']['max_length'])
        
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
    def create_data_loader(self, texts: List[str], labels: List[int], 
                          batch_size: int = None) -> DataLoader:
        if batch_size is None:
            batch_size = self.batch_size
            
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
        self.model.train()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        progress_bar = tqdm(train_loader, desc="训练中")
        
        for batch in progress_bar:
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
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            correct_predictions += (preds == labels).sum().item()
            total_predictions += labels.size(0)
            
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{correct_predictions/total_predictions:.4f}'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct_predictions / total_predictions
        
        return avg_loss, accuracy
    
    def evaluate(self, val_loader: DataLoader) -> Tuple[float, float, Dict]:
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
        logger.info("开始训练模型...")
        
        os.makedirs(save_dir, exist_ok=True)
        
        train_loader = self.create_data_loader(train_texts, train_labels)
        val_loader = self.create_data_loader(val_texts, val_labels)
        
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        total_steps = len(train_loader) * self.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
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
