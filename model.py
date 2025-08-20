import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, AutoConfig
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class SentimentClassifier(nn.Module):
    def __init__(self, model_name: str, num_labels: int, dropout: float = 0.1):
        super(SentimentClassifier, self).__init__()
        
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
        
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        outputs = (logits,)
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            outputs = (loss,) + outputs
        
        return outputs

class SentimentModel:
    def __init__(self, model_name: str, num_labels: int, device: str = None):
        self.model_name = model_name
        self.num_labels = num_labels
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"使用设备: {self.device}")
        
        self.model = None
        self.tokenizer = None
        self.label_mapping = {}
        
    def load_model(self, model_path: str = None):
        try:
            if model_path:
                self.model = SentimentClassifier.from_pretrained(model_path)
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                logger.info(f"从本地路径加载模型: {model_path}")
            else:
                self.model = SentimentClassifier(self.model_name, self.num_labels)
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                logger.info(f"加载预训练模型: {self.model_name}")
            
            self.model.to(self.device)
            self.model.eval()
            
            if hasattr(self.model, 'config') and hasattr(self.model.config, 'label2id'):
                self.label_mapping = self.model.config.label2id
            else:
                self.label_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}
            
            logger.info("模型加载成功")
            return True
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            return False
    
    def save_model(self, save_path: str):
        try:
            if self.model and self.tokenizer:
                self.model.save_pretrained(save_path)
                self.tokenizer.save_pretrained(save_path)
                logger.info(f"模型保存到: {save_path}")
                return True
            else:
                logger.error("模型未初始化，无法保存")
                return False
                
        except Exception as e:
            logger.error(f"模型保存失败: {e}")
            return False
    
    def predict(self, text: str, return_probs: bool = False) -> Dict:
        if not self.model or not self.tokenizer:
            logger.error("模型未加载")
            return {}
        
        try:
            inputs = self.tokenizer(
                text,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors='pt'
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs[0]
                probs = F.softmax(logits, dim=-1)
                predicted_class = torch.argmax(logits, dim=-1).item()
            
            label = self.label_mapping.get(predicted_class, f"class_{predicted_class}")
            
            result = {
                'text': text,
                'predicted_label': label,
                'confidence': probs[0][predicted_class].item()
            }
            
            if return_probs:
                result['probabilities'] = {
                    self.label_mapping.get(i, f"class_{i}"): probs[0][i].item()
                    for i in range(self.num_labels)
                }
            
            return result
            
        except Exception as e:
            logger.error(f"预测失败: {e}")
            return {}
    
    def batch_predict(self, texts: List[str], batch_size: int = 16) -> List[Dict]:
        if not self.model or not self.tokenizer:
            logger.error("模型未加载")
            return []
        
        results = []
        
        try:
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                inputs = self.tokenizer(
                    batch_texts,
                    truncation=True,
                    padding=True,
                    max_length=512,
                    return_tensors='pt'
                )
                
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs[0]
                    probs = F.softmax(logits, dim=-1)
                    predicted_classes = torch.argmax(logits, dim=-1)
                
                for j, (text, pred_class) in enumerate(zip(batch_texts, predicted_classes)):
                    label = self.label_mapping.get(pred_class.item(), f"class_{pred_class.item()}")
                    confidence = probs[j][pred_class].item()
                    
                    results.append({
                        'text': text,
                        'predicted_label': label,
                        'confidence': confidence
                    })
                
                logger.info(f"已处理 {min(i + batch_size, len(texts))}/{len(texts)} 条文本")
        
        except Exception as e:
            logger.error(f"批量预测失败: {e}")
        
        return results
