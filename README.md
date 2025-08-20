# 股票消息情感分析系统

这是一个基于深度学习的股票消息情感分析系统，能够自动识别中文金融新闻中的利好、利空和中性消息。

## 功能特点

- 🚀 基于BERT的中文文本情感分析
- 📊 支持MySQL数据库集成
- 🔄 批量离线标注功能
- 📈 完整的模型训练流程
- 🎯 专门针对金融领域优化
- 📝 详细的训练报告和可视化
- 🆕 支持标题和描述字段组合分析
- 🎯 支持置信度字段更新

## 系统架构

```
ai_odaily_news/
├── config.ini              # 配置文件
├── requirements.txt         # 依赖包列表
├── database.py             # 数据库管理模块
├── data_processor.py       # 数据预处理模块
├── model.py                # 情感分析模型
├── trainer.py              # 模型训练器
├── enhanced_trainer.py     # 增强版训练器（支持微调）
├── financial_vocab.py      # 金融词汇扩展模块
├── train_model.py          # 模型训练主程序
├── enhanced_train_model.py # 增强版训练程序
├── predict_batch.py        # 批量预测和标注
├── utils.py                # 工具函数
├── demo.py                 # 系统功能演示
├── demo_vocab_expansion.py # 词汇扩展演示
├── setup.py                # 安装和设置脚本
├── test_system.py          # 系统测试脚本
├── database_schema.sql     # 数据库表结构
└── README.md               # 项目说明
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 配置说明

1. 修改 `config.ini` 文件中的数据库连接信息：
```ini
[DATABASE]
host = localhost
port = 3306
user = your_username
password = your_password
database = your_database
charset = utf8mb4
```

2. 确保数据库中存在相应的数据表结构（参考 `database_schema.sql` 文件）

## 数据表结构

系统支持以下表结构：

```sql
CREATE TABLE `src_odaily_news_info` (
  `id` bigint NOT NULL COMMENT '新闻ID',
  `title` varchar(500) COLLATE utf8mb4_unicode_ci NOT NULL COMMENT '新闻标题',
  `description` text COLLATE utf8mb4_unicode_ci COMMENT '新闻描述',
  `images` json DEFAULT NULL COMMENT '图片数组',
  `is_important` tinyint(1) DEFAULT '0' COMMENT '是否重要',
  `publish_timestamp` bigint NOT NULL COMMENT '发布时间戳',
  `tags` json DEFAULT NULL COMMENT '标签数组',
  `news_url_type` varchar(50) COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '新闻链接类型',
  `news_url` text COLLATE utf8mb4_unicode_ci COMMENT '新闻链接',
  `is_collection` tinyint(1) DEFAULT NULL COMMENT '是否收藏',
  `is_like` tinyint(1) DEFAULT NULL COMMENT '是否点赞',
  `sentiment` varchar(20) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT '' COMMENT 'positive/negative/neutral',
  `confidence` decimal(10,2) DEFAULT '0.00' COMMENT '预测置信度',
  `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `updated_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='odaily快讯新闻表';
```

### 关键字段说明

- **`title`**: 新闻标题，用于情感分析的主要文本
- **`description`**: 新闻描述，补充信息，与标题组合分析
- **`sentiment`**: 情感标签，存储预测结果（positive/negative/neutral）
- **`confidence`**: 预测置信度，0.00-1.00之间的数值

## 使用方法

### 1. 模型训练

#### 基础训练
```bash
python train_model.py
```

#### 增强版训练（推荐）
```bash
python enhanced_train_model.py
```

训练过程会：
- 从数据库获取已标注的训练数据
- 自动组合标题和描述字段
- 自动分割训练集、验证集和测试集
- 训练BERT模型并进行验证
- 保存最佳模型和最终模型
- 生成训练报告和可视化图表

### 2. 批量预测和标注

```bash
python predict_batch.py
```

预测过程会：
- 从数据库获取未标注的数据
- 组合标题和描述进行文本分析
- 使用训练好的模型进行批量预测
- 自动更新数据库中的情感标签和置信度
- 输出预测统计信息

### 3. 词汇扩展演示

```bash
python demo_vocab_expansion.py
```

## 情感标签说明

- `positive`: 利好消息（如：业绩增长、股价上涨、获得合同等）
- `negative`: 利空消息（如：业绩下滑、股价下跌、政策收紧等）
- `neutral`: 中性消息（如：市场震荡、观望情绪、数据公布等）

## 模型说明

- **基础模型**: 使用中文BERT预训练模型（hfl/chinese-bert-wwm-ext）
- **分类头**: 在BERT基础上添加分类层，支持多分类任务
- **优化器**: AdamW优化器，带学习率调度
- **损失函数**: CrossEntropyLoss
- **评估指标**: 准确率、F1分数、分类报告

## 微调功能

系统支持多种微调策略：

### 1. 领域适应微调
- 在金融领域数据上继续训练
- 学习金融术语和表达方式
- 保持预训练知识的同时适应新领域

### 2. 词汇扩展
- 添加美股、币圈等专业术语
- 支持自定义金融词汇
- 动态扩展模型词汇表

### 3. 分层学习率
- BERT底层使用较低学习率（保持预训练知识）
- 分类头使用较高学习率（快速适应新任务）

## 训练参数

可在 `config.ini` 中调整：

```ini
[MODEL]
model_name = hfl/chinese-bert-wwm-ext
max_length = 512
batch_size = 16
learning_rate = 2e-5
epochs = 5
train_split = 0.8
random_seed = 42

[FINETUNE]
domain_adaptation = true
vocab_expansion = true
freeze_bert_layers = 0
gradient_accumulation_steps = 1
use_differential_lr = true
warmup_ratio = 0.1
weight_decay = 0.01
```

## 输出文件

训练完成后会生成：
- `models/best_model/` - 最佳模型文件
- `models/final_model/` - 最终模型文件
- `enhanced_models/` - 增强版模型文件（如果使用增强训练）
- `vocabulary/` - 词汇扩展相关文件
- `models/label_mapping.json` - 标签映射文件
- `models/training_curves.png` - 训练曲线图
- `training.log` - 训练日志

## 注意事项

1. **数据质量**: 确保训练数据质量，标注准确的情感标签
2. **数据平衡**: 尽量保持各类别样本数量平衡
3. **硬件要求**: 建议使用GPU进行训练，CPU训练速度较慢
4. **内存要求**: 根据batch_size和max_length调整，避免内存不足
5. **字段配置**: 确保config.ini中的字段名与数据库表结构一致

## 扩展功能

- 支持自定义情感类别
- 可集成其他预训练模型
- 支持增量训练和模型微调
- 可添加置信度阈值过滤
- 支持多语言扩展
- 支持标题和描述字段的灵活组合

## 常见问题

**Q: 训练时出现内存不足错误？**
A: 减小batch_size或max_length参数

**Q: 模型预测效果不好？**
A: 检查训练数据质量，增加训练样本，调整学习率和训练轮数

**Q: 如何添加新的情感类别？**
A: 修改数据库中的sentiment字段，重新训练模型

**Q: 如何调整标题和描述的组合方式？**
A: 修改data_processor.py中的combine_title_description方法

## 技术支持

如有问题，请检查：
1. 依赖包是否正确安装
2. 数据库连接是否正常
3. 数据格式是否符合要求
4. 配置文件中的字段名是否正确
5. 日志文件中的错误信息

## 许可证

本项目仅供学习和研究使用。
