# 股票消息情感分析系统 - Linux部署指南

## 🐧 为什么选择Linux服务器？

### 优势
- **依赖包安装简单**: 预编译包多，无需本地编译
- **性能更好**: 通常配置更高，支持GPU加速
- **环境稳定**: 长期运行更稳定，不会遇到macOS特有问题
- **资源管理**: 系统资源管理更高效

### 适用场景
- 生产环境部署
- 大规模数据处理
- 长期运行任务
- GPU加速训练

## 🚀 部署方案

### 直接部署（推荐）

#### 1. 上传代码到服务器
```bash
# 在本地打包代码
tar -czf sentiment-analysis.tar.gz ./*

# 上传到服务器
scp sentiment-analysis.tar.gz user@server:/path/to/deploy/

# 在服务器上解压
cd /path/to/deploy/
tar -xzf sentiment-analysis.tar.gz
```

#### 2. 运行安装脚本
```bash
# 给脚本添加执行权限
chmod +x deploy_linux.sh

# 运行安装脚本
./deploy_linux.sh
```

#### 3. 测试配置
```bash
python3 test_new_structure.py
```

#### 4. 开始训练
```bash
# 前台运行
python3 enhanced_train_model.py

# 后台运行（推荐）
nohup python3 enhanced_train_model.py > training.log 2>&1 &

# 查看日志
tail -f training.log
```

## 📋 服务器要求

### 最低配置
- **CPU**: 4核心
- **内存**: 8GB RAM
- **存储**: 20GB可用空间
- **系统**: Ubuntu 18.04+ / CentOS 7+

### 推荐配置
- **CPU**: 8核心以上
- **内存**: 16GB+ RAM
- **存储**: 50GB+ SSD
- **GPU**: NVIDIA GPU（可选，用于加速训练）
- **系统**: Ubuntu 20.04+ / CentOS 8+

## 🔧 配置说明

### 1. 数据库配置
确保 `config.ini` 中的数据库连接信息正确：
```ini
[DATABASE]
host = your_server_ip
port = 3306
user = pump
password = Pump#20250206
database = pump_web
charset = utf8mb4
```

### 2. 模型配置
```ini
[MODEL]
model_name = hfl/chinese-bert-wwm-ext
max_length = 512
batch_size = 16  # 根据内存调整
learning_rate = 2e-5
epochs = 5
```

### 3. 微调配置
```ini
[FINETUNE]
domain_adaptation = true
vocab_expansion = true
freeze_bert_layers = 0
gradient_accumulation_steps = 1
```

## 📊 运行监控

### 1. 查看训练进度
```bash
# 实时查看日志
tail -f training.log

# 查看GPU使用情况（如果有GPU）
nvidia-smi

# 查看系统资源
htop
```

### 2. 后台运行管理
```bash
# 查看后台进程
ps aux | grep python

# 停止训练
pkill -f enhanced_train_model.py

# 查看进程状态
jobs
```

## 🚨 常见问题

### 1. 样本数量不足
```bash
# 运行数据增强脚本
python3 data_augmentation.py

# 或者在训练前使用数据增强
from data_augmentation import DataAugmentor
augmentor = DataAugmentor()
augmented_data = augmentor.create_balanced_dataset(
    your_data, 'title', 'description', 'sentiment', 
    samples_per_class=20
)
```

### 2. 内存不足
```bash
# 调整batch_size
# 在config.ini中设置更小的batch_size
batch_size = 8  # 或更小
```

### 2. 磁盘空间不足
```bash
# 清理临时文件
rm -rf /tmp/*
rm -rf ~/.cache/pip

# 检查磁盘使用
df -h
```

### 3. 网络连接问题
```bash
# 测试数据库连接
mysql -h your_server_ip -u pump -p pump_web

# 检查防火墙设置
sudo ufw status
```

## 📈 性能优化

### 1. GPU加速（如果有NVIDIA GPU）
```bash
# 安装CUDA版本的PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. 多进程训练
```bash
# 在config.ini中设置
gradient_accumulation_steps = 4
```

### 3. 内存优化
```bash
# 调整batch_size和max_length
batch_size = 8
max_length = 256
```

## 🔄 更新和维护

### 1. 代码更新
```bash
# 停止服务
pkill -f enhanced_train_model.py

# 更新代码
git pull  # 如果使用Git
# 或重新上传代码包

# 重启服务
nohup python3 enhanced_train_model.py > training.log 2>&1 &
```

### 2. 模型备份
```bash
# 备份训练好的模型
tar -czf models_backup_$(date +%Y%m%d).tar.gz models/ enhanced_models/

# 定期清理旧模型
find models/ -name "*.pth" -mtime +30 -delete
```

## 📞 技术支持

如果遇到问题，请检查：
1. 系统日志: `journalctl -u docker`
2. 应用日志: `tail -f training.log`
3. 系统资源: `htop`, `df -h`
4. 网络连接: `ping`, `telnet`

## 🎯 总结

Linux服务器部署是生产环境的最佳选择，具有以下优势：
- 环境稳定，依赖包安装简单
- 性能更好，支持GPU加速
- 长期运行稳定，适合生产环境
- 资源管理高效，成本更低

直接部署方案简单可靠，适合大多数生产环境使用。
