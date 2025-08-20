#!/bin/bash

echo "股票消息情感分析系统 - Linux部署包打包脚本"
echo "=========================================="

# 设置版本号
VERSION=$(date +%Y%m%d_%H%M%S)
PACKAGE_NAME="sentiment-analysis-linux-${VERSION}"

echo "创建部署包: ${PACKAGE_NAME}"

# 创建临时目录
TEMP_DIR="/tmp/${PACKAGE_NAME}"
mkdir -p "${TEMP_DIR}"

# 复制必要的文件
echo "复制核心文件..."
cp -r *.py "${TEMP_DIR}/"
cp -r *.sh "${TEMP_DIR}/"
cp -r *.ini "${TEMP_DIR}/"
cp -r *.md "${TEMP_DIR}/"
cp -r *.sql "${TEMP_DIR}/"

# 创建目录结构
echo "创建目录结构..."
mkdir -p "${TEMP_DIR}/models"
mkdir -p "${TEMP_DIR}/enhanced_models"
mkdir -p "${TEMP_DIR}/vocabulary"
mkdir -p "${TEMP_DIR}/logs"

# 创建启动脚本
echo "创建启动脚本..."
cat > "${TEMP_DIR}/start_training.sh" << 'EOF'
#!/bin/bash

echo "启动股票消息情感分析训练..."
echo "================================"

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "错误: Python3未安装"
    exit 1
fi

# 检查依赖
echo "检查依赖包..."
python3 -c "
import sys
packages = ['torch', 'transformers', 'pandas', 'numpy', 'sklearn', 'jieba', 'pymysql', 'sqlalchemy']
missing = []
for pkg in packages:
    try:
        if pkg == 'sklearn':
            import sklearn
        else:
            __import__(pkg)
    except ImportError:
        missing.append(pkg)

if missing:
    print(f'缺少依赖包: {missing}')
    print('请先运行: ./deploy_linux.sh')
    sys.exit(1)
else:
    print('✓ 所有依赖包已安装')
"

if [ $? -ne 0 ]; then
    exit 1
fi

# 检查配置文件
if [ ! -f "config.ini" ]; then
    echo "错误: 配置文件config.ini不存在"
    exit 1
fi

echo "✓ 环境检查通过"
echo ""

# 开始训练
echo "开始训练模型..."
echo "训练日志将保存到: training.log"
echo "使用 Ctrl+C 停止训练，或使用 nohup 后台运行"

python3 enhanced_train_model.py
EOF

# 创建停止脚本
cat > "${TEMP_DIR}/stop_training.sh" << 'EOF'
#!/bin/bash

echo "停止股票消息情感分析训练..."

# 查找并停止训练进程
PIDS=$(ps aux | grep "enhanced_train_model.py" | grep -v grep | awk '{print $2}')

if [ -z "$PIDS" ]; then
    echo "没有找到运行中的训练进程"
else
    echo "找到训练进程: $PIDS"
    echo "正在停止..."
    echo "$PIDS" | xargs kill -TERM
    
    # 等待进程结束
    sleep 5
    
    # 强制停止仍在运行的进程
    REMAINING=$(ps aux | grep "enhanced_train_model.py" | grep -v grep | awk '{print $2}')
    if [ -n "$REMAINING" ]; then
        echo "强制停止剩余进程: $REMAINING"
        echo "$REMAINING" | xargs kill -KILL
    fi
    
    echo "✓ 训练已停止"
fi
EOF

# 创建状态检查脚本
cat > "${TEMP_DIR}/check_status.sh" << 'EOF'
#!/bin/bash

echo "股票消息情感分析系统状态检查"
echo "=============================="

# 检查Python进程
echo "1. 训练进程状态:"
TRAINING_PIDS=$(ps aux | grep "enhanced_train_model.py" | grep -v grep)
if [ -n "$TRAINING_PIDS" ]; then
    echo "✓ 训练进程正在运行:"
    echo "$TRAINING_PIDS"
else
    echo "✗ 没有训练进程在运行"
fi

echo ""

# 检查模型文件
echo "2. 模型文件状态:"
if [ -d "models" ] && [ "$(ls -A models)" ]; then
    echo "✓ 基础模型目录:"
    ls -la models/
else
    echo "✗ 基础模型目录为空或不存在"
fi

if [ -d "enhanced_models" ] && [ "$(ls -A enhanced_models)" ]; then
    echo "✓ 增强模型目录:"
    ls -la enhanced_models/
else
    echo "✗ 增强模型目录为空或不存在"
fi

echo ""

# 检查日志文件
echo "3. 日志文件状态:"
if [ -f "training.log" ]; then
    echo "✓ 训练日志文件存在"
    echo "最后10行日志:"
    tail -10 training.log
else
    echo "✗ 训练日志文件不存在"
fi

echo ""

# 检查系统资源
echo "4. 系统资源状态:"
echo "内存使用:"
free -h

echo ""
echo "磁盘使用:"
df -h .

echo ""
echo "CPU使用:"
top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1
EOF

# 设置执行权限
chmod +x "${TEMP_DIR}/start_training.sh"
chmod +x "${TEMP_DIR}/stop_training.sh"
chmod +x "${TEMP_DIR}/check_status.sh"

# 创建部署说明
cat > "${TEMP_DIR}/DEPLOY_README.txt" << 'EOF'
股票消息情感分析系统 - Linux部署说明
====================================

快速部署步骤:
1. 解压部署包: tar -xzf sentiment-analysis-linux-*.tar.gz
2. 进入目录: cd sentiment-analysis-linux-*
3. 运行安装脚本: ./deploy_linux.sh
4. 配置数据库: 编辑 config.ini 文件
5. 启动训练: ./start_training.sh

常用命令:
- 启动训练: ./start_training.sh
- 停止训练: ./stop_training.sh
- 检查状态: ./check_status.sh
- 后台运行: nohup ./start_training.sh > training.log 2>&1 &

注意事项:
- 确保服务器有足够的CPU和内存资源
- 确保数据库连接正常
- 建议使用screen或tmux进行会话管理
EOF

# 打包
echo "创建压缩包..."
cd /tmp
tar -czf "${PACKAGE_NAME}.tar.gz" "${PACKAGE_NAME}"

# 移动到当前目录
mv "${PACKAGE_NAME}.tar.gz" "${PWD}/"

# 清理临时文件
rm -rf "${TEMP_DIR}"

echo ""
echo "🎉 部署包创建完成!"
echo "文件名: ${PACKAGE_NAME}.tar.gz"
echo "位置: ${PWD}/${PACKAGE_NAME}.tar.gz"
echo ""
echo "部署包包含以下文件:"
echo "- 核心Python代码"
echo "- 安装脚本 (deploy_linux.sh)"
echo "- 启动脚本 (start_training.sh)"
echo "- 停止脚本 (stop_training.sh)"
echo "- 状态检查脚本 (check_status.sh)"
echo "- 配置文件模板 (config.ini)"
echo "- 数据库架构文件 (database_schema.sql)"
echo "- 详细说明文档"
echo ""
echo "现在可以将此文件上传到Linux服务器进行部署!"
