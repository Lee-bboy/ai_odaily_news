#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
股票消息情感分析系统安装和设置脚本
"""

import os
import sys
import subprocess
import configparser
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def install_dependencies():
    """安装项目依赖"""
    print("正在安装项目依赖...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("依赖安装完成!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"依赖安装失败: {e}")
        return False

def create_directories():
    """创建必要的目录"""
    directories = ['models', 'logs', 'data']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"创建目录: {directory}")

def setup_config():
    """设置配置文件"""
    config_file = 'config.ini'
    
    if os.path.exists(config_file):
        print(f"配置文件 {config_file} 已存在")
        return True
    
    print("创建配置文件...")
    
    config = configparser.ConfigParser()
    
    # 数据库配置
    config['DATABASE'] = {
        'host': 'localhost',
        'port': '3306',
        'user': 'your_username',
        'password': 'your_password',
        'database': 'your_database',
        'charset': 'utf8mb4'
    }
    
    # 模型配置
    config['MODEL'] = {
        'model_name': 'hfl/chinese-bert-wwm-ext',
        'max_length': '512',
        'batch_size': '16',
        'learning_rate': '2e-5',
        'epochs': '5',
        'train_split': '0.8',
        'random_seed': '42'
    }
    
    # 数据配置
    config['DATA'] = {
        'table_name': 'stock_news',
        'text_column': 'content',
        'label_column': 'sentiment',
        'id_column': 'id'
    }
    
    # 写入配置文件
    with open(config_file, 'w', encoding='utf-8') as f:
        config.write(f)
    
    print(f"配置文件 {config_file} 已创建，请根据实际情况修改数据库连接信息")
    return True

def create_database_schema():
    """创建数据库表结构"""
    print("生成数据库表结构SQL...")
    
    from utils import create_sample_database_schema
    sql = create_sample_database_schema()
    
    sql_file = 'database_schema.sql'
    with open(sql_file, 'w', encoding='utf-8') as f:
        f.write(sql)
    
    print(f"数据库表结构SQL已保存到: {sql_file}")
    print("请在MySQL中执行此SQL文件创建数据表")

def run_demo():
    """运行演示程序"""
    print("运行系统演示...")
    try:
        subprocess.check_call([sys.executable, "demo.py"])
    except subprocess.CalledProcessError as e:
        print(f"演示程序运行失败: {e}")

def main():
    """主设置函数"""
    print("股票消息情感分析系统 - 安装和设置")
    print("="*50)
    
    # 检查Python版本
    if sys.version_info < (3, 7):
        print("错误: 需要Python 3.7或更高版本")
        return
    
    print(f"Python版本: {sys.version}")
    
    # 安装依赖
    if not install_dependencies():
        print("依赖安装失败，请手动安装")
        return
    
    # 创建目录
    create_directories()
    
    # 设置配置
    setup_config()
    
    # 创建数据库表结构
    create_database_schema()
    
    print("\n" + "="*50)
    print("安装和设置完成!")
    print("\n下一步操作:")
    print("1. 修改 config.ini 中的数据库连接信息")
    print("2. 在MySQL中执行 database_schema.sql 创建数据表")
    print("3. 运行 python demo.py 查看系统演示")
    print("4. 运行 python train_model.py 开始训练模型")
    print("\n详细说明请查看 README.md")

if __name__ == "__main__":
    main()
