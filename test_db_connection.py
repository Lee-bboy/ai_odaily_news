#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
数据库连接测试脚本
测试数据库连接和基本操作
"""

import configparser
import pymysql
from sqlalchemy import create_engine
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_direct_connection():
    """测试直接PyMySQL连接"""
    print("=" * 50)
    print("测试直接PyMySQL连接...")
    
    try:
        # 读取配置
        config = configparser.ConfigParser()
        config.read('config.ini', encoding='utf-8')
        
        host = config['DATABASE']['host']
        port = int(config['DATABASE']['port'])
        user = config['DATABASE']['user']
        password = config['DATABASE']['password']
        database = config['DATABASE']['database']
        charset = config['DATABASE']['charset']
        
        print(f"连接信息:")
        print(f"  主机: {host}")
        print(f"  端口: {port}")
        print(f"  用户: {user}")
        print(f"  数据库: {database}")
        print(f"  字符集: {charset}")
        
        # 建立连接
        connection = pymysql.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
            charset=charset
        )
        
        print("✓ PyMySQL连接成功")
        
        # 测试查询
        with connection.cursor() as cursor:
            cursor.execute("SELECT VERSION()")
            version = cursor.fetchone()
            print(f"MySQL版本: {version[0]}")
            
            # 测试表是否存在
            table_name = config['DATA']['table_name']
            cursor.execute(f"SHOW TABLES LIKE '{table_name}'")
            table_exists = cursor.fetchone()
            
            if table_exists:
                print(f"✓ 表 {table_name} 存在")
                
                # 获取表结构
                cursor.execute(f"DESCRIBE {table_name}")
                columns = cursor.fetchall()
                print(f"表结构:")
                for col in columns:
                    print(f"  {col[0]} - {col[1]} - {col[2]}")
                
                # 获取数据统计
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                total_count = cursor.fetchone()[0]
                print(f"总数据量: {total_count}")
                
                # 获取已标注数据统计
                label_column = config['DATA']['label_column']
                cursor.execute(f"SELECT COUNT(*) FROM {table_name} WHERE {label_column} IS NOT NULL AND {label_column} != ''")
                labeled_count = cursor.fetchone()[0]
                print(f"已标注数据: {labeled_count}")
                
                # 获取未标注数据统计
                cursor.execute(f"SELECT COUNT(*) FROM {table_name} WHERE {label_column} IS NULL OR {label_column} = ''")
                unlabeled_count = cursor.fetchone()[0]
                print(f"未标注数据: {unlabeled_count}")
                
                # 显示情感分布
                if labeled_count > 0:
                    cursor.execute(f"SELECT {label_column}, COUNT(*) FROM {table_name} WHERE {label_column} IS NOT NULL AND {label_column} != '' GROUP BY {label_column}")
                    sentiment_dist = cursor.fetchall()
                    print(f"情感分布:")
                    for sentiment, count in sentiment_dist:
                        print(f"  {sentiment}: {count}")
                
            else:
                print(f"✗ 表 {table_name} 不存在")
        
        connection.close()
        print("✓ 连接测试完成")
        return True
        
    except Exception as e:
        print(f"✗ PyMySQL连接失败: {e}")
        return False

def test_sqlalchemy_connection():
    """测试SQLAlchemy连接"""
    print("\n" + "=" * 50)
    print("测试SQLAlchemy连接...")
    
    try:
        # 读取配置
        config = configparser.ConfigParser()
        config.read('config.ini', encoding='utf-8')
        
        host = config['DATABASE']['host']
        port = int(config['DATABASE']['port'])
        user = config['DATABASE']['user']
        password = config['DATABASE']['password']
        database = config['DATABASE']['database']
        charset = config['DATABASE']['charset']
        
        # 建立连接
        connection_string = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}?charset={charset}"
        engine = create_engine(connection_string)
        
        # 测试连接
        with engine.connect() as conn:
            result = conn.execute("SELECT VERSION()")
            version = result.fetchone()
            print(f"✓ SQLAlchemy连接成功")
            print(f"MySQL版本: {version[0]}")
        
        engine.dispose()
        return True
        
    except Exception as e:
        print(f"✗ SQLAlchemy连接失败: {e}")
        return False

def test_data_queries():
    """测试数据查询"""
    print("\n" + "=" * 50)
    print("测试数据查询...")
    
    try:
        # 读取配置
        config = configparser.ConfigParser()
        config.read('config.ini', encoding='utf-8')
        
        host = config['DATABASE']['host']
        port = int(config['DATABASE']['port'])
        user = config['DATABASE']['user']
        password = config['DATABASE']['password']
        database = config['DATABASE']['database']
        charset = config['DATABASE']['charset']
        
        # 建立连接
        connection_string = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}?charset={charset}"
        engine = create_engine(connection_string)
        
        table_name = config['DATA']['table_name']
        title_column = config['DATA']['title_column']
        description_column = config['DATA']['description_column']
        label_column = config['DATA']['label_column']
        
        # 测试查询未标注数据
        query = f"""
            SELECT {title_column}, {description_column}
            FROM {table_name}
            WHERE {label_column} IS NULL OR {label_column} = ''
            LIMIT 5
        """
        
        df = pd.read_sql(query, engine)
        print(f"✓ 查询未标注数据成功，获取 {len(df)} 条")
        
        if not df.empty:
            print("前3条未标注数据:")
            for i, row in df.head(3).iterrows():
                print(f"  {i+1}. 标题: {row[title_column]}")
                print(f"     描述: {row[description_column][:100]}...")
                print()
        
        # 测试查询已标注数据
        query = f"""
            SELECT {title_column}, {description_column}, {label_column}
            FROM {table_name}
            WHERE {label_column} IS NOT NULL AND {label_column} != ''
            LIMIT 5
        """
        
        df = pd.read_sql(query, engine)
        print(f"✓ 查询已标注数据成功，获取 {len(df)} 条")
        
        if not df.empty:
            print("前3条已标注数据:")
            for i, row in df.head(3).iterrows():
                print(f"  {i+1}. 标题: {row[title_column]}")
                print(f"     描述: {row[description_column][:100]}...")
                print(f"     情感: {row[label_column]}")
                print()
        
        engine.dispose()
        return True
        
    except Exception as e:
        print(f"✗ 数据查询失败: {e}")
        return False

def main():
    """主函数"""
    print("数据库连接测试")
    print("=" * 60)
    
    # 测试直接连接
    direct_success = test_direct_connection()
    
    # 测试SQLAlchemy连接
    sqlalchemy_success = test_sqlalchemy_connection()
    
    # 测试数据查询
    query_success = test_data_queries()
    
    # 总结
    print("\n" + "=" * 60)
    print("测试结果总结:")
    print(f"PyMySQL连接: {'✓ 成功' if direct_success else '✗ 失败'}")
    print(f"SQLAlchemy连接: {'✓ 成功' if sqlalchemy_success else '✗ 失败'}")
    print(f"数据查询: {'✓ 成功' if query_success else '✗ 失败'}")
    
    if direct_success and sqlalchemy_success and query_success:
        print("\n🎉 所有测试通过！数据库连接正常")
        print("现在可以运行自动标注脚本:")
        print("python3 smart_label.py")
    else:
        print("\n⚠️  部分测试失败，请检查数据库配置")
        print("检查 config.ini 中的数据库连接信息")

if __name__ == "__main__":
    main()
