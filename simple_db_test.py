#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
简化的数据库连接测试脚本
专注于核心功能测试
"""

import configparser
import pymysql
import pandas as pd
from sqlalchemy import create_engine
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_database_connection():
    """测试数据库连接"""
    print("=" * 60)
    print("数据库连接测试")
    print("=" * 60)
    
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
        print()
        
        # 测试PyMySQL连接
        print("1. 测试PyMySQL连接...")
        try:
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
                print(f"  MySQL版本: {version[0]}")
            
            connection.close()
            pymysql_success = True
            
        except Exception as e:
            print(f"✗ PyMySQL连接失败: {e}")
            pymysql_success = False
        
        # 测试SQLAlchemy连接
        print("\n2. 测试SQLAlchemy连接...")
        try:
            connection_string = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}?charset={charset}"
            engine = create_engine(connection_string)
            
            # 使用pandas测试连接
            df = pd.read_sql("SELECT 1 as test", engine)
            print("✓ SQLAlchemy连接成功")
            
            engine.dispose()
            sqlalchemy_success = True
            
        except Exception as e:
            print(f"✗ SQLAlchemy连接失败: {e}")
            sqlalchemy_success = False
        
        # 测试数据表
        print("\n3. 测试数据表...")
        try:
            table_name = config['DATA']['table_name']
            title_column = config['DATA']['title_column']
            description_column = config['DATA']['description_column']
            label_column = config['DATA']['label_column']
            
            print(f"  表名: {table_name}")
            print(f"  标题列: {title_column}")
            print(f"  描述列: {description_column}")
            print(f"  标签列: {label_column}")
            
            # 使用PyMySQL测试表
            connection = pymysql.connect(
                host=host, port=port, user=user, password=password,
                database=database, charset=charset
            )
            
            with connection.cursor() as cursor:
                # 检查表是否存在
                cursor.execute(f"SHOW TABLES LIKE '{table_name}'")
                table_exists = cursor.fetchone()
                
                if table_exists:
                    print(f"✓ 表 {table_name} 存在")
                    
                    # 获取数据统计
                    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                    total_count = cursor.fetchone()[0]
                    print(f"  总数据量: {total_count}")
                    
                    # 获取已标注数据统计
                    cursor.execute(f"SELECT COUNT(*) FROM {table_name} WHERE {label_column} IS NOT NULL AND {label_column} != ''")
                    labeled_count = cursor.fetchone()[0]
                    print(f"  已标注数据: {labeled_count}")
                    
                    # 获取未标注数据统计
                    cursor.execute(f"SELECT COUNT(*) FROM {table_name} WHERE {label_column} IS NULL OR {label_column} = ''")
                    unlabeled_count = cursor.fetchone()[0]
                    print(f"  未标注数据: {unlabeled_count}")
                    
                    # 显示情感分布
                    if labeled_count > 0:
                        cursor.execute(f"SELECT {label_column}, COUNT(*) FROM {table_name} WHERE {label_column} IS NOT NULL AND {label_column} != '' GROUP BY {label_column}")
                        sentiment_dist = cursor.fetchall()
                        print(f"  情感分布:")
                        for sentiment, count in sentiment_dist:
                            print(f"    {sentiment}: {count}")
                    
                    # 显示示例数据
                    if unlabeled_count > 0:
                        print(f"\n  未标注数据示例:")
                        cursor.execute(f"SELECT {title_column}, {description_column} FROM {table_name} WHERE {label_column} IS NULL OR {label_column} = '' LIMIT 3")
                        examples = cursor.fetchall()
                        for i, (title, desc) in enumerate(examples, 1):
                            print(f"    {i}. 标题: {title}")
                            if desc:
                                print(f"       描述: {desc[:100]}...")
                            print()
                    
                    table_success = True
                    
                else:
                    print(f"✗ 表 {table_name} 不存在")
                    table_success = False
            
            connection.close()
            
        except Exception as e:
            print(f"✗ 数据表测试失败: {e}")
            table_success = False
        
        # 总结
        print("\n" + "=" * 60)
        print("测试结果总结:")
        print(f"PyMySQL连接: {'✓ 成功' if pymysql_success else '✗ 失败'}")
        print(f"SQLAlchemy连接: {'✓ 成功' if sqlalchemy_success else '✗ 失败'}")
        print(f"数据表测试: {'✓ 成功' if table_success else '✗ 失败'}")
        
        if pymysql_success and table_success:
            print("\n🎉 核心功能测试通过！")
            print("现在可以运行自动标注脚本:")
            print("python3 smart_label.py")
            return True
        else:
            print("\n⚠️  部分测试失败，请检查配置")
            return False
            
    except Exception as e:
        print(f"\n❌ 测试脚本执行失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    success = test_database_connection()
    
    if not success:
        print("\n🔧 故障排除建议:")
        print("1. 检查 config.ini 中的数据库配置")
        print("2. 确保数据库服务器可访问")
        print("3. 检查用户名和密码是否正确")
        print("4. 确保数据库和表存在")

if __name__ == "__main__":
    main()
