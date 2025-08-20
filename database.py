import pymysql
import pandas as pd
from sqlalchemy import create_engine
import configparser
from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, config_path: str = 'config.ini'):
        self.config = configparser.ConfigParser()
        self.config.read(config_path, encoding='utf-8')
        
        self.host = self.config['DATABASE']['host']
        self.port = int(self.config['DATABASE']['port'])
        self.user = self.config['DATABASE']['user']
        self.password = self.config['DATABASE']['password']
        self.database = self.config['DATABASE']['database']
        self.charset = self.config['DATABASE']['charset']
        
        self.table_name = self.config['DATA']['table_name']
        self.title_column = self.config['DATA']['title_column']
        self.description_column = self.config['DATA']['description_column']
        self.label_column = self.config['DATA']['label_column']
        self.confidence_column = self.config['DATA']['confidence_column']
        self.id_column = self.config['DATA']['id_column']
        
        self.engine = None
        self.connection = None
        
        # 自动连接数据库
        self.connect()
        
    def connect(self):
        try:
            connection_string = f"mysql+pymysql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}?charset={self.charset}"
            self.engine = create_engine(connection_string)
            
            self.connection = pymysql.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database=self.database,
                charset=self.charset
            )
            
            logger.info("数据库连接成功")
            return True
            
        except Exception as e:
            logger.error(f"数据库连接失败: {e}")
            return False
    
    def disconnect(self):
        if self.connection:
            self.connection.close()
        if self.engine:
            self.engine.dispose()
        logger.info("数据库连接已关闭")
    
    def get_unlabeled_data(self, limit: int = None) -> pd.DataFrame:
        try:
            if not self.engine:
                logger.error("数据库引擎未初始化")
                return pd.DataFrame()
                
            query = f"""
                SELECT {self.id_column}, {self.title_column}, {self.description_column}
                FROM {self.table_name}
                WHERE {self.label_column} IS NULL OR {self.label_column} = ''
                ORDER BY {self.id_column} DESC
            """
            
            if limit:
                query += f" LIMIT {limit}"
            
            df = pd.read_sql(query, self.engine)
            logger.info(f"获取到 {len(df)} 条未标注数据")
            return df
            
        except Exception as e:
            logger.error(f"获取未标注数据失败: {e}")
            return pd.DataFrame()
    
    def get_labeled_data(self, limit: int = None) -> pd.DataFrame:
        try:
            if not self.engine:
                logger.error("数据库引擎未初始化")
                return pd.DataFrame()
                
            query = f"""
                SELECT {self.id_column}, {self.title_column}, {self.description_column}, {self.label_column}
                FROM {self.table_name}
                WHERE {self.label_column} IS NOT NULL AND {self.label_column} != ''
                ORDER BY {self.id_column} DESC
            """
            
            if limit:
                query += f" LIMIT {limit}"
            
            df = pd.read_sql(query, self.engine)
            logger.info(f"获取到 {len(df)} 条已标注数据")
            return df
            
        except Exception as e:
            logger.error(f"获取已标注数据失败: {e}")
            return pd.DataFrame()
    
    def update_sentiment(self, id_value: int, sentiment: str, confidence: float = None):
        try:
            with self.connection.cursor() as cursor:
                if confidence is not None:
                    sql = f"""
                        UPDATE {self.table_name}
                        SET {self.label_column} = %s, {self.confidence_column} = %s
                        WHERE {self.id_column} = %s
                    """
                    cursor.execute(sql, (sentiment, confidence, id_value))
                    logger.info(f"更新ID {id_value} 的情感标注为: {sentiment}, 置信度: {confidence}")
                else:
                    sql = f"""
                        UPDATE {self.table_name}
                        SET {self.label_column} = %s
                        WHERE {self.id_column} = %s
                    """
                    cursor.execute(sql, (sentiment, id_value))
                    logger.info(f"更新ID {id_value} 的情感标注为: {sentiment}")
                
                self.connection.commit()
                
        except Exception as e:
            logger.error(f"更新情感标注失败: {e}")
            self.connection.rollback()
    
    def batch_update_sentiment(self, updates: List[Dict[str, Any]]):
        try:
            with self.connection.cursor() as cursor:
                sql = f"""
                    UPDATE {self.table_name}
                    SET {self.label_column} = %s, {self.confidence_column} = %s
                    WHERE {self.id_column} = %s
                """
                
                for update in updates:
                    sentiment = update.get('sentiment', '')
                    confidence = update.get('confidence', 0.0)
                    id_value = update['id']
                    
                    cursor.execute(sql, (sentiment, confidence, id_value))
                
                self.connection.commit()
                logger.info(f"批量更新 {len(updates)} 条情感标注")
                
        except Exception as e:
            logger.error(f"批量更新情感标注失败: {e}")
            self.connection.rollback()
    
    def get_statistics(self) -> Dict[str, int]:
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(f"SELECT COUNT(*) FROM {self.table_name}")
                total = cursor.fetchone()[0]
                
                cursor.execute(f"""
                    SELECT COUNT(*) FROM {self.table_name}
                    WHERE {self.label_column} IS NOT NULL AND {self.label_column} != ''
                """)
                labeled = cursor.fetchone()[0]
                
                unlabeled = total - labeled
                
                cursor.execute(f"""
                    SELECT {self.label_column}, COUNT(*) 
                    FROM {self.table_name}
                    WHERE {self.label_column} IS NOT NULL AND {self.label_column} != ''
                    GROUP BY {self.label_column}
                """)
                
                sentiment_stats = {}
                for row in cursor.fetchall():
                    sentiment_stats[row[0]] = row[1]
                
                cursor.execute(f"""
                    SELECT 
                        AVG({self.confidence_column}) as avg_confidence,
                        MIN({self.confidence_column}) as min_confidence,
                        MAX({self.confidence_column}) as max_confidence
                    FROM {self.table_name}
                    WHERE {self.confidence_column} > 0
                """)
                
                confidence_stats = cursor.fetchone()
                
                return {
                    'total': total,
                    'labeled': labeled,
                    'unlabeled': unlabeled,
                    'sentiment_distribution': sentiment_stats,
                    'confidence_stats': {
                        'average': float(confidence_stats[0]) if confidence_stats[0] else 0.0,
                        'min': float(confidence_stats[1]) if confidence_stats[1] else 0.0,
                        'max': float(confidence_stats[2]) if confidence_stats[2] else 0.0
                    }
                }
                
        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {}
