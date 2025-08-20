-- 基于 src_odaily_news_info 表结构的数据库表
-- 用于股票消息情感分析系统

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
  PRIMARY KEY (`id`),
  KEY `idx_publish_timestamp` (`publish_timestamp`),
  KEY `idx_is_important` (`is_important`),
  KEY `idx_created_at` (`created_at`),
  KEY `idx_sentiment` (`sentiment`),
  KEY `idx_confidence` (`confidence`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='odaily快讯新闻表';

-- 插入示例数据用于测试
INSERT INTO `src_odaily_news_info` (`id`, `title`, `description`, `is_important`, `publish_timestamp`, `sentiment`, `confidence`) VALUES
(1, 'BTC突破50000美元大关', '比特币价格突破50000美元重要关口，市场情绪乐观，投资者信心增强', 1, 1642233600, 'positive', 0.95),
(2, '某公司业绩不及预期', '公司发布财报显示业绩下滑，股价可能承压，投资者担忧情绪上升', 1, 1642233600, 'negative', 0.88),
(3, '市场震荡调整', '今日市场出现震荡调整，投资者观望情绪浓厚，成交量略有下降', 0, 1642233600, 'neutral', 0.75),
(4, '科技股表现强劲', '科技板块今日表现强劲，多只股票涨停，市场看好科技股前景', 1, 1642233600, 'positive', 0.92),
(5, '行业政策收紧', '监管部门发布新政策，行业面临挑战，相关公司股价承压', 1, 1642233600, 'negative', 0.85);

-- 创建索引优化查询性能
CREATE INDEX `idx_sentiment_confidence` ON `src_odaily_news_info` (`sentiment`, `confidence`);
CREATE INDEX `idx_title_description` ON `src_odaily_news_info` (`title`(100), `description`(100));
