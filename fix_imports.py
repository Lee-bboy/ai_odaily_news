#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动修复导入问题的脚本
修复transformers版本兼容性问题
"""

import os
import re
import shutil
from pathlib import Path

def backup_file(file_path):
    """备份文件"""
    backup_path = f"{file_path}.backup"
    if not os.path.exists(backup_path):
        shutil.copy2(file_path, backup_path)
        print(f"✓ 已备份: {backup_path}")
    return backup_path

def fix_file_imports(file_path):
    """修复文件中的导入问题"""
    print(f"\n修复文件: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        changes_made = False
        
        # 修复AdamW导入
        if 'from transformers import' in content and 'AdamW' in content:
            print("  修复AdamW导入...")
            
            # 从transformers导入中移除AdamW
            content = re.sub(
                r'from transformers import\s*\(([^)]*),\s*AdamW([^)]*)\)',
                r'from transformers import (\1\2)',
                content
            )
            content = re.sub(
                r'from transformers import\s*([^,]*),\s*AdamW',
                r'from transformers import \1',
                content
            )
            content = re.sub(
                r'from transformers import\s*AdamW,\s*([^,]*)\s*',
                r'from transformers import \1',
                content
            )
            
            # 添加torch.optim.AdamW导入
            if 'from torch.optim import AdamW' not in content:
                # 找到第一个import语句
                import_match = re.search(r'^import\s+torch', content, re.MULTILINE)
                if import_match:
                    insert_pos = import_match.end()
                    content = content[:insert_pos] + '\nfrom torch.optim import AdamW' + content[insert_pos:]
                else:
                    # 如果没有找到torch导入，在文件开头添加
                    content = 'from torch.optim import AdamW\n' + content
            
            changes_made = True
        
        # 修复其他可能的导入问题
        if 'from transformers import' in content:
            # 检查是否有其他已移动的导入
            deprecated_imports = {
                'BertTokenizer': 'AutoTokenizer',
                'BertModel': 'AutoModel',
                'BertConfig': 'AutoConfig'
            }
            
            for old, new in deprecated_imports.items():
                if old in content:
                    print(f"  更新导入: {old} -> {new}")
                    content = content.replace(old, new)
                    changes_made = True
        
        if changes_made:
            # 备份原文件
            backup_file(file_path)
            
            # 写入修复后的内容
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"  ✓ 修复完成")
            return True
        else:
            print(f"  ✓ 无需修复")
            return False
            
    except Exception as e:
        print(f"  ✗ 修复失败: {e}")
        return False

def main():
    """主修复函数"""
    print("股票消息情感分析系统 - 导入问题自动修复")
    print("=" * 60)
    
    # 需要修复的文件列表
    files_to_fix = [
        'enhanced_trainer.py',
        'trainer.py',
        'model.py',
        'enhanced_train_model.py',
        'train_model.py',
        'predict_batch.py'
    ]
    
    print("开始修复导入问题...")
    
    fixed_count = 0
    total_files = len(files_to_fix)
    
    for file_path in files_to_fix:
        if os.path.exists(file_path):
            if fix_file_imports(file_path):
                fixed_count += 1
        else:
            print(f"⚠️  文件不存在: {file_path}")
    
    print("\n" + "=" * 60)
    print(f"修复完成: {fixed_count}/{total_files} 个文件")
    
    if fixed_count > 0:
        print("\n🎉 导入问题修复完成！")
        print("现在可以尝试运行:")
        print("python3 enhanced_train_model.py")
        
        # 建议运行检查脚本
        print("\n建议运行导入检查:")
        print("python3 check_imports.py")
    else:
        print("\n⚠️  没有发现需要修复的问题")
    
    return fixed_count > 0

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
