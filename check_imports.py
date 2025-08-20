#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
导入兼容性检查脚本
检查所有必要的模块导入是否正常
"""

import sys
import importlib

def check_import(module_name, import_name=None, fallback=None):
    """检查模块导入"""
    try:
        if import_name:
            module = importlib.import_module(module_name)
            if hasattr(module, import_name):
                print(f"✓ {module_name}.{import_name} 导入成功")
                return True
            else:
                print(f"✗ {module_name}.{import_name} 不存在")
                if fallback:
                    print(f"  尝试使用 {fallback}")
                    try:
                        fallback_module = importlib.import_module(fallback)
                        if hasattr(fallback_module, import_name):
                            print(f"✓ {fallback}.{import_name} 可用")
                            return True
                        else:
                            print(f"✗ {fallback}.{import_name} 也不存在")
                    except ImportError:
                        print(f"✗ {fallback} 模块导入失败")
                return False
        else:
            importlib.import_module(module_name)
            print(f"✓ {module_name} 导入成功")
            return True
    except ImportError as e:
        print(f"✗ {module_name} 导入失败: {e}")
        return False

def main():
    """主检查函数"""
    print("股票消息情感分析系统 - 导入兼容性检查")
    print("=" * 60)
    
    # 检查基础模块
    print("\n1. 基础模块检查:")
    basic_modules = [
        'torch', 'pandas', 'numpy', 'sklearn', 'jieba', 
        'pymysql', 'sqlalchemy', 'tqdm', 'matplotlib', 'seaborn'
    ]
    
    basic_success = 0
    for module in basic_modules:
        if check_import(module):
            basic_success += 1
    
    # 检查transformers相关
    print("\n2. Transformers模块检查:")
    transformers_success = 0
    
    # 检查transformers基础功能
    if check_import('transformers'):
        transformers_success += 1
    
    # 检查具体功能
    transformers_features = [
        ('transformers', 'AutoTokenizer'),
        ('transformers', 'AutoModel'),
        ('transformers', 'AutoConfig'),
        ('transformers', 'get_linear_schedule_with_warmup')
    ]
    
    for module, feature in transformers_features:
        if check_import(module, feature):
            transformers_success += 1
    
    # 检查AdamW
    print("\n3. 优化器检查:")
    optimizer_success = 0
    
    # 检查torch.optim.AdamW
    if check_import('torch.optim', 'AdamW'):
        optimizer_success += 1
        print("  使用 torch.optim.AdamW")
    else:
        print("  尝试从transformers导入AdamW...")
        if check_import('transformers', 'AdamW'):
            optimizer_success += 1
            print("  使用 transformers.AdamW")
        else:
            print("  ⚠️  AdamW导入失败")
    
    # 检查自定义模块
    print("\n4. 自定义模块检查:")
    custom_modules = [
        'database', 'data_processor', 'model', 'trainer', 
        'enhanced_trainer', 'financial_vocab', 'data_augmentation'
    ]
    
    custom_success = 0
    for module in custom_modules:
        if check_import(module):
            custom_success += 1
    
    # 总结
    print("\n" + "=" * 60)
    print("检查结果总结:")
    print(f"基础模块: {basic_success}/{len(basic_modules)} 通过")
    print(f"Transformers: {transformers_success}/{len(transformers_features) + 1} 通过")
    print(f"优化器: {optimizer_success}/1 通过")
    print(f"自定义模块: {custom_success}/{len(custom_modules)} 通过")
    
    total_tests = len(basic_modules) + len(transformers_features) + 1 + 1 + len(custom_modules)
    total_passed = basic_success + transformers_success + optimizer_success + custom_success
    
    print(f"\n总体结果: {total_passed}/{total_tests} 通过")
    
    if total_passed == total_tests:
        print("🎉 所有导入检查通过！系统可以正常运行")
        return True
    else:
        print("⚠️  部分导入检查失败，请安装缺失的模块")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
