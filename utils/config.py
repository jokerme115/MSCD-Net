"""
配置相关工具函数模块
包含配置文件加载、参数处理等
"""

import os
import yaml


def load_config(config_file):
    """从YAML配置文件加载参数"""
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"配置文件 {config_file} 不存在")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print(f"成功加载配置文件: {config_file}")
    return config


def merge_config_with_args(config, args):
    """合并配置文件和命令行参数"""
    # Merge config with command line arguments
    # Command line arguments take precedence over config file
    for key, value in config.items():
        if not hasattr(args, key) or getattr(args, key) is None:
            setattr(args, key, value)
        elif isinstance(getattr(args, key), dict) and isinstance(value, dict):
            # Merge dictionaries
            merged = value.copy()
            merged.update(getattr(args, key))
            setattr(args, key, merged)
        # For non-dict values, command line args take precedence, so we don't update
    
    return args