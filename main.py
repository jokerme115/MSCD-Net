import os
import random
import argparse
import numpy as np

import torch

# 直接从模块导入，而不是通过utils
from utils.config import load_config
from utils.batch_training import train_and_evaluate


def main():
    # 仅支持通过命令行参数指定配置文件路径
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default='config.yaml',
                        help="path to config file")
    parser.add_argument("--use_oversampling", action="store_true",
                        help="whether to use oversampling")
    parser.add_argument("--use_attention", action="store_true",
                        help="whether to use attention mechanism")
    args = parser.parse_args()
    
    # Load config from YAML file
    opts = load_config(args.config_file)
    
    # 添加过采样和注意力机制配置（仅当命令行参数启用时才覆盖配置文件）
    if args.use_oversampling:
        opts['use_oversampling'] = args.use_oversampling
    if args.use_attention:
        opts['use_attention'] = args.use_attention
    
    # 设置随机种子
    torch.manual_seed(opts['random_seed'])
    np.random.seed(opts['random_seed'])
    random.seed(opts['random_seed'])
    
    if opts['dataset'].lower() == 'cassava_rot':
        opts['num_classes'] = 2

    os.environ['CUDA_VISIBLE_DEVICES'] = str(opts['gpu_id'])  # 转换为字符串
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # 使用批量训练模式
    print("使用批量训练模式...")
    train_and_evaluate(opts)


if __name__ == '__main__':
    main()