"""
批量训练工具函数模块
包含多模型、多损失函数的批量训练功能
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# 注释掉引起问题的TensorBoard导入
# from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from torchvision import transforms
import json
import csv

import network
import utils
from utils.training import validate, save_ckpt
from utils.logging import log_training_progress
from utils.model_utils import get_loss_function
from utils.dataset_utils import get_dataset

# 创建一个虚拟的SummaryWriter类来避免错误
class SummaryWriter:
    def __init__(self, *args, **kwargs):
        pass
    
    def add_scalar(self, *args, **kwargs):
        pass
    
    def close(self):
        pass

def train_and_evaluate(opts):
    """训练并评估指定的模型和损失函数组合"""
    # 获取骨干网络和损失函数列表
    backbones = opts['backbones'] if isinstance(opts['backbones'], list) else [opts['backbones']]
    loss_functions = opts['loss_functions'] if isinstance(opts['loss_functions'], list) else [opts['loss_functions']]
    
    # 获取过采样和注意力机制配置
    use_oversample = opts.get('oversample', False)
    oversample_methods = opts.get('oversample_methods', ['none']) if use_oversample else ['none']
    # 如果只有一种过采样方法，确保它是列表形式
    if not isinstance(oversample_methods, list):
        oversample_methods = [oversample_methods]
        
    use_attention = opts.get('use_attention', False)
    
    print(f"过采样设置: {'启用' if use_oversample else '禁用'}")
    print(f"过采样方法列表: {oversample_methods}")
    print(f"注意力机制: {'启用' if use_attention else '禁用'}")
    
    # 创建实验根目录
    root_experiment_dir = os.path.join("experiments", opts['experiment_name'])
    os.makedirs(root_experiment_dir, exist_ok=True)
    
    # 存储所有实验结果的列表
    all_results = []
    
    # 遍历所有骨干网络、损失函数和过采样方法的组合
    for backbone in backbones:
        for loss_function in loss_functions:
            for oversample_method in oversample_methods:
                # 为每个组合创建独立的实验目录
                experiment_name = f"{opts['experiment_name']}_{backbone}_{loss_function}"
                if use_oversample and oversample_method != 'none':
                    experiment_name += f"_oversample_{oversample_method}"
                if use_attention:
                    experiment_name += "_attention"
                    
                experiment_dir = os.path.join(root_experiment_dir, experiment_name)
                os.makedirs(experiment_dir, exist_ok=True)
                
                # 在开始训练前提示当前组合
                print(f"\n{'='*50}")
                print(f"开始训练组合: {backbone} + {loss_function}")
                if use_oversample and oversample_method != 'none':
                    print(f"过采样方法: {oversample_method}")
                if use_attention:
                    print(f"注意力机制: 启用")
                print(f"实验目录: {experiment_dir}")
                if opts.get('verbose', False):
                    print(f"\n开始训练 - 骨干网络: {backbone}, 损失函数: {loss_function}, 过采样: {oversample_method}")
                
                # 创建TensorBoard日志目录
                log_dir = os.path.join(experiment_dir, 'logs')
                os.makedirs(log_dir, exist_ok=True)
                writer = SummaryWriter(log_dir=log_dir)
                
                # Setup device
                os.environ['CUDA_VISIBLE_DEVICES'] = str(opts['gpu_id'])  # 转换为字符串
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                if opts.get('verbose', False):
                    print(f"使用设备: {device}")

                # Setup dataloader
                # 传递过采样参数到数据集
                dataset_opts = opts.copy()
                dataset_opts['oversample'] = use_oversample and oversample_method != 'none'
                dataset_opts['oversample_method'] = oversample_method
                
                train_set, val_set = get_dataset(dataset_opts)
                train_loader = DataLoader(train_set, batch_size=opts['batch_size'], shuffle=True, num_workers=0,
                                        drop_last=True)
                val_loader = DataLoader(val_set, batch_size=opts['batch_size'], shuffle=False, num_workers=0)

                # Set up model
                model_map = {
                    'deeplabv3plus_resnet50': network.deeplabv3plus_resnet50,
                    'deeplabv3plus_resnet101': network.deeplabv3plus_resnet101,
                    'deeplabv3plus_resnet18': network.deeplabv3plus_resnet18,
                    'deeplabv3plus_resnet34': network.deeplabv3plus_resnet34,
                    'deeplabv3plus_resnet152': network.deeplabv3plus_resnet152,
                    'deeplabv3plus_mobilenet': network.deeplabv3plus_mobilenet,
                    'deeplabv3plus_xception': network.deeplabv3plus_xception,
                    'deeplabv3plus_hrnetv2_32': network.deeplabv3plus_hrnetv2_32,
                    'deeplabv3plus_hrnetv2_48': network.deeplabv3plus_hrnetv2_48
                }

                # 直接使用backbone变量（配置文件中已经包含了完整名称）
                model_name = backbone
                
                # 检查模型是否在model_map中
                if model_name not in model_map:
                    print(f"模型 {model_name} 不在支持的模型列表中")
                    print(f"支持的模型列表: {list(model_map.keys())}")
                    continue
                    
                # 设置output_stride参数
                output_stride = 4 if 'hrnetv2' in model_name else (8 if 'mobilenet' in model_name or 'xception' in model_name else 16)
                
                try:
                    # 对于HRNet，使用不同的output_stride，并设置pretrained=False
                    if 'hrnetv2' in model_name:
                        model = model_map[model_name](num_classes=opts['num_classes'], output_stride=4, pretrained_backbone=False, use_attention=use_attention)
                    elif 'mobilenet' in model_name:
                        # MobileNet使用默认的output_stride=8
                        model = model_map[model_name](num_classes=opts['num_classes'], output_stride=8, use_attention=use_attention)
                    else:
                        model = model_map[model_name](num_classes=opts['num_classes'], output_stride=16, use_attention=use_attention)
                except Exception as e:
                    print(f"模型 {model_name} 初始化失败: {str(e)}")
                    continue  # 继续下一个模型训练，而不是返回None

                utils.set_bn_momentum(model.backbone, momentum=0.01)
                
                # Set up metrics
                from metrics import StreamSegMetrics
                metrics = StreamSegMetrics(opts['num_classes'])

                # Set up optimizer
                # 确保学习率是浮点数类型
                lr = float(opts['lr'])
                optimizer = torch.optim.SGD(params=[
                    {'params': model.backbone.parameters(), 'lr': 0.1 * lr},
                    {'params': model.classifier.parameters(), 'lr': lr},
                ], lr=lr, momentum=0.9, weight_decay=1e-4, nesterov=True)
                
                # 设置学习率调度策略
                if opts['lr_policy'] == 'poly':
                    from utils.scheduler import PolyLR
                    scheduler = PolyLR(optimizer, opts['total_itrs'], power=0.9)
                elif opts['lr_policy'] == 'step':
                    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts['total_itrs'] // 3, gamma=0.1)
                elif opts['lr_policy'] == 'cosine':
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opts['total_itrs'], eta_min=1e-6)
                elif opts['lr_policy'] == 'cosine_with_warmup':
                    # 修复学习率调度器参数传递问题，确保参数类型正确
                    from utils.scheduler import CosineAnnealingWarmRestartsWithWarmup
                    # 确保T_0是正整数
                    T_0 = max(1, int(opts['total_itrs']//3))
                    scheduler = CosineAnnealingWarmRestartsWithWarmup(
                        optimizer, 
                        T_0=T_0, 
                        warmup_steps=int(opts['warmup_steps']), 
                        warmup_lr=float(opts['warmup_lr']))
                elif opts['lr_policy'] == 'reduce_on_plateau':
                    # 添加基于验证损失的自适应学习率调整
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, mode='min', factor=0.5, patience=5, verbose=True, min_lr=1e-7)
                else:
                    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts['total_itrs'] // 3, gamma=0.1)
                
                # 获取损失函数
                criterion = get_loss_function(loss_function)

                # 冻结骨干网络参数（如果指定）
                if opts['freeze_backbone'] > 0:
                    # freeze_backbone_parameters(model)
                    if opts.get('verbose', False):
                        print(f"冻结骨干网络前 {opts['freeze_backbone']} 个epoch的参数")

                # 加载检查点（如果存在）
                cur_itrs = 0
                best_score = 0.0
                cur_epochs = 0
                
                # Initialize Early Stopping
                from utils.training import EarlyStopping
                early_stopping = EarlyStopping(
                    patience=opts.get('early_stopping_patience', 10),
                    verbose=opts.get('early_stopping_verbose', True),
                    delta=opts.get('early_stopping_delta', 0.001),
                    save_path=os.path.join(experiment_dir, 'early_stop_best.pth'),
                    restore_best_weights=opts.get('early_stopping_restore_best_weights', True),
                    mode=opts.get('early_stopping_mode', 'max'),
                    avg_window=opts.get('early_stopping_avg_window', 3),
                    min_delta_patience=opts.get('early_stopping_min_delta_patience', 20)
                )
                
                # Setup visualization
                vis_sample_id = np.random.randint(0, len(val_loader), 1).item()
                
                # 训练循环
                interval_loss = 0
                pbar = tqdm(total=opts['total_itrs'], desc=f"训练 {backbone} + {loss_function} ({oversample_method})")
                pbar.update(cur_itrs)
                
                model = nn.DataParallel(model)
                model.to(device)
                model.train()
                while True: 
                    cur_epochs += 1
                    for (images, labels) in train_loader:
                        cur_itrs += 1
                        pbar.update(1)

                        images = images.to(device, dtype=torch.float32)
                        labels = labels.to(device, dtype=torch.long)

                        optimizer.zero_grad()
                        outputs = model(images)
                        
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()

                        np_loss = loss.detach().cpu().numpy()
                        interval_loss += np_loss
                        
                        if (cur_itrs) % opts['print_interval'] == 0:
                            # 记录训练日志
                            log_file = os.path.join(experiment_dir, 'training_log.csv')
                            # 确保目录存在
                            os.makedirs(os.path.dirname(log_file), exist_ok=True)
                            log_training_progress(log_file, cur_epochs, cur_itrs, interval_loss / opts['print_interval'])
                            print("Epoch %d, Itrs %d/%d, Loss=%f" %
                                (cur_epochs, cur_itrs, opts['total_itrs'], interval_loss / opts['print_interval']))
                            interval_loss = 0

                        if (cur_itrs) % opts['val_interval'] == 0:
                            save_ckpt_path = os.path.join(experiment_dir, 'latest_%s_%s_os%d.pth' %
                                                        (backbone, opts['dataset'], output_stride))
                            save_ckpt(save_ckpt_path, model=model, optimizer=optimizer, scheduler=scheduler, 
                                    best_score=best_score, cur_itrs=cur_itrs)
                            model.eval()
                            val_score = validate(
                                opts, model=model, val_loader=val_loader, metrics=metrics, device=device, criterion=criterion)
                            
                            # 记录验证日志
                            log_file = os.path.join(experiment_dir, 'validation_log.csv')
                            # 确保目录存在
                            os.makedirs(os.path.dirname(log_file), exist_ok=True)
                            log_training_progress(log_file, cur_epochs, cur_itrs, 0, val_score)
                            
                            # Check early stopping
                            early_stopping(val_score['Mean IoU'], model.module if hasattr(model, 'module') else model)
                            if early_stopping.early_stop:
                                print(f"Early stopping for {backbone} + {loss_function} ({oversample_method})")
                                pbar.close()
                                break
                            
                            if val_score['Mean IoU'] > best_score:  # save best model
                                best_score = val_score['Mean IoU']
                                best_ckpt_path = os.path.join(experiment_dir, 'best_%s_%s_os%d.pth' %
                                                            (backbone, opts['dataset'], output_stride))
                                save_ckpt(best_ckpt_path, model=model, optimizer=optimizer, scheduler=scheduler, 
                                        best_score=best_score, cur_itrs=cur_itrs)
                            
                            # 显示当前组合的性能
                            print(f"\n当前组合 {backbone} + {loss_function} ({oversample_method}) 验证结果:")
                            if use_attention:
                                print(f"  注意力机制: 启用")
                            print(f"  Overall Acc: {val_score['Overall Acc']:.4f}")
                            print(f"  Mean IoU: {val_score['Mean IoU']:.4f}")
                            print(f"  Class 1 IoU: {val_score['Class IoU'].get(1, 0):.4f}")
                            print(f"  Class 1 Sensitivity (Recall): {val_score['Class Sensitivity'].get(1, 0):.4f}")
                            print(f"  Class 1 Precision: {val_score['Class Precision'].get(1, 0):.4f}")
                            # 添加假阳性率输出
                            if 'Class FP Rate' in val_score:
                                print(f"  Class 1 FP Rate: {val_score['Class FP Rate'].get(1, 0):.4f}")
                            
                            model.train()
                        scheduler.step()

                        if cur_itrs >= opts['total_itrs']:
                            pbar.close()
                            break
                    if early_stopping.early_stop or cur_itrs >= opts['total_itrs']:
                        break
                
                # 记录实验结果
                result = {
                    'experiment_name': experiment_name,
                    'backbone': backbone,
                    'loss_function': loss_function,
                    'oversample_method': oversample_method,
                    'use_oversample': use_oversample and oversample_method != 'none',
                    'use_attention': use_attention,
                    'best_score': float(best_score),
                    'experiment_dir': experiment_dir,
                    'total_itrs': opts.get('total_itrs', 0),
                    'lr': opts.get('lr', 0.0),
                    'batch_size': opts.get('batch_size', 0),
                    'output_stride': output_stride
                }
                all_results.append(result)
    
    # 生成汇总文件 (CSV格式)
    summary_file = os.path.join(root_experiment_dir, 'experiment_summary.csv')
    if all_results:
        with open(summary_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = all_results[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for result in all_results:
                writer.writerow(result)
    
    # 生成详细指标汇总文件
    detailed_summary_file = os.path.join(root_experiment_dir, 'detailed_experiment_summary.csv')
    detailed_fieldnames = [
        'Epoch', 'Iteration', 'Loss', 'Overall_Acc', 'Mean_Acc', 'FreqW_Acc', 'Mean_IoU',
        'Class_0_IoU', 'Class_1_IoU', 'Class_0_Dice', 'Class_1_Dice',
        'Class_0_Sensitivity', 'Class_1_Sensitivity', 'Class_0_Specificity', 'Class_1_Specificity',
        'Class_0_F1_Score', 'Class_1_F1_Score', 'Class_0_Precision', 'Class_1_Precision',
        'Class_0_FP_Rate', 'Class_1_FP_Rate', 'TP_0', 'FP_0', 'TN_0', 'FN_0', 'TP_1', 'FP_1', 'TN_1', 'FN_1'
    ]
    
    with open(detailed_summary_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=detailed_fieldnames)
        writer.writeheader()
        
        # 遍历所有实验，从它们的验证日志中提取详细指标
        for result in all_results:
            experiment_dir = result['experiment_dir']
            validation_log_path = os.path.join(experiment_dir, 'validation_log.csv')
            
            if os.path.exists(validation_log_path):
                with open(validation_log_path, 'r', encoding='utf-8') as log_file:
                    reader = csv.reader(log_file)
                    headers = next(reader)  # 读取表头
                    header_map = {header: i for i, header in enumerate(headers)}  # 创建表头映射
                    
                    # 读取所有数据行
                    for row in reader:
                        detailed_row = {}
                        # 基本字段
                        detailed_row['Epoch'] = row[header_map['Epoch']]
                        detailed_row['Iteration'] = row[header_map['Iteration']]
                        detailed_row['Loss'] = row[header_map['Loss']]
                        detailed_row['Overall_Acc'] = row[header_map['Overall_Acc']]
                        detailed_row['Mean_Acc'] = row[header_map['Mean_Acc']]
                        detailed_row['FreqW_Acc'] = row[header_map['FreqW_Acc']]
                        detailed_row['Mean_IoU'] = row[header_map['Mean_IoU']]
                        
                        # 类别指标
                        for class_id in [0, 1]:
                            detailed_row[f'Class_{class_id}_IoU'] = row[header_map[f'Class_{class_id}_IoU']]
                            detailed_row[f'Class_{class_id}_Dice'] = row[header_map[f'Class_{class_id}_Dice']]
                            detailed_row[f'Class_{class_id}_Sensitivity'] = row[header_map[f'Class_{class_id}_Sensitivity']]
                            detailed_row[f'Class_{class_id}_Specificity'] = row[header_map[f'Class_{class_id}_Specificity']]
                            detailed_row[f'Class_{class_id}_F1_Score'] = row[header_map[f'Class_{class_id}_F1_Score']]
                            detailed_row[f'Class_{class_id}_Precision'] = row[header_map[f'Class_{class_id}_Precision']]
                            detailed_row[f'Class_{class_id}_FP_Rate'] = row[header_map[f'Class_{class_id}_FP_Rate']]
                            detailed_row[f'TP_{class_id}'] = row[header_map[f'TP_{class_id}']]
                            detailed_row[f'FP_{class_id}'] = row[header_map[f'FP_{class_id}']]
                            detailed_row[f'TN_{class_id}'] = row[header_map[f'TN_{class_id}']]
                            detailed_row[f'FN_{class_id}'] = row[header_map[f'FN_{class_id}']]
                        
                        writer.writerow(detailed_row)
    
    # 打印汇总信息
    print(f"\n{'='*50}")
    print("实验汇总:")
    for i, result in enumerate(all_results):
        print(f"{i+1}. {result['backbone']} + {result['loss_function']}")
        if result['use_oversample']:
            print(f"   过采样方法: {result['oversample_method']}")
        if result['use_attention']:
            print(f"   注意力机制: 启用")
        print(f"   最佳得分: {result['best_score']:.4f}")
        print(f"   实验目录: {result['experiment_dir']}")
    
    print(f"\n汇总信息已保存到: {summary_file}")
    print(f"详细指标已保存到: {detailed_summary_file}")
    return all_results