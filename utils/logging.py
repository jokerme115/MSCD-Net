"""
日志相关工具函数模块
包含训练日志记录、进度跟踪等
"""

import os
import csv


def log_training_progress(log_file, epoch, iteration, loss, val_score=None):
    """记录训练进度到日志文件"""
    # 确保目录存在
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    if val_score is None:
        # 记录训练损失
        with open(log_file, 'a', newline='', encoding='utf-8') as f:
            # 如果文件为空，写入表头
            if f.tell() == 0:
                headers = ['Epoch', 'Iteration', 'Loss']
                f.write(','.join(headers) + '\n')
            
            # 写入数据行
            row = [str(epoch), str(iteration), str(loss)]
            f.write(','.join(row) + '\n')
    else:
        # 记录验证结果
        with open(log_file, 'a', newline='', encoding='utf-8') as f:
            # 如果文件为空，写入表头
            if f.tell() == 0:
                headers = [
                    'Epoch', 'Iteration', 'Loss',
                    'Overall_Acc', 'Mean_Acc', 'FreqW_Acc', 'Mean_IoU',
                    'Class_0_IoU', 'Class_1_IoU', 'Class_0_Dice', 'Class_1_Dice',
                    'Class_0_Sensitivity', 'Class_1_Sensitivity',
                    'Class_0_Specificity', 'Class_1_Specificity',
                    'Class_0_F1_Score', 'Class_1_F1_Score',
                    'Class_0_Precision', 'Class_1_Precision',
                    'Class_0_FP_Rate', 'Class_1_FP_Rate',  # 添加假阳性率
                    'TP_0', 'FP_0', 'TN_0', 'FN_0',
                    'TP_1', 'FP_1', 'TN_1', 'FN_1'
                ]
                f.write(','.join(headers) + '\n')
            
            # 基本指标占位符（验证时不记录具体训练loss）
            row = [str(epoch), str(iteration), '0.0']
            
            # 添加验证指标
            row.extend([
                str(val_score['Overall Acc']), str(val_score['Mean Acc']), 
                str(val_score['FreqW Acc']), str(val_score['Mean IoU']),
                str(val_score['Class IoU'].get(0, 0)), str(val_score['Class IoU'].get(1, 0)),
                str(val_score['Class Dice'].get(0, 0)), str(val_score['Class Dice'].get(1, 0)),
                str(val_score['Class Sensitivity'].get(0, 0)), str(val_score['Class Sensitivity'].get(1, 0)),
                str(val_score['Class Specificity'].get(0, 0)), str(val_score['Class Specificity'].get(1, 0)),
                str(val_score['Class F1-Score'].get(0, 0)), str(val_score['Class F1-Score'].get(1, 0)),
                str(val_score['Class Precision'].get(0, 0)), str(val_score['Class Precision'].get(1, 0)),
                str(val_score['Class FP Rate'].get(0, 0)), str(val_score['Class FP Rate'].get(1, 0))  # 添加假阳性率
            ])
            
            # TP, FP, TN, FN
            cm = val_score.get('Confusion Matrix', {})
            for i in range(2):  # 对每个类别
                tp = cm.get(i, {}).get('TP', 0)
                fp = cm.get(i, {}).get('FP', 0)
                tn = cm.get(i, {}).get('TN', 0)
                fn = cm.get(i, {}).get('FN', 0)
                row.extend([str(tp), str(fp), str(tn), str(fn)])
            f.write(','.join(row) + '\n')