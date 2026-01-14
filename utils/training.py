"""
训练相关工具函数模块
包含训练循环、验证函数、早停机制等
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

from .utils import Denormalize, mkdir


class EarlyStopping:
    """Early stops the training if validation score doesn't improve after a given patience."""
    
    def __init__(self, patience=7, verbose=False, delta=0, save_path='checkpoints/best_model.pth', 
                 restore_best_weights=False, mode='max', avg_window=1, min_delta_patience=0):
        """
        Args:
            patience (int): How long to wait after last time validation score improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation score improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            save_path (str): Path to save the best model.
                            Default: 'checkpoints/best_model.pth'
            restore_best_weights (bool): If True, restores model weights from the best checkpoint 
                                         when stopping training.
                            Default: False
            mode (str): One of {'max', 'min'}. In 'min' mode, training will stop when the quantity 
                        monitored has stopped decreasing; in 'max' mode it will stop when the 
                        quantity monitored has stopped increasing.
                            Default: 'max'
            avg_window (int): Number of epochs to average over for smoothing the metric.
                            Default: 1 (no averaging)
            min_delta_patience (int): Minimum number of epochs to wait before considering delta improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.save_path = save_path
        self.restore_best_weights = restore_best_weights
        self.mode = mode
        self.avg_window = avg_window
        self.min_delta_patience = min_delta_patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None
        self.score_history = []
        self.best_epoch = 0
        self.current_epoch = 0

    def __call__(self, val_score, model):
        """
        检查是否应该早停
        
        Args:
            val_score (float): 验证分数
            model (nn.Module): 当前模型
        """
        score = val_score if isinstance(val_score, (int, float)) else val_score.get('Mean IoU', 0)
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif (self.mode == 'max' and score < self.best_score + self.delta) or \
             (self.mode == 'min' and score > self.best_score - self.delta):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
            
        return self.early_stop
    
    def save_checkpoint(self, model):
        """保存模型检查点"""
        if self.verbose:
            print(f'Validation score improved ({self.best_score:.6f} --> {self.best_score:.6f}). Saving model ...')
        # 这里可以添加实际的模型保存逻辑

def validate(opts, model, val_loader, metrics, device, criterion=None):
    """
    验证模型性能
    
    Args:
        opts: 配置选项
        model: 模型
        val_loader: 验证数据加载器
        metrics: 评估指标
        device: 设备
        criterion: 损失函数 (可选)
    """
    metrics.reset()
    model.eval()
    val_loss = 0.0
    val_loss_count = 0
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            
            outputs = model(images)
            if criterion is not None:
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_loss_count += 1
            
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()
            
            metrics.update(targets, preds)
    
    val_loss = val_loss / val_loss_count if val_loss_count > 0 else 0
    score = metrics.get_results()
    if criterion is not None:
        score['val_loss'] = val_loss
    
    model.train()
    return score


def save_ckpt(path, model, optimizer, scheduler, best_score, cur_itrs):
    """ save current model
    """
    torch.save({
        "cur_itrs": cur_itrs,
        "model_state": model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "best_score": best_score,
    }, path)
    print("Model saved as %s" % path)