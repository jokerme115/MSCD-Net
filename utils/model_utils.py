"""
模型处理工具函数模块
包含损失函数获取、模型设置等
"""

import torch.nn as nn
from . import loss
import network


def get_loss_function(loss_type):
    """根据指定的损失函数类型返回相应的损失函数"""
    if loss_type == 'focal_loss':
        return loss.FocalLoss(alpha=0.75, gamma=2, ignore_index=255, size_average=True)
    elif loss_type == 'ce_loss':
        return nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
    elif loss_type == 'bce_with_logits':
        return loss.BCEWithLogitsLoss()
    elif loss_type == 'dice_loss':
        return loss.DiceLoss()
    elif loss_type == 'tversky':
        # Tversky Loss更适合处理类别不平衡问题，调整参数更关注假阳性问题
        return loss.TverskyLoss(alpha=0.3, beta=0.7)
    elif loss_type == 'combined':
        # 组合损失函数：Tversky Loss + Focal Loss，调整参数更关注假阳性问题
        return loss.CombinedLoss(alpha_tversky=0.3, beta_tversky=0.7, weight_tversky=2.0, weight_focal=1.0)
    else:
        raise ValueError(f"不支持的损失函数类型: {loss_type}")


def get_model(model_name, num_classes, output_stride, separable_conv=False):
    """根据指定的模型名称返回相应的模型"""
    model_map = {
        'deeplabv3_resnet50': network.deeplabv3_resnet50,
        'deeplabv3plus_resnet50': network.deeplabv3plus_resnet50,
        'deeplabv3_resnet101': network.deeplabv3_resnet101,
        'deeplabv3plus_resnet101': network.deeplabv3plus_resnet101,
        'deeplabv3_mobilenet': network.deeplabv3_mobilenet,
        'deeplabv3plus_mobilenet': network.deeplabv3plus_mobilenet,
        'deeplabv3plus_xception': network.deeplabv3plus_xception,
        'deeplabv3plus_hrnetv2_32': network.deeplabv3plus_hrnetv2_32,
        'deeplabv3plus_hrnetv2_48': network.deeplabv3plus_hrnetv2_48
    }
    
    if model_name not in model_map:
        raise ValueError(f"不支持的模型类型: {model_name}")
    
    model = model_map[model_name](num_classes=num_classes, output_stride=output_stride)
    if separable_conv and 'plus' in model_name:
        network.convert_to_separable_conv(model.classifier)
    
    return model