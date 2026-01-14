import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

def set_bn_momentum(model, momentum=0.1):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = momentum

def Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Denormalize an image tensor with mean and standard deviation.
    """
    def denormalize(tensor):
        # 处理tensor类型
        if torch.is_tensor(tensor):
            for t, m, s in zip(tensor, mean, std):
                t.mul_(s).add_(m)
            return tensor
        # 处理numpy数组类型
        else:
            tensor = tensor.copy()
            for i, (m, s) in enumerate(zip(mean, std)):
                tensor[i] = tensor[i] * s + m
            return tensor
    return denormalize

def mkdir(dir):
    """
    Create a directory if it does not exist.
    """
    import os
    if not os.path.exists(dir):
        os.makedirs(dir)