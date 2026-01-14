"""
扩展的 torchvision.transforms 模块
"""

import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image, ImageOps, ImageFilter

class ExtCompose:
    """组合多个变换"""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, lbl):
        for t in self.transforms:
            img, lbl = t(img, lbl)
        return img, lbl

class ExtToTensor:
    """将PIL图像转换为Tensor"""

    def __init__(self):
        self.to_tensor = transforms.ToTensor()

    def __call__(self, pic, lbl):
        return self.to_tensor(pic), torch.from_numpy(np.array(lbl, dtype=np.uint8))

class ExtNormalize:
    """标准化张量图像"""

    def __init__(self, mean, std):
        self.normalize = transforms.Normalize(mean, std)

    def __call__(self, tensor, lbl):
        return self.normalize(tensor), lbl

class ExtRandomScale:
    """随机缩放图像"""

    def __init__(self, scale_range):
        self.scale_range = scale_range

    def __call__(self, img, lbl):
        scale = random.uniform(self.scale_range[0], self.scale_range[1])
        new_size = (int(img.size[1] * scale), int(img.size[0] * scale))
        img = img.resize(new_size, Image.BILINEAR)
        lbl = lbl.resize(new_size, Image.NEAREST)
        return img, lbl

class ExtRandomCrop:
    """随机裁剪图像"""

    def __init__(self, size, pad_if_needed=True):
        self.size = size
        self.pad_if_needed = pad_if_needed

    def __call__(self, img, lbl):
        w, h = img.size
        tw, th = self.size
        if w < tw or h < th:
            padw = max((tw - w + 1) // 2, 0)
            padh = max((th - h + 1) // 2, 0)
            img = ImageOps.expand(img, border=(padw, padh), fill=0)
            lbl = ImageOps.expand(lbl, border=(padw, padh), fill=0)
        left = random.randint(0, w - tw)
        top = random.randint(0, h - th)
        img = img.crop((left, top, left + tw, top + th))
        lbl = lbl.crop((left, top, left + tw, top + th))
        return img, lbl

class ExtRandomHorizontalFlip:
    """随机水平翻转图像"""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, lbl):
        if random.random() < self.p:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            lbl = lbl.transpose(Image.FLIP_LEFT_RIGHT)
        return img, lbl

class ExtRandomVerticalFlip:
    """随机垂直翻转图像"""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, lbl):
        if random.random() < self.p:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            lbl = lbl.transpose(Image.FLIP_TOP_BOTTOM)
        return img, lbl

class ExtColorJitter:
    """颜色抖动"""

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.color_jitter = transforms.ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, img, lbl):
        img = self.color_jitter(img)
        return img, lbl

class ExtResize:
    """调整图像大小"""

    def __init__(self, size):
        self.size = size

    def __call__(self, img, lbl):
        img = img.resize(self.size, Image.BILINEAR)
        lbl = lbl.resize(self.size, Image.NEAREST)
        return img, lbl
"""
数据集处理工具函数模块
包含数据集加载、数据增强处理等
"""

import torch
from torch.utils import data
import torchvision.transforms as transforms
import os

from datasets import CassavaRotDataset
from utils import ext_transforms as et
# 添加腐烂区域增强模块
from utils.rot_enhancement import RotPercentageEnhancer

def get_dataset(opts):
    """ Dataset And Augmentation
    """
    # 确保所有必需参数都有默认值
    crop_size = opts.get('crop_size', 512)
    data_root = opts.get('data_root', './datasets/data')
    use_smote_dataset = opts.get('use_smote_dataset', False)
    oversample = opts.get('oversample', False)
    oversample_method = opts.get('oversample_method', 'default')
    # 获取腐烂区域增强参数
    enhance_rot_percentage = opts.get('enhance_rot_percentage', False)
    target_rot_percentage = opts.get('target_rot_percentage', 2.0)
    
    # 构建训练变换
    train_transform_list = [
        et.ExtRandomScale((0.5, 2.0)),
        et.ExtRandomCrop(size=(crop_size, crop_size), pad_if_needed=True),
        et.ExtRandomHorizontalFlip(),
        et.ExtRandomVerticalFlip(),  # 添加垂直翻转增强
        et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
    ]
    
    # 如果启用腐烂区域增强，则添加到变换列表开头
    if enhance_rot_percentage:
        train_transform_list.insert(0, RotPercentageEnhancer(target_percentage=target_rot_percentage))
    
    # 添加张量转换和标准化
    train_transform_list.extend([
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])
    
    train_transform = et.ExtCompose(train_transform_list)
    
    val_transform = et.ExtCompose([
        et.ExtCenterCrop((crop_size, crop_size)),
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])
    
    # 使用更新后的CassavaRotDataset，支持过采样选项
    train_dst = CassavaRotDataset(root=data_root, image_set='train',
                                 transform=train_transform, 
                                 oversample=oversample,
                                 oversample_method=oversample_method)
    
    val_dst = CassavaRotDataset(root=data_root, image_set='val',
                               transform=val_transform)
    
    return train_dst, val_dst

def setup_data_loaders(train_dst, val_dst, opts):
    """
    设置数据加载器
    """
    batch_size = opts.get('batch_size', 4)
    val_batch_size = opts.get('val_batch_size', batch_size)
    
    train_loader = data.DataLoader(
        train_dst, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2,
        drop_last=True)  # 添加drop_last以避免最后一个不完整的batch
    
    val_loader = data.DataLoader(
        val_dst, 
        batch_size=val_batch_size, 
        shuffle=False, 
        num_workers=2)
    
    return train_loader, val_loader
