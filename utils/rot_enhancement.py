"""
腐烂区域增强工具模块
包含确保每张图片腐烂面积百分比的工具函数
"""

import numpy as np
from PIL import Image
import random
import math

def calculate_rot_percentage(label):
    """
    计算标签中腐烂区域的百分比
    
    Args:
        label (np.ndarray or PIL.Image): 标签图像
        
    Returns:
        float: 腐烂区域占总像素的百分比
    """
    if isinstance(label, Image.Image):
        label = np.array(label)
    
    total_pixels = label.size
    rot_pixels = np.sum(label == 1)  # 腐烂区域标签值为1
    return (rot_pixels / total_pixels) * 100 if total_pixels > 0 else 0

def enhance_rot_percentage(img, lbl, target_percentage=2.0):
    """
    增强图像中的腐烂区域百分比，确保达到目标百分比
    
    Args:
        img (PIL.Image): 输入图像
        lbl (PIL.Image): 输入标签
        target_percentage (float): 目标腐烂区域百分比 (默认2.0%)
        
    Returns:
        PIL.Image: 增强后的图像
        PIL.Image: 增强后的标签
    """
    # 计算当前腐烂区域百分比
    current_percentage = calculate_rot_percentage(lbl)
    
    # 如果已经满足条件，直接返回
    if current_percentage >= target_percentage:
        return img, lbl
    
    # 将PIL图像转换为numpy数组
    img_np = np.array(img)
    lbl_np = np.array(lbl)
    
    # 查找腐烂区域（标签值为1的区域）
    rot_mask = (lbl_np == 1)
    
    # 如果没有腐烂区域，直接返回
    if not np.any(rot_mask):
        return img, lbl
    
    # 获取腐烂区域的边界框
    coords = np.where(rot_mask)
    if len(coords[0]) == 0:
        return img, lbl
    
    min_row, max_row = np.min(coords[0]), np.max(coords[0])
    min_col, max_col = np.min(coords[1]), np.max(coords[1])
    
    # 提取腐烂区域
    rot_region_img = img_np[min_row:max_row+1, min_col:max_col+1]
    rot_region_mask = rot_mask[min_row:max_row+1, min_col:max_col+1]
    
    # 获取图像尺寸
    h, w = img_np.shape[:2]
    region_h, region_w = rot_region_mask.shape
    
    # 计算需要增加的腐烂区域像素数
    total_pixels = h * w
    current_rot_pixels = np.sum(lbl_np == 1)
    target_rot_pixels = int(total_pixels * target_percentage / 100)
    needed_rot_pixels = target_rot_pixels - current_rot_pixels
    
    # 估计每次复制增加的像素数
    pixels_per_copy = np.sum(rot_region_mask)
    
    # 计算需要复制的次数，确保不低于目标值
    if pixels_per_copy > 0:
        # 使用 math.ceil 确保复制足够的次数以达到或超过目标值
        num_copies = max(1, math.ceil(needed_rot_pixels / pixels_per_copy))
    else:
        # 如果无法计算pixels_per_copy，使用默认值
        num_copies = max(1, math.ceil(needed_rot_pixels / 100))  # 假设每次复制100个像素
    
    # 执行复制粘贴操作
    for _ in range(num_copies):
        # 随机选择粘贴位置
        paste_row = random.randint(0, h - region_h)
        paste_col = random.randint(0, w - region_w)
        
        # 创建粘贴区域的掩码
        paste_mask_local = np.zeros((h, w), dtype=bool)
        paste_mask_local[paste_row:paste_row+region_h, paste_col:paste_col+region_w] = rot_region_mask
        
        # 在图像上粘贴腐烂区域
        for i in range(3):  # 对于每个颜色通道
            img_np[paste_row:paste_row+region_h, paste_col:paste_col+region_w, i][rot_region_mask] = \
                rot_region_img[:, :, i][rot_region_mask]
        
        # 在标签上更新腐烂区域
        lbl_np[paste_row:paste_row+region_h, paste_col:paste_col+region_w][rot_region_mask] = 1
    
    # 将numpy数组转换回PIL图像
    img_pil = Image.fromarray(img_np)
    # 确保标签值不超过1（只保留0和1）
    lbl_np = np.clip(lbl_np, 0, 1)
    lbl_pil = Image.fromarray(lbl_np.astype(np.uint8))
    
    return img_pil, lbl_pil

class RotPercentageEnhancer:
    """
    腐烂区域百分比增强器
    确保每张图片的腐烂区域百分比达到指定阈值
    """
    
    def __init__(self, target_percentage=2.0):
        """
        初始化增强器
        
        Args:
            target_percentage (float): 目标腐烂区域百分比
        """
        self.target_percentage = target_percentage
    
    def __call__(self, img, lbl):
        """
        应用增强
        
        Args:
            img (PIL.Image): 输入图像
            lbl (PIL.Image): 输入标签
            
        Returns:
            PIL.Image: 增强后的图像
            PIL.Image: 增强后的标签
        """
        return enhance_rot_percentage(img, lbl, self.target_percentage)
    
    def __repr__(self):
        return f"{self.__class__.__name__}(target_percentage={self.target_percentage})"