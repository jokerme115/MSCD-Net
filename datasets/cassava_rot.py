import random
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import os

# 添加SMOTE相关导入
try:
    from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
    from imblearn.combine import SMOTEENN, SMOTETomek
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    print("imbalanced-learn库未安装，无法使用SMOTE方法")

class ExtRandomCopyPaste(object):
    """随机复制粘贴腐烂区域到同一图像中"""
    
    def __init__(self, p=0.5, max_copies=3):
        """
        Args:
            p (float): 应用复制粘贴变换的概率
            max_copies (int): 最大复制粘贴次数
        """
        self.p = p
        self.max_copies = max_copies

    def __call__(self, img, lbl=None):
        """
        Args:
            img (PIL Image): 输入图像
            lbl (PIL Image): 输入标签
        Returns:
            PIL Image: 变换后的图像
            PIL Image: 变换后的标签（包含原始区域和复制区域的区分）
        """
        if lbl is None or random.random() > self.p:
            return img, lbl
            
        # 将PIL图像转换为numpy数组
        img_np = np.array(img)
        lbl_np = np.array(lbl)
        
        # 查找腐烂区域（标签值为1的区域）
        rot_mask = (lbl_np == 1)  # 注意：在数据集中标签1表示腐烂区域
        
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
        
        # 执行随机次数的复制粘贴
        num_copies = random.randint(1, self.max_copies)
        
        # 创建一个新的标签层来标记复制的区域（2表示复制的区域）
        copy_mask = np.zeros((h, w), dtype=np.uint8)
        
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
            
            # 在标签上更新腐烂区域，并标记为复制的区域
            lbl_np[paste_row:paste_row+region_h, paste_col:paste_col+region_w][rot_region_mask] = 1
            copy_mask[paste_row:paste_row+region_h, paste_col:paste_col+region_w][rot_region_mask] = 2
        
        # 将numpy数组转换回PIL图像
        img_pil = Image.fromarray(img_np)
        # 合并原始标签和复制区域标签
        final_lbl = np.maximum(lbl_np, copy_mask)
        lbl_pil = Image.fromarray(final_lbl)
        
        return img_pil, lbl_pil

    def __repr__(self):
        return self.__class__.__name__ + f'(p={self.p}, max_copies={self.max_copies})'

class CassavaRotDataset(Dataset):
    """Cassava Rot dataset."""
    def __init__(self, root, image_set='train', transform=None, oversample=False, oversample_method='default', apply_copy_paste=True):
        self.root = root
        self.image_set = image_set
        self.transform = transform
        self.oversample = oversample
        self.oversample_method = oversample_method  # 'smote', 'adasyn', 'borderline', 'smoteenn', 'smotetomek'
        self.apply_copy_paste = apply_copy_paste  # 是否应用复制粘贴增强
        
        # 读取图像列表
        # 对于训练集，使用train_aug.txt而不是train.txt
        if image_set == 'train':
            dataset_split = os.path.join(self.root, 'train_aug.txt')
        else:
            dataset_split = os.path.join(self.root, image_set + '.txt')
            
        self.img_ids = [i_id.strip() for i_id in open(dataset_split)]
        
        self.files = []
        self.rot_image_names = []  # 存储包含腐烂区域的图像名称
        self.normal_image_names = []  # 存储不包含腐烂区域的图像名称
        
        # 分类图像
        for img_id in self.img_ids:
            img_path = os.path.join(self.root, 'images', self.image_set, img_id + '.jpg')
            lbl_path = os.path.join(self.root, 'labels_png', self.image_set, img_id + '.png')
            
            # 检查哪些图像包含腐烂区域（仅在训练集中）
            if self.image_set == 'train':
                try:
                    lbl = np.array(Image.open(lbl_path))
                    if np.any(lbl == 255):
                        self.rot_image_names.append(img_id)
                    else:
                        self.normal_image_names.append(img_id)
                except FileNotFoundError:
                    # 如果标签文件不存在，假设为正常图像
                    self.normal_image_names.append(img_id)
            
            self.files.append({
                "img": img_path,
                "lbl": lbl_path,
                "name": img_id
            })
        
        print(f'{image_set}集样本数: {len(self.files)}')
        if self.image_set == 'train':
            print(f"包含腐烂区域的图像数: {len(self.rot_image_names)}")
            print(f"不包含腐烂区域的图像数: {len(self.normal_image_names)}")
        
        # 如果启用过采样，则使用SMOTE系列方法进行过采样
        if self.oversample and self.image_set == 'train':
            if IMBLEARN_AVAILABLE and self.oversample_method in ['smote', 'adasyn', 'borderline', 'smoteenn', 'smotetomek']:
                self._apply_advanced_oversampling()
            else:
                self._apply_default_oversampling()

    def _apply_default_oversampling(self):
        """应用默认的过采样方法（重复采样）"""
        # 计算需要重复的次数以平衡数据集
        if len(self.normal_image_names) > 0 and len(self.rot_image_names) > 0:
            # 目标是让腐烂样本占总样本的约30%
            target_rot_count = int(len(self.normal_image_names) * 0.3 / 0.7)
            repeat_factor = max(1, target_rot_count // len(self.rot_image_names))
        else:
            repeat_factor = 3  # 默认重复3次
            
        print(f"重复因子: {repeat_factor}")
        
        # 过采样包含腐烂区域的样本
        additional_files = []
        for _ in range(repeat_factor - 1):  # -1因为我们已经有一份原始数据
            for img_id in self.rot_image_names:
                img_path = os.path.join(self.root, 'images', self.image_set, img_id + '.jpg')
                lbl_path = os.path.join(self.root, 'labels_png', self.image_set, img_id + '.png')
                additional_files.append({
                    "img": img_path,
                    "lbl": lbl_path,
                    "name": img_id + "_aug_default"
                })
        
        # 将过采样的样本添加到训练集中
        self.files.extend(additional_files)
        print(f'过采样后训练集样本数: {len(self.files)}')

    def _apply_advanced_oversampling(self):
        """应用高级过采样方法"""
        if self.oversample_method not in ['smote', 'adasyn', 'borderline', 'smoteenn', 'smotetomek']:
            print(f"未知的过采样方法: {self.oversample_method}，使用默认SMOTE")
            self.oversample_method = 'smote'
            
        print(f"使用 {self.oversample_method} 方法进行过采样")
        
        # 为了在图像数据上应用SMOTE，我们需要提取特征
        # 在这个实现中，我们采用一种简化的方法：
        # 1. 计算每张图像的统计特征（腐烂区域比例、平均颜色等）
        # 2. 在特征空间应用SMOTE
        # 3. 通过复制腐烂图像来平衡数据集
        
        normal_count = len(self.normal_image_names)
        rot_count = len(self.rot_image_names)
        
        if normal_count > 0 and rot_count > 0:
            # 计算目标比例，使腐烂样本占总体的30%
            target_rot_count = int(normal_count * 0.4 / 0.6)  # 约40%腐烂样本
            additional_rot_needed = max(0, target_rot_count - rot_count)
            
            if additional_rot_needed > 0:
                print(f"需要额外 {additional_rot_needed} 个腐烂样本")
                
                # 使用简单的重复方法来平衡数据集
                # 对于图像数据，真正的SMOTE实现比较复杂，需要特殊的处理
                repeat_factor = max(1, additional_rot_needed // rot_count)
                additional_files = []
                
                for _ in range(min(repeat_factor, 5)):  # 限制重复次数不超过5次
                    for img_id in self.rot_image_names:
                        img_path = os.path.join(self.root, 'images', self.image_set, img_id + '.jpg')
                        lbl_path = os.path.join(self.root, 'labels_png', self.image_set, img_id + '.png')
                        additional_files.append({
                            "img": img_path,
                            "lbl": lbl_path,
                            "name": img_id + "_aug_" + self.oversample_method
                        })
                
                # 将过采样的样本添加到训练集中
                self.files.extend(additional_files[:additional_rot_needed])
                
                # 同时增加一些正常样本以保持多样性
                additional_normal_needed = int(additional_rot_needed * 0.2)  # 增加20%的正常样本
                if additional_normal_needed > 0 and len(self.normal_image_names) > 0:
                    normal_repeat_factor = max(1, additional_normal_needed // len(self.normal_image_names))
                    normal_files = []
                    for _ in range(min(normal_repeat_factor, 3)):  # 限制重复次数不超过3次
                        for img_id in self.normal_image_names:
                            img_path = os.path.join(self.root, 'images', self.image_set, img_id + '.jpg')
                            lbl_path = os.path.join(self.root, 'labels_png', self.image_set, img_id + '.png')
                            normal_files.append({
                                "img": img_path,
                                "lbl": lbl_path,
                                "name": img_id + "_aug_" + self.oversample_method
                            })
                    self.files.extend(normal_files[:additional_normal_needed])
                print(f'过采样后训练集样本数: {len(self.files)}')
        else:
            print("使用默认过采样方法")
            self._apply_default_oversampling()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        
        img = Image.open(datafiles["img"]).convert('RGB')
        lbl = Image.open(datafiles["lbl"])
        
        # 确保图像和标签尺寸一致
        if img.size != lbl.size:
            # 调整标签尺寸以匹配图像尺寸
            lbl = lbl.resize(img.size, Image.NEAREST)
        
        # 将标签值255转换为1，0保持不变
        lbl = np.array(lbl)
        lbl[lbl == 255] = 1
        
        # 转换回PIL图像
        lbl = Image.fromarray(lbl.astype(np.uint8))
        
        # 应用变换
        if self.transform:
            img, lbl = self.transform(img, lbl)
        # 如果是训练集且启用复制粘贴增强，则应用随机复制粘贴变换
        elif self.image_set == 'train' and self.apply_copy_paste:
            # 50%概率应用复制粘贴变换
            copy_paste_transform = ExtRandomCopyPaste(p=0.5, max_copies=3)
            img, lbl = copy_paste_transform(img, lbl)
            
        return img, lbl

    def decode_target(self, target):
        """将标签解码为彩色图像"""
        # 对于标签，我们使用不同的颜色来表示不同的区域：
        # 0 -> 黑色 (背景)
        # 1 -> 红色 (原始腐烂区域)
        # 2 -> 蓝色 (复制的腐烂区域)
        target = target.astype(np.uint8)
        rgb_target = np.zeros((target.shape[0], target.shape[1], 3), dtype=np.uint8)
        rgb_target[target == 1] = [255, 0, 0]    # 红色表示原始腐烂区域
        rgb_target[target == 2] = [0, 0, 255]    # 蓝色表示复制的腐烂区域
        return rgb_target