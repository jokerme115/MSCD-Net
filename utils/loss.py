import torch.nn as nn
import torch.nn.functional as F
import torch 

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2, size_average=True, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.size_average = size_average

    def forward(self, inputs, targets):
        # 处理忽略索引
        if self.ignore_index is not None:
            valid_mask = targets != self.ignore_index
            targets = targets * valid_mask
        
        # 处理二分类情况
        if inputs.size(1) == 2:  # 二分类任务
            # 确保目标标签为long类型
            targets = targets.squeeze(1) if targets.dim() == 4 else targets
            targets = targets.long()
            
            # 计算交叉熵损失
            ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index if self.ignore_index is not None else -100)
            pt = torch.exp(-ce_loss)
            focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
            
            # 为标签1（腐烂区域）设置更高的权重，进一步增强对小目标的关注
            weights = torch.ones_like(targets, dtype=torch.float)
            weights[targets == 1] = 20.0  # 进一步提高对腐烂区域的关注度
            
            focal_loss = focal_loss * weights
            
            # 只计算有效区域的损失
            if self.ignore_index is not None:
                focal_loss = focal_loss[valid_mask]
            
            if self.size_average:
                return focal_loss.mean()
            else:
                return focal_loss.sum()
        else:
            # 多分类情况
            ce_loss = F.cross_entropy(
                inputs, targets, reduction='none', ignore_index=self.ignore_index if self.ignore_index is not None else -100)
            pt = torch.exp(-ce_loss)
            focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
            if self.ignore_index is not None:
                focal_loss = focal_loss[valid_mask]
            if self.size_average:
                return focal_loss.mean()
            else:
                return focal_loss.sum()

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # 如果是多分类情况，使用softmax
        if inputs.size(1) > 1:
            inputs = F.softmax(inputs, dim=1)
            # 将targets转换为one-hot编码
            targets_one_hot = torch.zeros_like(inputs)
            targets = targets.unsqueeze(1) if targets.dim() == 3 else targets
            targets_one_hot.scatter_(1, targets, 1)
            targets = targets_one_hot
        
        # 确保输入和目标有相同的形状
        if inputs.dim() != targets.dim():
            if inputs.dim() == 4 and targets.dim() == 3:
                targets = targets.unsqueeze(1)
            elif inputs.dim() == 3 and targets.dim() == 4:
                inputs = inputs.unsqueeze(1)
        
        # 使用reshape代替view
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)
        
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        return 1. - dice

class BCEWithLogitsLoss(nn.BCEWithLogitsLoss):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None):
        super(BCEWithLogitsLoss, self).__init__(weight, size_average, reduce, reduction, pos_weight)
        
    def forward(self, input, target):
        # 如果target比input多一个维度（通道维度），则移除该维度
        if target.dim() == 4 and input.dim() == 4 and target.size(1) == 1 and input.size(1) > 1:
            target = target.squeeze(1)
            
        # 如果target是单通道而input是多通道，需要扩展target
        if target.dim() == 3 and input.dim() == 4 and input.size(1) > 1:
            # 创建一个与input形状相同的target
            new_target = torch.zeros_like(input)
            # 对于二分类，将前景类设为1，背景类保持为0
            new_target[:, 1, :, :] = (target > 0).float()
            new_target[:, 0, :, :] = (target == 0).float()
            target = new_target
            
        return F.binary_cross_entropy_with_logits(input, target, 
                                                  self.weight, 
                                                  pos_weight=self.pos_weight, 
                                                  reduction=self.reduction)

# 添加一个专门用于二分类的Focal Loss实现
class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, size_average=True, ignore_index=255):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.size_average = size_average
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        # 输入应该是(N, 1, H, W)或(N, H, W)
        # 目标应该是(N, H, W)或(N, 1, H, W)
        
        if inputs.size(1) == 1:  # 二分类情况
            # 确保目标是正确的形状和类型
            if targets.dim() == 4 and targets.size(1) == 1:
                targets = targets.squeeze(1)
            targets = targets.float()
            
            # 计算BCE损失
            bce_loss = F.binary_cross_entropy_with_logits(inputs.squeeze(1), targets, reduction='none')
            pt = torch.exp(-bce_loss)
            focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
            
            if self.size_average:
                return focal_loss.mean()
            else:
                return focal_loss.sum()
        else:
            raise ValueError("BinaryFocalLoss requires single channel input")

# 添加Tversky Loss实现，专门用于处理类别不平衡问题
class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-5):
        """
        Tversky Loss用于处理类别不平衡问题
        alpha: 控制假阳性的惩罚权重 (针对健康区域)
        beta: 控制假阴性的惩罚权重 (针对腐烂区域)
        当前设置更关注假阳性问题，降低假阴性惩罚
        """
        super(TverskyLoss, self).__init__()
        self.alpha = alpha  # 增加假阳性惩罚
        self.beta = beta    # 减少假阴性惩罚
        self.smooth = smooth

    def forward(self, inputs, targets):
        # 如果输入是多分类，使用softmax
        if inputs.size(1) > 1:
            inputs = F.softmax(inputs, dim=1)
            # 取第二类(腐烂区域)的概率
            inputs = inputs[:, 1, :, :]
        
        # 确保输入和目标形状一致
        if targets.dim() == 4 and targets.size(1) == 1:
            targets = targets.squeeze(1)
        
        # 转换为float类型
        inputs = inputs.float()
        targets = targets.float()
        
        # 使用reshape代替view
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)
        
        # 计算TP, FP, FN
        TP = (inputs * targets).sum()
        FP = (inputs * (1 - targets)).sum()
        FN = ((1 - inputs) * targets).sum()
        
        # 计算Tversky指数
        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        
        # 返回Tversky Loss
        return 1 - tversky

# 组合损失函数：Tversky Loss + Focal Loss
class CombinedLoss(nn.Module):
    def __init__(self, alpha_tversky=0.3, beta_tversky=0.7, alpha_focal=0.75, gamma_focal=2, weight_tversky=1.0, weight_focal=1.0):
        """
        组合损失函数：Tversky Loss + Focal Loss
        修改参数以更关注假阳性问题
        """
        super(CombinedLoss, self).__init__()
        self.tversky_loss = TverskyLoss(alpha=alpha_tversky, beta=beta_tversky)
        self.focal_loss = FocalLoss(alpha=alpha_focal, gamma=gamma_focal)
        self.weight_tversky = weight_tversky
        self.weight_focal = weight_focal

    def forward(self, inputs, targets):
        # 确保目标是正确的形状和类型
        if targets.dim() == 4 and targets.size(1) == 1:
            targets = targets.squeeze(1)
        targets = targets.long()
        
        # 计算组合损失
        tversky_loss = self.tversky_loss(inputs, targets)
        focal_loss = self.focal_loss(inputs, targets)
        
        return self.weight_tversky * tversky_loss + self.weight_focal * focal_loss