"""
DeepLabV3+ 模型变体实现
支持不同的架构变体用于实验比较
"""

import torch
from torch import nn
from torch.nn import functional as F

from .utils import IntermediateLayerGetter
from ._deeplab import DeepLabHead, DeepLabHeadV3Plus, DeepLabV3
from .backbone import resnet

def _segm_resnet_variant(name, backbone_name, num_classes, output_stride=8, pretrained_backbone=True, 
                         use_cbam=False, use_oversample=False):
    """
    构建不同变体的ResNet DeepLabV3+模型
    
    Args:
        name: 模型名称 ('deeplabv3' 或 'deeplabv3plus')
        backbone_name: 骨干网络名称 (如 'resnet101')
        num_classes: 类别数
        output_stride: 输出步长
        pretrained_backbone: 是否使用预训练骨干网络
        use_cbam: 是否使用CBAM注意力机制
        use_oversample: 是否使用过采样 (这个参数主要在数据加载时使用)
    """
    if output_stride == 8:
        replace_stride_with_dilation = [False, True, True]
    elif output_stride == 16:
        replace_stride_with_dilation = [False, False, True]
    else:
        raise ValueError("output_stride must be 8 or 16")
        
    backbone = resnet.__dict__[backbone_name](pretrained=pretrained_backbone, 
                                              replace_stride_with_dilation=replace_stride_with_dilation)
    
    if name=='deeplabv3plus':
        if output_stride == 8:
            aspp_dilate = [12, 24, 36]
        elif output_stride == 16:
            aspp_dilate = [6, 12, 18]
        
        # 处理不同ResNet模型的通道数
        if backbone_name in ['resnet18', 'resnet34']:
            low_level_planes = 64   # BasicBlock的输出通道数
            inplanes = 512          # ResNet-18/34的layer4输出通道数
        else:
            low_level_planes = 256  # Bottleneck的输出通道数
            inplanes = 2048         # ResNet-50/101/152的layer4输出通道数
            
        return_layers = {'layer4': 'out', 'layer1': 'low_level'}
        classifier = DeepLabHeadV3PlusVariant(inplanes, low_level_planes, num_classes, aspp_dilate, use_cbam=use_cbam)
    elif name=='deeplabv3':
        if output_stride == 8:
            aspp_dilate = [12, 24, 36]
        elif output_stride == 16:
            aspp_dilate = [6, 12, 18]
            
        # 处理不同ResNet模型的inplanes值
        if backbone_name in ['resnet18', 'resnet34']:
            inplanes = 512          # ResNet-18/34的layer4输出通道数
        else:
            inplanes = 2048         # ResNet-50/101/152的layer4输出通道数
            
        return_layers = {'layer4': 'out'}
        classifier = DeepLabHead(inplanes, num_classes, aspp_dilate)

    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    model = DeepLabV3(backbone, classifier)
    return model


def _load_model_variant(arch_type, backbone, num_classes, output_stride=8, pretrained_backbone=True,
                        use_cbam=False, use_oversample=False):
    """加载指定变体的模型"""
    if backbone.startswith('resnet'):
        model = _segm_resnet_variant(arch_type, backbone, num_classes, output_stride=output_stride, 
                                     pretrained_backbone=pretrained_backbone, use_cbam=use_cbam,
                                     use_oversample=use_oversample)
    else:
        raise NotImplementedError(f"当前仅支持ResNet骨干网络，不支持 {backbone}")
    return model


# DeepLabV3+ 变体模型定义
def deeplabv3plus_resnet101_variant(num_classes=21, output_stride=8, pretrained_backbone=True,
                                    use_cbam=False, use_oversample=False):
    """构建DeepLabV3+ ResNet-101变体模型
    
    Args:
        num_classes: 类别数
        output_stride: 输出步长
        pretrained_backbone: 是否使用预训练骨干网络
        use_cbam: 是否使用CBAM注意力机制
        use_oversample: 是否使用过采样 (主要用于标识)
    """
    return _load_model_variant('deeplabv3plus', 'resnet101', num_classes, output_stride=output_stride,
                               pretrained_backbone=pretrained_backbone, use_cbam=use_cbam,
                               use_oversample=use_oversample)


class DeepLabHeadV3PlusVariant(nn.Module):
    """支持CBAM的DeepLabV3+头部变体"""
    def __init__(self, in_channels, low_level_channels, num_classes, aspp_dilate=[12, 24, 36], use_cbam=False):
        super(DeepLabHeadV3PlusVariant, self).__init__()
        self.use_cbam = use_cbam
        
        self.project = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.aspp = ASPPVariant(in_channels, aspp_dilate)
        
        # 根据是否使用CBAM决定是否添加注意力模块
        if self.use_cbam:
            from .utils import CBAM
            self.cbam = CBAM(304)  # 256 (ASPP输出) + 48 (低级特征) = 304

        self.classifier = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, num_classes, 1)
        )
        self._init_weight()

    def forward(self, feature):
        low_level_feature = self.project(feature['low_level'])
        output_feature = self.aspp(feature['out'])
        output_feature = F.interpolate(output_feature, size=low_level_feature.shape[2:], mode='bilinear', align_corners=False)
        output_feature = torch.cat([low_level_feature, output_feature], dim=1)
        
        # 根据配置决定是否应用CBAM注意力机制
        if self.use_cbam:
            output_feature = self.cbam(output_feature)
        
        return self.classifier(output_feature)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class ASPPVariant(nn.Module):
    """ASPP变体实现"""
    def __init__(self, in_channels, atrous_rates):
        super(ASPPVariant, self).__init__()
        out_channels = 256
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),)

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)