from .utils import IntermediateLayerGetter
from ._deeplab import DeepLabHead, DeepLabHeadV3Plus, DeepLabV3
from .backbone import (
    resnet,
    mobilenetv2,
    hrnetv2,
    xception
)

def _segm_hrnet(name, backbone_name, num_classes, pretrained_backbone, use_attention=False):
    """为DeepLabV3/V3+构建HRNet主干网络"""
    
    # 创建HRNet主干网络
    backbone = hrnetv2.__dict__[backbone_name](pretrained_backbone)
    
    # 解析HRNet配置
    # 最终输出通道数取决于最高分辨率通道配置(c)
    hrnet_channels = int(backbone_name.split('_')[-1])
    
    # HRNet输出是四个分支的连接:
    # 分支0: c通道
    # 分支1: 2*c通道  
    # 分支2: 4*c通道
    # 分支3: 8*c通道
    # 总计: c + 2*c + 4*c + 8*c = 15*c
    inplanes = 15 * hrnet_channels  # HRNet输出连接后的通道数
    
    # HRNet的layer1输出通道数是256（Bottleneck结构的输出）
    low_level_planes = 256  # HRNet的layer1输出通道数是256（经过Bottleneck处理后）
    
    # HRNet通常使用固定的aspp_dilate设置
    aspp_dilate = [12, 24, 36]

    # 根据模型类型配置返回层和分类器
    if name == 'deeplabv3plus':
        # DeepLabV3+需要低级特征用于解码
        return_layers = {'layer1': 'low_level'}
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate, use_attention=use_attention)
    elif name == 'deeplabv3':
        # DeepLabV3只需要高级特征
        return_layers = {}
        classifier = DeepLabHead(inplanes, num_classes, aspp_dilate, use_attention=use_attention)

    # 创建中间层获取器
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers, hrnet_flag=True)
    
    # 创建DeepLabV3模型
    model = DeepLabV3(backbone, classifier)
    return model

def _segm_resnet(name, backbone_name, num_classes, output_stride=8, pretrained_backbone=True, use_attention=False):
    if output_stride == 8:
        replace_stride_with_dilation = [False, True, True]
    elif output_stride == 16:
        replace_stride_with_dilation = [False, False, True]
    else:
        raise ValueError("output_stride must be 8 or 16")
        
    backbone = resnet.__dict__[backbone_name](pretrained=pretrained_backbone, 
                                              replace_stride_with_dilation=replace_stride_with_dilation)
    
    # 确保输出通道数正确
    if name=='deeplabv3plus':
        if output_stride == 8:
            aspp_dilate = [12, 24, 36]
        elif output_stride == 16:
            aspp_dilate = [6, 12, 18]
        
        # 修改低级特征层以匹配ResNet结构
        # BasicBlock (ResNet-18/34) 的layer1输出通道数是64
        # Bottleneck (ResNet-50/101/152) 的layer1输出通道数是256
        if backbone_name in ['resnet18', 'resnet34']:
            low_level_planes = 64   # BasicBlock的输出通道数
            inplanes = 512          # ResNet-18/34的layer4输出通道数 (512 = 512 * BasicBlock.expansion)
        else:
            low_level_planes = 256  # Bottleneck的输出通道数
            inplanes = 2048         # ResNet-50/101/152的layer4输出通道数 (2048 = 512 * Bottleneck.expansion)
            
        return_layers = {'layer4': 'out', 'layer1': 'low_level'}
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate, use_attention=use_attention)
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
        classifier = DeepLabHead(inplanes, num_classes, aspp_dilate, use_attention=use_attention)

    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    model = DeepLabV3(backbone, classifier)
    return model


def _segm_xception(name, backbone_name, num_classes, output_stride=8, pretrained_backbone=True, use_attention=False):
    if output_stride == 8:
        replace_stride_with_dilation=[False, False, False, True]
    elif output_stride == 16:
        replace_stride_with_dilation=[False, False, True, True]
    else:
        raise ValueError("output_stride must be 8 or 16")
        
    backbone = xception.__dict__[backbone_name](pretrained='imagenet' if pretrained_backbone else False,
                                                replace_stride_with_dilation=replace_stride_with_dilation)
    
    # Xception的输出通道数
    if name=='deeplabv3plus':
        if output_stride == 8:
            aspp_dilate = [12, 24, 36]
        elif output_stride == 16:
            aspp_dilate = [6, 12, 18]
            
        low_level_planes = 128  # Xception低级特征的通道数
        return_layers = {'block1': 'low_level', 'conv4': 'out'}
        classifier = DeepLabHeadV3Plus(2048, low_level_planes, num_classes, aspp_dilate, use_attention=use_attention)
    elif name=='deeplabv3':
        if output_stride == 8:
            aspp_dilate = [12, 24, 36]
        elif output_stride == 16:
            aspp_dilate = [6, 12, 18]
            
        return_layers = {'conv4': 'out'}
        classifier = DeepLabHead(2048, num_classes, aspp_dilate, use_attention=use_attention)

    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    model = DeepLabV3(backbone, classifier)
    return model


def _segm_mobilenet(name, backbone_name, num_classes, output_stride=8, pretrained_backbone=True, use_attention=False):
    # 修改MobileNetv2的实现以确保正确输出
    backbone = mobilenetv2.mobilenet_v2(pretrained=pretrained_backbone)
    
    # 获取最后的特征通道数
    inplanes = 1280
    low_level_planes = 24
    
    if name=='deeplabv3plus':
        # 正确指定低级特征层
        return_layers = {'features': 'out', 'features.2.conv.2': 'low_level'}  # features作为out，features.2.conv.2作为低级特征
        aspp_dilate = [12, 24, 36]
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate, use_attention=use_attention)
    elif name=='deeplabv3':
        return_layers = {'features': 'out'}
        aspp_dilate = [12, 24, 36]
        classifier = DeepLabHead(inplanes, num_classes, aspp_dilate, use_attention=use_attention)

    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    model = DeepLabV3(backbone, classifier)
    return model

def _load_model(arch_type, backbone, num_classes, output_stride=8, pretrained_backbone=True, use_attention=False):

    if backbone=='mobilenetv2':
        model = _segm_mobilenet(arch_type, backbone, num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone, use_attention=use_attention)
    elif backbone.startswith('resnet'):
        model = _segm_resnet(arch_type, backbone, num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone, use_attention=use_attention)
    elif backbone.startswith('hrnetv2'):
        model = _segm_hrnet(arch_type, backbone, num_classes, pretrained_backbone=pretrained_backbone, use_attention=use_attention)
    elif backbone=='xception':
        model = _segm_xception(arch_type, backbone, num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone, use_attention=use_attention)
    else:
        raise NotImplementedError
    return model


# Deeplab v3
def deeplabv3_hrnetv2_48(num_classes=21, output_stride=4, pretrained_backbone=False, use_attention=False): # no pretrained backbone yet
    return _load_model('deeplabv3', 'hrnetv2_48', num_classes, output_stride, pretrained_backbone=pretrained_backbone, use_attention=use_attention)

def deeplabv3_hrnetv2_32(num_classes=21, output_stride=4, pretrained_backbone=True, use_attention=False):
    return _load_model('deeplabv3', 'hrnetv2_32', num_classes, output_stride, pretrained_backbone=pretrained_backbone, use_attention=use_attention)

def deeplabv3_resnet50(num_classes=21, output_stride=8, pretrained_backbone=True, use_attention=False):
    """Constructs a DeepLabV3 model with a ResNet-50 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
        use_attention (bool): If True, use attention mechanism.
    """
    return _load_model('deeplabv3', 'resnet50', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone, use_attention=use_attention)

def deeplabv3_resnet101(num_classes=21, output_stride=8, pretrained_backbone=True, use_attention=False):
    """Constructs a DeepLabV3 model with a ResNet-101 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
        use_attention (bool): If True, use attention mechanism.
    """
    return _load_model('deeplabv3', 'resnet101', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone, use_attention=use_attention)

def deeplabv3_mobilenet(num_classes=21, output_stride=8, pretrained_backbone=True, use_attention=False, **kwargs):
    """Constructs a DeepLabV3 model with a MobileNetv2 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
        use_attention (bool): If True, use attention mechanism.
    """
    return _load_model('deeplabv3', 'mobilenetv2', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone, use_attention=use_attention)

def deeplabv3_xception(num_classes=21, output_stride=8, pretrained_backbone=True, use_attention=False, **kwargs):
    """Constructs a DeepLabV3 model with a Xception backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
        use_attention (bool): If True, use attention mechanism.
    """
    return _load_model('deeplabv3', 'xception', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone, use_attention=use_attention)


# Deeplab v3+
def deeplabv3plus_hrnetv2_48(num_classes=21, output_stride=4, pretrained_backbone=False, use_attention=False): # no pretrained backbone yet
    return _load_model('deeplabv3plus', 'hrnetv2_48', num_classes, output_stride, pretrained_backbone=pretrained_backbone, use_attention=use_attention)

def deeplabv3plus_hrnetv2_32(num_classes=21, output_stride=4, pretrained_backbone=True, use_attention=False):
    return _load_model('deeplabv3plus', 'hrnetv2_32', num_classes, output_stride, pretrained_backbone=pretrained_backbone, use_attention=use_attention)

def deeplabv3plus_resnet50(num_classes=21, output_stride=8, pretrained_backbone=True, use_attention=False):
    """Constructs a DeepLabV3 model with a ResNet-50 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
        use_attention (bool): If True, use attention mechanism.
    """
    return _load_model('deeplabv3plus', 'resnet50', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone, use_attention=use_attention)


def deeplabv3plus_resnet101(num_classes=21, output_stride=8, pretrained_backbone=True, use_attention=False):
    """Constructs a DeepLabV3+ model with a ResNet-101 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
        use_attention (bool): If True, use attention mechanism.
    """
    return _load_model('deeplabv3plus', 'resnet101', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone, use_attention=use_attention)


# 添加ResNet-18模型支持
def deeplabv3plus_resnet18(num_classes=21, output_stride=8, pretrained_backbone=True, use_attention=False):
    """Constructs a DeepLabV3+ model with a ResNet-18 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
        use_attention (bool): If True, use attention mechanism.
    """
    return _load_model('deeplabv3plus', 'resnet18', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone, use_attention=use_attention)


# 添加ResNet-34模型支持
def deeplabv3plus_resnet34(num_classes=21, output_stride=8, pretrained_backbone=True, use_attention=False):
    """Constructs a DeepLabV3+ model with a ResNet-34 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
        use_attention (bool): If True, use attention mechanism.
    """
    return _load_model('deeplabv3plus', 'resnet34', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone, use_attention=use_attention)


# 添加ResNet-152模型支持
def deeplabv3plus_resnet152(num_classes=21, output_stride=8, pretrained_backbone=True, use_attention=False):
    """Constructs a DeepLabV3+ model with a ResNet-152 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
        use_attention (bool): If True, use attention mechanism.
    """
    return _load_model('deeplabv3plus', 'resnet152', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone, use_attention=use_attention)


def deeplabv3plus_mobilenet(num_classes=21, output_stride=8, pretrained_backbone=True, use_attention=False):
    """Constructs a DeepLabV3+ model with a MobileNetv2 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
        use_attention (bool): If True, use attention mechanism.
    """
    return _load_model('deeplabv3plus', 'mobilenetv2', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone, use_attention=use_attention)

def deeplabv3plus_xception(num_classes=21, output_stride=8, pretrained_backbone=True, use_attention=False):
    """Constructs a DeepLabV3+ model with a Xception backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
        use_attention (bool): If True, use attention mechanism.
    """
    return _load_model('deeplabv3plus', 'xception', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone, use_attention=use_attention)