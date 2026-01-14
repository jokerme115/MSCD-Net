import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict

class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model

    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.

    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.

    Arguments:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).

    Examples::

        >>> m = torchvision.models.resnet18(pretrained=True)
        >>> # extract layer1 and layer3, giving as names `feat1` and feat2`
        >>> new_m = torchvision.models._utils.IntermediateLayerGetter(m,
        >>>     {'layer1': 'feat1', 'layer3': 'feat2'})
        >>> out = new_m(torch.rand(1, 3, 224, 224))
        >>> print([(k, v.shape) for k, v in out.items()])
        >>>     [('feat1', torch.Size([1, 256, 56, 56])),
        >>>      ('feat2', torch.Size([1, 256, 14, 14]))]
    """
    def __init__(self, model, return_layers, hrnet_flag=False):
        # For HRNet, we need to explicitly specify the layers we want to use
        if hrnet_flag:
            # Create a new OrderedDict with the HRNet layers in the correct order
            layers = OrderedDict()
            for name in ['conv1', 'bn1', 'relu', 'conv2', 'bn2', 'relu', 
                        'layer1', 'transition1', 'stage2', 'transition2', 
                        'stage3', 'transition3', 'stage4']:
                if name in model._modules:
                    layers[name] = model._modules[name]
            
            # Check if all required layers are present
            required_layers = ['conv1', 'bn1', 'conv2', 'bn2', 'layer1']
            for layer in required_layers:
                if layer not in layers:
                    raise ValueError(f"Required layer {layer} not found in model")
            
            # Set return_layers for HRNet
            self.return_layers = return_layers
        else:
            # Original logic for other models
            if not set(return_layers).issubset([name for name, _ in model.named_children()]):
                raise ValueError("return_layers are not present in model")

            orig_return_layers = return_layers
            return_layers = {k: v for k, v in return_layers.items()}
            layers = OrderedDict()
            for name, module in model.named_children():
                layers[name] = module
                if name in return_layers:
                    del return_layers[name]
                if not return_layers:
                    break

        super(IntermediateLayerGetter, self).__init__(layers)
        if not hrnet_flag:
            self.return_layers = orig_return_layers
        self.hrnet_flag = hrnet_flag

    def forward(self, x):
        out = OrderedDict()
        
        # For HRNet, we need to manually handle the forward pass
        if self.hrnet_flag:
            # Forward pass through HRNet backbone
            # Stem
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu(x)

            # Stage 1
            x = self.layer1(x)
            layer1_output = x  # This is the low-level feature (256 channels)
            
            # Handle transitions and stages
            x_list = [trans(x) for trans in self.transition1]
            x_list = self.stage2(x_list)
            x_list.append(self.transition2(x_list[-1]))
            x_list = self.stage3(x_list)
            x_list.append(self.transition3(x_list[-1]))
            x_list = self.stage4(x_list)

            # Final concatenation (this should give us 480 channels)
            output_h, output_w = x_list[0].size(2), x_list[0].size(3)
            x1 = F.interpolate(x_list[1], size=(output_h, output_w), mode='bilinear', align_corners=False)
            x2 = F.interpolate(x_list[2], size=(output_h, output_w), mode='bilinear', align_corners=False)
            x3 = F.interpolate(x_list[3], size=(output_h, output_w), mode='bilinear', align_corners=False)
            final_output = torch.cat([x_list[0], x1, x2, x3], dim=1)  # Should be 480 channels
            
            # Assign outputs according to return_layers
            for name, return_name in self.return_layers.items():
                if name == 'layer1':
                    out[return_name] = layer1_output
                # We don't have other named layers in HRNet case
            
            # Add the final output
            out['out'] = final_output
            
            return out
        else:
            # Original logic for other backbones
            for name, module in self.named_children():
                x = module(x)
                if name in self.return_layers:
                    out[self.return_layers[name]] = x
            return out


# CBAM注意力模块实现
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_planes, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out