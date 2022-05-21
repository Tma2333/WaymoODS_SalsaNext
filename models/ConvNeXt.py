# Partially reference from https://github.com/facebookresearch/ConvNeXt

from types import TracebackType
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import BaseEncoder
from .decoder import BaseDecoder

class ConvNeXtContextBlock(nn.Module):
    def __init__(self, filters, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(filters, filters, kernel_size=7, padding=3, groups=filters) # depthwise conv
        self.norm = LayerNorm(filters, eps=1e-6)
        self.pwconv1 = nn.Linear(filters, 4 * filters) # pointwise/1x1 convs, implemented with linear layers
        self.act2 = nn.GELU()
        self.pwconv2 = nn.Linear(4 * filters, filters)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((filters)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act2(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + x
        return x


class ConvNeXtBlock(nn.Module):
    def __init__(self, in_filters, out_filters, drop_path=0., layer_scale_init_value=1e-6, downsample=False,
                       dropout_rate = 0.2, drop_out=False):
        super().__init__()
        self.pwconv0 = nn.Conv2d(in_filters, out_filters, kernel_size=1, stride=1)
        self.dwconv = nn.Conv2d(out_filters, out_filters, kernel_size=7, padding=3, groups=out_filters) # depthwise conv
        self.ln1 = LayerNorm(out_filters, eps=1e-6)
        self.pwconv1 = nn.Linear(out_filters, 4 * out_filters) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * out_filters, out_filters)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((out_filters)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.downsample = downsample
        self.drop_out = drop_out
        # if self.downsample:
        #     self.ln2 = LayerNorm(out_filters, eps=1e-6, data_format="channels_first")
        #     self.down_conv = nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=2, padding=1)
        if downsample:
            self.dropout = nn.Dropout2d(p=dropout_rate)
            self.pool = nn.AvgPool2d(kernel_size=(3,3), stride=2, padding=1)
        else:
            self.dropout = nn.Dropout2d(p=dropout_rate)

    def forward(self, x):
        x = self.pwconv0(x)
        skip = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.ln1(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = skip + self.drop_path(x)
        # if self.downsample:
        #     x = self.ln2(x)
        #     down_x = self.down_conv(x)
        #     return down_x, x
        # else:
        #     return x

        if self.downsample:
            if self.drop_out:
                down_x = self.dropout(x)
            else:
                down_x = x
            down_x = self.pool(down_x)

            return down_x, x
        else:
            if self.drop_out:
                down_x = self.dropout(x)
            else:
                down_x = x
            return down_x

class ConvNeXtEncoder (BaseEncoder):
    def __init__ (self, num_inchannel):
        super().__init__()
        self._forward_connection = ['context', 'layer1', 'layer2', 'layer3', 'layer4']

        self.init_conv = nn.Conv2d(num_inchannel, 32, kernel_size=(7, 7), stride=1, padding=3, bias=False)
        self.act = nn.GELU()

        self.convNeXtBlock1 = ConvNeXtBlock(32, 64, downsample=True, drop_out=False)

        self.convNeXtBlock2_1 = ConvNeXtBlock(64, 128, downsample=False, dropout_rate=0.2, drop_out=True)
        self.convNeXtBlock2_2 = ConvNeXtBlock(128, 128, downsample=True, dropout_rate=0.2, drop_out=True)

        self.convNeXtBlock3_1 = ConvNeXtBlock(128, 256, downsample=False, dropout_rate=0.2, drop_out=True)
        self.convNeXtBlock3_2 = ConvNeXtBlock(256, 256, downsample=False, dropout_rate=0.2, drop_out=True)
        self.convNeXtBlock3_3 = ConvNeXtBlock(256, 256, downsample=True, dropout_rate=0.2, drop_out=True)

        self.convNeXtBlock4_1 = ConvNeXtBlock(256, 512, downsample=False, dropout_rate=0.2, drop_out=True)
        self.convNeXtBlock4_2 = ConvNeXtBlock(512, 512, downsample=True, dropout_rate=0.2, drop_out=True)

        self.convNeXtBlock5 = ConvNeXtBlock(512, 512, downsample=False, dropout_rate=0.2, drop_out=True)


    def forward(self, x):
        out = {}
        downCntx = self.init_conv(x)
        downCntx = self.act(downCntx)

        down0, skip0 = self.convNeXtBlock1(downCntx)
        out[self._forward_connection[1]] = skip0

        down1 = self.convNeXtBlock2_1(down0)
        down1, skip1 = self.convNeXtBlock2_2(down1)
        out[self._forward_connection[2]] = skip1

        down2 = self.convNeXtBlock3_1(down1)
        down2 = self.convNeXtBlock3_2(down2)
        down2, skip2= self.convNeXtBlock3_3(down2)
        out[self._forward_connection[3]] = skip2

        down3 = self.convNeXtBlock4_1(down2)
        down3, skip3= self.convNeXtBlock4_2(down3)
        out[self._forward_connection[4]] = skip3

        down4 = self.convNeXtBlock5(down3)
        out[self._forward_connection[0]] = down4

        return out


    @property
    def _connection(self):
        return self._forward_connection


class ConvNeXtDecoder (BaseDecoder):
    def __init__(self):
        super().__init__()

        self._receive_connection = ['context', 'layer1', 'layer2', 'layer3', 'layer4']

        self.upBlock1 = UpBlock(512, 256, 0.2)
        self.upBlock2 = UpBlock(256, 128, 0.2)
        self.upBlock3 = UpBlock(128, 64, 0.2)
        self.upBlock4 = UpBlock(64, 32, 0.2, drop_out=False)      
        self.conv = nn.Conv2d(480, 32, kernel_size=3, stride=1, padding=1)


    def forward (self, x):
        context = x[self._receive_connection[0]]
        rev4 = x[self._receive_connection[4]]
        rev3 = x[self._receive_connection[3]]
        rev2 = x[self._receive_connection[2]]
        rev1 = x[self._receive_connection[1]]
        up4e = self.upBlock1(context, rev4)
        up3e = self.upBlock2(up4e, rev3)
        up2e = self.upBlock3(up3e, rev2)
        up1e = self.upBlock4(up2e, rev1)

        return up1e

    @property
    def _connection(self):
        return self._receive_connection

# class ConvNeXtEncoder (BaseEncoder):
#     def __init__ (self, num_inchannel):
#         super().__init__()
#         self._forward_connection = ['context', 'layer1', 'layer2', 'layer3', 'layer4']

#         self.init_conv = nn.Conv2d(num_inchannel, 32, kernel_size=(1, 1), stride=1)
#         self.act = nn.GELU()
#         self.downCntx1 = ConvNeXtContextBlock(32)
#         self.downCntx2 = ConvNeXtContextBlock(32)
#         self.downCntx3 = ConvNeXtContextBlock(32)

#         self.convNeXtBlock1 = ConvNeXtBlock(32, 64, downsample=True)
#         self.convNeXtBlock2 = ConvNeXtBlock(64, 128, downsample=True)
#         self.convNeXtBlock3 = ConvNeXtBlock(128, 256, downsample=True)
#         self.convNeXtBlock4 = ConvNeXtBlock(256, 256, downsample=True)
#         self.convNeXtBlock5 = ConvNeXtBlock(256, 256, downsample=False)


#     def forward(self, x):
#         out = {}
#         downCntx = self.init_conv(x)
#         downCntx = self.act(downCntx)
#         downCntx = self.downCntx1(downCntx)
#         downCntx = self.downCntx2(downCntx)
#         downCntx = self.downCntx3(downCntx)
#         down0, skip0 = self.convNeXtBlock1(downCntx)
#         out[self._forward_connection[1]] = skip0
#         down1, skip1 = self.convNeXtBlock2(down0)
#         out[self._forward_connection[2]] = skip1
#         down2, skip2 = self.convNeXtBlock3(down1)
#         out[self._forward_connection[3]] = skip2
#         down3, skip3= self.convNeXtBlock4(down2)
#         out[self._forward_connection[4]] = skip3
#         down4 = self.convNeXtBlock5(down3)
#         out[self._forward_connection[0]] = down4

#         return out


#     @property
#     def _connection(self):
#         return self._forward_connection


class UpBlock(nn.Module):
    def __init__(self, in_filters, out_filters, dropout_rate, drop_out=True):
        super(UpBlock, self).__init__()
        self.drop_out = drop_out
        self.in_filters = in_filters
        self.out_filters = out_filters

        self.dropout1 = nn.Dropout2d(p=dropout_rate)

        self.dropout2 = nn.Dropout2d(p=dropout_rate)

        self.conv1 = nn.Conv2d(in_filters//4 + 2*out_filters, out_filters, (3,3), padding=1)
        self.act1 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_filters)

        self.conv2 = nn.Conv2d(out_filters, out_filters, (3,3),dilation=2, padding=2)
        self.act2 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm2d(out_filters)

        self.conv3 = nn.Conv2d(out_filters, out_filters, (2,2), dilation=2,padding=1)
        self.act3 = nn.LeakyReLU()
        self.bn3 = nn.BatchNorm2d(out_filters)


        self.conv4 = nn.Conv2d(out_filters*3,out_filters,kernel_size=(1,1))
        self.act4 = nn.LeakyReLU()
        self.bn4 = nn.BatchNorm2d(out_filters)

        self.dropout3 = nn.Dropout2d(p=dropout_rate)

    def forward(self, x, skip):
        upA = nn.PixelShuffle(2)(x)
        if self.drop_out:
            upA = self.dropout1(upA)

        w_diff = skip.shape[-1] - upA.shape[-1]
        h_diff = skip.shape[-2] - upA.shape[-2]
        # truncate for odd dimension
        if w_diff < 0 and h_diff < 0:
            upB = torch.cat((upA[:, :, :h_diff, :w_diff], skip),dim=1)
        elif w_diff < 0:
            upB = torch.cat((upA[:, :, :, :w_diff], skip),dim=1)
        elif h_diff < 0:
            upB = torch.cat((upA[:, :, :h_diff, :], skip),dim=1)
        else:
            upB = torch.cat((upA, skip),dim=1)
        
        if self.drop_out:
            upB = self.dropout2(upB)

        upE = self.conv1(upB)
        upE = self.act1(upE)
        upE1 = self.bn1(upE)

        upE = self.conv2(upE1)
        upE = self.act2(upE)
        upE2 = self.bn2(upE)

        upE = self.conv3(upE2)
        upE = self.act3(upE)
        upE3 = self.bn3(upE)

        concat = torch.cat((upE1,upE2,upE3),dim=1)
        upE = self.conv4(concat)
        upE = self.act4(upE)
        upE = self.bn4(upE)
        if self.drop_out:
            upE = self.dropout3(upE)

        return upE

    
class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

# Following is necessary part from timm module
# Following DropPath implementation is taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)