"""
Code is adapted from: https://github.com/chenfengxu714/SqueezeSegV3 
"""

from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from .encoder import BaseEncoder
from .decoder import BaseDecoder


class SACBlock(nn.Module):
  def __init__(self, inplanes, expand1x1_planes1):

    super(SACBlock, self).__init__()
    self.inplanes = inplanes

    self.attention_x = nn.Sequential(
            nn.Conv2d(3, 9 * self.inplanes, kernel_size = 7, padding = 3),
            nn.BatchNorm2d(9 * self.inplanes, momentum = 0.1),
            )

    self.position_mlp_2 = nn.Sequential(
            nn.Conv2d(9 * self.inplanes, self.inplanes, kernel_size = 1),
            nn.BatchNorm2d(self.inplanes, momentum = 0.1),
            nn.ReLU(inplace = True),
            nn.Conv2d(self.inplanes, self.inplanes, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(self.inplanes, momentum = 0.1),
            nn.ReLU(inplace = True),
            )

  def forward(self, input):
    xyz = input[0]
    new_xyz= input[1]
    feature = input[2]
    N, C, H, W = feature.size()

    new_feature = F.unfold(feature, kernel_size = 3, padding = 1).view(N, -1, H, W)
    attention = F.sigmoid(self.attention_x(new_xyz))
    new_feature = new_feature * attention
    new_feature = self.position_mlp_2(new_feature)
    fuse_feature = new_feature + feature
   
    return xyz, new_xyz, fuse_feature

# Encoder
model_blocks = {
  21: [1, 1, 2, 2, 1]
}

class SqueezeSegEncoder(BaseEncoder):
    """
    SqueezeSegEncoder that builds on SAC blocks
    """
    def __init__ (self, num_inchannel, drop_prob):
        super().__init__()
        self._forward_connection = ['context', 'layer1', 'layer2', 'layer3', 'layer4', 'layer5']

        self.use_xyz = True  # This is some kind of parameter from the parameter-settings

        self.drop_prob = drop_prob
        self.input_depth = 0
        self.input_idxs = []
        if self.use_xyz:
          self.input_depth += 3
          self.input_idxs.extend([1, 2, 3])
        
        self.strides = [2, 2, 2, 1, 1]

        self.layers = 21  # Hard code this here
        assert self.layers in model_blocks.keys()


        self.blocks = model_blocks[self.layers]

        self.conv1 = nn.Conv2d(self.input_depth, 32, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32, momentum=0.1)  # momentum=self.bn_d
        self.relu1 = nn.LeakyReLU(0.1)

        self.enc1 = self._make_enc_layer(SACBlock, [32, 64], self.blocks[0],
                                        stride=self.strides[0], DS=True)
        self.enc2 = self._make_enc_layer(SACBlock, [64, 128], self.blocks[1],
                                        stride=self.strides[1], DS=True)
        self.enc3 = self._make_enc_layer(SACBlock, [128, 256], self.blocks[2],
                                        stride=self.strides[2], DS=True)
        self.enc4 = self._make_enc_layer(SACBlock, [256, 256], self.blocks[3],
                                        stride=self.strides[3], DS=False)
        self.enc5 = self._make_enc_layer(SACBlock, [256, 256], self.blocks[4],
                                        stride=self.strides[4], DS=False)

        self.dropout = nn.Dropout2d(self.drop_prob)

        self.last_channels = 256

    def run_layer(self, xyz, feature, layer, flag=True):
      """
      Helper function for doing forward pass
      """
      new_xyz = xyz    
      if flag == True:
        xyz, new_xyz, y = layer[:-3]([xyz, new_xyz, feature])
        y = layer[-3:](y)
        xyz = F.upsample_bilinear(xyz, size=[xyz.size()[2], xyz.size()[3]//2])
      else:
        xyz,new_xyz,y = layer([xyz, new_xyz, feature])
      if y.shape[2] < feature.shape[2] or y.shape[3] < feature.shape[3]:
        skips = feature.detach()
      feature = self.dropout(y)
      return xyz, feature, skips

    def _make_enc_layer(self, block, planes, blocks, stride, DS, bn_d=0.1):
      """
      Helper function for creating a layer
      """
      layers = []

      inplanes = planes[0]
      for i in range(0, blocks):
        layers.append(("residual_{}".format(i),
                      block(inplanes, planes)))
      if DS==True:
          layers.append(("conv", nn.Conv2d(planes[0], planes[1],
                                      kernel_size=3,
                                      stride=[1, stride], dilation=1,
                                      padding=1, bias=False)))
          layers.append(("bn", nn.BatchNorm2d(planes[1], momentum=bn_d)))
          layers.append(("relu", nn.LeakyReLU(0.1)))
      
      return nn.Sequential(OrderedDict(layers))


    def forward(self, feature):
        out = {}
        print("feature.keys()", feature.keys())
        xyz = feature[:,1:4,:,:]
        feature = self.relu1(self.bn1(self.conv1(feature)))

        out[self._forward_connection[0]] = feature
        xyz,feature, skips = self.run_layer(xyz,feature, self.enc1)
        out[self._forward_connection[1]] = skips
        xyz,feature, skips = self.run_layer(xyz,feature, self.enc2)
        out[self._forward_connection[2]] = skips
        xyz,feature, skips = self.run_layer(xyz,feature, self.enc3)
        out[self._forward_connection[3]] = skips
        xyz,feature, skips = self.run_layer(xyz,feature, self.enc4, flag=False)
        out[self._forward_connection[4]] = skips
        xyz,feature, skips = self.run_layer(xyz,feature, self.enc5, flag=False)
        out[self._forward_connection[5]] = skips

        return feature, skips


    @property
    def _connection(self):
        return self._forward_connection



# Decode (right now based totally on SalsaNext setup)
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

class SqueezeSegDecoder(BaseDecoder):
    def __init__(self):
        super().__init__()

        self._receive_connection = ['context', 'layer1', 'layer2', 'layer3', 'layer4', 'layer5']

        self.upBlock1 = UpBlock(2 * 4 * 32, 4 * 32, 0.2)
        self.upBlock2 = UpBlock(4 * 32, 4 * 32, 0.2)
        self.upBlock3 = UpBlock(4 * 32, 2 * 32, 0.2)
        self.upBlock4 = UpBlock(2 * 32, 32, 0.2, drop_out=False)      


    def forward (self, x):
        context = x[self._receive_connection[0]]
        rev5 = x[self._receive_connection[5]]
        rev4 = x[self._receive_connection[4]]
        rev3 = x[self._receive_connection[3]]
        rev2 = x[self._receive_connection[2]]
        rev1 = x[self._receive_connection[1]]
        up5e = self.upBlock1(context, rev5)
        up4e = self.upBlock1(up5e, rev4)
        up3e = self.upBlock2(up4e, rev3)
        up2e = self.upBlock3(up3e, rev2)
        up1e = self.upBlock4(up2e, rev1)

        return up1e

    @property
    def _connection(self):
        return self._receive_connection



class SqueezeSegHead(nn.Module):
    def __init__(self, num_features, num_cls):
        super().__init__()

        self.logits = nn.Conv2d(num_features, num_cls, kernel_size=(1, 1))

    
    def forward(self, x):
        logits = self.logits(x)

        logits = F.softmax(logits, dim=1)

        return logits
