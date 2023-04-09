# -*- coding: utf-8 -*-
"""
Created on Fri May 27 20:34:14 2022

@author: Marmalade
"""

from torch import nn
import torch as t
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    ### 残差单元
    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        ### 卷积
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 3, stride, 1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.right = shortcut

    def forward(self, x):
        ### 先恒等映射，然后加上卷积后的out再relu
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)


class ResNet18(nn.Module):
    def __init__(self, num_classes=12):
        super(ResNet18, self).__init__()
        ### 先做 7x7 卷积
        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),  ### 输入 3 通道，输出 64 通道，卷积核7x7，步长2，padding 3
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 1, 1)  ### inchannel，outchannel，padding
        )
        ### 共32层
        self.layer1 = self._make_layer(64, 128, 2)  ### 3 个 64 通道的残差单元，输出 128通道，共6层
        self.layer2 = self._make_layer(128, 256, 2, stride=2)  ### 4 个 128通道的残差单元，输出 256通道，共8层
        self.layer3 = self._make_layer(256, 512, 2, stride=2)  ### 6 个 256通道的残差单元，输出 512通道，共12层
        self.layer4 = self._make_layer(512, 512, 2, stride=2)  ### 3 个 512通道的残差单元，输出 512通道，共6层
        ### fc，1层
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, inchannel, outchannel, block_num, stride=1):
        ### 1x1 卷积 改变通道数
        shortcut = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 1, stride, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        layers = []
        layers.append(ResidualBlock(inchannel, outchannel, stride, shortcut))  ### 先来一个残差单元，主要是改变通道数
        ### 再接几个同样的残差单元，通道数不变
        for i in range(1, block_num + 1):  ### block_num
            layers.append(ResidualBlock(outchannel, outchannel))
        return nn.Sequential(*layers)

    def forward(self, x):
        ### 第1层
        x = self.pre(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        ### 最后的池化把一个 feature map 变成一个特征，故池化野大小等于最后 x 的大小
        x = F.avg_pool2d(x, 14)  ###

        x = x.view(x.size(0), -1)
        return self.fc(x)