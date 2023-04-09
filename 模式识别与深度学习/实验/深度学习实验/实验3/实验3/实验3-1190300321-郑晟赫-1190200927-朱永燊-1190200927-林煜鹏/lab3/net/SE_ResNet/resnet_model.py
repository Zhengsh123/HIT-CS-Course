# -*- coding: utf-8 -*-
"""
Created on Fri May 27 20:34:14 2022

@author: Marmalade
"""
import torch
from torch import nn
import torch as t
from torch.nn import functional as F


# 残差单元
class ResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, shortcut=None):
        # 卷积
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channel)
        )
        self.right = shortcut

    def forward(self, x):
        # 先恒等映射，然后加上卷积后的out再relu
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)


# SE模块
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化，输入BCHW -> 输出 B*C*1*1
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),  # 可以看到channel得被reduction整除，否则可能出问题
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)  # 得到B*C*1*1,然后转成B*C，才能送入到FC层中。
        y = self.fc(y).unsqueeze(2).unsqueeze(2)  # 得到B*C的向量，C个值就表示C个通道的权重。把B*C变为B*C*1*1是为了与四维的x运算。
        return x * y  # 先把B*C*1*1变成B*C*H*W大小，其中每个通道上的H*W个值都相等。*表示对应位置相乘。


# SE基本块，结构和残差块类似，只是内部加上SELayer
class SEBasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, shortcut=None):
        # 卷积
        super(SEBasicBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            SELayer(out_channel)
        )
        self.right = shortcut

    def forward(self, x):
        # 先恒等映射，然后加上卷积后的out再relu
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)


class SEResNet18(nn.Module):
    def __init__(self, num_classes=12):
        super(SEResNet18, self).__init__()
        # 先做 7x7 卷积
        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),  # 输入 3 通道，输出 64 通道，卷积核7x7，步长2，padding 3
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 1, 1)  # in_channel，out_channel，padding
        )
        # 共32层
        self.layer1 = self.make_layer(64, 128, 2)  # 2 个 64 通道的残差单元，输出 128通道，共6层
        self.layer2 = self.make_layer(128, 256, 2, stride=2)  # 2 个 128通道的残差单元，输出 256通道，共8层
        self.layer3 = self.make_layer(256, 512, 2, stride=2)  # 2 个 256通道的残差单元，输出 512通道，共12层
        self.layer4 = self.make_layer(512, 512, 2, stride=2)  # 2 个 512通道的残差单元，输出 512通道，共6层
        # fc，1层
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, in_channel, out_channel, block_num, stride=1):
        # 1x1 卷积 改变通道数
        shortcut = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channel)
        )
        layers = [SEBasicBlock(in_channel, out_channel, stride, shortcut)]
        # 再接几个同样的残差单元，通道数不变
        for i in range(1, block_num):  # block_num
            layers.append(SEBasicBlock(out_channel, out_channel))
        return nn.Sequential(*layers)

    def forward(self, x):
        # 第1层
        x = self.pre(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # 最后的池化把一个 feature map 变成一个特征，故池化野大小等于最后 x 的大小
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class ResNet18(nn.Module):
    def __init__(self, num_classes=12):
        super(ResNet18, self).__init__()
        # 先做 7x7 卷积
        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),  # 输入 3 通道，输出 64 通道，卷积核7x7，步长2，padding 3
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 1, 1)  # in_channel，out_channel，padding
        )
        # 共16层
        self.layer1 = self.make_layer(64, 128, 2)  # 2 个 64 通道的残差单元，输出 128通道，共6层
        self.layer2 = self.make_layer(128, 256, 2, stride=2)  # 2 个 128通道的残差单元，输出 256通道，共8层
        self.layer3 = self.make_layer(256, 512, 2, stride=2)  # 2 个 256通道的残差单元，输出 512通道，共12层
        self.layer4 = self.make_layer(512, 512, 2, stride=2)  # 2 个 512通道的残差单元，输出 512通道，共6层
        # fc和global_avg，1层
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, in_channel, out_channel, block_num, stride=1):
        # 1x1 卷积 改变通道数
        shortcut = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channel)
        )
        layers = [ResidualBlock(in_channel, out_channel, stride, shortcut)]
        # 再接几个同样的残差单元，通道数不变
        for i in range(1, block_num):  # block_num
            layers.append(ResidualBlock(out_channel, out_channel))
        return nn.Sequential(*layers)

    def forward(self, x):
        # 第1层
        x = self.pre(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # 最后的池化把一个 feature map 变成一个特征，故池化野大小等于最后 x 的大小
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

