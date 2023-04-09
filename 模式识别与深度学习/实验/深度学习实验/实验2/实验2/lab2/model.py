#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/5/2 10:39
# @Author  : ZSH
import torch
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self, config,is_init=True):
        super(AlexNet, self).__init__()
        self.class_num=config['class_num']
        self.is_init=is_init
        self.inner=nn.Sequential(
            # input [3,224,224]
            nn.Conv2d(in_channels=3,out_channels=96,kernel_size=(11,11),stride=(4,4),padding=2),
            nn.ReLU(inplace=True),
            # output[96,55,55]
            nn.MaxPool2d(kernel_size=(3,3),stride=2),
            # output[96,27,27]
            nn.LocalResponseNorm(size=5,alpha=1e-4, beta=0.75, k=2),
            # output[96,27,27]

            nn.Conv2d(in_channels=96,out_channels=256,kernel_size=(5,5),stride=(1,1),padding=2),
            # output [256,27,27]
            nn.ReLU(inplace=True),
            # output [256,27,27]
            nn.MaxPool2d(kernel_size=(3,3),stride=2),
            # output [256,13,13]
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
            # output [256,13,13]

            nn.Conv2d(in_channels=256,out_channels=384,kernel_size=(3,3),stride=(1,1),padding=1),
            # output [384,13,13]
            nn.ReLU(inplace=True),
            # output [384,13,13]

            nn.Conv2d(in_channels=384,out_channels=384,kernel_size=(3,3),stride=(1,1),padding=1),
            nn.ReLU(inplace=True),
            # output [384,13,13]

            nn.Conv2d(in_channels=384,out_channels=256,kernel_size=(3,3),stride=(1,1),padding=1),
            nn.ReLU(inplace=True),
            # output [256,13,13]
            nn.MaxPool2d(kernel_size=(3,3),stride=2)
            # output [256,6,6]
        )
        self.flattern=nn.Flatten()
        self.linear1=nn.Sequential(
            nn.Linear(in_features=256*6*6,out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5)
        )
        self.linear2 = nn.Sequential(
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5)
        )
        self.linear3=nn.Linear(in_features=4096,out_features=self.class_num)

        if is_init:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)  # 正态分布赋值
                nn.init.constant_(m.bias, 0)

    def forward(self,x):
        x=self.inner(x)
        x=self.flattern(x)
        x=self.linear1(x)
        x=self.linear2(x)
        x=self.linear3(x)
        return x


