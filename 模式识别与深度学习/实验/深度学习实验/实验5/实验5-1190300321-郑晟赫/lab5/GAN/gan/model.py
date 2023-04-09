#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/5/29 21:42
# @Author  : ZSH
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.input_size = config.input_size
        self.output_size = config.output_size
        self.layer = nn.Sequential(
            nn.Linear(self.input_size, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, self.output_size),
        )

    def forward(self, x):
        return self.layer(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(2, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.layer(x)


if __name__=="__main__":
    from GAN.gan.config import GanConfig
    import torch
    config=GanConfig()
    ge=Generator(config)
    a=torch.rand((150,6))
    print(ge(a))