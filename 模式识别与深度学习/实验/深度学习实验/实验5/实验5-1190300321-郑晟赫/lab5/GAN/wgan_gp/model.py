#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/5/30 17:09
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


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(2, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        return self.layer(x)