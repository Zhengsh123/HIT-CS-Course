#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/5/30 15:56
# @Author  : ZSH

import torch


class WGanConfig():
    def __init__(self):
        self.data_path='../points.mat'
        self.input_size=4 # 生成器输入噪声维度
        self.output_size=2 # 拟合图片的维度
        self.clamp=0.1 # 截断范围
        # 训练设置
        self.batch_size=200
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lr = 0.00002
        self.epoch=500

        # 保存路径
        self.generator_save_path='../train/WGAN_generator.pt'
        self.discriminator_save_path = '../train/WGAN_discriminator.pt'
        self.log_path='../train/log/WGAN/'
        self.picture_save_path='../res/WGAN/'



