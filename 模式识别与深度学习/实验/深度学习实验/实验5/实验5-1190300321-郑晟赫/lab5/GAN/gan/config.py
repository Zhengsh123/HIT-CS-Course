#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/5/29 21:23
# @Author  : ZSH

import torch


class GanConfig():
    def __init__(self):
        self.data_path='../points.mat'
        self.input_size=4 # 生成器输入噪声维度
        self.output_size=2 # 拟合图片的维度
        # 训练设置
        self.batch_size=200
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lr = 0.00002
        self.epoch=100

        # 保存路径
        self.generator_save_path='../train/GAN_generator.pt'
        self.discriminator_save_path = '../train/GAN_discriminator.pt'
        self.log_path='../train/log/GAN/'
        self.picture_save_path='../res/GAN/'



