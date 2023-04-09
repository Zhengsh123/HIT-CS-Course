#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/5/30 15:56
# @Author  : ZSH

import torch


class WGanGPConfig():
    def __init__(self):
        self.data_path='../points.mat'
        self.input_size=4 # 生成器输入噪声维度
        self.output_size=2 # 拟合图片的维度
        self.gp_lambda=1 # WGAN_GP的参数
        # 训练设置
        self.batch_size=200
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lr = 0.00006
        self.epoch=500

        # 保存路径
        self.generator_save_path='../train/WGANGP_generator.pt'
        self.discriminator_save_path = '../train/WGANGP_discriminator.pt'
        self.log_path='../train/log/WGAN_GP/'
        self.picture_save_path='../res/WGAN_GP/'



