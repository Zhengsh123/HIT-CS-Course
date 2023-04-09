#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/5/29 21:08
# @Author  : ZSH
import torch
import torch.nn as nn
from GAN.gan.model import Generator, Discriminator
import matplotlib.pyplot as plt
import numpy as np
from GAN.util import back_draw,scatter_draw


def train(config, data,generator, discriminator):
    '''
    通用训练函数
    :param config:
    :param data:
    :param generator:
    :param discriminator:
    :return:
    '''
    batch_size=config.batch_size
    epoch_num=config.epoch
    data_size=len(data)
    optimizer_g=torch.optim.RMSprop(generator.parameters(),lr=config.lr)
    optimizer_d = torch.optim.RMSprop(discriminator.parameters(), lr=config.lr)
    for epoch in range(epoch_num):
        for i in range(data_size//batch_size):
            target=data[i * batch_size:(i + 1) * batch_size]
            g_input=torch.randn(batch_size,config.input_size).to(config.device)
            g_out = generator(g_input)
            # 计算loss
            # 训练判别器
            for j in range(5):
                d_target = discriminator(target)
                d_gene = discriminator(g_out.detach())
                d_loss = -torch.mean(torch.log(d_target)) - torch.mean(torch.log(1. - d_gene))
                optimizer_d.zero_grad()
                d_loss.backward(retain_graph=True)
                optimizer_d.step()
            d_gene = discriminator(g_out)
            g_loss = -torch.mean(torch.log(d_gene))
            # 训练分类器
            optimizer_g.zero_grad()
            g_loss.backward()
            optimizer_g.step()

            print("[%d,%d]: \t d_loss: %.3f \t g_loss: %.3f " % (epoch + 1, i + 1, d_loss.item(), g_loss.item()))
        if (epoch + 1) % 5 == 0:
            background = test(config,data,generator,discriminator)
            plt.savefig(config.picture_save_path + str(epoch + 1))
            background.remove()
            plt.cla()


def test(config,data,generator,discriminator):
    input=torch.randn(1000,config.input_size).to(config.device)
    out=np.array(generator(input).cpu().data)
    data= np.array(data.cpu().data)
    # 计算数据点边界
    x_min = min(np.min(data[:, 0]), np.min(out[:, 0]))
    x_max = max(np.max(data[:, 0]), np.max(out[:, 0]))
    y_min = min(np.min(data[:, 1]), np.min(out[:, 1]))
    y_max = max(np.max(data[:, 1]), np.max(out[:, 1]))
    boundary=(x_min, x_max, y_min, y_max)
    # 背景
    background=back_draw(config, discriminator, boundary)
    # 画拟合图
    scatter_draw(data, 'b', boundary)
    scatter_draw(out,  'r', boundary)

    return background


