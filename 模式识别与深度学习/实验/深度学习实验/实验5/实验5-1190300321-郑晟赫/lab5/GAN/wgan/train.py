#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/5/30 16:03
# @Author  : ZSH

import torch
import matplotlib.pyplot as plt
from GAN.util import test


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
                d_loss = -torch.mean(d_target) + torch.mean(d_gene)
                optimizer_d.zero_grad()
                d_loss.backward(retain_graph=True)
                optimizer_d.step()
                for p in discriminator.parameters():
                    p.data.clamp_(-config.clamp, config.clamp)
            d_gene = discriminator(g_out)
            g_loss = -torch.mean(d_gene)
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





