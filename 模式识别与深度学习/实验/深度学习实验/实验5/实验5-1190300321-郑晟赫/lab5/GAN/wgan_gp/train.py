#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/5/30 16:03
# @Author  : ZSH

import torch
import matplotlib.pyplot as plt
from GAN.util import test
import torch.autograd as autograd
import numpy as np


def train(config, data, generator, critic):
    '''
    通用训练函数
    :param config:
    :param data:
    :param generator:
    :param critic:
    :return:
    '''
    batch_size=config.batch_size
    epoch_num=config.epoch
    data_size=len(data)
    optimizer_g=torch.optim.Adam(generator.parameters(),lr=config.lr)
    optimizer_d = torch.optim.Adam(critic.parameters(), lr=config.lr)
    for epoch in range(epoch_num):
        for i in range(data_size//batch_size):
            target=data[i * batch_size:(i + 1) * batch_size]
            g_input=torch.randn(batch_size,config.input_size).to(config.device)
            g_out = generator(g_input)
            for j in range(5):
                d_target = critic(target)
                d_gene = critic(g_out.detach())
                d_penalty=cal_penalty(target,g_out,critic,config)
                d_loss = -torch.mean(d_target) + torch.mean(d_gene)+config.gp_lambda*d_penalty
                optimizer_d.zero_grad()
                d_loss.backward(retain_graph=True)
                optimizer_d.step()
            d_gene = critic(g_out)
            g_loss = -torch.mean(d_gene)
            # 训练分类器
            optimizer_g.zero_grad()
            g_loss.backward()
            optimizer_g.step()

            print("[%d,%d]: \t d_loss: %.3f \t g_loss: %.3f " % (epoch + 1, i + 1, d_loss.item(), g_loss.item()))
        if (epoch + 1) % 5 == 0:
            background = test(config, data, generator, critic)
            plt.savefig(config.picture_save_path + str(epoch + 1))
            background.remove()
            plt.cla()


def cal_penalty(real,generate,critic,config):
    '''
    计算loss中的惩罚项
    :param real:真实的点分布
    :param generate:生成的点分布
    :param config:设置
    :return:tensor 惩罚项
    '''
    epsilon=torch.rand(real.shape).to(config.device)
    x_hat=(epsilon*real+(1.-epsilon)*generate).requires_grad_(True)
    c_gene=critic(x_hat)
    grad_output=autograd.Variable(torch.from_numpy(np.ones((real.shape[0],1))),requires_grad=False)
    gradients = autograd.grad(
        outputs=c_gene,
        inputs=x_hat,
        grad_outputs=grad_output,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return penalty





