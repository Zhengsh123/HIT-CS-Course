#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/5/29 22:36
# @Author  : ZSH
'''
一些通用函数
'''
import torch
import numpy as np
import matplotlib.pyplot as plt
import imageio
import os


def back_draw(config, discriminator,boundary):
    x_min, x_max, y_min, y_max=boundary
    background = []
    for i in np.arange(x_min, x_max, 0.01):
        for j in np.arange(y_min, y_max, 0.01):
            background.append([i, j])

    background.append([x_max, y_max])
    color = discriminator(torch.Tensor(background).to(config.device))
    background = np.array(background)
    cm = plt.cm.get_cmap('binary')
    sc = plt.scatter(background[:, 0], background[:, 1], c= np.squeeze(color.cpu().data), cmap=cm)
    bar = plt.colorbar(sc)
    return bar


def scatter_draw(data, color, boundary):
    x_min, x_max, y_min, y_max=boundary
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.title('Scatter Plot')
    plt.xlabel("X", fontsize=10)
    plt.ylabel("Y", fontsize=10)
    plt.scatter(data[:, 0], data[:, 1], c=color, s=10)


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


def generate_gif(data_path,save_path):
    """
    根据图片生成gif
    :param data_path:图片存储的地址文件夹
    :param save_path:gif存储的地址
    :return:
    """
    dirs = os.listdir(data_path)
    length=len(dirs)
    file=[]
    for i in range(length):
        file.append(imageio.imread(data_path+str(5*(i+1))+'.png'))
    imageio.mimsave(save_path, file, fps=5)  # 转化为gif动画


if __name__=="__main__":
    generate_gif('./res/GAN/RMSprop/','./GAN.gif')
    # generate_gif('./res/WGAN/', './WGAN.gif')
    # generate_gif('./res/WGAN_GP/RMSprop/', './WGAN_GP.gif')