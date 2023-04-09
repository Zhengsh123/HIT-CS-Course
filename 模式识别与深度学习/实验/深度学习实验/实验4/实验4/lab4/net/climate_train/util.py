#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/5/25 22:15
# @Author  : ZSH
"""
用于climate数据集上的训练
"""
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def train(config, model, train_loader,t):
    writer = SummaryWriter(config.log_path)
    start_time=time.time()
    optim=torch.optim.Adam(model.parameters(),lr=config.lr)
    criterian=nn.SmoothL1Loss()
    train_loss = 0.0
    test_step=0
    best_loss_test=10000.0
    device=config.device
    for epoch in range(config.epoch):
        for i, data in enumerate(train_loader):
            input = data['data']
            target = data['target']
            input, target = input.to(device), target.to(device)
            optim.zero_grad()
            output = model(input)
            loss = criterian(output, target)
            loss.backward()
            optim.step()
            train_loss += loss.item()  # 只取loss的值，不作为tensor加减

            writer.add_scalar(tag="training_loss", scalar_value=loss.item(),
                              global_step=epoch * len(train_loader) + i)

            if i % 10 == 9:
                print("[%d,%d],loss:%.3f" % (epoch, i, train_loss/10))
                train_loss=0.0

            if i%50==49:
                output = output.cpu().data.numpy()[0]
                target = target.cpu().data.numpy()[0]
                max_t=t[0]
                min_t = t[1]
                max_min=max_t-min_t
                output=output*max_min+min_t
                target = target * max_min + min_t

                x = range(len(output))

                plt.plot(x, output)
                plt.plot(x, target, color='red', linewidth=1, linestyle='--')

                plt.show()
    end_time = time.time()
    print("Train Time : {:.3f} s,best loss in test:{}".format(end_time-start_time,best_loss_test))


def test(config, model, test_loader,t):
    test_loss_l1=0.0
    test_loss_l2=0.0
    criterian1 = nn.L1Loss()
    criterian2= nn.MSELoss()
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            input = data['data']
            target = data['target']
            input, target = input.to(config.device), target.to(config.device)
            output = model(input)

            max_t = t[0]
            min_t = t[1]

            max_min = max_t - min_t
            output = output * max_min + min_t
            target = target * max_min + min_t
            input = input * max_min + min_t

            test_loss_l1 += criterian1(output, target)
            test_loss_l2 += criterian2(output, target)

            output = output.cpu().data.numpy()[0]
            target = target.cpu().data.numpy()[0]
            input= input.cpu().data.numpy()[0,:,0]

            x = range(len(input))
            x2 = range(len(input),len(input)+len(output),1)
            mpl.rcParams["font.sans-serif"] = ["SimHei"]
            mpl.rcParams["axes.unicode_minus"] = False
            plt.plot(x, input,color='blue')
            plt.plot(x2, target,color='green',label="真实值")
            plt.plot(x2, output, color='red', linewidth=1,label="预测值")
            plt.legend(loc="upper right")
            plt.show()
    return test_loss_l1,test_loss_l2
