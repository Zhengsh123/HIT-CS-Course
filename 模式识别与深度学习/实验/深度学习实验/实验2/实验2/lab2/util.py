#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/5/2 12:35
# @Author  : ZSH
import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('./log')
# 随机种子设置
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train(config,model,train_loader,dev_loader):
    # 启用DropOut
    model.train()
    optim=torch.optim.Adam(model.parameters(),lr=config['lr'])
    criterian=nn.CrossEntropyLoss()
    device=config['device']

    train_loss = 0.0
    dev_step=0
    for epoch in range(config['epoch']):
        for i, data in enumerate(train_loader):
            input = data['data']
            target = data['label']

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
                print("[%d,%d],loss:%.3f" % (epoch, i, loss.item()))
                train_loss = 0.0
                dev(config,model,dev_loader,dev_step)
                dev_step+=1


def dev(config,model,dev_loader,dev_step):
    correct = 0
    total = 0
    model.eval()
    dev_loss=0
    criterian = nn.CrossEntropyLoss()
    with torch.no_grad():
        for i, data in enumerate(dev_loader):
            input = data['data']
            target = data['label']
            input, target = input.to(config['device']), target.to(config['device'])
            output = model(input)
            loss = criterian(output, target)
            dev_loss += loss.item()
            _, predict = torch.max(output, dim=1)
            total += output.shape[0]
            correct += (predict == target).sum().item()
    writer.add_scalar(tag="dev_loss", scalar_value=dev_loss/len(dev_loader),
                      global_step=dev_step)
    writer.add_scalar(tag="dev_acc", scalar_value=correct/total,
                      global_step=dev_step)


def test(config,model,test_loader):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            input=data['data']
            target=data['label']
            input, target = input.to(config['device']), target.to(config['device'])
            output = model(input)
            _, predict = torch.max(output, dim=1)
            total += output.shape[0]
            correct += (predict == target).sum().item()
    acc = correct / total
    print('accuracy on test = %.3f %%' %(acc*100))
    return acc