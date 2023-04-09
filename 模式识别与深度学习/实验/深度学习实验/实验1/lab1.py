#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/4/21 18:38
# @Author  : ZSH

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random
writer = SummaryWriter('./log')
# 是否使用GPU加速
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 64
# 将读入的数据转为tensor并归一化，其中0.1307，0.3081分别是均值和方差
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

# 读入数据
train_data = datasets.MNIST(root='../data/mnist', download=True, train=True, transform=transform)
test_data = datasets.MNIST(root='../data/mnist', download=True, train=False, transform=transform)

# Data loader
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# 随机种子设置
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
# 定义MLP
class MLP(nn.Module):
    def __init__(self):
        super(MLP,self).__init__()
        # 定义线性层
        self.l1=torch.nn.Linear(784,512)
        self.l2 = torch.nn.Linear(512, 256)
        self.l3 = torch.nn.Linear(256, 128)
        self.l4 = torch.nn.Linear(128, 10)
        # 定义激活函数
        self.activate=torch.nn.ReLU()

    def forward(self,x):
        x=x.view(-1,784) # 784=28*28
        x=self.activate(self.l1(x))
        x=self.activate(self.l2(x))
        x = self.activate(self.l3(x))
        x=self.l4(x)
        return x

def train(epoch):
    running_loss=0.0
    for i,data in enumerate(train_loader):
        input,target=data
        input,target=input.to(device),target.to(device)
        optim.zero_grad()
        output=model(input)
        loss=criterian(output,target)
        loss.backward()
        optim.step()
        running_loss+=loss.item() # 只取loss的值，不作为tensor加减

        writer.add_scalar(tag="training_loss", scalar_value=loss.item(),
                          global_step=epoch * len(train_loader) + i)

        if i%300==299:
            print("[%d,%d],loss:%.3f"%(epoch,i,running_loss/300))
            running_loss=0.0

def test(epoch):
    correct=0
    total=0
    for i,data in enumerate(test_loader):
        input, target = data
        input, target = input.to(device), target.to(device)
        output=model(input)
        _,predict=torch.max(output,dim=1)
        total+=output.shape[0]
        correct+=(predict==target).sum().item()
    acc=correct/total
    print('accuracy on test = %.3f %%' %(acc*100))
    writer.add_scalar(tag="test_acc", scalar_value=acc,
                      global_step=epoch)
    return acc

if __name__=="__main__":
    # train model
    setup_seed(20)
    model=MLP()
    model=model.to(device)
    criterian=torch.nn.CrossEntropyLoss()
    optim=torch.optim.SGD(model.parameters(),lr=0.005,momentum=0.5)

    for epoch in range(18):
        train(epoch)
        test(epoch)

    torch.save(model, './mlp.pt')

    ## load model
    # model=torch.load('./mlp.pt')
    # test(0)