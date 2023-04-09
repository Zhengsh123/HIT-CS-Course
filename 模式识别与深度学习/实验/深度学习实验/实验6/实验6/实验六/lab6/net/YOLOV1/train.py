#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/6/2 16:05
# @Author  : ZSH
import torch
from torch.utils.tensorboard import SummaryWriter
import time
from net.YOLOLoss import YoloLoss
from net.util import draw_box


def train(config,model,train_loader,device):
    writer = SummaryWriter(config.log_path)
    start_time=time.time()
    optim=torch.optim.SGD(model.parameters(),lr=config.lr,momentum=0.9, weight_decay=5e-4)
    criterian=YoloLoss(S=7,B=2,l_coord=5,l_noobj=0.5,device=device)
    model.train()
    train_loss = 0.0
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

            if (i+1) % 5 == 0:
                print("[%d,%d],loss:%.3f" % (epoch, i, train_loss/5))
                train_loss = 0.0
            if (i+1)%50==0:
                model.eval()
                print(i)
                draw_box(model,'./dog.jpg','result.jpg')
                model.train()
    end_time = time.time()
    print("train time %f"%(end_time-start_time))


def test(config,model,test_loader,device):
    model.eval()
    test_start=time.time()
    criterian = YoloLoss(S=7,B=2,l_coord=0.5,l_noobj=0.5,device=device)
    test_loss = 0.0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            input=data['data']
            target = data['target']
            input = input.to(device)
            target=target.to(device)
            output = model(input)
            loss = criterian(output,target )
            test_loss += loss.item()
            print('batch %s of total batch %s' % (i, len(test_loader)),
                  'Loss: %.3f ' % (test_loss / (i + 1)))

    test_last=time.time()-test_start
    print(test_last)