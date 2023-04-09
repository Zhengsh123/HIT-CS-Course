#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/5/24 11:02
# @Author  : ZSH
"""
用于三种模型的训练
"""
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import time
import numpy as np
from sklearn import metrics


def train(config,model,train_loader,dev_loader):
    writer = SummaryWriter(config.log_path)
    start_time=time.time()
    optim=torch.optim.Adam(model.parameters(),lr=config.lr)
    criterian=nn.CrossEntropyLoss()
    device=config.device

    train_loss = 0.0
    dev_step=0
    best_acc_dev=0.0
    best_f1_dev=0.0
    for epoch in range(config.epoch):
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
                cur_acc,cur_f1=dev(config,model,dev_loader,dev_step,writer)
                best_acc_dev=max(best_acc_dev,cur_acc)
                best_f1_dev = max(best_f1_dev, cur_f1)
                dev_step+=1
                model.train()
    end_time = time.time()
    print("Train Time : {:.3f} s,best acc in dev:{},best f1_score in dev {}".format(end_time-start_time,best_acc_dev,best_f1_dev))

def dev(config,model,dev_loader,dev_step,writer):
    dev_loss=0
    predict_all=np.array([],dtype=int)
    label_all=np.array([],dtype=int)
    criterian = nn.CrossEntropyLoss()
    with torch.no_grad():
        for i, data in enumerate(dev_loader):
            input = data['data']
            target = data['label']
            input, target = input.to(config.device), target.to(config.device)
            output = model(input)
            loss = criterian(output, target)
            dev_loss += loss.item()
            _, predict = torch.max(output, dim=1)
            predict_all=np.append(predict_all,predict.cpu().data.numpy())
            label_all=np.append(label_all,target.cpu().data.numpy())
    acc=metrics.accuracy_score(label_all,predict_all)
    f1_score=metrics.f1_score(label_all,predict_all,average='macro')

    writer.add_scalar(tag="dev_loss", scalar_value=dev_loss/len(dev_loader),
                      global_step=dev_step)
    writer.add_scalar(tag="dev_acc", scalar_value=acc,
                      global_step=dev_step)
    writer.add_scalar(tag="dev_f1_score", scalar_value=f1_score,
                      global_step=dev_step)
    return acc,f1_score


def test(config,model,test_loader):
    test_loss=0.0
    criterian = nn.CrossEntropyLoss()
    label_all=np.array([], dtype=int)
    predict_all=np.array([], dtype=int)
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            input = data['data']
            target = data['label']
            input, target = input.to(config.device), target.to(config.device)
            output = model(input)
            loss = criterian(output, target)
            test_loss += loss.item()
            _, predict = torch.max(output, dim=1)
            predict_all = np.append(predict_all, predict.cpu().data.numpy())
            label_all = np.append(label_all, target.cpu().data.numpy())
    acc = metrics.accuracy_score(label_all, predict_all)
    f1_score = metrics.f1_score(label_all, predict_all, average='macro')
    print("acc in test %f,f1_score in test %f"%(acc,f1_score))
    return acc, f1_score