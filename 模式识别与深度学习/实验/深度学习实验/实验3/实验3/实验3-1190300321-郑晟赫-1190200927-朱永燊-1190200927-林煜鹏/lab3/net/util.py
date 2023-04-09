#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/5/27 19:00
# @Author  : ZSH
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import time
import numpy as np
from sklearn import metrics
import pandas as pd


def train(config,model,train_loader,dev_loader):
    writer = SummaryWriter(config.log_path)
    start_time=time.time()
    optim=torch.optim.Adam(model.parameters(),lr=config.lr)
    criterian=nn.CrossEntropyLoss()
    device=config.device
    model.train()
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
                print("[%d,%d],loss:%.3f" % (epoch, i, train_loss/10))
                train_loss = 0.0

        cur_acc, cur_f1 = dev(config, model, dev_loader, dev_step, writer)
        best_acc_dev = max(best_acc_dev, cur_acc)
        best_f1_dev = max(best_f1_dev, cur_f1)
        dev_step += 1
        print("[%f,%f] "% (best_acc_dev,best_f1_dev))
        model.train()
    end_time = time.time()
    print("Train Time : {:.3f} s,best acc in dev:{},best f1_score in dev {}".format(end_time-start_time,best_acc_dev,best_f1_dev))


def dev(config,model,dev_loader,dev_step,writer):
    model.eval()
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
    f1_score=metrics.f1_score(label_all,predict_all,average='micro')

    writer.add_scalar(tag="dev_loss", scalar_value=dev_loss/len(dev_loader),
                      global_step=dev_step)
    writer.add_scalar(tag="dev_acc", scalar_value=acc,
                      global_step=dev_step)
    writer.add_scalar(tag="dev_f1_score", scalar_value=f1_score,
                      global_step=dev_step)
    return acc,f1_score


def test(config,model,test_loader,lable2class,test_files):
    model.eval()
    predict_all=np.array([], dtype=int)
    test_start=time.time()
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            input=data['data']
            input = input.to(config.device)
            output = model(input)
            _, predict = torch.max(output, dim=1)
            predict_all=np.append(predict_all,predict.cpu().data.numpy())
    test_last=time.time()-test_start
    print("test time %f"%(test_last))
    predict_res=[lable2class[item] for item in predict_all]
    final_pred = {'file': test_files, 'species': predict_res}
    final_pred = pd.DataFrame(final_pred)
    final_pred.to_csv(config.predict_save, index=False)