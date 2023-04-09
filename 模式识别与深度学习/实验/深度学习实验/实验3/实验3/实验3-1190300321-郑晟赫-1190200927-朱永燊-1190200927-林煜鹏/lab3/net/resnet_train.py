# -*- coding: utf-8 -*-
"""
Created on Fri May 27 20:48:48 2022

@author: Marmalade
"""

import time
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from sklearn import metrics
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from net.ResNet.config import *
from net.ResNet.model import *
from net.ResNet.generate_data import *


def train(config,model,train_iter,dev_iter):
    # 记录时间
    start_time = time.time()
    # 启用dropout
    model.train()
    # 设置adam优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    total_batch = 0                         # 记录总共训练的批次
    dev_best_loss = float('inf')            # 记录验证集上最低的loss
    dev_best_acc = float(0)                 # 记录验证集上最高的acc
    dev_best_f1score = float(0)             # 记录验证集上最高的f1score
    last_improve = 0                        # 记录上一次dev的loss下降时的批次
    flag = False                            # 是否结束训练
    writer = SummaryWriter(config.log_path)
    for epoch in range(config.epoch):
        print("Epoch [{}/{}]".format(epoch+1, config.epoch))
        for index,data in enumerate(train_iter):
            trains = data['data'].to(config.device)
            labels = data['label'].to(config.device)
            outputs = model(trains)
            model.zero_grad()
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            # 输出当前效果
            if total_batch % 10 == 0:
                ground_truth = labels.data.cpu()
                predict_labels = torch.argmax(outputs,dim=1).cpu().numpy()
                train_acc = metrics.accuracy_score(ground_truth, predict_labels)
                dev_acc, dev_loss, dev_f1score = evaluate(config,model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                if dev_acc > dev_best_acc:
                    dev_best_acc = dev_acc
                if dev_f1score > dev_best_f1score:
                    dev_best_f1score = dev_f1score
                    torch.save(model.state_dict(), config.save_model_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                print("Iter:{:4d} TrainLoss:{:.12f} TrainAcc:{:.5f} DevLoss:{:.12f} DevAcc:{:.5f} DevF1Score:{:.2f} Improve:{}".format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, dev_f1score, improve))
                writer.add_scalar("loss/train", loss.item(), total_batch)
                writer.add_scalar("loss/dev", dev_loss, total_batch)
                writer.add_scalar("acc/train", train_acc, total_batch)
                writer.add_scalar("acc/dev", dev_acc, total_batch)
                model.train()
            total_batch += 1
            if total_batch - last_improve > config.maxiter_without_improvement:
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
    writer.close()
    end_time = time.time()
    print("Train Time : {:.3f} min , The Best Acc in Dev : {} % , The Best f1-score in Dev : {}".format(((float)((end_time-start_time))/60), dev_best_acc,dev_best_f1score))

def evaluate(config,model, dev_iter):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for index,data in enumerate(dev_iter):
            trains = data['data'].to(config.device)
            labels = data['label'].to(config.device)
            outputs = model(trains)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            ground_truth = labels.cpu().data.numpy()
            predict_labels = torch.argmax(outputs,dim=1).cpu().numpy()
            labels_all = np.append(labels_all, ground_truth)
            predict_all = np.append(predict_all, predict_labels)
    acc = metrics.accuracy_score(labels_all, predict_all)
    f1score = metrics.f1_score(labels_all, predict_all, average='micro')
    return acc, loss_total / len(dev_iter), f1score


def test(config,model,test_iter):
    model.load_state_dict(torch.load(config.save_model_path))
    model.to(config.device)
    model.eval()
    predict_all = np.array([], dtype=int)
    with torch.no_grad():
        for index,data in enumerate(test_iter):
            trains = data['data'].to(config.device)
            outputs = model(trains)
            predict_all = np.append(predict_all, torch.max(outputs.cpu().data, 1)[1].numpy())
    predict_result = np.column_stack([test_iter.ids, predict_all])
    

if __name__ == '__main__':
    myConfig = resnet_Config()
    print('load data\n')
    train_data, dev_data, class2lable, lable2class = generate_train_data(myConfig.train_path)
    #test_data = generate_test_data(myConfig.test_path)
    trainDataSet = MyTrainDataset(train_data, myConfig)
    devDataSet = MyTrainDataset(dev_data, myConfig)
    #testDataSet = MyTestDataset(test_data, myConfig)
    trainDataLoader = DataLoader(dataset=trainDataSet, batch_size=myConfig.batch_size, shuffle=True)
    devDataLoader = DataLoader(dataset=devDataSet, batch_size=myConfig.batch_size, shuffle=False)
    #testDataLoader = DataLoader(dataset=testDataSet, batch_size=myConfig.batch_size, shuffle=False)
    resnet = ResNet18().to(myConfig.device)
    print('train unlock\n')
    train(myConfig, resnet, trainDataLoader, devDataLoader)
    