#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/5/25 22:12
# @Author  : ZSH
'''
生成气温数据的训练集和测试集
'''
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader,Dataset, random_split
from sklearn import preprocessing


def generate_data(config, data_path):
    pd_all = pd.read_csv(data_path)
    max_t = pd_all.max()['T']
    min_t = pd_all.min()['T']
    pd_all=pd_all.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    dataset=pd_all.values
    data,labels=multivariate_data(dataset,dataset[:,0],0,len(dataset)-config.output_size,config.history_size,config.output_size,step=config.step)
    return (data,labels),(max_t,min_t)


def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
    data = []
    labels = []
    start_index = start_index + history_size

    if end_index is None:
        end_index = len(dataset) - target_size
    for i in range(start_index, end_index,144):
        indices = range(i - history_size, i, step)
        data.append(dataset[indices])
        if single_step:
            labels.append(target[i + target_size])
        else:
            labels.append(target[i:i + target_size])
    return np.array(data).reshape(-1,history_size//step,3), np.array(labels).reshape(-1,target_size)


class MyDataset(Dataset):
    def __init__(self,data,config):
        self.device=config.device
        self.data=data
    def __getitem__(self, index):
        data = {
            'data': self._totensor(self.data[0][index]),
            'target': self._totensor(self.data[1][index])
        }
        return data

    def __len__(self):
        return len(self.data[0])

    def _totensor(self,data):

        data = torch.Tensor(data.astype(float)).float().to(self.device)
        return data

if __name__=="__main__":
    from net.climate_train.config import *
    config=LstmClimateConfig()

    train_data, temp = generate_data(config, config.train_path)
    train_data = MyDataset(train_data, config)
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=False)
    for i, data in enumerate(train_loader):
        print(data['data'])
