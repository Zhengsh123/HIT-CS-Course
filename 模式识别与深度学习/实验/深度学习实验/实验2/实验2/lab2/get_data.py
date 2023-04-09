#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/5/1 17:10
# @Author  : ZSH
'''
本文件用于读取数据集并生成测试集、验证集、测试集，8:1:1，每个类分别划分
'''

from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader,Dataset, random_split
import numpy as np
import os
from PIL import Image

class GenerateData():
    def __init__(self,config):
        self.data_path=config['data_path']
        self.train_dataset,self.test_dataset,self.dev_dataset,self.class2lable=self.generateData()

    def generateData(self):
        dirs=os.listdir(self.data_path)
        class2lable={}
        train_dataset=[]
        test_dataset=[]
        dev_dataset=[]
        assert len(dirs)==101
        for i,fileDir in enumerate(dirs):
            path=os.path.join(self.data_path,fileDir)
            class2lable[fileDir]=i
            fileData=[os.path.join(path,file) for file in os.listdir(path)]
            train,test=train_test_split(fileData,train_size=0.8,test_size=0.2)
            test,dev=train_test_split(test,train_size=0.5,test_size=0.5)
            train_dataset.extend([(item,i)for item in train])
            test_dataset.extend([(item, i) for item in test])
            dev_dataset.extend([(item, i) for item in dev])
        return train_dataset,test_dataset,dev_dataset,class2lable


class MyDataset(Dataset):
    def __init__(self,data,config):
        self.transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((config['image_size'],config['image_size'])),
            # transforms.Normalize((config['mean1'],config['mean2'],config['mean3']),(config['std1'],config['std2'],config['std3']))
        ])
        self.dataset = [(self.transform(Image.open(item[0]).convert('RGB')), item[1]) for item in data]
        self.len=len(self.dataset)

    def __getitem__(self, index):
        data={
            'data':self.dataset[index][0],
            'label':self.dataset[index][1]
        }
        return data

    def __len__(self):
        return self.len

if __name__=="__main__":
    pass