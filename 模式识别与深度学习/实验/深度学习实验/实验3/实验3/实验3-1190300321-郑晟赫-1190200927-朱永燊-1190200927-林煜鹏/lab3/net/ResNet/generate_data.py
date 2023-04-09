# -*- coding: utf-8 -*-
"""
Created on Fri May 27 16:25:23 2022

@author: Marmalade
"""

#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/5/27 11:11
# @Author  : ZSH
'''
自定义Dataset
'''

from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import os
from PIL import Image
from tqdm import tqdm

def generate_train_data(data_path):
    """
    生成训练集和开发集数据
    :param data_path: 数据集地址
    :return:
    """
    dirs = os.listdir(data_path)
    class2lable = {}
    lable2class = {}
    train_dataset = []
    dev_dataset = []
    for i, fileDir in enumerate(dirs):
        lable2class[i] = fileDir
        path = os.path.join(data_path, fileDir)
        class2lable[fileDir] = i
        file_data = [os.path.join(path, file) for file in os.listdir(path)]
        train, dev = train_test_split(file_data, train_size=0.8, test_size=0.2)
        train_dataset.extend([(item, i) for item in train])
        dev_dataset.extend([(item, i) for item in dev])
    return train_dataset, dev_dataset, class2lable, lable2class


def generate_test_data(data_path):
    """
    生成测试集数据集
    :param data_path: 数据集地址
    :return:
    """
    
    lable2class = {}
    dirs = os.listdir("./data/train")
    for i, fileDir in enumerate(dirs):
        lable2class[i] = fileDir
    
    test_data = [os.path.join(data_path, file) for file in os.listdir(data_path)]
    test_files = [file for file in os.listdir(data_path)]
    return test_data, lable2class, test_files


class MyTrainDataset(Dataset):
    def __init__(self, data, config):
        self.transform = transforms.Compose([
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.dataset = [(self.transform(Image.open(item[0]).convert('RGB')), item[1]) for item in tqdm(data)]
        self.len = len(self.dataset)

    def __getitem__(self, index):
        data = {
            'data': self.dataset[index][0],
            'label': self.dataset[index][1]
        }
        return data

    def __len__(self):
        return self.len


class MyTestDataset(Dataset):
    def __init__(self, data, config):
        self.transform = transforms.Compose([
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.dataset = [self.transform(Image.open(item).convert('RGB')) for item in tqdm(data)]
        self.len = len(self.dataset)

    def __getitem__(self, index):
        data = {
            'data': self.dataset[index],
        }
        return data

    def __len__(self):
        return self.len

if __name__ == '__main__':
    x, y, z, n =generate_train_data("./data/train")
    print(x)