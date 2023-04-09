#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/5/27 11:11
# @Author  : ZSH
'''
自定义Dataset
'''
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import Dataset
import os
from PIL import Image


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
        path = os.path.join(data_path, fileDir)
        class2lable[fileDir] = i
        lable2class[i]=fileDir
        file_data = [os.path.join(path, file) for file in os.listdir(path)]
        train, dev = train_test_split(file_data, train_size=0.8, test_size=0.2)
        train_dataset.extend([(item, i) for item in train])
        dev_dataset.extend([(item, i) for item in dev])
    return train_dataset, dev_dataset, lable2class


def generate_test_data(data_path):
    """
    生成测试集数据集
    :param data_path: 数据集地址
    :return:
    """
    test_data = [os.path.join(data_path, file) for file in os.listdir(data_path)]
    file_data=[file for file in os.listdir(data_path)]
    return test_data,file_data


class MyTrainDataset(Dataset):
    def __init__(self, data, config):
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.RandomResizedCrop(config.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5]),
        ])
        self.dataset = data
        self.len = len(self.dataset)

    def __getitem__(self, index):
        data = {
            'data': self.transform(Image.open(self.dataset[index][0]).convert('RGB')),
            'label': self.dataset[index][1]
        }
        return data

    def __len__(self):
        return self.len


class MyTestDataset(Dataset):
    def __init__(self, data, config):
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.RandomResizedCrop(config.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5]),
        ])
        self.dataset = data
        self.len = len(self.dataset)

    def __getitem__(self, index):

        data = {
            'data': self.transform(Image.open(self.dataset[index]).convert('RGB')),
        }
        return data

    def __len__(self):
        return self.len


if __name__ == "__main__":
    pass





