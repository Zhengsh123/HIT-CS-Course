#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/5/23 22:36
# @Author  : ZSH
'''
生成在线购物数据集测试与训练数据
'''
import torch
import jieba
import json
from torch.utils.data import DataLoader,Dataset, random_split
import word2vec
import numpy as np
import pandas as pd
import torch.nn as nn
PAD = '<PAD>'
UNK = '<UNK>'


def generate_data(config, data_path):
    """
    构造训练数据集与开发集
    :param config: 配置文件，class形式
    :param data_path: 训练文件地址
    :return: 训练数据集
    """
    label_dic= {'书籍':0,'平板':1,'手机':2,'水果':3,'洗发水':4,'热水器':5,
                '蒙牛':6,'衣服':7,'计算机':8,'酒店':9}
    train=[]
    for word in config.vocab:
        jieba.add_word(word)
    pd_all = pd.read_csv(data_path)

    for index, row in pd_all.iterrows():
        cat = label_dic[row['cat']]
        review =('' if pd.isnull(row['review']) else row['review'].strip())
        token = list(jieba.cut(review))
        seq_len = len(token)
        if seq_len < config.padding_size:
            token.extend([PAD] * (config.padding_size - seq_len))

        elif seq_len > config.padding_size:
            token = token[:config.padding_size]
            seq_len = config.padding_size
        word_id = [config.vocab_dic.get(word, config.vocab_dic.get(UNK)) for word in token]
        train.append((word_id, int(cat), seq_len))
    return train


class MyDataset(Dataset):
    def __init__(self,data,config):
        self.device=config.device
        self.data=self._totensor(data,config)

    def __getitem__(self, index):
        data = {
            'data': self.data[0][index],
            'label': self.data[1][index]
        }
        return data

    def __len__(self):
        return len(self.data[0])

    def _totensor(self,train_data,config):
        emb = nn.Embedding.from_pretrained(config.vectors, freeze=True).to(self.device)
        content=torch.Tensor([data[0] for data in train_data]).long().to(self.device)
        content_emb=emb(content)
        label=torch.Tensor([data[1] for data in train_data]).long().to(self.device)
        seq_len=torch.LongTensor([data[2] for data in train_data]).long().to(self.device)
        return (content_emb,label,seq_len)



if __name__=="__main__":
    from net.rnn.config import *
    ls=RnnShoppingConfig()
    train=generate_data(ls,'./data/online_shopping/train.csv')
    train_data=MyDataset(train,ls)
    train_loader=DataLoader(train_data,batch_size=50,shuffle=True)
    for i,data in enumerate(train_loader):
        print(data)


