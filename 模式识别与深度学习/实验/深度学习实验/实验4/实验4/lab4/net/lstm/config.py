#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/5/24 21:45
# @Author  : ZSH

import torch
import numpy as np
import word2vec

class LstmShoppingConfig():
    def __init__(self):
        self.data_path='../../data/online_shopping/'
        self.train_path=self.data_path+'train.csv'
        self.dev_path = self.data_path + "dev.csv"
        self.test_path = self.data_path + "test.csv"
        self.save_path='../../res/lstm_shopping_predict.csv'
        self.emb_path="../../vocab/"

        # LSTM 网络配置
        self.hidden_size=64
        self.dropout=0.3
        self.output_size=10

        # 训练设置
        self.batch_size=64
        self.padding_size=30 # 控制每句话的长度，不够的补足，太长的截断
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lr = 0.005
        self.epoch=2

        # 载入词典
        emb_model=word2vec.load(self.emb_path+'sgns.weibo.word.txt')
        self.vocab=emb_model.vocab # 词表中所有的单词
        self.vocab_dic=emb_model.vocab_hash # 词表中每个词对应的编号
        self.vocab_dic.update({'<UNK>': len(self.vocab), '<PAD>': len(self.vocab) + 1})
        self.vocab=np.append(self.vocab,'<UNK>')
        self.vocab = np.append(self.vocab, '<PAD>')
        self.vectors=emb_model.vectors
        self.vectors = torch.Tensor(np.append(np.append(self.vectors, self.vectors.mean(axis=0).reshape(1, -1),
                                                        axis=0), self.vectors.mean(axis=0).reshape(1, -1), axis=0))
        self.emb_size=self.vectors.shape[1]
        self.vocab_size=self.vectors.shape[0]

        # 保存路径
        self.model_save_path='../../train/LSTM_shopping.pt'
        self.log_path='../../train/log/LSTM_shopping/'