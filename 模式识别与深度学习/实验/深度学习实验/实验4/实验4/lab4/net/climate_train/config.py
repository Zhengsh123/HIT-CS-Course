#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/5/25 22:17
# @Author  : ZSH

import torch
import numpy as np
import pandas as pd
class LstmClimateConfig():
    def __init__(self):
        self.data_path='../../data/climate/'
        self.train_path=self.data_path+'train.csv'
        self.test_path = self.data_path + "test.csv"

        # LSTM 网络配置
        self.hidden_size=64
        self.output_size=288 #6*24*2
        self.history_size=720 #6*24*5
        self.input_size=3 # 使用3个特征
        self.step=1 # 重采样长度（3个点中取一个点）

        # 训练设置
        self.batch_size=32
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lr = 0.001
        self.epoch=20

        # 保存路径
        self.model_save_path='../../train/LSTM_climate.pth'
        self.log_path='../../train/log/LSTM_climate/'



