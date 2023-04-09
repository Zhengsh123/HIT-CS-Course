#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/5/27 15:03
# @Author  : ZSH

import torch


class VggConfig():
    def __init__(self,args):
        self.train_path="../data/train"
        self.test_path="../data/test"

        self.image_size=224
        self.batch_size=args.batch_size
        self.lr = args.lr
        self.epoch = args.epoch

        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device=('cpu')
        self.class_num=12
        # 保存路径
        self.model_save_path='../train/VGG.pt'
        self.log_path='../train/log/VGG/'
        self.predict_save='../res/VGG.csv'


if __name__=="__main__":
    pass