# -*- coding: utf-8 -*-
"""
Created on Fri May 27 16:27:57 2022

@author: Marmalade
"""

import torch


class resnet_Config():
    def __init__(self):
        self.train_path = "../data/train"
        self.test_path = "../data/test"
        self.image_size = 224
        self.model_name = 'resnet'

        self.batch_size = 8
        self.learning_rate = 1e-5
        self.class_num = 12
        self.epoch = 600
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_save_path = '../train/resnet.ckpt'
        self.log_path = '../train/log/ResNet/'
        self.maxiter_without_improvement = 1000
        self.predict_save = '../res/ResNet.csv'