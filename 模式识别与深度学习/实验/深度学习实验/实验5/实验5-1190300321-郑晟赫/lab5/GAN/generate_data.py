#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/5/29 21:08
# @Author  : ZSH
import scipy.io as scio
from sklearn.model_selection import train_test_split
from GAN.gan.config import GanConfig
import torch
import numpy as np


def generate_data(config):
    data_path=config.data_path
    data = scio.loadmat(data_path).get('xx')
    np.random.shuffle(data)
    data=torch.from_numpy(data).float().to(config.device)
    return data


if __name__=="__main__":
    pass
