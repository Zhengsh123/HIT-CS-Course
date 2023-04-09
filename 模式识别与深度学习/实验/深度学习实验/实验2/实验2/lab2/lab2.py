#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/5/1 15:57
# @Author  : ZSH
from torch.utils.data import DataLoader
from util import *
from model import AlexNet
from get_data import GenerateData,MyDataset
if __name__=="__main__":
    config = {
        'data_path': '../data/caltech/caltech101',
        'mean1': 0.545,
        'mean2': 0.528,
        "mean3": 0.502,
        'std1': 0.249,
        'std2': 0.246,
        'std3': 0.247,
        "image_size": 224,
        'class_num': 101,

        'batch_size': 64,
        'epoch': 8,
        'lr': 0.0001,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'save_path': './model.pt'
    }
    ge=GenerateData(config)
    train_data=MyDataset(ge.train_dataset,config)
    dev_data=MyDataset(ge.dev_dataset,config)
    test_data=MyDataset(ge.test_dataset,config)
    train_loader=DataLoader(train_data,batch_size=config['batch_size'],shuffle=True)
    test_loader = DataLoader(test_data, batch_size=config['batch_size'], shuffle=False)
    dev_loader = DataLoader(dev_data, batch_size=config['batch_size'], shuffle=False)

    print("load data is ok")
    model=AlexNet(config).to(config['device'])
    print("model init is ok")
    # 训练
    # train(config,model,train_loader,dev_loader)
    # test(config,model,test_loader)
    #
    # torch.save(model.state_dict(),config['save_path'])
    # 加载模型
    model.load_state_dict(torch.load(config['save_path']))
    test(config, model, test_loader)




