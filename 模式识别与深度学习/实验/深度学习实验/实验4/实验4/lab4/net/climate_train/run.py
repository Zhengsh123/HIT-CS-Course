#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/5/25 22:16
# @Author  : ZSH

from net.climate_train.config import *
from net.climate_train.util import *
from net.climate_train.generate_data import *
from net.climate_train.model import *

if __name__=="__main__":
    config=LstmClimateConfig()
    model=LSTM(config,bidirectional=True).to(config.device)
    train_data,t= generate_data(config, config.train_path)
    # train_data = MyDataset(train_data, config)
    # train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    # train(config, model, train_loader, t)
    # torch.save(model.state_dict(),config.model_save_path)

    model.load_state_dict(torch.load(config.model_save_path))
    test_data, t_test= generate_data(config, config.test_path)
    test_data = MyDataset(test_data, config)
    test_loader=DataLoader(test_data, batch_size=1, shuffle=False)

    test_loss_l1,test_loss_l2=test(config,model,test_loader,t)
    print(test_loss_l1,test_loss_l2)

