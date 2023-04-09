#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/5/25 11:04
# @Author  : ZSH

from net.gru.config import *
from net.util import *
from net.generate_shopping_data import *
from net.gru.model import *
if __name__=="__main__":
    config=GruShoppingConfig()
    model=Gru(config).to(config.device)
    # 训练
    # train_data = generate_data(config, config.train_path)
    # train_data = MyDataset(train_data, config)
    # train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    # dev_data = generate_data(config, config.dev_path)
    # dev_data = MyDataset(dev_data, config)
    # dev_loader = DataLoader(dev_data, batch_size=config.batch_size, shuffle=False)
    # train(config,model,train_loader,dev_loader)
    # torch.save(model.state_dict(),config.model_save_path)
    # 加载模型
    test_data = generate_data(config, config.test_path)
    test_data = MyDataset(test_data, config)
    test_loader = DataLoader(test_data, batch_size=config.batch_size, shuffle=False)
    model.load_state_dict(torch.load(config.model_save_path))
    test(config,model,test_loader)


