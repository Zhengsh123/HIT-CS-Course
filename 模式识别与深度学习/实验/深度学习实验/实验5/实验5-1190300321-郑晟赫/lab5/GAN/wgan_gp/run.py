#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/5/30 17:09
# @Author  : ZSH
from GAN.wgan_gp.config import WGanGPConfig
from GAN.wgan_gp.model import Generator,Critic
from GAN.generate_data import generate_data
from GAN.wgan_gp.train import train

if __name__=="__main__":
    config=WGanGPConfig()
    data=generate_data(config)
    generator=Generator(config)
    critic=Critic()
    train(config,data,generator,critic)