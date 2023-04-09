#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/5/29 23:09
# @Author  : ZSH
from GAN.wgan.config import WGanConfig
from GAN.wgan.model import Generator,Discriminator
from GAN.generate_data import generate_data
from GAN.wgan.train import train

if __name__=="__main__":
    config=WGanConfig()
    data=generate_data(config)
    generator=Generator(config)
    discriminator=Discriminator()
    train(config,data,generator,discriminator)

