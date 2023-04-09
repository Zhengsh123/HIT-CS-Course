#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/5/23 11:39
# @Author  : ZSH

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable
class GruCell(nn.Module):
    def __init__(self,config):
        super(GruCell,self).__init__()
        self.config=config
        self.emb_size=self.config.emb_size
        self.hidden_size=self.config.hidden_size
        self.W_z=nn.Linear(self.config.hidden_size+self.config.emb_size,self.config.hidden_size)
        self.W_r=nn.Linear(self.config.hidden_size+self.emb_size,self.hidden_size)
        self.W=nn.Linear(self.config.hidden_size+self.emb_size,self.hidden_size)
        self.sigmoid=nn.Sigmoid()
        self.tanh=nn.Tanh()

    def forward(self,x,h):
        combined = torch.cat((x, h), 1)
        z_t=self.sigmoid(self.W_z(combined))
        r_t=self.sigmoid(self.W_r(combined))
        combined_h=torch.cat((r_t*h,x), 1)
        h_t_hat=self.tanh(self.W(combined_h))
        h_t=(1-z_t)*h+z_t*h_t_hat

        return h_t

class Gru(nn.Module):
    def __init__(self,config):
        super(Gru,self).__init__()

        self.cell = GruCell(config)
        self.fc = nn.Linear(config.hidden_size, 10)
        self.config = config
        self.layers=1

    def forward(self,input):
        initial_states = None
        len = input.size(1)
        output = []
        if initial_states is None:
            zeros = Variable(torch.zeros(input.size(0), self.config.hidden_size)).to(self.config.device)
            initial_states = [(zeros), ] * self.layers
        states=initial_states

        for t in range(len):
            x = input[:, t, :]
            hidden = self.cell(x, states[0])
            states[0]=hidden
            output.append(hidden)

        return self.fc(output[-1])
