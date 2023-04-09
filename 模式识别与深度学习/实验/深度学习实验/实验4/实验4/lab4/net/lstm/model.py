#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/5/23 11:39
# @Author  : ZSH
import torch.nn as nn
import torch
from torch.autograd import Variable

class LstmCell(nn.Module):
    def __init__(self,config):
        super(LstmCell,self).__init__()
        self.config=config
        self.W_f=nn.Linear(config.hidden_size+config.emb_size,config.hidden_size)
        self.W_i=nn.Linear(config.hidden_size+config.emb_size,config.hidden_size)
        self.W_c=nn.Linear(config.hidden_size+config.emb_size,config.hidden_size)
        self.W_o=nn.Linear(config.hidden_size+config.emb_size,config.hidden_size)
        self.tanh=nn.Tanh()
        self.sigmoid=nn.Sigmoid()

    def forward(self,input,h,c):
        combined = torch.cat((input, h), 1) # [batch_size,1,hidden_size+emb_size]
        f_t=self.sigmoid(self.W_f(combined))
        i_t=self.sigmoid(self.W_i(combined))
        C_t_hat=self.tanh(self.W_c(combined))
        C_t=f_t*c+i_t*C_t_hat
        o_t=self.sigmoid(self.W_o(combined))
        h_t=o_t*self.tanh(C_t)
        return h_t,C_t


class LSTM(nn.Module):
    def __init__(self,config,bidirectional=False):
        super(LSTM,self).__init__()
        self.config=config
        self.cell=LstmCell(config)
        self.bidirectional=bidirectional
        # self.emb = nn.Embedding.from_pretrained(config.vectors, freeze=False)
        self.layers=1
        if bidirectional:
            self.fc = nn.Linear(2*config.hidden_size, 10)
        else:
            self.fc=nn.Linear(config.hidden_size,10)
    def forward(self, input,initial_states=None):
        # input = self.emb(input)  # [batch_size,seq_len,emb_size]
        len = input.size(1)
        output_f = [] # 正向数据
        output_b = [] # 反向数据
        if initial_states is None:
            zeros = Variable(torch.zeros(input.size(0), self.config.hidden_size)).to(self.config.device)
            initial_states = [(zeros),(zeros),(zeros),(zeros), ] * self.layers
        states=initial_states

        for t in range(len):
            x_f = input[:, t, :]
            h_t_f,c_t_f = self.cell(x_f, states[0],states[1])
            states[0] = h_t_f.data
            states[1]=c_t_f
            output_f.append(h_t_f)

            if self.bidirectional:
                x_b = input[:, len-t-1, :]
                h_t_b, c_t_b = self.cell(x_b, states[2], states[3])
                states[2] = h_t_b.data
                states[3] = c_t_b
                output_b.append(h_t_b)
        if self.bidirectional:
            x=self.fc(torch.cat((output_f[-1], output_b[-1]), 1))
        else:
            x=self.fc(output_f[-1])
        return x