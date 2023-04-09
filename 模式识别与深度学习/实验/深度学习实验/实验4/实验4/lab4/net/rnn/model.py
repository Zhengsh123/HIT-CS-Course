#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/5/23 11:39
# @Author  : ZSH
import torch
import torch.nn as nn
import numpy as np
import word2vec
from torch.autograd import Variable

class RnnCell(nn.Module):
    def __init__(self,config):
        super(RnnCell,self).__init__()
        self.hidden_size=config.hidden_size
        self.emb_size=config.emb_size
        self.output_size=config.output_size
        self.W=nn.Linear(self.emb_size,self.hidden_size,bias=True)
        self.U=nn.Linear(self.hidden_size,self.hidden_size,bias=True)
        self.activate=nn.Tanh()

    def forward(self,input,hidden):
        hidden = self.W(input)+self.U(hidden)
        hidden=self.activate(hidden)
        return hidden


class Rnn(nn.Module):
    def __init__(self,config):
        super(Rnn,self).__init__()
        # self.emb=nn.Embedding.from_pretrained(config.vectors,freeze=False)
        self.layers=1
        self.fc=nn.Linear(config.hidden_size,10,bias=True)
        self.config=config
        self.activate=nn.Softmax()
        self.layer0=RnnCell(config)

    def forward(self, input,initial_states=None):
        # input=self.emb(input)  # [batch_size,seq_len,emb_size]
        len=input.size(1)
        output=[]
        if initial_states is None:
            zeros = Variable(torch.zeros(input.size(0), self.config.hidden_size)).to(self.config.device)
            initial_states = [(zeros), ] * self.layers
        states=initial_states
        for t in range(len):
            x=input[:,t,:]
            hidden=self.layer0(x, states[0])
            states[0]=hidden
            output.append(hidden)
        return self.fc(output[-1])

class RnnTorch(nn.Module):
    """
    用于测试自定义的RNN结果是否基本正确
    """
    def __init__(self,config):
        super(RnnTorch,self).__init__()
        # self.emb = nn.Embedding.from_pretrained(config.vectors, freeze=False)
        self.rnn= nn.RNN(input_size=config.emb_size, hidden_size=config.hidden_size,batch_first=True)
        self.fc = nn.Linear(config.hidden_size, 10)

    def forward(self,x):
        # x = self.emb(x)  # [batch_size,seq_len,emb_size]
        x,_ = self.rnn(x)
        x = self.fc(x[:, -1, :])
        return x

if __name__=="__main__":
    pass