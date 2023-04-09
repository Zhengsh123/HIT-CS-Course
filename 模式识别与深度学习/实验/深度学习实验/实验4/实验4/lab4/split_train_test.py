#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/5/23 20:56
# @Author  : ZSH
'''
用于划分两个任务的训练集、测试集、验证集
'''

import pandas as pd
import numpy as np
import jieba
import re
import datetime
def split_shopping(path,train_path,test_path,dev_path):
    """
    分割在线购物数据集，删除一部分不需要的词汇和停用词
    :param path: 原始数据集存储地址
    :param train_path: 训练集需要存储的地址
    :param test_path: 测试集需要存储的地址
    :param dev_path: 开发集需要存储的地址
    :return:
    """
    pd_all = pd.read_csv(path)
    train=[]
    dev=[]
    test=[]
    pd_all['review'] = pd_all['review'].apply(remove_punctuation)
    stopwords = [line.strip() for line in open("./vocab/chineseStopWords.txt", 'r', encoding='utf-8').readlines()]
    pd_all['review'] = pd_all['review'].apply(lambda x: " ".join([w for w in list(jieba.cut(x)) if w not in stopwords]))
    for index, row in pd_all.iterrows():
        temp_row=[]
        temp_row.append(row['cat'])
        temp_row.append(row['review'])
        if index%5==4:
            dev.append(temp_row)
        elif index%5==0:
            test.append(temp_row)
        else:
            train.append(temp_row)
    df_train = pd.DataFrame(train, columns=['cat', 'review'])
    df_dev = pd.DataFrame(dev, columns=['cat', 'review'])
    df_test = pd.DataFrame(test, columns=['cat', 'review'])

    df_train.to_csv(train_path)
    df_test.to_csv(test_path)
    df_dev.to_csv(dev_path)
#定义删除除字母,数字，汉字以外的所有符号的函数
def remove_punctuation(line):
    line = str(line)
    if line.strip()=='':
        return ''
    rule = re.compile(u"[^a-zA-Z0-9\u4E00-\u9FA5]")
    line = rule.sub('',line)
    return line

def split_climate(path,train_path,test_path):
    '''
    分割气温数据集，09~14年数据为训练集
    :param path:数据集地址
    :param train_path:训练集地址
    :param test_path:测试集地址
    :return:
    '''
    pd_all = pd.read_csv(path)
    train_index=315731 # 对应2009-1014年6年数据
    train = []
    test = []
    for index,row in pd_all.iterrows():
        temp_row = []
        temp_row.append(row['Date Time'])
        temp_row.append(row['T (degC)'])
        temp_row.append(row['p (mbar)'])
        temp_row.append(row['rho (g/m**3)'])
        if index < train_index:
            train.append(temp_row)
        else:
            test.append(temp_row)
    df_train = pd.DataFrame(train, columns=['Time', 'T','p','rho'])
    df_test = pd.DataFrame(test, columns=['Time', 'T','p','rho'])

    df_train.to_csv(train_path)
    df_test.to_csv(test_path)

if __name__=="__main__":
    # path="./data/online_shopping_10_cats.csv"
    # train_path='./data/online_shopping/train.csv'
    # dev_path = './data/online_shopping/dev.csv'
    # test_path = './data/online_shopping/test.csv'
    #
    # split_shopping(path,train_path,dev_path,test_path)
    path="./data/jena_climate_2009_2016.csv"
    train_path='./data/climate/train.csv'
    test_path = './data/climate/test.csv'
    split_climate(path,train_path,test_path)

