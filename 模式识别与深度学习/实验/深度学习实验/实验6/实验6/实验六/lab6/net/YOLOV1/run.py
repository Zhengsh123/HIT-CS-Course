#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/6/2 16:32
# @Author  : ZSH
from net.YOLOV1.model import resnet50
from net.YOLOV1.train import train,test
from generate_data import MyDataset
from torch.utils.data import DataLoader
from net.util import draw_box
import cv2
import torch
import argparse
import warnings

parser = argparse.ArgumentParser(description='')
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--train_root',default='../../data/train/JPEGImages/', type=str, help='for train images')
parser.add_argument('--test_root',default='../../data/test/JPEGImages/', type=str, help='for test images')
parser.add_argument('--train_info',default='../../data/train.txt', type=str, help='for train images')
parser.add_argument('--test_info',default='../../data/test_info.txt', type=str, help='for test images')
parser.add_argument('--log_path',default='../../train/YOLOV1/log', type=str, help='')
parser.add_argument('--batch_size',default=4, type=int, help='batch size')
parser.add_argument('--epoch',default=20, type=int, help='training length')
parser.add_argument('--model_path',default='./checkpoint/optim.pth', type=str, help="")
parser.add_argument('--test_img_path',default='./testimg/dog.jpg', type=str, help="")
parser.add_argument('--test_img_res_path',default='./testimg/dog_result.jpg', type=str, help="")
args = parser.parse_args()
device=torch.device(device = 'cuda' if torch.cuda.is_available() else 'cpu')

if __name__=="__main__":
    warnings.filterwarnings('ignore')
    model=resnet50().to(device)
    model.load_state_dict((torch.load(args.model_path)))
    print("load ok")
    draw_box(model,args.test_img_path,args.test_img_res_path)

    # train_data=MyDataset(args.train_root,args.train_info,is_train=True)
    # test_data=MyDataset(args.test_root,args.test_info,is_train=False)
    # train_loader=DataLoader(train_data,args.batch_size,shuffle=True)
    # test_loader=DataLoader(test_data,args.batch_size,shuffle=False)
    # train(args,model,train_loader,device)
    # torch.save(model.state_dict(),'./YOLOV1.pt')


