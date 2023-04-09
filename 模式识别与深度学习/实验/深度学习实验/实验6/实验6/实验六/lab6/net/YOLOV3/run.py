#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/6/3 10:17
# @Author  : ZSH
from net.YOLOV3.model import Darknet
import torch
from net.YOLOV3.detector import draw_box

if __name__=="__main__":
    device = torch.device(device='cuda' if torch.cuda.is_available() else 'cpu')
    # model = Darknet()
    # model.load_weights('./checkpoint/yolov3.weights')
    # model=model.to(device)
    weight_path='./checkpoint/yolov3.weights'
    test_img_path='./testimg/dog.jpg'
    test_img_result_image='./testimg/result.jpg'
    draw_box(weight_path,test_img_path,test_img_result_image)