#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/6/2 23:27
# @Author  : ZSH
"""
YOLO V5只作为体验，因此直接载入官方模型与参数
"""
import cv2
import torch
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
img=cv2.imread('./test_img/dog.jpg')[:, :, ::-1]
results = model(img, size=640)
results.print()
results.save()
