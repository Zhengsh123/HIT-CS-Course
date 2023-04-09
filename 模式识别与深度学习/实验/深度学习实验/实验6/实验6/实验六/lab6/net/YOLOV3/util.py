#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/6/3 10:54
# @Author  : ZSH
import torch
import numpy as np
import PIL.Image as Image
import PIL.ImageDraw as Draw
import random


def cal_iou(box, boxes, isMin=False):
    temp_box = box.detach().numpy()
    temp_boxes = boxes.detach().numpy()
    x1, x2 = temp_box[0] - temp_box[2] / 2, temp_box[0] + temp_box[2] / 2
    y1, y2 = temp_box[1] - temp_box[3] / 2, temp_box[1] + temp_box[3] / 2
    x1_temp, x2_temp = temp_boxes[:, 0] - temp_boxes[:, 2] / 2, temp_boxes[:, 0] + temp_boxes[:, 2] / 2
    y1_temp, y2_temp = temp_boxes[:, 1] - temp_boxes[:, 3] / 2, temp_boxes[:, 1] + temp_boxes[:, 3] / 2
    box_area, boxes_area = temp_box[2] * temp_box[3], temp_boxes[:, 2] * temp_boxes[:, 3]
    xx1, xx2 = np.maximum(x1, x1_temp), np.minimum(x2, x2_temp)
    yy1, yy2 = np.maximum(y1, y1_temp), np.minimum(y2, y2_temp)
    inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
    if isMin is True:
        iou = np.true_divide(inter, np.minimum(box_area, boxes_area))
    else:
        iou = np.true_divide(inter, (box_area + boxes_area - inter))
    return iou


def nms(box, thresh=0.2):
    if box.shape[0] == 0:
        return torch.Tensor([])
    box_sort = box[box[:, 6].argsort(descending=True)]
    res_box = []
    while box_sort.shape[0] > 1:
        a_box, b_box = box_sort[0], box_sort[1:]
        res_box.append(a_box.unsqueeze(0))
        index = np.where(cal_iou(a_box[1:5], b_box[:, 1:5]) < thresh)
        box_sort = b_box[index]
    if box_sort.shape[0] > 0:
        res_box.append(box_sort[0].unsqueeze(0))
    res_box = torch.cat(res_box, dim=0)
    return res_box


def resize_img(im):
    """
    缩放图片成416*416
    :param im:  原图
    :return:
    """
    resize_size = (416, 416)
    img = Image.new(mode="RGB", size=resize_size, color=(128, 128, 128))
    W, H = im.size
    max_side, min_side = max(W, H), min(W, H)
    scale = 416 / max_side
    Z = max_side - min_side
    im.thumbnail(resize_size)
    xy = (0, int((Z * scale) / 2)) if W > H else (int((Z * scale) / 2), 0)
    img.paste(im, xy)
    return W, H, scale, img


def cal_box(W, H, scale, box):
    """
    将resize后的框转换到原图上
    :param W: 原图的W
    :param H: 原图的H
    :param scale: 缩放比例
    :param box: 网络输出的框
    :return: 转换到原图上的框
    """
    if box.shape[0] == 0:
        return torch.Tensor([])
    x, y = box[:, 1:2], box[:, 2:3]
    x, y = (x / scale, y / scale - (W - H) / 2) if W >= H else (x / scale - (H - W) / 2, y / scale)
    box[:, 1:2], box[:, 2:3] = x, y
    box[:, 3:5] = box[:, 3:5] / scale
    return box


def resize_box(W, H, scale, box):
    """
    将resize之后的标注也resize
    :param W: 原图W
    :param H: 原图H
    :param scale: 缩放比例
    :param box: 原图的标签框
    :return: 缩放后的标签框
    """
    cent_x,cent_y = box[:, 1:2],box[:, 2:3]
    cent_x,cent_y=(cent_x * scale,(cent_y + (W - H) / 2) * scale) if W > H \
        else ( (cent_x + (H - W) / 2) * scale,cent_y * scale)
    box[:, 1:2],box[:, 2:3] = cent_x,cent_y
    box[:, 3:5] = box[:, 3:5]*scale
    return box


def draw(box, image):
    """
    画图函数：把框在原图上画出
    :param box: 实际框
    :param image: 原图
    :return: 画了框的图
    """
    fp = open(r'coco.names', "r")
    res= fp.read().split("\n")[:-1]
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(res))]
    draw = Draw.ImageDraw(image)
    W, H = image.size
    for i in range(len(box)):
        cx,cy = box[i][1],box[i][2]
        w,h = box[i][3],box[i][4]
        num_class = int(box[i][5])
        cls = float('%.2f' % box[i][6])
        x1 = max(int(cx - w / 2),0)
        y1 = max(int(cy - h / 2),0)
        x2 = min(int(cx + w / 2),W)
        y2 = min(int(cy + h / 2),H)
        xy = (x1, y1, x2, y2)
        xy_ = (x1, y1 - 15, x2, y1)
        draw.rectangle(xy, fill=None, outline=tuple(colors[num_class]), width=3)
        draw.rectangle(xy_, fill=tuple(colors[num_class]), outline=tuple(colors[num_class]), width=3)
        draw.text(xy=(x1 + 2, y1 - 12), text=res[num_class] + " " + str(cls), fill="black", font=None)
    return image
