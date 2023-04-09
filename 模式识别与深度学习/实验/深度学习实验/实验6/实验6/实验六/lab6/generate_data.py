#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/6/1 22:17
# @Author  : ZSH
# @Reference  : https://github.com/abeardear/pytorch-YOLO-v1,本文件数据增强部分参考该项目

'''
用于读取VOC数据中的信息，生成训练集和测试集
'''

import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import cv2
import numpy as np
import random


class MyDataset(Dataset):
    def __init__(self, root_path, idx_file_path, is_train, snumber=14, bnumber=2, cnumber=20, image_size=448):
        """
        初始化自定义Dataset
        :param root_path: 需要建立的文件的root地址，例如'./data/train/JPEGImages/'
        :param idx_file_path: 从xml转换来的txt地址
        :param is_train: True时表示是训练集，需要做transform
        :param snumber:每张图片分为的格子数量(s*s)
        :param bnumber:每个格子能检测的物体数量
        :param cnumber:物体总量
        :param image_size:图片大小
        """
        self.root_path = root_path
        self.S = snumber
        self.B = bnumber
        self.C = cnumber
        self.mean = (123, 117, 104)  # RGB
        self.image_size = (image_size, image_size)
        self.is_train = is_train
        self.file_name = []  # 存储每条数据对应的图片文件名
        self.box = []  # 存储每条数据对应的框的取值范围
        self.class_label = []  # 存储每条数据对应的包含的物体的分类
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        with open(idx_file_path, 'r')as fr:
            lines = fr.readlines()
        for line in lines:
            line_split = line.strip().split()
            self.file_name.append(line_split[0])
            box_num = (len(line_split) - 1) // 5
            box = []
            label = []
            for i in range(box_num):
                x, y, x2, y2, c = (float(line_split[j + 5 * i]) for j in range(1, 6, 1))
                box.append([x, y, x2, y2])
                label.append(int(c))
            self.box.append(torch.tensor(box))
            self.class_label.append(torch.LongTensor(label))
        self.num = len(self.box)

    def __getitem__(self, idx):
        name = self.file_name[idx]
        img = cv2.imread(os.path.join(self.root_path, name))
        boxes = self.box[idx].clone()
        labels = self.class_label[idx].clone()
        if self.is_train is True:
            img, boxes, labels = self.__my_transform(img, boxes, labels)
        h, w, _ = img.shape
        boxes /= torch.Tensor([w, h, w, h]).expand_as(boxes)
        target = self.convert(labels, boxes)
        target = torch.tensor(target).float()
        img = cv2.resize(img, self.image_size)
        img = self.transform(self.__bgr2rgb(img))
        data = {
            'data': img,
            'target': target
        }
        return data

    def __len__(self):
        return self.num

    def __my_transform(self, img, boxes, labels):
        img, boxes = self.random_flip(img, boxes)
        img, boxes = self.randomScale(img, boxes)
        img = self.randomBlur(img)
        img = self.RandomBrightness(img)
        img = self.RandomHue(img)
        img = self.RandomSaturation(img)
        img, boxes, labels = self.randomShift(img, boxes, labels)
        img, boxes, labels = self.randomCrop(img, boxes, labels)
        return img, boxes, labels

    def __bgr2rgb(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def box_to_center(self, bboxes):
        """
        将候选框从四角坐标转换成中心坐标+宽和高
        :param bboxes:原始候选框
        :return:
        """
        res = []
        for bbox in bboxes:
            x_center, y_center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
            width, height = (bbox[0] - bbox[2]), (bbox[1] - bbox[3])
            res.append([x_center, y_center, width, height])
        return res

    def convert(self, labels, boxes):
        """
        将每张图对应的box和label转换成这张图对应的向量
        :param labels: list，对应每个box的物体类型
        :param boxes: list，对应每个box的坐标
        :return: array  [self.S, self.S, self.B*5+self.C]
        """
        boxes = self.box_to_center(boxes)
        num_res = self.B * 5 + self.C
        num_boxes = len(boxes)
        if num_boxes == 0:
            return np.zeros((self.S, self.S, num_res))
        labels = np.array(labels, dtype=np.int)
        boxes = np.array(boxes, dtype=np.float)
        np_target = np.zeros((self.S, self.S, num_res))
        np_class = np.zeros((num_boxes, self.C))
        for i in range(num_boxes):
            np_class[i, labels[i]] = 1
        x_center = boxes[:, 0].reshape(-1, 1)
        y_center = boxes[:, 1].reshape(-1, 1)
        w = boxes[:, 2].reshape(-1, 1)
        h = boxes[:, 3].reshape(-1, 1)
        x_idx = np.ceil(x_center * self.S) - 1
        y_idx = np.ceil(y_center * self.S) - 1
        x_idx[x_idx < 0] = 0
        y_idx[y_idx < 0] = 0
        x_center = x_center - x_idx / self.S - 1 / (2 * self.S)
        y_center = y_center - y_idx / self.S - 1 / (2 * self.S)
        conf = np.ones_like(x_center)
        temp = np.concatenate([x_center, y_center, w, h, conf], axis=1)
        temp = np.repeat(temp, self.B, axis=0).reshape(num_boxes, -1)
        temp = np.concatenate([temp, np_class], axis=1)
        for i in range(num_boxes):
            np_target[int(y_idx[i]), int(x_idx[i])] = temp[i]
        return np_target

    def BGR2HSV(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    def HSV2BGR(self, img):
        return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

    def RandomBrightness(self, bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h, s, v = cv2.split(hsv)
            adjust = random.choice([0.5, 1.5])
            v = v * adjust
            v = np.clip(v, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h, s, v))
            bgr = self.HSV2BGR(hsv)
        return bgr

    def RandomSaturation(self, bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h, s, v = cv2.split(hsv)
            adjust = random.choice([0.5, 1.5])
            s = s * adjust
            s = np.clip(s, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h, s, v))
            bgr = self.HSV2BGR(hsv)
        return bgr

    def RandomHue(self, bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h, s, v = cv2.split(hsv)
            adjust = random.choice([0.5, 1.5])
            h = h * adjust
            h = np.clip(h, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h, s, v))
            bgr = self.HSV2BGR(hsv)
        return bgr

    def randomBlur(self, bgr):
        if random.random() < 0.5:
            bgr = cv2.blur(bgr, (5, 5))
        return bgr

    def randomShift(self, bgr, boxes, labels):
        # 平移变换
        center = (boxes[:, 2:] + boxes[:, :2]) / 2
        if random.random() < 0.5:
            height, width, c = bgr.shape
            after_shfit_image = np.zeros((height, width, c), dtype=bgr.dtype)
            after_shfit_image[:, :, :] = (104, 117, 123)  # bgr
            shift_x = random.uniform(-width * 0.2, width * 0.2)
            shift_y = random.uniform(-height * 0.2, height * 0.2)
            # print(bgr.shape,shift_x,shift_y)
            # 原图像的平移
            if shift_x >= 0 and shift_y >= 0:
                after_shfit_image[int(shift_y):, int(shift_x):, :] = bgr[:height - int(shift_y), :width - int(shift_x),
                                                                     :]
            elif shift_x >= 0 and shift_y < 0:
                after_shfit_image[:height + int(shift_y), int(shift_x):, :] = bgr[-int(shift_y):, :width - int(shift_x),
                                                                              :]
            elif shift_x < 0 and shift_y >= 0:
                after_shfit_image[int(shift_y):, :width + int(shift_x), :] = bgr[:height - int(shift_y), -int(shift_x):,
                                                                             :]
            elif shift_x < 0 and shift_y < 0:
                after_shfit_image[:height + int(shift_y), :width + int(shift_x), :] = bgr[-int(shift_y):,
                                                                                      -int(shift_x):, :]

            shift_xy = torch.FloatTensor([[int(shift_x), int(shift_y)]]).expand_as(center)
            center = center + shift_xy
            mask1 = (center[:, 0] > 0) & (center[:, 0] < width)
            mask2 = (center[:, 1] > 0) & (center[:, 1] < height)
            mask = (mask1 & mask2).view(-1, 1)
            boxes_in = boxes[mask.expand_as(boxes)].view(-1, 4)
            if len(boxes_in) == 0:
                return bgr, boxes, labels
            box_shift = torch.FloatTensor([[int(shift_x), int(shift_y), int(shift_x), int(shift_y)]]).expand_as(
                boxes_in)
            boxes_in = boxes_in + box_shift
            labels_in = labels[mask.view(-1)]
            return after_shfit_image, boxes_in, labels_in
        return bgr, boxes, labels

    def randomCrop(self, bgr, boxes, labels):
        if random.random() < 0.5:
            center = (boxes[:, 2:] + boxes[:, :2]) / 2
            height, width, c = bgr.shape
            h = random.uniform(0.6 * height, height)
            w = random.uniform(0.6 * width, width)
            x = random.uniform(0, width - w)
            y = random.uniform(0, height - h)
            x, y, h, w = int(x), int(y), int(h), int(w)

            center = center - torch.FloatTensor([[x, y]]).expand_as(center)
            mask1 = (center[:, 0] > 0) & (center[:, 0] < w)
            mask2 = (center[:, 1] > 0) & (center[:, 1] < h)
            mask = (mask1 & mask2).view(-1, 1)

            boxes_in = boxes[mask.expand_as(boxes)].view(-1, 4)
            if (len(boxes_in) == 0):
                return bgr, boxes, labels
            box_shift = torch.FloatTensor([[x, y, x, y]]).expand_as(boxes_in)

            boxes_in = boxes_in - box_shift
            boxes_in[:, 0] = boxes_in[:, 0].clamp_(min=0, max=w)
            boxes_in[:, 2] = boxes_in[:, 2].clamp_(min=0, max=w)
            boxes_in[:, 1] = boxes_in[:, 1].clamp_(min=0, max=h)
            boxes_in[:, 3] = boxes_in[:, 3].clamp_(min=0, max=h)

            labels_in = labels[mask.view(-1)]
            img_croped = bgr[y:y + h, x:x + w, :]
            return img_croped, boxes_in, labels_in
        return bgr, boxes, labels

    def randomScale(self, bgr, boxes):
        if random.random() < 0.5:
            scale = random.uniform(0.8, 1.2)
            height, width, c = bgr.shape
            bgr = cv2.resize(bgr, (int(width * scale), height))
            scale_tensor = torch.FloatTensor([[scale, 1, scale, 1]]).expand_as(boxes)
            boxes = boxes * scale_tensor
            return bgr, boxes
        return bgr, boxes

    def subMean(self, bgr, mean):
        mean = np.array(mean, dtype=np.float32)
        bgr = bgr - mean
        return bgr

    def random_flip(self, im, boxes):
        if random.random() < 0.5:
            im_lr = np.fliplr(im).copy()
            h, w, _ = im.shape
            xmin = w - boxes[:, 2]
            xmax = w - boxes[:, 0]
            boxes[:, 0] = xmin
            boxes[:, 2] = xmax
            return im_lr, boxes
        return im, boxes

    def random_bright(self, im, delta=16):
        alpha = random.random()
        if alpha > 0.3:
            im = im * alpha + random.randrange(-delta, delta)
            im = im.clip(min=0, max=255).astype(np.uint8)
        return im


if __name__ == "__main__":
    root_path = './data/train/JPEGImages/'
    idx_file_path = './data/train_info.txt'
    train_dataset = MyDataset(root_path, idx_file_path, True)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    for i, data in enumerate(train_loader):
        print(data['data'])
