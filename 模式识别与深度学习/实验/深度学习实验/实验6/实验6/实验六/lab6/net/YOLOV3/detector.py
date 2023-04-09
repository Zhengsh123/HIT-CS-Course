#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/6/3 10:53
# @Author  : ZSH
from net.YOLOV3.model import Darknet
import torchvision
from net.YOLOV3.util import *
import torch


class Detector(torch.nn.Module):

    def __init__(self,save_net):
        super(Detector, self).__init__()
        self.net = Darknet()
        self.net.load_weights(save_net)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.eval()

    def forward(self, input, thresh, anchors):
        input_ = input

        output_13, output_26, output_52 = self.net(input_)
        idxs_13, vecs_13 = self._filter(output_13, thresh)
        boxes_13 = self._parse(idxs_13, vecs_13, 32, anchors[13])
        idxs_26, vecs_26 = self._filter(output_26, thresh)
        boxes_26 = self._parse(idxs_26, vecs_26, 16, anchors[26])
        idxs_52, vecs_52 = self._filter(output_52, thresh)
        boxes_52 = self._parse(idxs_52, vecs_52, 8, anchors[52])
        box = torch.cat([boxes_13, boxes_26, boxes_52], dim=0)
        box = nms(box)
        return box

    def _filter(self, output, thresh):
        output = output.permute(0, 2, 3, 1)
        output = output.reshape(output.size(0), output.size(1), output.size(2), 3, -1)
        mask = torch.sigmoid(output[..., 4]) > thresh
        idxs = mask.nonzero()
        vecs = output[mask]
        return idxs, vecs

    def _parse(self, idxs, vecs, t, anchors):
        anchors = torch.Tensor(anchors)
        n = idxs[:, 0]
        a = idxs[:, 3]
        cy = (idxs[:, 1].float() + torch.sigmoid(vecs[:, 1])) * t
        cx = (idxs[:, 2].float() + torch.sigmoid(vecs[:, 0])) * t
        w = anchors[a, 0] * torch.exp(vecs[:, 2])
        h = anchors[a, 1] * torch.exp(vecs[:, 3])

        cls = torch.sigmoid(vecs[:,4])


        if len(vecs[:,5:85]) > 0:
            _,pred = torch.max(vecs[:,5:85],dim=1)
            box = torch.stack([n.float(), cx, cy, w, h,pred.float(),cls], dim=1)
        else:
            box = torch.stack([n.float(), cx, cy, w, h, h,cls], dim=1)

        return box


def draw_box(weight_path,test_img_path,res_img_path):
    ANCHORS_GROUP = {
        13: [[116, 90], [156, 198], [373, 326]],
        26: [[30, 61], [62, 45], [59, 119]],
        52: [[10, 13], [16, 30], [33, 23]]
    }
    transforms = torchvision.transforms.Compose([  # 归一化，Tensor处理
        torchvision.transforms.ToTensor()
    ])
    detector = Detector(weight_path)
    image = Image.open(test_img_path)
    image_c = image.copy()
    W, H, scale, img = resize_img(image)
    img_data = transforms(img).unsqueeze(0)
    box = detector(img_data, 0.10, ANCHORS_GROUP)
    box = cal_box(W, H, scale, box)
    image_out = draw(box, image_c)
    image_out.save(res_img_path)

if __name__ == '__main__':
    pass

