#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/6/3 10:01
# @Author  : ZSH
import torch
import torch.nn as nn
import numpy as np


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False):
        super(ConvLayer, self).__init__()
        self.sub_module = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        return self.sub_module(x)


class ResLayer(nn.Module):
    def __init__(self, in_channels):
        super(ResLayer, self).__init__()
        self.sub_module = torch.nn.Sequential(
            ConvLayer(in_channels, in_channels // 2, 1, 1, 0),
            ConvLayer(in_channels // 2, in_channels, 3, 1, 1),
        )

    def forward(self, x):
        return x + self.sub_module(x)


class UpsampleLayer(nn.Module):

    def __init__(self):
        super(UpsampleLayer, self).__init__()

    def forward(self, x):
        return torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')


class DownsampleLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownsampleLayer, self).__init__()

        self.sub_module = torch.nn.Sequential(
            ConvLayer(in_channels, out_channels, 3, 2, 1)
        )

    def forward(self, x):
        return self.sub_module(x)


class ConvSet(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvSet, self).__init__()
        time_channel = out_channels * 2
        self.sub_module = torch.nn.Sequential(
            ConvLayer(in_channels, out_channels, 1, 1, 0),
            ConvLayer(out_channels, time_channel, 3, 1, 1),
            ConvLayer(time_channel, out_channels, 1, 1, 0),
            ConvLayer(out_channels, time_channel, 3, 1, 1),
            ConvLayer(time_channel, out_channels, 1, 1, 0),
        )

    def forward(self, x):
        return self.sub_module(x)


class Darknet(nn.Module):
    def __init__(self, cls=80):
        super(Darknet, self).__init__()
        output_channel = 3 * (5 + cls)
        self.trunk52 = nn.Sequential(
            ConvLayer(3, 32, 3, 1, 1),
            DownsampleLayer(32, 64),
            ResLayer(64),
            DownsampleLayer(64, 128),
            ResLayer(128),
            ResLayer(128),
            DownsampleLayer(128, 256),
            ResLayer(256),
            ResLayer(256),
            ResLayer(256),
            ResLayer(256),
            ResLayer(256),
            ResLayer(256),
            ResLayer(256),
            ResLayer(256),
        )

        self.trunk26 = nn.Sequential(
            DownsampleLayer(256, 512),

            ResLayer(512),
            ResLayer(512),
            ResLayer(512),
            ResLayer(512),
            ResLayer(512),
            ResLayer(512),
            ResLayer(512),
            ResLayer(512),
        )

        self.trunk13 = nn.Sequential(
            DownsampleLayer(512, 1024),
            ResLayer(1024),
            ResLayer(1024),
            ResLayer(1024),
            ResLayer(1024),
        )

        self.con_set13 = nn.Sequential(
            ConvSet(1024, 512)
        )

        self.predict_one = nn.Sequential(
            ConvLayer(512, 1024, 3, 1, 1),
            nn.Conv2d(1024, output_channel, 1, 1, 0)
        )

        self.up_to_26 = nn.Sequential(
            ConvLayer(512, 256, 1, 1, 0),
            UpsampleLayer()
        )

        self.con_set26 = nn.Sequential(
            ConvSet(768, 256)
        )

        self.predict_two = nn.Sequential(
            ConvLayer(256, 512, 3, 1, 1),
            nn.Conv2d(512, output_channel, 1, 1, 0)
        )

        self.up_to_52 = nn.Sequential(
            ConvLayer(256, 128, 1, 1, 0),
            UpsampleLayer()
        )

        self.con_set52 = nn.Sequential(
            ConvSet(384, 128)
        )

        self.predict_three = nn.Sequential(
            ConvLayer(128, 256, 3, 1, 1),
            nn.Conv2d(256, output_channel, 1, 1, 0)
        )

    def forward(self, x):
        feature_52 = self.trunk52(x)
        feature_26 = self.trunk26(feature_52)
        feature_13 = self.trunk13(feature_26)

        con_set_13_out = self.con_set13(feature_13)
        detection_13_out = self.predict_one(con_set_13_out)

        up_26_out = self.up_to_26(con_set_13_out)
        route_26_out = torch.cat((up_26_out, feature_26), dim=1)
        con_set_26_out = self.con_set26(route_26_out)
        detection_26_out = self.predict_two(con_set_26_out)

        up_52_out = self.up_to_52(con_set_26_out)
        route_52_out = torch.cat((up_52_out, feature_52), dim=1)
        con_set_52_out = self.con_set52(route_52_out)
        detection_52_out = self.predict_three(con_set_52_out)

        return detection_13_out, detection_26_out, detection_52_out

    def load_weights(self, weightfile):
        """
        读取.weights文件标准函数
        :param weightfile:
        :return:
        """
        # Open the weights file
        fp = open(weightfile, "rb")
        weights = np.fromfile(fp, dtype=np.float32)  # 加载 np.ndarray 中的剩余权重，权重是以float32类型存储的
        weights = weights[5:]

        model_list = []
        for model in self.modules():
            if isinstance(model, nn.BatchNorm2d):
                model_list.append(model)
            if isinstance(model, nn.Conv2d):
                model_list.append(model)

        ptr = 0
        is_continue = False
        for i in range(0, len(model_list)):
            if is_continue:
                is_continue = False
                continue

            conv = model_list[i]
            # print(i // 2, conv)

            if i < len(model_list) - 1 and isinstance(model_list[i + 1], nn.BatchNorm2d):
                is_continue = True

                bn = model_list[i + 1]
                # print(bn)
                num_bn_biases = bn.bias.numel()
                # print(num_bn_biases, weights[ptr:ptr + 4 * num_bn_biases])

                bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                ptr += num_bn_biases

                bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                ptr += num_bn_biases

                bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                ptr += num_bn_biases

                bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                ptr += num_bn_biases

                bn_biases = bn_biases.view_as(bn.bias.data)
                bn_weights = bn_weights.view_as(bn.weight.data)
                bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                bn_running_var = bn_running_var.view_as(bn.running_var)

                bn.bias.data.copy_(bn_biases)
                bn.weight.data.copy_(bn_weights)
                bn.running_mean.copy_(bn_running_mean)
                bn.running_var.copy_(bn_running_var)
            else:
                is_continue = False

                num_biases = conv.bias.numel()
                conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                ptr = ptr + num_biases
                conv_biases = conv_biases.view_as(conv.bias.data)
                conv.bias.data.copy_(conv_biases)

            num_weights = conv.weight.numel()
            conv_weights = torch.from_numpy(weights[ptr:ptr + num_weights])
            ptr = ptr + num_weights
            conv_weights = conv_weights.view_as(conv.weight.data)
            conv.weight.data.copy_(conv_weights)

        fp.close()


class MyYoLo3(nn.Module):
    def __init__(self,pretrain_path):
        super(MyYoLo3,self).__init__()
        self.net=Darknet()
        self.net.load_weights(pretrain_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
