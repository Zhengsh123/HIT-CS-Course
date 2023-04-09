import torch.nn.functional as F
import torch
import torch.nn as nn

'''
Inception-Resnet V2 网络超参数均参考原论文
'''


class BasicConv2(nn.Module):
    def __init__(self, in_size, out_size, kernel_size, stride, padding=0):
        super(BasicConv2, self).__init__()
        self.conv = nn.Conv2d(in_size, out_size, kernel_size=kernel_size,stride=stride, padding=padding, bias=False)
        self.batch_norm = nn.BatchNorm2d(out_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        return x


class Inception_Resnet_A(nn.Module):
    def __init__(self, in_size, scale=1.0):
        super(Inception_Resnet_A, self).__init__()
        self.scale = scale
        self.branch_0 = BasicConv2(in_size, 32, kernel_size=1, stride=1)
        self.branch_1 = nn.Sequential(
            BasicConv2(in_size, 32, kernel_size=1, stride=1),
            BasicConv2(32, 32, kernel_size=3, stride=1, padding=1)
        )
        self.branch_2 = nn.Sequential(
            BasicConv2(in_size, 32, kernel_size=1, stride=1),
            BasicConv2(32, 48, kernel_size=3, stride=1, padding=1),
            BasicConv2(48, 64, kernel_size=3, stride=1, padding=1)
        )
        self.conv = nn.Conv2d(128, 320, stride=(1,1), kernel_size=(1,1))
        self.relu = nn.ReLU()

    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        out = torch.cat((x0, x1, x2), dim=1)
        out = self.conv(out)
        return self.relu(x + self.scale * out)


class Reduction_A(nn.Module):
    def __init__(self, in_size, k, l, m, n):
        super(Reduction_A, self).__init__()
        self.branch_0 = BasicConv2(in_size, n, kernel_size=3, stride=2)
        self.branch_1 = nn.Sequential(
            BasicConv2(in_size, k, kernel_size=1, stride=1),
            BasicConv2(k, l, kernel_size=3, stride=1, padding=1),
            BasicConv2(l, m, kernel_size=3, stride=2)
        )
        self.branch_2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        return torch.cat((x0, x1, x2), dim=1)


class Inception_Resnet_B(nn.Module):
    def __init__(self, in_size, scale=1.0):
        super(Inception_Resnet_B, self).__init__()
        self.scale = scale
        self.branch_0 = BasicConv2(in_size, 192, kernel_size=1, stride=1)
        self.branch_1 = nn.Sequential(
            BasicConv2(in_size, 128, kernel_size=1, stride=1),
            BasicConv2(128, 160, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            BasicConv2(160, 192, kernel_size=(7, 1), stride=1, padding=(3, 0))
        )
        self.conv = nn.Conv2d(384, 1088, kernel_size=(1,1), stride=(1,1))
        self.relu = nn.ReLU()

    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        out = torch.cat((x0, x1), dim=1)
        out = self.conv(out)
        return self.relu(out * self.scale + x)


class Reduction_B(nn.Module):
    def __init__(self, in_size):
        super(Reduction_B, self).__init__()
        self.branch_0 = nn.Sequential(
            BasicConv2(in_size, 256, kernel_size=1, stride=1),
            BasicConv2(256, 384, kernel_size=3, stride=2)
        )
        self.branch_1 = nn.Sequential(
            BasicConv2(in_size, 256, kernel_size=1, stride=1),
            BasicConv2(256, 288, kernel_size=3, stride=2)
        )
        self.branch_2 = nn.Sequential(
            BasicConv2(in_size, 256, kernel_size=1, stride=1),
            BasicConv2(256, 288, kernel_size=3, stride=1, padding=1),
            BasicConv2(288, 320, kernel_size=3, stride=2)
        )
        self.branch_3 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x3 = self.branch_3(x)
        return torch.cat((x0, x1, x2, x3), dim=1)


class Inception_Resnet_C(nn.Module):
    def __init__(self, in_size, scale=1.0, activation=False):
        super(Inception_Resnet_C, self).__init__()
        self.scale = scale
        self.activation = activation
        self.branch_0 = BasicConv2(in_size, 192, kernel_size=1, stride=1)
        self.branch_1 = nn.Sequential(
            BasicConv2(in_size, 192, kernel_size=1, stride=1),
            BasicConv2(192, 224, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            BasicConv2(224, 256, kernel_size=(3, 1), stride=1, padding=(1, 0))
        )
        self.conv = nn.Conv2d(448, 2080, kernel_size=(1,1), stride=(1,1))
        self.relu = nn.ReLU()

    def forward(self, x):
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        out = torch.cat((x0, x1), dim=1)
        out = self.conv(out)
        if self.activation:
            return self.relu(out * self.scale + x)
        return out * self.scale + x


class InceptionResnetV2(torch.nn.Module):
    def __init__(self, num_classes):
        super(InceptionResnetV2, self).__init__()
        # Stem
        self.conv_1a = BasicConv2(3, 32, kernel_size=3, stride=2)
        self.conv_2a = BasicConv2(32, 32, kernel_size=3, stride=1)
        self.conv_2b = BasicConv2(32, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool_3a = nn.MaxPool2d(3, stride=2)
        self.conv_3b = BasicConv2(64, 80, kernel_size=1, stride=1)
        self.conv_3c = BasicConv2(80, 192, kernel_size=3, stride=1)
        self.maxpool_4a = nn.MaxPool2d(3, stride=2)
        self.branch_0 = BasicConv2(192, 96, kernel_size=1, stride=1)
        self.branch_1 = nn.Sequential(
            BasicConv2(192, 48, kernel_size=1, stride=1),
            BasicConv2(48, 64, kernel_size=5, stride=1, padding=2)
        )
        self.branch_2 = nn.Sequential(
            BasicConv2(192, 64, kernel_size=1, stride=1),
            BasicConv2(64, 96, kernel_size=3, stride=1, padding=1),
            BasicConv2(96, 96, kernel_size=3, stride=1, padding=1)
        )
        self.branch_3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv2(192, 64, kernel_size=1, stride=1)
        )
        # Inception A
        self.inception_a = nn.Sequential(
            Inception_Resnet_A(320, scale=0.17),
            Inception_Resnet_A(320, scale=0.17),
            Inception_Resnet_A(320, scale=0.17),
            Inception_Resnet_A(320, scale=0.17),
            Inception_Resnet_A(320, scale=0.17),
            Inception_Resnet_A(320, scale=0.17),
            Inception_Resnet_A(320, scale=0.17),
            Inception_Resnet_A(320, scale=0.17),
            Inception_Resnet_A(320, scale=0.17),
            Inception_Resnet_A(320, scale=0.17)
        )
        self.reduction_a = Reduction_A(320, 256, 256, 384, 384)
        self.inception_b = nn.Sequential(
            Inception_Resnet_B(1088, scale=0.10),
            Inception_Resnet_B(1088, scale=0.10),
            Inception_Resnet_B(1088, scale=0.10),
            Inception_Resnet_B(1088, scale=0.10),
            Inception_Resnet_B(1088, scale=0.10),
            Inception_Resnet_B(1088, scale=0.10),
            Inception_Resnet_B(1088, scale=0.10),
            Inception_Resnet_B(1088, scale=0.10),
            Inception_Resnet_B(1088, scale=0.10),
            Inception_Resnet_B(1088, scale=0.10),
            Inception_Resnet_B(1088, scale=0.10),
            Inception_Resnet_B(1088, scale=0.10),
            Inception_Resnet_B(1088, scale=0.10),
            Inception_Resnet_B(1088, scale=0.10),
            Inception_Resnet_B(1088, scale=0.10),
            Inception_Resnet_B(1088, scale=0.10),
            Inception_Resnet_B(1088, scale=0.10),
            Inception_Resnet_B(1088, scale=0.10),
            Inception_Resnet_B(1088, scale=0.10),
            Inception_Resnet_B(1088, scale=0.10)
        )
        self.reduction_b = Reduction_B(1088)
        self.inception_c = nn.Sequential(
            Inception_Resnet_C(2080, scale=0.20),
            Inception_Resnet_C(2080, scale=0.20),
            Inception_Resnet_C(2080, scale=0.20),
            Inception_Resnet_C(2080, scale=0.20),
            Inception_Resnet_C(2080, scale=0.20),
            Inception_Resnet_C(2080, scale=0.20),
            Inception_Resnet_C(2080, scale=0.20),
            Inception_Resnet_C(2080, scale=0.20),
            Inception_Resnet_C(2080, scale=0.20)
        )
        self.inception_c_last = Inception_Resnet_C(2080, scale=0.20, activation=True)
        self.conv = BasicConv2(2080, 1536, kernel_size=1, stride=1)
        self.global_average_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.liner = nn.Linear(1536, num_classes)

        # 初始化
        for m in self.modules():
             if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
             elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
             elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def features(self, input):
        # Stem
        x = self.conv_1a(input)
        x = self.conv_2a(x)
        x = self.conv_2b(x)
        x = self.maxpool_3a(x)
        x = self.conv_3b(x)
        x = self.conv_3c(x)
        x = self.maxpool_4a(x)
        x0 = self.branch_0(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x3 = self.branch_3(x)
        x = torch.cat((x0, x1, x2, x3), dim=1)
        # Inception A
        x = self.inception_a(x)
        # Reduction A
        x = self.reduction_a(x)
        # Inception B
        x = self.inception_b(x)
        # Reduction B
        x = self.reduction_b(x)
        # Inception C
        x = self.inception_c(x)
        x = self.inception_c_last(x)
        x = self.conv(x)
        return x

    def out(self, features):
        x = self.global_average_pooling(features)
        x = x.view(x.size(0), -1)
        x = self.liner(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.out(x)
        return x
