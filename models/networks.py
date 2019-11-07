import torch
import torch.nn as nn
import numpy as np
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import math
import torch.nn.init as init


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilate=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=0.01)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               dilation=dilate, padding=dilate, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=0.01)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out




class L2Norm(nn.Module):
    def __init__(self, n_channels, scale):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.weight, self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
        x = torch.div(x, norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out


class ResNet(nn.Module):

    def __init__(self, block_name, layers):
        self.inplanes = 64
        if block_name == 'Bottleneck':
            self.block = Bottleneck

        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(self.block, 64, layers[0], stride=1, dilate=1)
        self.layer2 = self._make_layer(self.block, 128, layers[1], stride=2, dilate=1)
        self.layer3 = self._make_layer(self.block, 256, layers[2], stride=2, dilate=1)
        self.layer4 = self._make_layer(self.block, 512, layers[3], stride=1,dilate=2)

        self.norm2 = L2Norm(64*self.block.expansion,10)

        self.deconv3 = nn.Sequential(L2Norm(128*self.block.expansion,10),
                           nn.ConvTranspose2d(128*self.block.expansion, 256, kernel_size=4, stride=2, padding=1))
        self.deconv4 = nn.Sequential(L2Norm(256*self.block.expansion,10),
                          nn.ConvTranspose2d(256*self.block.expansion, 256, kernel_size=4, stride=4, padding=0))
        self.deconv5 = nn.Sequential(L2Norm(512*self.block.expansion,10),
                        nn.ConvTranspose2d(512*self.block.expansion, 256, kernel_size=4, stride=4, padding=0))
        # if self.block == BasicBlock:
        #     self.concat_sizes = self.layer2[layers[0]-1].conv2.out_channels+self.layer3[layers[1]-1].conv2.out_channels+self.layer4[layers[2]-1].conv2.out_channels+self.layer4[layers[3]-1].conv2.out_channels
        # elif self.block == Bottleneck:
        #     self.concat_sizes = self.layer1[layers[0]-1].conv3.out_channels+self.layer2[layers[1]-1].conv3.out_channels+self.layer3[layers[2]-1].conv3.out_channels+ self.layer4[layers[3]-1].conv3.out_channels


        self.cat = nn.Conv2d(256*4,256,kernel_size=3,stride=1,padding=1)
        self.cat_bn = nn.BatchNorm2d(256, momentum=0.01)
        self.cat_act = nn.ReLU(inplace=True)

        self.heat_conv = nn.Conv2d(256,1,kernel_size=1,stride=1,padding=0)
        self.scale_conv = nn.Conv2d(256,1,kernel_size=1,stride=1,padding=0)

        self.heat_act = nn.Sigmoid()



        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.in_features * m.out_features
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m,nn.ConvTranspose2d):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=0.01),
            )

        # downsample是对identity 映射x的操作确定是否要对x 下采样

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilate, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, dilate))

        return nn.Sequential(*layers)




    def forward(self, inputs):
        #ipdb.set_trace()
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x1 = self.maxpool(x)

        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)

        x2 = self.norm2(x2)
        x3 = self.deconv3(x3)
        x4 = self.deconv4(x4)
        x5 = self.deconv5(x5)

        map = torch.cat([x2,x3,x4,x5],dim=1)
        map = self.cat(map)
        map = self.cat_bn(map)
        map = self.cat_act(map)

        center_map = self.heat_conv(map)
        center_map = self.heat_act(center_map)

        scale_map = self.scale_conv(map)


        return center_map,scale_map
