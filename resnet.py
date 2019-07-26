#coding=utf-8

from __future__ import print_function
from __future__ import division

import torch as t
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels:

            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, input):
        out = F.relu(self.bn1(self.conv1(input)))
        out = self.bn2(self.conv2(out))

        out += self.shortcut(input)

        out = F.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()

        self.in_channels = 16
        self.conv1 = nn.Conv2d(3, self.in_channels, 3, 1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)

        self.layer1 = self._make_layer(block, 16, 16, num_blocks[0], 1)
        self.layer2 = self._make_layer(block, 16, 32, num_blocks[1], 2)
        self.layer3 = self._make_layer(block, 32, 64, num_blocks[2], 2)

        self.linear = nn.Linear(64, num_classes)

    def _make_layer(self, block, in_channels, out_channels, num_blocks, stride):

        layers = []
        layers.append(block(in_channels, out_channels, stride))
        for i in range(1, num_blocks):
            layers.append(block(out_channels, out_channels, 1))

        return nn.Sequential(*layers)


    def forward(self, input):
        out = F.relu(self.bn1(self.conv1(input)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = nn.AdaptiveAvgPool2d(1)(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out

def resnet56():
    return ResNet(BasicBlock, num_blocks=[9, 9, 9])  # 3 * 9 * 2 + 2 = 56


if __name__ == '__main__':
    resnet = resnet56()

    print(resnet)