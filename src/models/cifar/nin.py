from __future__ import print_function
from __future__ import division
from models.layers import FCBlock, ConvBlock, InitialBlock, FinalBlock
import math
import torch.nn as nn


class NIN(nn.Module):
    def __init__(self, opt):
        super(NIN, self).__init__()
        self.layers = nn.Sequential(*self._createFeatures(opt))

    def forward(self, x):
        output = self.layers(x)
        output = output.view(x.size(0), -1)
        return output

    def _createFeatures(self, opt):
        hidden_size = opt.hidden_size
        layers = [InitialBlock(opt=opt, out_channels=192, kernel_size=5, stride=1, padding=2)]
        layers += [ConvBlock(opt=opt, in_channels=192, out_channels=160, kernel_size=1)]
        layers += [ConvBlock(opt=opt, in_channels=160, out_channels=96, kernel_size=1)]
        layers += [getattr(nn, opt.pooltype)(kernel_size=3, stride=2)]
        #layers += [nn.Dropout(opt.drop_rate)]
        layers += [ConvBlock(opt=opt, in_channels=96, out_channels=192, kernel_size=5, stride=1, padding=2)]
        layers += [ConvBlock(opt=opt, in_channels=192, out_channels=192, kernel_size=1)]
        layers += [ConvBlock(opt=opt, in_channels=192, out_channels=192, kernel_size=1)]
        layers += [getattr(nn, opt.pooltype)(kernel_size=3, stride=2)]
        #layers += [nn.Dropout(opt.drop_rate)]
        layers += [ConvBlock(opt=opt, in_channels=192, out_channels=192, kernel_size=3, stride=1, padding=1)]
        layers += [ConvBlock(opt=opt, in_channels=192, out_channels=192, kernel_size=1)]
        layers += [ConvBlock(opt=opt, in_channels=192, out_channels=opt.num_classes, kernel_size=1)]
        layers += [nn.AdaptiveAvgPool2d(1)]
        return layers
