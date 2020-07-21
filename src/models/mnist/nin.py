import torch.nn as nn
from models.layers import ConvBlock, InitialBlock

class NIN(nn.Module):
    def __init__(self, opt):
        super(NIN, self).__init__()
        self.layers = nn.Sequential(*self._createFeatures(opt))

    def forward(self, x):
        output = self.layers(x)
        output = output.view(x.size(0), -1)
        return output

    def _createFeatures(self, opt):
        layers = [InitialBlock(opt=opt, out_channels=128, kernel_size=5, stride=1, padding=2)]
        layers += [ConvBlock(opt=opt, in_channels=128, out_channels=96, kernel_size=1)]
        layers += [ConvBlock(opt=opt, in_channels=96, out_channels=48, kernel_size=1)]
        layers += [getattr(nn, opt.pooltype)(kernel_size=3, stride=2)]
        #layers += [nn.Dropout(opt.drop_rate)]
        layers += [ConvBlock(opt=opt, in_channels=48, out_channels=128, kernel_size=5, stride=1, padding=2)]
        layers += [ConvBlock(opt=opt, in_channels=128, out_channels=96, kernel_size=1)]
        layers += [ConvBlock(opt=opt, in_channels=96, out_channels=48, kernel_size=1)]
        layers += [getattr(nn, opt.pooltype)(kernel_size=3, stride=2)]
        #layers += [nn.Dropout(opt.drop_rate)]
        layers += [ConvBlock(opt=opt, in_channels=48, out_channels=128, kernel_size=3, stride=1, padding=1)]
        layers += [ConvBlock(opt=opt, in_channels=128, out_channels=96, kernel_size=1)]
        layers += [ConvBlock(opt=opt, in_channels=96, out_channels=opt.num_classes, kernel_size=1)]
        layers += [nn.AdaptiveAvgPool2d(1)]
        return layers
