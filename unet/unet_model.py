""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        #kernel size changed for the first layer
        self.inc = DoubleConv(n_channels, 64, kernel=(5, 5), padding=2)
        self.down1 = Down(64, 96)
        self.down2 = Down(96, 128)
        #Kernel size altered for the second conv operation
        self.down3 = Down(128, 256, kernel=(3, 3))
        self.down4 = Down(256, 512)
        #The last layer is not deployed
        self.up1 = Up(768, 256, bilinear)
        self.up2 = Up(384, 128, bilinear)
        self.up3 = Up(224, 96, bilinear)
        self.up4 = Up(160, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return torch.sigmoid(logits)
        #return torch.nn.LogSigmoid()(logits)

        if self.n_classes > 1:
            return F.softmax(x, dim=1)
        else:
            return torch.sigmoid(x)
