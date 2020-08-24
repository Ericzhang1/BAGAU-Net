""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from unet_parts import *

import numpy as np
#import matplotlib.pyplot as plt
class Dk_UNet_XR(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, aging=True, size=3):
        super(Dk_UNet_XR, self).__init__()
        channel = 2 if aging and n_channels == 4 else 1
        self.n_channels = n_channels - channel
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.encoding_channel = channel
        self.kernel = (3, 3) if size == 3 else (5, 5)
        self.padding = 1 if size == 3 else 2

        #kernel size changed for the first layer
        self.inc = DoubleConv(self.n_channels, 64)
        self.MaxPool1 = nn.MaxPool2d(2)
        self.down1 = DoubleConv(64, 96)
        self.MaxPool2 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(96, 128)
        self.MaxPool3 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(128, 256)
        self.MaxPool4 = nn.MaxPool2d(2)
        self.down4 = DoubleConv(256, 512)
        #more attention channels if using large
        self.up1 = Attention_up_with_atlas_XR(160, 64, 32, 96)
        self.up2 = Attention_up_with_atlas_XR(224, 96, 64, 128)
        self.up3 = Attention_up_with_atlas_XR(384, 128, 96, 256)
        self.up4 = Attention_up_with_atlas_XR(768, 256, 128, 512)
        #self.outc = OutConv(64, n_classes)

        #domain knowledge encoder
        self.encode_inc = DoubleConv(channel, 64, kernel=self.kernel, padding=self.padding)
        self.encode_MaxPool1 = nn.MaxPool2d(2)
        self.encode_down1 = DoubleConv(64, 96, kernel=self.kernel, padding=self.padding)
        self.encode_MaxPool2 = nn.MaxPool2d(2)
        self.encode_down2 = DoubleConv(96, 128, kernel=self.kernel, padding=self.padding)
        self.encode_MaxPool3 = nn.MaxPool2d(2)
        self.encode_down3 = DoubleConv(128, 256, kernel=self.kernel, padding=self.padding)
        self.encode_MaxPool4 = nn.MaxPool2d(2)
        self.encode_down4 = DoubleConv(256, 512, kernel=self.kernel, padding=self.padding)

        self.encode_up1 = Up_X(512, 256, bilinear, kernel=self.kernel, padding=self.padding)
        self.encode_up2 = Up_X(256, 128, bilinear, kernel=self.kernel, padding=self.padding)
        self.encode_up3 = Up_X(128, 96, bilinear, kernel=self.kernel, padding=self.padding)
        self.encode_up4 = Up_X(96, 64, bilinear, kernel=self.kernel, padding=self.padding)
        #self.final = Attention_block_atlas(64, 64, 32)
        self.final = Attention_fusion_XR(64, n_classes)

    def forward(self, x):
        #spliting orginal image with atlas
        image = x[:, 0:-self.encoding_channel, :, :]
        atlas = x[:, -self.encoding_channel:, : , :]
        atlas_mask = (atlas > 0.5).float()

        x1 = self.inc(image)
        x2 = self.MaxPool1(x1)
        x2 = self.down1(x2)
        x3 = self.MaxPool1(x2)
        x3 = self.down2(x3)
        x4 = self.MaxPool1(x3)
        x4 = self.down3(x4)
        x5 = self.MaxPool1(x4)
        x5 = self.down4(x5)

        #Encoding of atlas
        atlas1 = self.encode_inc(atlas)
        atlas2 = self.encode_MaxPool1(atlas1)
        atlas2 = self.encode_down1(atlas2)
        atlas3 = self.encode_MaxPool2(atlas2)
        atlas3 = self.encode_down2(atlas3)
        atlas4 = self.encode_MaxPool3(atlas3)
        atlas4 = self.encode_down3(atlas4)
        atlas5 = self.encode_MaxPool4(atlas4)
        atlas5 = self.encode_down4(atlas5)

        #atlas_hist = plt.hist(atlas5.flatten().numpy(), alpha=0.5, label='atlas')
        #normal_hist = plt.hist(x5.flatten().numpy(), alpha=0.5, label='normal')
        #bins = np.linspace(0, 10, 100)
        #_ = plt.hist([x5.flatten().numpy(), bins, atlas5.flatten().numpy()], alpha=0.5, label=['normal', 'atlas'])
        #plt.legend(loc='upper right')
        #plt.show()
        x = self.up4(x5, x4, atlas5)
        x_encode = self.encode_up1(atlas5, atlas4)
        x = self.up3(x, x3, atlas4)
        x_encode = self.encode_up2(x_encode, atlas3)
        x = self.up2(x, x2, atlas3)
        x_encode = self.encode_up3(x_encode, atlas2)
        x = self.up1(x, x1, atlas2)
        x_encode = self.encode_up4(x_encode, atlas1)
        logits = self.final(x, x_encode)
        #logits = self.outc(x)
    
        return torch.sigmoid(logits)
        if self.n_classes > 1:
            return F.softmax(x, dim=1)
        else:
            return torch.sigmoid(x)
