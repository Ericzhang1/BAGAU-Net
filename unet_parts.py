""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, kernel=(3, 3), padding=1):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel[0], padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel[1], padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, kernel=(3, 3)):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, kernel=kernel)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            #model uses nearest instead of bilinear
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            #self.up = nn.Upsample(scale_factor=2, mode='nearest')
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class Attention_block(nn.Module):
    """
    Attention Block
    """
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            #make the channel of key and query consistent
            nn.Conv2d(F_l, F_g, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out

class Attention_up(nn.Module):
    '''upsampling with attention'''
    def __init__(self, in_channels, out_channels, att_in, att_out, bilinear=True):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            #model uses nearest instead of bilinear
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            #self.up = nn.Upsample(scale_factor=2, mode='nearest')
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.att = Attention_block(F_g=out_channels, F_l=att_in, F_int=att_out)
        self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x2 = self.att(g=x1, x=x2)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class Attention_block_atlas(nn.Module):
    """
    Attention Block
    """
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block_atlas, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out

class Attention_up_with_atlas_XR(nn.Module):
    '''upsampling with attention'''
    def __init__(self, in_channels, out_channels, att_out, encode_in, bilinear=True):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            #model uses nearest instead of bilinear
            self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            #self.up = nn.Upsample(scale_factor=2, mode='nearest')
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            
        self.att1 = Attention_block_atlas(F_g=out_channels, F_l=encode_in, F_int=att_out)
        self.att2 = Attention_block_atlas(F_g=out_channels, F_l=encode_in, F_int=att_out)
        self.out_conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.out_conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2, atlas):
        decoding1 = self.up1(x1)
        decoding2 = self.up2(atlas)
        # input is CHW
        diffY = x2.size()[2] - decoding1.size()[2]
        diffX = x2.size()[3] - decoding1.size()[3]

        decoding1 = F.pad(decoding1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        decoding2 = F.pad(decoding2, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        out1 = self.att1(decoding1, x2)
        out2 = self.att2(decoding2, x2)
        out = self.out_conv1(out1) + self.out_conv2(out2)
        x = torch.cat([out, decoding1], dim=1)
        return self.conv(x)

class Up_X(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels=0, out_channels=0, bilinear=True, kernel=(3, 3), padding=1):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            #model uses nearest instead of bilinear
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            #self.up = nn.Upsample(scale_factor=2, mode='nearest')
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.double_conv = DoubleConv(out_channels, out_channels, kernel=kernel, padding=padding)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        out = self.conv(x1)
        return self.double_conv(out + x2)
        
class Attention_fusion_XR(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, padding=0)
        self.attention_gate = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        x_attention = self.attention_gate(x2) * x1
        return self.conv(torch.cat([x1, x_attention], dim=1))


class Dsv_up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor):
        super().__init__()
        self.dsv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
            nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        )

    def forward(self, x):
        return self.dsv(x)

class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, act_func=nn.ReLU(inplace=True)):
        super(VGGBlock, self).__init__()
        self.act_func = act_func
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act_func(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act_func(out)

        return out

def up_sampling(x1, x2):
    up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    x1 = up(x1)
    # input is CHW
    diffY = x2.size()[2] - x1.size()[2]
    diffX = x2.size()[3] - x1.size()[3]
    x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                    diffY // 2, diffY - diffY // 2])
    return x1        