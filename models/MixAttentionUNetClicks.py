from __future__ import print_function, division
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch

class ConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.conv = nn.Sequential(
                                  nn.Conv2d(ch_in, ch_out,
                                            kernel_size=3, stride=1,
                                            padding=1, bias=True),
                                  nn.BatchNorm2d(ch_out),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(ch_out, ch_out,
                                            kernel_size=3, stride=1,
                                            padding=1, bias=True),
                                  nn.BatchNorm2d(ch_out),
                                  nn.ReLU(inplace=True),
        )
        
    def forward(self, x):
        x = self.conv(x)
        return x

class UpConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.up = nn.Sequential(
                                nn.Upsample(scale_factor=2),
                                nn.Conv2d(ch_in, ch_out,
                                         kernel_size=3,stride=1,
                                         padding=1, bias=True),
                                nn.BatchNorm2d(ch_out),
                                nn.ReLU(inplace=True),
        )
        
    def forward(self, x):
        x = x = self.up(x)
        return x

class SAModule(nn.Module):
    def __init__(self, in_channels=2):
        super(SAModule, self).__init__()
        self.kernel_size = 7
        self.conv = nn.Conv2d(in_channels, out_channels=1, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_feature):
        cbam_feature = input_feature
        avg_pool = torch.mean(cbam_feature, dim=1, keepdim=True)
        max_pool = torch.max(cbam_feature, dim=1, keepdim=True)[0]
        concat = torch.cat([avg_pool, max_pool], dim=1)
        cbam_feature = self.conv(concat)        
        return input_feature * self.sigmoid(cbam_feature)
    
class AttentionBlock(nn.Module):
    def __init__(self, f_g, f_l, f_int):
        super().__init__()
        
        self.w_g = nn.Sequential(
                                nn.Conv2d(f_g, f_int,
                                         kernel_size=1, stride=1,
                                         padding=0, bias=True),
                                nn.BatchNorm2d(f_int)
        )
        
        self.w_x = nn.Sequential(
                                nn.Conv2d(f_l, f_int,
                                         kernel_size=1, stride=1,
                                         padding=0, bias=True),
                                nn.BatchNorm2d(f_int)
        )
        
        self.psi = nn.Sequential(
                                nn.Conv2d(f_int, 1,
                                         kernel_size=1, stride=1,
                                         padding=0,  bias=True),
                                nn.BatchNorm2d(1),
                                nn.Sigmoid(),
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.w_g(g)
        x1 = self.w_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)
        
        return psi*x

class ReduceBlock(nn.Module):
    def __init__(self):
        super(ReduceBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)  # Max pooling with kernel size 2 and stride 2

    def forward(self, clicks):
        # Apply max pooling iteratively to reduce dimensionality
        clicks_64 = self.maxpool(clicks)  # Reduce to 64x64
        clicks_32 = self.maxpool(clicks_64)  # Reduce to 32x32
        clicks_16 = self.maxpool(clicks_32)  # Reduce to 16x16
        return clicks_64, clicks_32, clicks_16

class MixAttentionUNetClicks(nn.Module):
    def __init__(self, n_classes=1, in_channel=3, out_channel=1):
        super().__init__() 

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv1 = ConvBlock(ch_in=in_channel, ch_out=64)
        self.conv2 = ConvBlock(ch_in=64, ch_out=128)
        self.conv3 = ConvBlock(ch_in=128, ch_out=256)
        self.conv4 = ConvBlock(ch_in=256, ch_out=512)
        self.conv5 = ConvBlock(ch_in=512, ch_out=1024)
        
        self.up5 = UpConvBlock(ch_in=1024, ch_out=512)
        self.att5 = AttentionBlock(f_g=512, f_l=512, f_int=256)
        self.att5CL = AttentionBlock(f_g=1, f_l=1024, f_int=256)
        self.upconv5 = ConvBlock(ch_in=1024, ch_out=512)
        
        self.up4 = UpConvBlock(ch_in=512, ch_out=256)
        self.att4 = AttentionBlock(f_g=256, f_l=256, f_int=128)
        self.att4CL = AttentionBlock(f_g=1, f_l=512, f_int=256)
        self.upconv4 = ConvBlock(ch_in=512, ch_out=256)
        
        self.up3 = UpConvBlock(ch_in=256, ch_out=128)
        self.att3 = AttentionBlock(f_g=128, f_l=128, f_int=64)
        self.att3CL = AttentionBlock(f_g=1, f_l=256, f_int=256)
        self.upconv3 = ConvBlock(ch_in=256, ch_out=128)
        
        self.up2 = UpConvBlock(ch_in=128, ch_out=64)
        self.att2 = AttentionBlock(f_g=64, f_l=64, f_int=32)
        self.att2CL = AttentionBlock(f_g=1, f_l=128, f_int=256)
        self.upconv2 = ConvBlock(ch_in=128, ch_out=64)
        
        self.conv_1x1 = nn.Conv2d(64, out_channel,
                                  kernel_size=1, stride=1, padding=0)
        self.attention = SAModule()
        self.reduce = ReduceBlock()
        
    def forward(self, x):
        # encoder
        clicks = x[:, 1:2, :, :]
        x = x[:, 0:1, :, :]
        x1 = self.conv1(x)
        c2,c3,c4 = self.reduce(clicks)

        x2 = self.maxpool(x1)
        x2 = self.conv2(x2)

        x3 = self.maxpool(x2)
        x3 = self.conv3(x3)

        x4 = self.maxpool(x3)
        x4 = self.conv4(x4)

        x5 = self.maxpool(x4)
        x5 = self.conv5(x5)

        # decoder + concat
        d5 = self.up5(x5)
        x4 = self.att5(g=d5, x=x4)        
        #x4 = self.attention(x4)
        d5 = torch.cat((x4, d5), dim=1)

        d5 = self.att5CL(g=c4, x=d5)
        d5 = self.upconv5(d5)
        
        d4 = self.up4(d5)
        x3 = self.att4(g=d4, x=x3)
        #x3 = self.attention(x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.att4CL(g=c3, x=d4)
        d4 = self.upconv4(d4)
        
        d3 = self.up3(d4)
        x2 = self.att3(g=d3, x=x2)
        #x2 = self.attention(x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.att3CL(g=c2, x=d3)
        d3 = self.upconv3(d3)
        
        d2 = self.up2(d3)
        x1 = self.att2(g=d2, x=x1)
        #x1 = self.attention(x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.att2CL(g=clicks, x=d2)
        d2 = self.upconv2(d2)
        
        d1 = self.conv_1x1(d2)
        
        return d1