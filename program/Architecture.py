# Architecture.py
# This module defines the architecture of a neural network for image processing.

import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, groups=8):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(groups, out_channels)
        )
        self.se = ChannelAttention(out_channels)
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()
            
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.se(x)
        x += residual
        return self.activation(x)

class MultiScaleBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.branch5 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.fusion = nn.Conv2d(3*channels, channels, 1)

    def forward(self, x):
        b1 = self.branch1(x)
        b3 = self.branch3(x)
        b5 = self.branch5(x)
        fused = torch.cat([b1, b3, b5], dim=1)
        return self.fusion(fused) + x

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * 4, 3, padding=1)
        self.shuffle = nn.PixelShuffle(2)
        self.norm = nn.GroupNorm(8, out_channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.shuffle(x)
        return self.act(self.norm(x))

class AdvancedUNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=3):
        super().__init__()

        self.init_conv = nn.Conv2d(in_channels, 64, 3, padding=1)

        self.enc1 = nn.Sequential(
            ResidualConvBlock(64, 64),
            MultiScaleBlock(64)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            ResidualConvBlock(128, 128),
            MultiScaleBlock(128)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            ResidualConvBlock(256, 256),
            MultiScaleBlock(256)
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            ResidualConvBlock(512, 512),
            MultiScaleBlock(512)
        )

        self.bottleneck = nn.Sequential(
            ResidualConvBlock(512, 512),
            MultiScaleBlock(512),
            ResidualConvBlock(512, 512),
            ChannelAttention(512)
        )

        self.up1 = UpsampleBlock(512, 256)
        self.dec1 = nn.Sequential(
            ResidualConvBlock(512, 256),
            MultiScaleBlock(256)
        )
        
        self.up2 = UpsampleBlock(256, 128)
        self.dec2 = nn.Sequential(
            ResidualConvBlock(256, 128),
            MultiScaleBlock(128)
        )
        
        self.up3 = UpsampleBlock(128, 64)
        self.dec3 = nn.Sequential(
            ResidualConvBlock(128, 64),
            MultiScaleBlock(64)
        )

        self.skip1 = nn.Conv2d(256, 256, 1)
        self.skip2 = nn.Conv2d(128, 128, 1)
        self.skip3 = nn.Conv2d(64, 64, 1)
        
        self.out_conv = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, out_channels, 3, padding=1)
        )
        
    def forward(self, x):
        x0 = F.leaky_relu(self.init_conv(x), 0.2)

        e1 = self.enc1(x0)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        b = self.bottleneck(e4)
        
        d1 = self.up1(b)
        d1 = torch.cat([d1, self.skip1(e3)], dim=1) 
        d1 = self.dec1(d1)
        
        d2 = self.up2(d1)
        d2 = torch.cat([d2, self.skip2(e2)], dim=1)
        d2 = self.dec2(d2)
        
        d3 = self.up3(d2)
        d3 = torch.cat([d3, self.skip3(e1)], dim=1)
        d3 = self.dec3(d3)

        return torch.tanh(self.out_conv(d3))