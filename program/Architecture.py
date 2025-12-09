# Architecture.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ResBlockNoNorm(nn.Module):
    def __init__(self, channels, use_se=True):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=True)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=True)
        self.se = SEBlock(channels) if use_se else nn.Identity()

    def forward(self, x):
        res = self.conv1(x)
        res = self.act(res)
        res = self.conv2(res)
        res = self.se(res)
        return x + res


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.unshuffle = nn.PixelUnshuffle(2)
        self.conv = nn.Conv2d(in_channels * 4, out_channels, 1, bias=True)

    def forward(self, x):
        x = self.unshuffle(x)
        x = self.conv(x)
        return x


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * 4, 3, padding=1, bias=True)
        self.shuffle = nn.PixelShuffle(2)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.shuffle(x)
        x = self.act(x)
        return x


class SOTARestorationUNet(nn.Module):

    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        base_channels=32,
        use_checkpointing=True,
        use_global_residual=True,
    ):
        super().__init__()
        self.use_checkpointing = use_checkpointing
        self.use_global_residual = use_global_residual

        self.intro = nn.Conv2d(in_channels, base_channels, 3, padding=1, bias=True)

        self.enc1 = nn.Sequential(
            ResBlockNoNorm(base_channels), ResBlockNoNorm(base_channels)
        )
        self.down1 = DownsampleBlock(base_channels, base_channels * 2)

        self.enc2 = nn.Sequential(
            ResBlockNoNorm(base_channels * 2), ResBlockNoNorm(base_channels * 2)
        )
        self.down2 = DownsampleBlock(base_channels * 2, base_channels * 4)

        self.enc3 = nn.Sequential(
            ResBlockNoNorm(base_channels * 4), ResBlockNoNorm(base_channels * 4)
        )
        self.down3 = DownsampleBlock(base_channels * 4, base_channels * 8)

        self.bottleneck = nn.Sequential(
            ResBlockNoNorm(base_channels * 8),
            ResBlockNoNorm(base_channels * 8),
            ResBlockNoNorm(base_channels * 8),
            ResBlockNoNorm(base_channels * 8),
        )

        self.up3 = UpsampleBlock(base_channels * 8, base_channels * 4)
        self.dec3 = nn.Sequential(
            nn.Conv2d(base_channels * 8, base_channels * 4, 1),
            ResBlockNoNorm(base_channels * 4),
            ResBlockNoNorm(base_channels * 4),
        )

        self.up2 = UpsampleBlock(base_channels * 4, base_channels * 2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 2, 1),
            ResBlockNoNorm(base_channels * 2),
            ResBlockNoNorm(base_channels * 2),
        )

        self.up1 = UpsampleBlock(base_channels * 2, base_channels)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, 1),
            ResBlockNoNorm(base_channels),
            ResBlockNoNorm(base_channels),
        )

        self.final_conv = nn.Conv2d(
            base_channels, out_channels, 3, padding=1, bias=True
        )
        self.final_act = nn.Sigmoid()

    def run_checkpoint(self, module, x):
        if self.training and self.use_checkpointing:
            return checkpoint(module, x, use_reentrant=False)
        return module(x)

    def forward(self, x, return_internals=False):
        input_image = x

        x1 = self.intro(x)

        e1 = self.run_checkpoint(self.enc1, x1)
        x2 = self.down1(e1)

        e2 = self.run_checkpoint(self.enc2, x2)
        x3 = self.down2(e2)

        e3 = self.run_checkpoint(self.enc3, x3)
        x4 = self.down3(e3)

        b = self.run_checkpoint(self.bottleneck, x4)

        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        out = self.final_conv(d1)

        if self.use_global_residual:
            out = out + input_image

        out = self.final_act(out)

        if return_internals:
            return out, {"bottleneck": b}

        return out


class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, n_layers=3):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, base_channels, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        for i in range(1, n_layers):
            in_ch = base_channels * (2 ** (i - 1))
            out_ch = base_channels * (2**i)
            layers.extend(
                [
                    nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(out_ch),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
        in_ch = base_channels * (2 ** (n_layers - 1))
        out_ch = base_channels * (2**n_layers)
        layers.extend(
            [
                nn.Conv2d(in_ch, out_ch, 4, stride=1, padding=1),
                nn.InstanceNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        )
        layers.append(nn.Conv2d(out_ch, 1, 4, stride=1, padding=1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def get_model(config: dict) -> nn.Module:
    return SOTARestorationUNet
