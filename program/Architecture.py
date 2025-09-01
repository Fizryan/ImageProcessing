# Architecture.py
# This module defines the architecture of a neural network for image processing.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torchvision import models
import math


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(x)


class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels) if norm else nn.Identity()
        self.norm2 = nn.GroupNorm(8, out_channels) if norm else nn.Identity()
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.res_conv = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        identity = self.res_conv(x)
        out = self.act(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.act(out + identity)


class ResidualDenseBlock(nn.Module):
    def __init__(self, channels, growth=32, num_layers=4):
        super().__init__()
        self.layers = nn.ModuleList()
        chs = channels
        for i in range(num_layers):
            self.layers.append(nn.Conv2d(chs, growth, 3, padding=1))
            chs += growth
        self.lff = nn.Conv2d(chs, channels, 1)

    def forward(self, x):
        feats = [x]
        for layer in self.layers:
            out = F.leaky_relu(layer(torch.cat(feats, 1)), 0.2)
            feats.append(out)
        return self.lff(torch.cat(feats, 1)) + x


class MultiScaleBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(channels, channels, 1), nn.LeakyReLU(0.2, inplace=True)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1), nn.LeakyReLU(0.2, inplace=True)
        )
        self.branch5 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fusion = nn.Conv2d(3 * channels, channels, 1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, a=0.2, mode="fan_in", nonlinearity="leaky_relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        b1 = self.branch1(x)
        b3 = self.branch3(x)
        b5 = self.branch5(x)
        fused = torch.cat([b1, b3, b5], dim=1)
        return self.fusion(fused) + x


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)

    def forward(self, x):
        return self.up(x)


class PerceptualLoss(nn.Module):
    def __init__(self, layers=None):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features.eval()
        for p in vgg.parameters():
            p.requires_grad = False

        self.vgg = vgg
        self.layer_indices = layers if layers is not None else [3, 8, 15, 22]
        self.criterion = nn.L1Loss()
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, source, target):
        source_denorm = (source + 1) / 2
        target_denorm = (target + 1) / 2
        source_norm = (source_denorm - self.mean) / self.std

        with torch.no_grad():
            target_norm = (target_denorm - self.mean) / self.std

        loss = 0.0
        source_feat = source_norm
        target_feat = target_norm

        for i, layer in enumerate(self.vgg):
            source_feat = layer(source_feat)
            with torch.no_grad():
                target_feat = layer(target_feat)

            if i in self.layer_indices:
                loss = loss + self.criterion(source_feat, target_feat)
        return loss


class EdgeLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        k_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        k_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        self.kernel = torch.cat([k_x, k_y], dim=0).to(device)
        self.criterion = nn.L1Loss()

    def forward(self, pred, target):
        pred_gray = 0.299 * pred[:, 0:1] + 0.587 * pred[:, 1:2] + 0.114 * pred[:, 2:3]
        target_gray = (
            0.299 * target[:, 0:1] + 0.587 * target[:, 1:2] + 0.114 * target[:, 2:3]
        )

        pred_edges = F.conv2d(pred_gray, self.kernel, padding=1)
        with torch.no_grad():
            target_edges = F.conv2d(target_gray, self.kernel, padding=1)

        return self.criterion(pred_edges, target_edges)


class MultiScaleLoss(nn.Module):
    def __init__(self, scales=[1, 0.5, 0.25], weights=[1.0, 0.5, 0.25]):
        super().__init__()
        self.scales = scales
        self.weights = weights
        self.criterion = nn.L1Loss()

    def forward(self, pred, target):
        total_loss = 0
        for scale, weight in zip(self.scales, self.weights):
            if scale != 1:
                pred_scaled = F.interpolate(
                    pred, scale_factor=scale, mode="bilinear", align_corners=False
                )
                target_scaled = F.interpolate(
                    target, scale_factor=scale, mode="bilinear", align_corners=False
                )
            else:
                pred_scaled, target_scaled = pred, target

            total_loss += weight * self.criterion(pred_scaled, target_scaled)
        return total_loss


class FrequencyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.L1Loss()

    def forward(self, pred, target):
        pred_fft = torch.fft.rfft2(pred, norm="ortho")
        target_fft = torch.fft.rfft2(target, norm="ortho")
        return self.criterion(torch.abs(pred_fft), torch.abs(target_fft))


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, n_layers=3):
        super().__init__()

        layers = [
            nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        nf_mult = 1
        for n in range(1, n_layers + 1):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            layers += [
                nn.Conv2d(
                    base_channels * nf_mult_prev,
                    base_channels * nf_mult,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                ),
                nn.GroupNorm(8, base_channels * nf_mult),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        layers += [nn.Conv2d(base_channels * nf_mult, 1, kernel_size=4, padding=1)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# 47.000.000+ Parameters and more Heavy
class AdvancedUNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=3):
        super().__init__()

        self.init_conv = nn.Conv2d(in_channels, 64, 3, padding=1)

        self.enc1 = nn.Sequential(ResidualConvBlock(64, 64), MultiScaleBlock(64))
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            ResidualConvBlock(128, 128),
            MultiScaleBlock(128),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            ResidualConvBlock(256, 256),
            MultiScaleBlock(256),
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            ResidualConvBlock(512, 512),
            MultiScaleBlock(512),
        )

        self.bottleneck = nn.Sequential(
            ResidualConvBlock(512, 512),
            MultiScaleBlock(512),
            ResidualConvBlock(512, 512),
            ChannelAttention(512),
        )

        self.up1 = UpsampleBlock(512, 256)
        self.dec1 = nn.Sequential(
            ResidualConvBlock(256 + 256, 256), MultiScaleBlock(256)
        )

        self.up2 = UpsampleBlock(256, 128)
        self.dec2 = nn.Sequential(
            ResidualConvBlock(128 + 128, 128), MultiScaleBlock(128)
        )

        self.up3 = UpsampleBlock(128, 64)
        self.dec3 = nn.Sequential(ResidualConvBlock(64 + 64, 64), MultiScaleBlock(64))

        self.skip1 = nn.Conv2d(256, 256, 1)
        self.skip2 = nn.Conv2d(128, 128, 1)
        self.skip3 = nn.Conv2d(64, 64, 1)

        self.out_conv = nn.Conv2d(64, out_channels, 3, padding=1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, a=0.2, mode="fan_in", nonlinearity="leaky_relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

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


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3)

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        maxv, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.sigmoid(self.conv(torch.cat([avg, maxv], 1)))
        return x * attn


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
        self.conv6 = nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6)
        self.conv12 = nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12)
        self.conv18 = nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18)
        self.out_conv = nn.Conv2d(out_channels * 4, out_channels, 1)

    def forward(self, x):
        out = torch.cat(
            [self.conv1(x), self.conv6(x), self.conv12(x), self.conv18(x)], 1
        )
        return self.out_conv(out)


class EnhancedMultiScaleBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.dilated1 = nn.Conv2d(channels, channels // 4, 3, padding=2, dilation=2)
        self.dilated2 = nn.Conv2d(channels, channels // 4, 3, padding=3, dilation=3)
        self.dilated3 = nn.Conv2d(channels, channels // 4, 3, padding=4, dilation=4)
        self.conv1x1 = nn.Conv2d(channels // 4 * 3 + channels, channels, 1)
        self.spatial_attn = SpatialAttention()

    def forward(self, x):
        identity = x
        d1 = F.leaky_relu(self.dilated1(x), 0.2)
        d2 = F.leaky_relu(self.dilated2(x), 0.2)
        d3 = F.leaky_relu(self.dilated3(x), 0.2)
        combined = torch.cat([identity, d1, d2, d3], dim=1)
        out = self.conv1x1(combined)
        return self.spatial_attn(out)


class GatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.feature_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding
        )
        self.mask_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding
        )
        nn.init.kaiming_normal_(self.feature_conv.weight, a=0.2)
        nn.init.kaiming_normal_(self.mask_conv.weight, a=0.2)

    def forward(self, x):
        feat = self.feature_conv(x)
        mask = torch.sigmoid(self.mask_conv(x))
        return feat * mask


# More stable than AdvancedUNet
class UNetLite(nn.Module):
    def __init__(
        self, in_channels=4, out_channels=3, base_channels=16, use_checkpointing=True
    ):
        super().__init__()
        self.use_checkpointing = use_checkpointing

        self.init_conv = GatedConv2d(in_channels, base_channels)

        self.enc1 = ResidualDenseBlock(base_channels)
        self.enc2 = nn.Sequential(
            GatedConv2d(base_channels, base_channels * 2, 4, stride=2, padding=1),
            ResidualDenseBlock(base_channels * 2),
        )
        self.enc3 = nn.Sequential(
            GatedConv2d(base_channels * 2, base_channels * 4, 4, stride=2, padding=1),
            ResidualDenseBlock(base_channels * 4),
        )
        self.enc4 = nn.Sequential(
            GatedConv2d(base_channels * 4, base_channels * 8, 4, stride=2, padding=1),
            ResidualDenseBlock(base_channels * 8),
        )

        self.bottleneck = nn.Sequential(
            ASPP(base_channels * 8, base_channels * 8),
            ResidualDenseBlock(base_channels * 8),
            ChannelAttention(base_channels * 8),
            SpatialAttention(),
        )

        self.up1 = UpsampleBlock(base_channels * 8, base_channels * 4)
        self.dec1 = nn.Sequential(
            ResidualDenseBlock(base_channels * 8),
            EnhancedMultiScaleBlock(base_channels * 8),
        )

        self.up2 = UpsampleBlock(base_channels * 8, base_channels * 2)
        self.dec2 = nn.Sequential(
            ResidualDenseBlock(base_channels * 4),
            EnhancedMultiScaleBlock(base_channels * 4),
        )

        self.up3 = UpsampleBlock(base_channels * 4, base_channels)
        self.dec3 = nn.Sequential(
            ResidualDenseBlock(base_channels * 2),
            EnhancedMultiScaleBlock(base_channels * 2),
        )

        self.out_conv_final = nn.Conv2d(base_channels * 2, out_channels, 3, padding=1)
        self.out_conv_d1 = nn.Conv2d(base_channels * 8, out_channels, 1)
        self.out_conv_d2 = nn.Conv2d(base_channels * 4, out_channels, 1)
        self.out_conv_d3 = nn.Conv2d(base_channels * 2, out_channels, 1)

    def run_checkpoint(self, module, *inputs):
        if self.training and self.use_checkpointing:
            return checkpoint(module, *inputs, use_reentrant=False)
        else:
            return module(*inputs)

    def forward(self, x):
        x0 = F.leaky_relu(self.init_conv(x), negative_slope=0.2, inplace=True)

        e1 = self.run_checkpoint(self.enc1, x0)
        e2 = self.run_checkpoint(self.enc2, e1)
        e3 = self.run_checkpoint(self.enc3, e2)
        e4 = self.run_checkpoint(self.enc4, e3)

        b = self.run_checkpoint(self.bottleneck, e4)

        up1_out = self.up1(b)
        d1 = self.run_checkpoint(self.dec1, torch.cat([up1_out, e3], dim=1))

        up2_out = self.up2(d1)
        d2 = self.run_checkpoint(self.dec2, torch.cat([up2_out, e2], dim=1))

        up3_out = self.up3(d2)
        d3 = self.run_checkpoint(self.dec3, torch.cat([up3_out, e1], dim=1))

        final_out = torch.tanh(self.out_conv_final(d3))

        out_d1 = torch.tanh(self.out_conv_d1(d1))
        out_d2 = torch.tanh(self.out_conv_d2(d2))
        out_d3 = torch.tanh(self.out_conv_d3(d3))

        return final_out, out_d1, out_d2, out_d3


# Not implemented yet
class SuperResolutionNet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        num_features=64,
        num_res_blocks=16,
        upscale_factor=4,
    ):
        super().__init__()

        if upscale_factor not in [2, 4, 8] or not (
            upscale_factor & (upscale_factor - 1) == 0
        ):
            raise ValueError("Upscale factor must be a power of 2 (e.g., 2, 4, 8).")

        self.head = nn.Sequential(
            nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.body = nn.Sequential(
            *[
                ResidualConvBlock(num_features, num_features)
                for _ in range(num_res_blocks)
            ]
        )

        upscalers = []
        for _ in range(int(math.log2(upscale_factor))):
            upscalers.append(UpsampleBlock(num_features, num_features))

        self.tail = nn.Sequential(
            *upscalers,
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_features, out_channels, kernel_size=3, padding=1)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, a=0.2, mode="fan_in", nonlinearity="leaky_relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.head(x)
        residual = x
        x = self.body(x)
        x = x + residual
        output = self.tail(x)

        return torch.tanh(output)
