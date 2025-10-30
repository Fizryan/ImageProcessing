# Architecture.py
# This module defines an improved U-Net architecture for image restoration tasks,
# incorporating best practices like PixelShuffle upsampling and residual blocks.

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.InstanceNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x):
        return x + self.conv(x)


class ChannelSpatialAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        # Channel attention
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid(),
        )
        # Spatial attention
        self.spatial_att = nn.Sequential(nn.Conv2d(2, 1, 7, padding=3), nn.Sigmoid())

    def forward(self, x):
        # Channel attention
        ca = self.channel_att(x)
        x_att = x * ca

        # Spatial attention
        avg_out = torch.mean(x_att, dim=1, keepdim=True)
        max_out, _ = torch.max(x_att, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_out, max_out], dim=1)
        sa = self.spatial_att(spatial_input)

        return x_att * sa


class ResidualDenseBlock(nn.Module):
    """Residual Dense Block for enhanced feature extraction."""

    def __init__(self, channels, growth_rate=32):
        super().__init__()
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.conv1 = nn.Conv2d(channels, growth_rate, 3, padding=1)
        self.conv2 = nn.Conv2d(channels + growth_rate, growth_rate, 3, padding=1)
        self.conv3 = nn.Conv2d(channels + 2 * growth_rate, growth_rate, 3, padding=1)
        self.conv4 = nn.Conv2d(channels + 3 * growth_rate, channels, 3, padding=1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        return x4 * 0.2 + x  # Scaled residual


class FeatureRefinementModule(nn.Module):
    """Refines features from skip connections to preserve details."""

    def __init__(self, channels):
        super().__init__()
        self.refinement = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            ChannelSpatialAttention(channels),
        )

    def forward(self, x):
        return x + self.refinement(x)


class SharpnessEnhancementModule(nn.Module):
    """Enhances sharpness in the final features."""

    def __init__(self, channels):
        super().__init__()
        self.enhancement = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # High-pass filter inspired component
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        enhanced = self.enhancement(x)
        return x + enhanced  # Residual connection to preserve information


def get_model(config: dict) -> nn.Module:
    """Factory function to get the appropriate model based on config."""
    model_name = config.get("model_size", "efficient")
    if model_name == "detail_preserving":
        return DetailPreservationUNet
    # Add other models here if needed
    # Default to EfficientUNet
    return EfficientUNet


class EfficientUNet(nn.Module):

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

        # Encoder
        self.enc1 = self._encoder_block(in_channels, base_channels)
        self.enc2 = self._encoder_block(base_channels, base_channels * 2)
        self.enc3 = self._encoder_block(base_channels * 2, base_channels * 4)
        self.enc4 = self._encoder_block(base_channels * 4, base_channels * 8)

        self.bottleneck = nn.Sequential(
            ResidualBlock(base_channels * 8),
            ChannelSpatialAttention(base_channels * 8),
            ResidualBlock(base_channels * 8),
        )

        # Decoder
        self.dec1 = self._decoder_block(base_channels * 8, base_channels * 4)
        self.dec2 = self._decoder_block(base_channels * 8, base_channels * 2)
        self.dec3 = self._decoder_block(base_channels * 4, base_channels)
        self.dec4 = self._decoder_block(base_channels * 2, base_channels)

        self.final_conv = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, out_channels, 3, padding=1),
        )
        self.final_act = nn.Sigmoid()

    def _encoder_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1),
            ResidualBlock(out_ch),
            ResidualBlock(out_ch),
        )

    def _decoder_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch * 4, 3, padding=1),
            nn.PixelShuffle(2),
            ResidualBlock(out_ch),
            ResidualBlock(out_ch),
        )

    def run_checkpoint(self, module, *inputs):
        if self.training and self.use_checkpointing:
            return checkpoint(module, *inputs, use_reentrant=False)
        return module(*inputs)

    def forward(self, x, return_internals=False):
        input_image = x
        internals = {}

        # Encoder
        e1 = self.run_checkpoint(self.enc1, x)
        e2 = self.run_checkpoint(self.enc2, e1)
        e3 = self.run_checkpoint(self.enc3, e2)
        e4 = self.run_checkpoint(self.enc4, e3)
        if return_internals:
            internals["enc1"] = e1
            internals["enc2"] = e2
            internals["enc3"] = e3
            internals["enc4"] = e4

        # Bottleneck
        b = self.run_checkpoint(self.bottleneck, e4)
        if return_internals:
            internals["bottleneck"] = b

        # Decoder
        d1 = self.dec1(b)
        d1 = torch.cat([d1, e3], dim=1)
        if return_internals:
            internals["dec1"] = d1

        d2 = self.dec2(d1)
        d2 = torch.cat([d2, e2], dim=1)
        if return_internals:
            internals["dec2"] = d2

        d3 = self.dec3(d2)
        d3 = torch.cat([d3, e1], dim=1)
        if return_internals:
            internals["dec3"] = d3

        d4 = self.dec4(d3)

        out = self.final_conv(d4)

        if self.use_global_residual and input_image.shape == out.shape:
            out = out + input_image

        final_output = self.final_act(out)

        if return_internals:
            return final_output, internals
        return final_output


class PatchGANDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator network."""

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


class DetailPreservationUNet(EfficientUNet):
    """Enhanced U-Net with better detail preservation capabilities."""

    def __init__(self, *args, **kwargs):
        base_channels = kwargs.get("base_channels", 32)
        super().__init__(*args, **kwargs)

        self.bottleneck = nn.Sequential(
            *[ResidualDenseBlock(base_channels * 8) for _ in range(3)]
        )

        self.skip_refinement1 = FeatureRefinementModule(base_channels * 2)
        self.skip_refinement2 = FeatureRefinementModule(base_channels * 4)
        self.skip_refinement3 = FeatureRefinementModule(base_channels * 8)

        self.sharpness_enhancer = SharpnessEnhancementModule(base_channels)

    def forward(self, x, return_internals=False):
        e1 = self.run_checkpoint(self.enc1, x)
        e2 = self.run_checkpoint(self.enc2, e1)
        e3 = self.run_checkpoint(self.enc3, e2)
        e4 = self.run_checkpoint(self.enc4, e3)

        b = self.run_checkpoint(self.bottleneck, e4)

        d1 = self.dec1(b)
        e3_refined = self.run_checkpoint(self.skip_refinement3, e3)
        d1 = torch.cat([d1, e3_refined], dim=1)

        d2 = self.dec2(d1)
        e2_refined = self.run_checkpoint(self.skip_refinement2, e2)
        d2 = torch.cat([d2, e2_refined], dim=1)

        d3 = self.dec3(d2)
        e1_refined = self.run_checkpoint(self.skip_refinement1, e1)
        d3 = torch.cat([d3, e1_refined], dim=1)

        d4 = self.dec4(d3)

        d4_enhanced = self.sharpness_enhancer(d4)

        out = self.final_conv(d4_enhanced)

        if self.use_global_residual and x.shape == out.shape:
            out = out + x

        final_output = self.final_act(out)

        return final_output
