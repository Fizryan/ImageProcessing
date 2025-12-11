# Losses.py
# Loss functions for image restoration training

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import lpips
except ImportError:
    lpips = None


class LightPerceptualLoss(nn.Module):
    """Lightweight perceptual loss using a simple CNN feature extractor."""

    def __init__(self, device):
        super().__init__()
        self.feature_extractor = self._build_feature_extractor().to(device).eval()
        self.criterion = nn.L1Loss()

        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def _build_feature_extractor(self):
        layers = []
        in_channels = 3
        for out_channels in [32, 64, 128]:
            layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                ]
            )
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, pred, target):
        pred_features = self.feature_extractor(pred)
        target_features = self.feature_extractor(target)
        return self.criterion(pred_features, target_features)


class AdvancedRestorationLoss(nn.Module):
    """Advanced loss combining L1, perceptual, FFT, and gradient losses."""

    def __init__(self, device):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = LightPerceptualLoss(device)

    def fft_loss(self, pred, target):
        pred_fft = torch.fft.fft2(pred, dim=(-2, -1))
        target_fft = torch.fft.fft2(target, dim=(-2, -1))
        return F.l1_loss(pred_fft.real, target_fft.real) + F.l1_loss(
            pred_fft.imag, target_fft.imag
        )

    def gradient_loss(self, pred, target):
        pred_grad_x = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        pred_grad_y = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        target_grad_x = target[:, :, :, 1:] - target[:, :, :, :-1]
        target_grad_y = target[:, :, 1:, :] - target[:, :, :-1, :]

        return F.l1_loss(pred_grad_x, target_grad_x) + F.l1_loss(
            pred_grad_y, target_grad_y
        )

    def forward(self, pred, target, current_epoch):
        losses = {
            "l1": self.l1_loss(pred, target),
            "perc": self.perceptual_loss(pred, target),
            "fft": self.fft_loss(pred, target),
            "grad": self.gradient_loss(pred, target),
        }

        if current_epoch < 25:
            weights = {"l1": 0.6, "perc": 0.2, "fft": 0.1, "grad": 0.1}
        else:
            weights = {"l1": 0.2, "perc": 0.4, "fft": 0.2, "grad": 0.2}

        total_loss = sum(weights.get(k, 0) * v for k, v in losses.items())
        loss_dict = {k: v.item() for k, v in losses.items()}
        loss_dict["total"] = total_loss.item()
        return total_loss, loss_dict


class SharpnessOptimizedLoss(nn.Module):
    """Loss optimized for image sharpness with edge-aware and frequency components."""

    def __init__(self, device):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = LightPerceptualLoss(device)

        laplacian_kernel = torch.tensor(
            [[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        self.laplacian_kernel = laplacian_kernel.repeat(3, 1, 1, 1).to(device)

    def edge_aware_loss(self, pred, target):
        pred_edges = F.conv2d(pred, self.laplacian_kernel, padding=1, groups=3)
        target_edges = F.conv2d(target, self.laplacian_kernel, padding=1, groups=3)
        return F.l1_loss(pred_edges, target_edges)

    def frequency_band_loss(self, pred, target, low_freq_ratio=0.3):
        pred_fft = torch.fft.fft2(pred, dim=(-2, -1))
        target_fft = torch.fft.fft2(target, dim=(-2, -1))

        h, w = pred.shape[-2:]
        y_freq = torch.fft.fftfreq(h, device=pred.device).abs().view(-1, 1)
        x_freq = torch.fft.fftfreq(w, device=pred.device).abs().view(1, -1)
        freq_mask = torch.sqrt(y_freq**2 + x_freq**2)

        high_freq_mask = (freq_mask > low_freq_ratio).float()
        high_freq_loss = F.l1_loss(
            pred_fft * high_freq_mask, target_fft * high_freq_mask
        )
        return high_freq_loss

    def forward(self, pred, target, current_epoch):
        l1_loss = self.l1_loss(pred, target)
        perc_loss = self.perceptual_loss(pred, target)
        edge_loss = self.edge_aware_loss(pred, target)
        freq_loss = self.frequency_band_loss(pred, target)

        if current_epoch < 25:
            weights = {"l1": 0.5, "perc": 0.2, "edge": 0.2, "freq": 0.1}
        elif current_epoch < 60:
            weights = {"l1": 0.3, "perc": 0.3, "edge": 0.25, "freq": 0.15}
        else:
            weights = {"l1": 0.2, "perc": 0.3, "edge": 0.3, "freq": 0.2}

        total_loss = (
            weights["l1"] * l1_loss
            + weights["perc"] * perc_loss
            + weights["edge"] * edge_loss
            + weights["freq"] * freq_loss
        )

        losses = {
            "total": total_loss.item(),
            "l1": l1_loss.item(),
            "perc": perc_loss.item(),
            "edge": edge_loss.item(),
            "freq": freq_loss.item(),
        }
        return total_loss, losses
