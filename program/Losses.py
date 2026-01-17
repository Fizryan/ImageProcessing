import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from program.LoggingManager import LoggingManager

logger = LoggingManager.setup_logging(__name__)


class LightPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = self._build_feature_extractor()

        self.feature_extractor.eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.criterion = nn.L1Loss()

    def _build_feature_extractor(self) -> nn.Sequential:
        layers = []
        in_channels = 3
        for out_channels in [24, 48, 96]:
            layers.extend(
                [
                    nn.Conv2d(
                        in_channels, out_channels, kernel_size=3, padding=1, bias=False
                    ),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2),
                ]
            )
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        self.feature_extractor.eval()

        pred_features = self.feature_extractor(pred)
        target_features = self.feature_extractor(target)

        return self.criterion(pred_features, target_features)


class AdvancedRestorationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = LightPerceptualLoss()

        self.weights_early = {"l1": 0.6, "perc": 0.2, "fft": 0.1, "grad": 0.1}
        self.weights_late = {"l1": 0.2, "perc": 0.4, "fft": 0.2, "grad": 0.2}

    def fft_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_fft = torch.fft.rfft2(pred, dim=(-2, -1))
        target_fft = torch.fft.rfft2(target, dim=(-2, -1))

        return F.l1_loss(pred_fft.real, target_fft.real) + F.l1_loss(
            pred_fft.imag, target_fft.imag
        )

    def gradient_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_grad_x = pred[..., 1:] - pred[..., :-1]
        target_grad_x = target[..., 1:] - target[..., :-1]

        pred_grad_y = pred[..., 1:, :] - pred[..., :-1, :]
        target_grad_y = target[..., 1:, :] - target[..., :-1, :]

        return F.l1_loss(pred_grad_x, target_grad_x) + F.l1_loss(
            pred_grad_y, target_grad_y
        )

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, current_epoch: int
    ) -> Tuple[torch.Tensor, Dict[str, float]]:

        l1_val = self.l1_loss(pred, target)
        perc_val = self.perceptual_loss(pred, target)
        fft_val = self.fft_loss(pred, target)
        grad_val = self.gradient_loss(pred, target)

        weights = self.weights_early if current_epoch < 25 else self.weights_late

        total_loss = (
            weights["l1"] * l1_val
            + weights["perc"] * perc_val
            + weights["fft"] * fft_val
            + weights["grad"] * grad_val
        )

        loss_dict = {
            "l1": l1_val.item(),
            "perc": perc_val.item(),
            "fft": fft_val.item(),
            "grad": grad_val.item(),
            "total": total_loss.item(),
        }
        return total_loss, loss_dict


class SharpnessOptimizedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = LightPerceptualLoss()

        kernel_data = torch.tensor(
            [[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=torch.float32
        ).view(1, 1, 3, 3)

        self.register_buffer("laplacian_kernel", kernel_data.repeat(3, 1, 1, 1))

        self.weights_early = {"l1": 0.5, "perc": 0.2, "edge": 0.2, "freq": 0.1}
        self.weights_mid = {"l1": 0.3, "perc": 0.3, "edge": 0.25, "freq": 0.15}
        self.weights_late = {"l1": 0.2, "perc": 0.3, "edge": 0.3, "freq": 0.2}

        self.cached_freq_mask: Optional[torch.Tensor] = None
        self.cached_input_shape: Optional[Tuple[int, int]] = None

    def edge_aware_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_edges = F.conv2d(pred, self.laplacian_kernel, padding=1, groups=3)
        target_edges = F.conv2d(target, self.laplacian_kernel, padding=1, groups=3)
        return F.l1_loss(pred_edges, target_edges)

    def frequency_band_loss(
        self, pred: torch.Tensor, target: torch.Tensor, low_freq_ratio: float = 0.3
    ) -> torch.Tensor:
        pred_fft = torch.fft.rfft2(pred, dim=(-2, -1))
        target_fft = torch.fft.rfft2(target, dim=(-2, -1))

        h, w = pred.shape[-2], pred.shape[-1]
        fft_w = pred_fft.shape[-1]

        if self.cached_freq_mask is None or self.cached_input_shape != (h, w):
            y_freq = torch.fft.fftfreq(h, device=pred.device).abs().view(-1, 1)
            x_freq = torch.linspace(0, 0.5, fft_w, device=pred.device).view(1, -1)

            freq_dist = torch.sqrt(y_freq**2 + x_freq**2)
            self.cached_freq_mask = (freq_dist > low_freq_ratio).float()
            self.cached_input_shape = (h, w)

        high_freq_loss = F.l1_loss(
            pred_fft.abs() * self.cached_freq_mask,
            target_fft.abs() * self.cached_freq_mask,
        )
        return high_freq_loss

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, current_epoch: int
    ) -> Tuple[torch.Tensor, Dict[str, float]]:

        l1_loss = self.l1_loss(pred, target)
        perc_loss = self.perceptual_loss(pred, target)
        edge_loss = self.edge_aware_loss(pred, target)
        freq_loss = self.frequency_band_loss(pred, target)

        if current_epoch < 25:
            weights = self.weights_early
        elif current_epoch < 60:
            weights = self.weights_mid
        else:
            weights = self.weights_late

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
