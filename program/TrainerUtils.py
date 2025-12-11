# TrainerUtils.py
# Utility methods for Trainer class to reduce main file size

import torch
import torch.nn.functional as F
from typing import Tuple, Optional


class TrainerUtils:
    """Static utility methods for training operations."""

    @staticmethod
    def generate_blend_mask(
        patch_size: Tuple[int, int], device: torch.device
    ) -> torch.Tensor:
        """
        Generate Hann window blend mask for smooth patch stitching.

        Args:
            patch_size: (width, height) of the patch
            device: torch device

        Returns:
            Blend mask tensor of shape (1, 1, height, width)
        """
        patch_w, patch_h = patch_size
        hann_h = torch.hann_window(patch_h * 2, periodic=False, device=device)[:patch_h]
        hann_w = torch.hann_window(patch_w * 2, periodic=False, device=device)[:patch_w]
        blend_mask = hann_h.unsqueeze(1) * hann_w.unsqueeze(0)
        return blend_mask.view(1, 1, patch_h, patch_w)

    @staticmethod
    def compute_combined_loss(
        pred: torch.Tensor,
        target: torch.Tensor,
        l1_loss_fn,
        lpips_metric,
        config: dict,
        device: torch.device,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute combined loss with configurable weights.

        Args:
            pred: Predicted tensor
            target: Target tensor
            l1_loss_fn: L1 loss function
            lpips_metric: LPIPS metric (can be None)
            config: Configuration dict with loss weights
            device: torch device

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        loss_dict = {}
        total_loss = torch.tensor(0.0, device=device)

        l1_weight = config.get("l1_weight", 1.0)
        if l1_weight > 0:
            l1_loss = l1_loss_fn(pred, target)
            total_loss += l1_weight * l1_loss
            loss_dict["l1"] = l1_loss.item()

        lpips_weight = config.get("lpips_weight", 0.0)
        if lpips_metric and lpips_weight > 0:
            lpips_loss = lpips_metric(pred * 2 - 1, target * 2 - 1).mean()
            total_loss += lpips_weight * lpips_loss
            loss_dict["lpips"] = lpips_loss.item()

        fft_weight = config.get("fft_weight", 0.0)
        if fft_weight > 0:
            pred_fft = torch.fft.fft2(pred, dim=(-2, -1))
            target_fft = torch.fft.fft2(target, dim=(-2, -1))
            fft_loss = F.l1_loss(pred_fft.real, target_fft.real) + F.l1_loss(
                pred_fft.imag, target_fft.imag
            )
            total_loss += fft_weight * fft_loss
            loss_dict["fft"] = fft_loss.item()

        loss_dict["total_recon"] = total_loss.item()
        return total_loss, loss_dict

    @staticmethod
    def get_current_ohem_percent(epoch: int, config: dict) -> float:
        """
        Get OHEM percentage for current epoch based on schedule.

        Args:
            epoch: Current epoch number
            config: Configuration dict with 'ohem_schedule' as [(epoch_ratio, percent), ...]
                   where epoch_ratio is a float between 0.0 and 1.0 representing
                   the fraction of total epochs (e.g., 0.5 = 50% of num_epochs)

        Returns:
            OHEM percentage for current epoch
        """
        schedule = config.get("ohem_schedule", [])
        if not schedule:
            return config.get("ohem_percent", 1.0)

        num_epochs = config.get("num_epochs", 100)

        # Convert ratio-based schedule to actual epoch numbers
        actual_schedule = []
        for epoch_ratio, percent in schedule:
            actual_epoch = int(epoch_ratio * num_epochs)
            actual_schedule.append((actual_epoch, percent))

        current_percent = config.get("ohem_percent", 1.0)
        for schedule_epoch, percent in sorted(actual_schedule, key=lambda x: x[0]):
            if epoch >= schedule_epoch:
                current_percent = percent
            else:
                break
        return current_percent
