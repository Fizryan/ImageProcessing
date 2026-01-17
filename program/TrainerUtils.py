import torch
import torch.nn.functional as F
from typing import Tuple, Dict, List, Optional, Any

from program.LoggingManager import LoggingManager

logger = LoggingManager.setup_logging(__name__)


class TrainerUtils:
    @staticmethod
    def generate_blend_mask(
        patch_size: Tuple[int, int], device: torch.device
    ) -> torch.Tensor:
        patch_w, patch_h = patch_size

        hann_h = torch.hann_window(patch_h, periodic=False, device=device)
        hann_w = torch.hann_window(patch_w, periodic=False, device=device)

        blend_2d = torch.outer(hann_h, hann_w)

        return blend_2d.view(1, 1, patch_h, patch_w)

    @staticmethod
    def compute_combined_loss(
        pred: torch.Tensor,
        target: torch.Tensor,
        l1_loss_fn: Any,
        lpips_metric: Optional[Any],
        config: Dict[str, Any],
        device: torch.device,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        loss_dict = {}
        total_loss = torch.tensor(0.0, device=device)

        l1_weight = config.get("l1_weight", 1.0)
        if l1_weight > 0:
            l1_loss = l1_loss_fn(pred, target)
            total_loss += l1_weight * l1_loss
            loss_dict["l1"] = l1_loss.item()

        lpips_weight = config.get("lpips_weight", 0.0)
        if lpips_metric is not None and lpips_weight > 0:
            lpips_val = lpips_metric(pred * 2 - 1, target * 2 - 1).mean()
            total_loss += lpips_weight * lpips_val
            loss_dict["lpips"] = lpips_val.item()

        fft_weight = config.get("fft_weight", 0.0)
        if fft_weight > 0:
            pred_fft = torch.fft.rfft2(pred, dim=(-2, -1))
            target_fft = torch.fft.rfft2(target, dim=(-2, -1))

            fft_loss = F.l1_loss(pred_fft.real, target_fft.real) + F.l1_loss(
                pred_fft.imag, target_fft.imag
            )

            total_loss += fft_weight * fft_loss
            loss_dict["fft"] = fft_loss.item()

        loss_dict["total_recon"] = total_loss.item()
        return total_loss, loss_dict

    @staticmethod
    def get_current_ohem_percent(epoch: int, config: Dict[str, Any]) -> float:
        schedule = config.get("ohem_schedule", [])
        default_percent = config.get("ohem_percent", 1.0)

        if not schedule:
            return default_percent

        num_epochs = config.get("num_epochs", 100)

        current_percent = default_percent

        sorted_schedule = sorted(schedule, key=lambda x: x[0])

        for ratio, percent in sorted_schedule:
            target_epoch = int(ratio * num_epochs)
            if epoch >= target_epoch:
                current_percent = percent
            else:
                break

        return current_percent

    @staticmethod
    def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor) -> float:
        mse = F.mse_loss(img1, img2)
        if mse == 0:
            return 100.0
        return 20 * torch.log10(1.0 / torch.sqrt(mse)).item()
