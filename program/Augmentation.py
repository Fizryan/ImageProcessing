import random
import torch
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple
from torchvision import transforms

from program.LoggingManager import LoggingManager

logger = LoggingManager.setup_logging(__name__)


class RobustDegradation:
    def __init__(self, config: Dict[str, Any]):
        self.p = config.get("robust_degradation_prob", 0.5)

        deg_cfg = config.get("robust_degradation_config", {})
        self.blur_prob = deg_cfg.get("blur_prob", 0.3)
        self.noise_prob = deg_cfg.get("noise_prob", 0.3)
        self.lowres_prob = deg_cfg.get("jpeg_prob", 0.3)

        self.noise_std_range = deg_cfg.get("noise_std_range", [0.01, 0.05])
        self.downscale_range = deg_cfg.get("jpeg_scale_range", [0.5, 0.9])

    def __call__(self, img_tensor: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return img_tensor

        if random.random() < self.blur_prob:
            kernel_size = random.choice([3, 5])
            sigma = random.uniform(0.1, 2.0)
            img_tensor = transforms.functional.gaussian_blur(
                img_tensor, kernel_size, sigma=[sigma, sigma]
            )

        if random.random() < self.noise_prob:
            noise_std = random.uniform(*self.noise_std_range)
            noise = torch.randn_like(img_tensor) * noise_std
            img_tensor = (img_tensor + noise).clamp(0.0, 1.0)

        if random.random() < self.lowres_prob:
            _, h, w = img_tensor.shape
            scale_factor = random.uniform(*self.downscale_range)

            small = F.interpolate(
                img_tensor.unsqueeze(0),
                scale_factor=scale_factor,
                mode="bilinear",
                align_corners=False,
            )

            img_tensor = F.interpolate(
                small, size=(h, w), mode="bilinear", align_corners=False
            ).squeeze(0)

        return img_tensor


def create_pixelated_mosaic(
    rgb_tensor: torch.Tensor, block_size: int = 16, use_grid_shift: bool = False
) -> torch.Tensor:
    _, h, w = rgb_tensor.shape

    effective_block_size = min(block_size, h, w)
    if effective_block_size < 2:
        return rgb_tensor

    if use_grid_shift:
        shift_x = random.randint(0, effective_block_size - 1)
        shift_y = random.randint(0, effective_block_size - 1)

        padded = F.pad(rgb_tensor, (shift_x, 0, shift_y, 0), mode="reflect")
        ph, pw = padded.shape[1], padded.shape[2]

        small_tensor = F.interpolate(
            padded.unsqueeze(0),
            size=(ph // effective_block_size, pw // effective_block_size),
            mode="area",
        )

        pixelated_full = F.interpolate(
            small_tensor, size=(ph, pw), mode="nearest"
        ).squeeze(0)

        pixelated_tensor = pixelated_full[
            :, shift_y : shift_y + h, shift_x : shift_x + w
        ]
    else:
        small_tensor = F.interpolate(
            rgb_tensor.unsqueeze(0),
            size=(h // effective_block_size, w // effective_block_size),
            mode="area",
        )
        pixelated_tensor = F.interpolate(
            small_tensor, size=(h, w), mode="nearest"
        ).squeeze(0)

    return pixelated_tensor
