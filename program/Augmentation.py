# Augmentation.py
# Data augmentation techniques for training

import random
import torch
import torch.nn.functional as F
from typing import Dict, Any
from torchvision import transforms


class RobustDegradation:
    """
    Applies robust degradation (blur, noise, JPEG compression) to simulate
    real-world low-quality images for blind restoration training.
    """

    def __init__(self, config: Dict[str, Any]):
        self.p = config.get("robust_degradation_prob", 0.5)
        degradation_cfg = config.get("robust_degradation_config", {})
        self.blur_prob = degradation_cfg.get("blur_prob", 0.3)
        self.noise_prob = degradation_cfg.get("noise_prob", 0.3)
        self.jpeg_prob = degradation_cfg.get("jpeg_prob", 0.3)
        self.noise_std_range = degradation_cfg.get("noise_std_range", [0.01, 0.05])
        self.jpeg_scale_range = degradation_cfg.get("jpeg_scale_range", [0.5, 0.9])

    def __call__(self, img_tensor):
        if random.random() > self.p:
            return img_tensor

        if random.random() < self.blur_prob:
            kernel_size = random.choice([3, 5])
            sigma = random.uniform(0.1, 2.0)
            img_tensor = transforms.functional.gaussian_blur(
                img_tensor, kernel_size, sigma
            )

        if random.random() < self.noise_prob:
            noise_std = random.uniform(*self.noise_std_range)
            noise = torch.randn_like(img_tensor) * noise_std
            img_tensor = (img_tensor + noise).clamp(0, 1)

        if random.random() < self.jpeg_prob:
            _, h, w = img_tensor.shape
            scale_factor = random.uniform(*self.jpeg_scale_range)
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
    """
    Creates pixelated mosaic effect with optional grid shifting to prevent
    overfitting to fixed grid positions.
    """
    _, h, w = rgb_tensor.shape

    if use_grid_shift:
        shift_x = random.randint(0, block_size - 1)
        shift_y = random.randint(0, block_size - 1)

        padded = F.pad(rgb_tensor, (shift_x, 0, shift_y, 0), mode="reflect")
        ph, pw = padded.shape[1], padded.shape[2]

        small_tensor = F.interpolate(
            padded.unsqueeze(0),
            size=(ph // block_size, pw // block_size),
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
            size=(h // block_size, w // block_size),
            mode="area",
        )
        pixelated_tensor = F.interpolate(
            small_tensor, size=(h, w), mode="nearest"
        ).squeeze(0)

    return pixelated_tensor
