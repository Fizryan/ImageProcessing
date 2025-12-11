# RestorationDataset.py
# Dataset class for image restoration training

import random
import torch
from pathlib import Path
from typing import Tuple
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from program.Augmentation import create_pixelated_mosaic
from program.LoggingSetup import setup_logger

logger = setup_logger(__name__)


class RandomScale:
    """
    Randomly scale images BEFORE cropping to simulate various video resolutions.
    from a single 1080p source dataset - no need to create multiple resolution copies.

    Example: Training with 1080p images:
        - Epoch 1: Scale 1.0 (1920x1080)
        - Epoch 2: Scale 0.5 (960x540)
        - Epoch 3: Scale 0.66 (1280x720)
    """

    def __init__(self, scale_range=(0.5, 1.0), target_crop_size=256):
        """
        Args:
            scale_range: (min_scale, max_scale) for random scaling
            target_crop_size: Minimum size after scaling (safety check)
        """
        self.scale_range = scale_range
        self.target_crop_size = target_crop_size

    def __call__(self, img):
        """
        Args:
            img: PIL Image

        Returns:
            Randomly scaled PIL Image
        """
        w, h = img.size

        scale = random.uniform(*self.scale_range)

        new_w = int(w * scale)
        new_h = int(h * scale)

        if new_w < self.target_crop_size or new_h < self.target_crop_size:
            return img

        return img.resize((new_w, new_h), Image.Resampling.BILINEAR)


class RestorationDataset(Dataset):
    """
    Dataset for image restoration tasks (demosaic, inpainting).
    Applies degradation effects to clean images based on masks.
    """

    def __init__(
        self,
        clean_dir: Path,
        mask_dir: Path,
        image_size: Tuple[int, int],
        transform=None,
        mosaic_block_size_range: Tuple[int, int] = (16, 16),
        mosaic_opacity_range: Tuple[float, float] = (1.0, 1.0),
        use_masks=True,
        task_type="demosaic",
        keep_original_size=False,
        use_mosaic_grid_shift: bool = False,
        robust_degradation=None,
    ):
        self.clean_paths = sorted(
            [
                p
                for p in clean_dir.iterdir()
                if p.suffix.lower() in [".png", ".jpg", ".jpeg"]
            ]
        )
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.transform = transform
        self.mosaic_block_size_range = mosaic_block_size_range
        self.mosaic_opacity_range = mosaic_opacity_range
        self.use_masks = use_masks
        self.task_type = task_type
        self.keep_original_size = keep_original_size
        self.use_mosaic_grid_shift = use_mosaic_grid_shift
        self.robust_degradation = robust_degradation

    def __len__(self):
        return len(self.clean_paths)

    def __getitem__(self, idx):
        clean_path = self.clean_paths[idx]
        mask_path = self.mask_dir / clean_path.name

        try:
            clean_img = Image.open(clean_path).convert("RGB")

            if not self.keep_original_size:
                if self.transform:
                    clean_tensor = self.transform(clean_img)
                else:
                    clean_tensor = transforms.ToTensor()(clean_img)
                    clean_tensor = transforms.functional.resize(
                        clean_tensor, self.image_size
                    )

                try:
                    mask_img = Image.open(mask_path).convert("L")
                    if self.transform:
                        torch.manual_seed(idx)
                        mask_tensor = self.transform(mask_img)
                        if mask_tensor.shape[0] == 3:
                            mask_tensor = mask_tensor[0:1]
                    else:
                        mask_tensor = transforms.ToTensor()(mask_img)
                        mask_tensor = transforms.functional.resize(
                            mask_tensor,
                            self.image_size,
                            interpolation=transforms.InterpolationMode.NEAREST,
                        )
                except FileNotFoundError:
                    mask_tensor = torch.ones(1, *clean_tensor.shape[1:])

                degraded_tensor = self._apply_degradation_with_mask(
                    clean_tensor, mask_tensor
                )
            else:
                clean_tensor = transforms.ToTensor()(clean_img)

                try:
                    mask_img = Image.open(mask_path).convert("L")
                    mask_tensor = transforms.ToTensor()(mask_img)
                except FileNotFoundError:
                    mask_tensor = torch.ones(1, *clean_tensor.shape[1:])

                degraded_tensor = self._apply_degradation_with_mask(
                    clean_tensor, mask_tensor
                )

            return degraded_tensor, clean_tensor

        except Exception as e:
            logger.warning(f"Error processing {clean_path.name}: {e}, skipping.")
            new_idx = (idx + 1) % len(self)
            return self.__getitem__(new_idx)

    def _apply_degradation_with_mask(self, clean_tensor, mask_tensor):
        """Apply task-specific degradation effects."""

        if self.task_type == "demosaic":
            degraded_base = clean_tensor
            if self.robust_degradation is not None:
                degraded_base = self.robust_degradation(clean_tensor)

            block_size = random.randint(*self.mosaic_block_size_range)
            pixelated_tensor = create_pixelated_mosaic(
                degraded_base,
                block_size=block_size,
                use_grid_shift=self.use_mosaic_grid_shift,
            )
            opacity = random.uniform(*self.mosaic_opacity_range)

            mosaic_blend = (opacity * pixelated_tensor) + (
                (1 - opacity) * degraded_base
            )

            mask_binary = (mask_tensor > 0.5).float()
            degraded_tensor = torch.where(
                mask_binary > 0.5, mosaic_blend, degraded_base
            )
            return degraded_tensor

        elif self.task_type == "inpainting":
            mask_binary = (mask_tensor > 0.5).float()
            return torch.where(
                mask_binary > 0.5, torch.ones_like(clean_tensor), clean_tensor
            )

        return clean_tensor
