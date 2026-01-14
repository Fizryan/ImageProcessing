# MosaicGenerator.py

import random
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from program.Augmentation import RobustDegradation, create_pixelated_mosaic
from program.LoggingSetup import setup_logger

logger = setup_logger(__name__)


class MosaicGenerator:
    """
    Generate mosaic/degraded images from clean images using masks.
    Supports demosaic and inpainting degradation types.
    """

    def __init__(
        self,
        clean_dir: str,
        mask_dir: str,
        output_dir: str,
        task_type: str = "demosaic",
        mosaic_block_size_range: tuple = (20, 60),
        mosaic_opacity_range: tuple = (1.0, 1.0),
        use_mosaic_grid_shift: bool = False,
        use_robust_degradation: bool = False,
        robust_degradation_config: dict = None,
    ):
        """
        Args:
            clean_dir: Directory containing clean images
            mask_dir: Directory containing mask images
            output_dir: Directory to save degraded images
            task_type: 'demosaic' or 'inpainting'
            mosaic_block_size_range: (min, max) block size for mosaic
            mosaic_opacity_range: (min, max) opacity for mosaic blend
            use_mosaic_grid_shift: Apply random grid shift to mosaic
            use_robust_degradation: Apply blur, noise, JPEG compression
            robust_degradation_config: Config dict for RobustDegradation
        """
        self.clean_dir = Path(clean_dir)
        self.mask_dir = Path(mask_dir)
        self.output_dir = Path(output_dir)
        self.task_type = task_type
        self.mosaic_block_size_range = mosaic_block_size_range
        self.mosaic_opacity_range = mosaic_opacity_range
        self.use_mosaic_grid_shift = use_mosaic_grid_shift

        self.robust_degradation = None
        if use_robust_degradation:
            default_config = {
                "robust_degradation_prob": 1.0,
                "robust_degradation_config": robust_degradation_config
                or {
                    "blur_prob": 0.3,
                    "noise_prob": 0.3,
                    "jpeg_prob": 0.3,
                    "noise_std_range": [0.01, 0.03],
                    "jpeg_scale_range": [0.7, 0.95],
                },
            }
            self.robust_degradation = RobustDegradation(default_config)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.clean_paths = sorted(
            [
                p
                for p in self.clean_dir.iterdir()
                if p.suffix.lower() in [".png", ".jpg", ".jpeg"]
            ]
        )

        logger.info(f"MosaicGenerator initialized:")
        logger.info(f"  - Clean images: {len(self.clean_paths)}")
        logger.info(f"  - Task type: {task_type}")
        logger.info(f"  - Block size range: {mosaic_block_size_range}")
        logger.info(f"  - Opacity range: {mosaic_opacity_range}")
        logger.info(f"  - Grid shift: {use_mosaic_grid_shift}")
        logger.info(f"  - Robust degradation: {use_robust_degradation}")

    def _apply_degradation_with_mask(
        self, clean_tensor: torch.Tensor, mask_tensor: torch.Tensor
    ) -> torch.Tensor:
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

    def process_single_image(self, clean_path: Path) -> bool:
        """
        Process a single image and save the degraded version.

        Returns:
            True if successful, False otherwise
        """
        try:
            mask_path = self.mask_dir / clean_path.name
            output_path = self.output_dir / clean_path.name

            clean_img = Image.open(clean_path).convert("RGB")
            clean_tensor = transforms.ToTensor()(clean_img)

            if mask_path.exists():
                mask_img = Image.open(mask_path).convert("L")
                mask_tensor = transforms.ToTensor()(mask_img)
            else:
                mask_tensor = torch.ones(
                    1, clean_tensor.shape[1], clean_tensor.shape[2]
                )
                logger.debug(f"No mask found for {clean_path.name}, using full mask")

            degraded_tensor = self._apply_degradation_with_mask(
                clean_tensor, mask_tensor
            )

            degraded_img = transforms.ToPILImage()(degraded_tensor.clamp(0, 1))

            if clean_path.suffix.lower() == ".png":
                degraded_img.save(output_path, "PNG")
            else:
                degraded_img.save(output_path, "JPEG", quality=95)

            return True

        except Exception as e:
            logger.error(f"Error processing {clean_path.name}: {e}")
            return False

    def process_all(self) -> tuple:
        """
        Process all images in the clean directory.

        Returns:
            (success_count, fail_count)
        """
        logger.info(f"Starting to process {len(self.clean_paths)} images...")

        success_count = 0
        fail_count = 0

        for clean_path in tqdm(self.clean_paths, desc="Generating degraded images"):
            if self.process_single_image(clean_path):
                success_count += 1
            else:
                fail_count += 1

        logger.info(f"Processing complete:")
        logger.info(f"  - Success: {success_count}")
        logger.info(f"  - Failed: {fail_count}")
        logger.info(f"  - Output: {self.output_dir}")

        return success_count, fail_count
