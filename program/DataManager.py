import random
import torch
from pathlib import Path
from typing import Tuple, Optional, Callable, List, Union
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF

from program.LoggingManager import LoggingManager
from program.Augmentation import create_pixelated_mosaic

logger = LoggingManager.setup_logging(__name__)

ImageFile.LOAD_TRUNCATED_IMAGES = True


class RandomScale:
    def __init__(
        self, scale_range: Tuple[float, float] = (0.5, 1.0), target_crop_size: int = 256
    ):
        self.scale_range = scale_range
        self.target_crop_size = target_crop_size

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        scale = random.uniform(*self.scale_range)

        new_w = int(w * scale)
        new_h = int(h * scale)

        if new_w < self.target_crop_size or new_h < self.target_crop_size:
            return img

        return img.resize((new_w, new_h), Image.Resampling.BILINEAR)


class ExternalDataset(Dataset):
    def __init__(
        self,
        external_dir: Path,
        clean_dir: Path,
        image_size: Tuple[int, int],
        transform: Optional[Callable] = None,
        keep_original_size: bool = False,
    ):
        self.external_paths = sorted(
            [
                p
                for p in Path(external_dir).iterdir()
                if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]
            ]
        )
        self.clean_dir = Path(clean_dir)
        self.image_size = image_size
        self.transform = transform
        self.keep_original_size = keep_original_size

        logger.info(
            f"ExternalDataset initialized: {len(self.external_paths)} pairs found."
        )

    def __len__(self) -> int:
        return len(self.external_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        external_path = self.external_paths[idx]
        clean_path = self.clean_dir / external_path.name

        try:
            ext_img = Image.open(external_path).convert("RGB")
            clean_img = Image.open(clean_path).convert("RGB")

            if not self.keep_original_size:
                if self.transform:
                    seed = torch.randint(0, 2**32, (1,)).item()

                    torch.manual_seed(seed)
                    random.seed(seed)
                    ext_tensor = self.transform(ext_img)

                    torch.manual_seed(seed)
                    random.seed(seed)
                    clean_tensor = self.transform(clean_img)
                else:
                    ext_tensor = TF.to_tensor(ext_img)
                    clean_tensor = TF.to_tensor(clean_img)
                    ext_tensor = TF.resize(ext_tensor, self.image_size, antialias=True)
                    clean_tensor = TF.resize(
                        clean_tensor, self.image_size, antialias=True
                    )
            else:
                ext_tensor = TF.to_tensor(ext_img)
                clean_tensor = TF.to_tensor(clean_img)

            return ext_tensor, clean_tensor

        except (OSError, FileNotFoundError) as e:
            logger.warning(
                f"Skipping corrupt/missing file pair: {external_path.name} ({e})"
            )
            return self.__getitem__((idx + 1) % len(self))


class RestorationDataset(Dataset):
    def __init__(
        self,
        clean_dir: Path,
        mask_dir: Path,
        image_size: Tuple[int, int],
        transform: Optional[Callable] = None,
        mosaic_block_size_range: Tuple[int, int] = (16, 16),
        mosaic_opacity_range: Tuple[float, float] = (1.0, 1.0),
        task_type: str = "demosaic",
        keep_original_size: bool = False,
        use_mosaic_grid_shift: bool = False,
        robust_degradation: Optional[Callable] = None,
    ):
        self.clean_paths = sorted(
            [
                p
                for p in Path(clean_dir).iterdir()
                if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]
            ]
        )
        self.mask_dir = Path(mask_dir)
        self.image_size = image_size
        self.transform = transform

        self.mosaic_block_size_range = mosaic_block_size_range
        self.mosaic_opacity_range = mosaic_opacity_range
        self.task_type = task_type
        self.keep_original_size = keep_original_size
        self.use_mosaic_grid_shift = use_mosaic_grid_shift
        self.robust_degradation = robust_degradation

        logger.info(
            f"RestorationDataset initialized: {len(self.clean_paths)} images found. Task: {task_type}"
        )

    def __len__(self) -> int:
        return len(self.clean_paths)

    def _load_mask(self, mask_path: Path, target_shape: torch.Size) -> torch.Tensor:
        try:
            mask_img = Image.open(mask_path).convert("L")

            if self.transform and not self.keep_original_size:
                return self.transform(mask_img)

            tensor = TF.to_tensor(mask_img)
            if not self.keep_original_size:
                tensor = TF.resize(
                    tensor,
                    self.image_size,
                    interpolation=transforms.InterpolationMode.NEAREST,
                )
            return tensor

        except FileNotFoundError:
            return torch.ones(1, target_shape[1], target_shape[2])

    def _apply_degradation(
        self, clean_tensor: torch.Tensor, mask_tensor: torch.Tensor
    ) -> torch.Tensor:
        mask_binary = (mask_tensor > 0.5).float()

        if self.task_type == "demosaic":
            base = clean_tensor.clone()

            if self.robust_degradation:
                base = self.robust_degradation(base)

            block_size = random.randint(*self.mosaic_block_size_range)
            pixelated = create_pixelated_mosaic(
                base, block_size=block_size, use_grid_shift=self.use_mosaic_grid_shift
            )

            opacity = random.uniform(*self.mosaic_opacity_range)
            mosaic_effect = (opacity * pixelated) + ((1 - opacity) * base)

            return torch.where(mask_binary > 0.5, mosaic_effect, base)

        elif self.task_type == "inpainting":
            return torch.where(
                mask_binary > 0.5, torch.ones_like(clean_tensor), clean_tensor
            )

        return clean_tensor

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        clean_path = self.clean_paths[idx]
        mask_path = self.mask_dir / clean_path.name

        try:
            clean_img = Image.open(clean_path).convert("RGB")

            if not self.keep_original_size:
                if self.transform:
                    seed = torch.random.initial_seed()
                    clean_tensor = self.transform(clean_img)
                else:
                    clean_tensor = TF.to_tensor(clean_img)
                    clean_tensor = TF.resize(
                        clean_tensor, self.image_size, antialias=True
                    )
            else:
                clean_tensor = TF.to_tensor(clean_img)

            mask_tensor = self._load_mask(mask_path, clean_tensor.shape)

            degraded_tensor = self._apply_degradation(clean_tensor, mask_tensor)

            return degraded_tensor, clean_tensor

        except Exception as e:
            logger.warning(f"Error processing {clean_path.name}: {e}, skipping.")
            return self.__getitem__((idx + 1) % len(self))


def get_dataloader(
    dataset_type: str,
    config: dict,
    batch_size: int = 16,
    num_workers: int = 4,
    shuffle: bool = True,
) -> DataLoader:
    base_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    if dataset_type == "external":
        ds = ExternalDataset(
            external_dir=Path(config["external_dir"]),
            clean_dir=Path(config["clean_dir"]),
            image_size=config.get("image_size", (256, 256)),
            transform=base_transform,
        )

    elif dataset_type == "restoration":
        ds = RestorationDataset(
            clean_dir=Path(config["clean_dir"]),
            mask_dir=Path(config["mask_dir"]),
            image_size=config.get("image_size", (256, 256)),
            transform=base_transform,
            task_type=config.get("task_type", "demosaic"),
            mosaic_block_size_range=config.get("mosaic_block", (16, 16)),
            mosaic_opacity_range=config.get("opacity", (1.0, 1.0)),
        )
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
