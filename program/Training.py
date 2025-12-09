# Training.py

import logging
import time
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import json
import torch
import warnings
import torch.nn as nn

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
import random
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import DataLoader, Dataset
from torchmetrics.image import (
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)
from torchvision import transforms, models
from torchvision.utils import make_grid, save_image
from tqdm.auto import tqdm

from program.Architecture import SOTARestorationUNet, get_model, PatchGANDiscriminator
from program.Utils import check_gpu_temp, load_model_weights

try:
    import lpips
except ImportError:
    lpips = None


class LightPerceptualLoss(nn.Module):
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


class RobustDegradation:
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


def compute_image_gradient(x):
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    dx = F.pad(dx, (0, 1, 0, 0))
    dy = F.pad(dy, (0, 0, 0, 1))
    return torch.sqrt(dx**2 + dy**2 + 1e-8)


def create_pixelated_mosaic(
    rgb_tensor: torch.Tensor, block_size: int = 16, use_grid_shift: bool = False
) -> torch.Tensor:
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


class RestorationDataset(Dataset):
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
            logging.warning(f"Error processing {clean_path.name}: {e}, skipping.")
            new_idx = (idx + 1) % len(self)
            return self.__getitem__(new_idx)

    def _apply_degradation_with_mask(self, clean_tensor, mask_tensor):
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


class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (
                    1.0 - self.decay
                ) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        self.backup = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}


class Trainer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.setup_logging()

        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        self.setup_directories()

        self._log_augmentation_status()

        self.setup_data_loaders()

        self.initialize_model_only()
        if hasattr(torch, "compile") and self.config.get("compile_mode"):
            self.generator = torch.compile(
                self.generator, mode=self.config["compile_mode"]
            )

        self.initialize_optimizers_and_schedulers()
        self.initialize_remaining_components()

        self.load_checkpoint()

    def _log_augmentation_status(self):
        if self.config.get("use_robust_degradation", False):
            self.logger.info("✓ Robust Degradation: ENABLED (Blind Restoration)")
            self.logger.info(
                f"  - Activation Probability: {self.config.get('robust_degradation_prob', 0.5)}"
            )
            cfg = self.config.get("robust_degradation_config", {})
            self.logger.info(
                f"  - Blur (unfocused camera): {cfg.get('blur_prob', 0.3)}"
            )
            self.logger.info(f"  - Noise (high ISO): {cfg.get('noise_prob', 0.3)}")
            self.logger.info(f"  - JPEG (compression): {cfg.get('jpeg_prob', 0.3)}")
            self.logger.info("  → Simulates real-world low-quality images")
        else:
            self.logger.info("✗ Robust Degradation: DISABLED")

        if self.config.get("use_geometric_augmentation", False):
            self.logger.info("✓ Geometric Augmentation: ENABLED")
            if self.config.get("use_vertical_flip", False):
                self.logger.info("  - Vertical Flip: ENABLED")
            self.logger.info("  → Provides geometric invariance")
        else:
            self.logger.info("✗ Geometric Augmentation: DISABLED")

        if self.config.get("use_mosaic_grid_shift", False):
            self.logger.info("✓ Mosaic Grid Shift: ENABLED")
            self.logger.info("  → Prevents overfitting to fixed grid positions")
            self.logger.info("  → Model learns position-agnostic mosaic removal")
        else:
            self.logger.info("✗ Mosaic Grid Shift: DISABLED")

        enabled_features = []
        if self.config.get("use_robust_degradation", False):
            enabled_features.append("Blind Restoration")
        if self.config.get("use_geometric_augmentation", False):
            enabled_features.append("Geometric Aug")
        if self.config.get("use_mosaic_grid_shift", False):
            enabled_features.append("Grid Shift")

        if enabled_features:
            self.logger.info(f"Active Features: {', '.join(enabled_features)}")
            self.logger.info("Training Mode: ADVANCED GENERALIZATION")
        else:
            self.logger.info("Training Mode: BASIC (No advanced augmentation)")

    def setup_logging(self):
        log_file = self.config.get("log_file", "Training/training.log")
        Path(log_file).parent.mkdir(exist_ok=True, parents=True)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)-7s | %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(),
            ],
            force=True,
        )
        self.logger = logging.getLogger(__name__)

        logging.getLogger("PIL").setLevel(logging.WARNING)
        logging.getLogger("matplotlib").setLevel(logging.WARNING)
        logging.getLogger("torch").setLevel(logging.WARNING)
        logging.getLogger("torchvision").setLevel(logging.WARNING)

        if "tensorboard_log_dir" in self.config and "trial_number" in self.config:
            log_dir = (
                Path(self.config["tensorboard_log_dir"])
                / f"trial_{self.config['trial_number']}"
            )
        else:
            log_dir = Path(self.config["checkpoint_dir"]) / "runs"
        self.writer = SummaryWriter(log_dir=str(log_dir))

    def setup_directories(self):
        self.checkpoint_dir = Path(self.config["checkpoint_dir"])
        self.preview_dir = Path(self.config["preview_dir"])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.preview_dir.mkdir(parents=True, exist_ok=True)

    def initialize_model_only(self):
        model_class = get_model(self.config)
        self.logger.info(f"Using generator architecture: {model_class.__name__}")
        self.generator = model_class(**self.config.get("model_params", {})).to(
            self.device
        )

        if self.config.get("use_channels_last", True):
            self.generator = self.generator.to(memory_format=torch.channels_last)

        if self.config.get("use_gan"):
            self.discriminator = PatchGANDiscriminator().to(self.device)
            if self.config.get("use_channels_last", True):
                self.discriminator = self.discriminator.to(
                    memory_format=torch.channels_last
                )

    def initialize_optimizers_and_schedulers(self):
        self.optimizer_G = optim.AdamW(
            self.generator.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config.get("weight_decay", 1e-4),
            betas=(0.9, 0.999),
        )

        scheduler_type = self.config.get("scheduler", "onecycle")
        if scheduler_type == "onecycle":
            total_steps = len(self.train_loader) * self.config["num_epochs"]
            self.logger.info(
                f"Initializing OneCycleLR: total_steps={total_steps}, "
                f"max_lr={self.config['learning_rate']}, "
                f"batches_per_epoch={len(self.train_loader)}"
            )
            self.scheduler = optim.lr_scheduler.OneCycleLR(
                self.optimizer_G,
                max_lr=self.config["learning_rate"],
                total_steps=total_steps,
                **self.config.get("onecycle_params", {}),
            )
        elif scheduler_type == "cosine_restarts":
            params = self.config.get("cosine_restarts_params", {})
            if "T_0" in params:
                params["T_0"] = params["T_0"] * len(self.train_loader)
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer_G, **params
            )
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

        if self.config.get("use_gan"):
            self.optimizer_D = optim.Adam(
                self.discriminator.parameters(),
                lr=self.config.get(
                    "discriminator_lr", self.config["learning_rate"] * 0.5
                ),
                betas=(0.5, 0.999),
            )

    def initialize_remaining_components(self):
        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = LightPerceptualLoss(self.device)

        if self.config.get("use_sharpness_loss"):
            self.criterion = SharpnessOptimizedLoss(self.device)
        elif self.config.get("use_advanced_loss"):
            self.criterion = AdvancedRestorationLoss(self.device)
        else:
            self.criterion = None

        if self.config.get("use_gan"):
            self.gan_loss = nn.BCEWithLogitsLoss()

        self.scaler_G = GradScaler(enabled=self.config.get("use_amp", True))
        if self.config.get("use_gan"):
            self.scaler_D = GradScaler(enabled=self.config.get("use_amp", True))
        self.psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(
            self.device
        )

        if lpips:
            self.lpips_metric = lpips.LPIPS(net="vgg").to(self.device)
        else:
            self.lpips_metric = None
            self.logger.warning("LPIPS not available. Skipping LPIPS computation.")

        if self.config.get("use_ema"):
            self.ema = ModelEMA(self.generator, decay=0.999)
        else:
            self.ema = None

        self.best_psnr = float("-inf")
        self.best_lpips = float("inf")
        self.start_epoch = 0
        self.global_step = 0

    @staticmethod
    def _generate_blend_mask_training(
        patch_size: Tuple[int, int], device: torch.device
    ):
        patch_w, patch_h = patch_size
        hann_h = torch.hann_window(patch_h * 2, periodic=False, device=device)[:patch_h]
        hann_w = torch.hann_window(patch_w * 2, periodic=False, device=device)[:patch_w]
        blend_mask = hann_h.unsqueeze(1) * hann_w.unsqueeze(0)
        return blend_mask.view(1, 1, patch_h, patch_w)

    def _inference_with_tiling(
        self,
        model: nn.Module,
        image_tensor: torch.Tensor,
        tile_size: Optional[Tuple[int, int]] = None,
        overlap: int = 32,
    ) -> torch.Tensor:
        b, c, h, w = image_tensor.shape

        if tile_size is None:
            patch_w = self.config.get("img_width", 448)
            patch_h = self.config.get("img_height", 256)
        else:
            patch_w, patch_h = tile_size

        if h <= patch_h and w <= patch_w:
            return model(image_tensor)

        if overlap and overlap > 0:
            stride_w = max(1, patch_w - overlap)
            stride_h = max(1, patch_h - overlap)
        else:
            stride_h = max(1, patch_h // 2)
            stride_w = max(1, patch_w // 2)

        pad_h = (stride_h - (h - patch_h) % stride_h) % stride_h
        pad_w = (stride_w - (w - patch_w) % stride_w) % stride_w
        padded_tensor = F.pad(image_tensor, (0, pad_w, 0, pad_h), "reflect")
        _, _, padded_h, padded_w = padded_tensor.shape

        result_accumulator = torch.zeros_like(padded_tensor)
        divisor = torch.zeros_like(padded_tensor)

        blend_mask = self._generate_blend_mask_training((patch_w, patch_h), self.device)

        for y in range(0, padded_h - patch_h + 1, stride_h):
            for x in range(0, padded_w - patch_w + 1, stride_w):
                patch = padded_tensor[:, :, y : y + patch_h, x : x + patch_w]
                patch_result = model(patch)

                result_accumulator[:, :, y : y + patch_h, x : x + patch_w] += (
                    patch_result * blend_mask
                )
                divisor[:, :, y : y + patch_h, x : x + patch_w] += blend_mask

        divisor = torch.where(divisor == 0, torch.ones_like(divisor), divisor)

        final_tensor = (result_accumulator / divisor).clamp(0, 1)
        return final_tensor[:, :, :h, :w]

    def setup_data_loaders(self):
        image_size = (self.config["img_height"], self.config["img_width"])

        train_transform_list = [
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(0.5),
        ]

        if self.config.get("use_geometric_augmentation", False) and self.config.get(
            "use_vertical_flip", False
        ):
            train_transform_list.append(transforms.RandomVerticalFlip(0.5))

        train_transform_list.extend(
            [
                transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
                transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
                transforms.ToTensor(),
            ]
        )

        train_transform = transforms.Compose(train_transform_list)

        dataset_robust_degradation = None
        if self.config.get("use_robust_degradation", False):
            dataset_robust_degradation = RobustDegradation(self.config)

        self.train_dataset = RestorationDataset(
            Path(self.config["train_clean_dir"]),
            Path(self.config["train_mask_dir"]),
            image_size,
            transform=train_transform,
            mosaic_block_size_range=self.config.get(
                "mosaic_block_size_range", [16, 16]
            ),
            mosaic_opacity_range=self.config.get("mosaic_opacity_range", [1.0, 1.0]),
            use_masks=self.config.get("use_masks", True),
            task_type=self.config.get("task_type", "demosaic"),
            keep_original_size=False,
            use_mosaic_grid_shift=self.config.get("use_mosaic_grid_shift", False),
            robust_degradation=dataset_robust_degradation,
        )

        self.val_dataset = RestorationDataset(
            Path(self.config["val_clean_dir"]),
            Path(self.config["val_mask_dir"]),
            image_size,
            transform=None,
            mosaic_block_size_range=self.config.get(
                "mosaic_block_size_range", [16, 16]
            ),
            mosaic_opacity_range=self.config.get("mosaic_opacity_range", [1.0, 1.0]),
            use_masks=self.config.get("use_masks", True),
            task_type=self.config.get("task_type", "demosaic"),
            keep_original_size=True,
            use_mosaic_grid_shift=False,
            robust_degradation=None,
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config["dataloader_params"]["batch_size"],
            shuffle=True,
            num_workers=self.config["dataloader_params"]["num_workers"],
            pin_memory=True,
            drop_last=True,
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )
        self.val_iter = iter(self.val_loader)

    def load_checkpoint(self):
        finetune_path = self.config.get("finetune_checkpoint_path")
        if finetune_path and Path(finetune_path).exists():
            self.logger.info(
                f"--- Starting fine-tuning session from: {finetune_path} ---"
            )
            try:
                checkpoint = torch.load(
                    finetune_path, map_location=self.device, weights_only=False
                )
                load_model_weights(self.generator, checkpoint["model_state_dict"])
                if (
                    self.config.get("use_gan")
                    and "discriminator_state_dict" in checkpoint
                ):
                    load_model_weights(
                        self.discriminator, checkpoint["discriminator_state_dict"]
                    )

                if "optimizer_G_state_dict" in checkpoint:
                    self.optimizer_G.load_state_dict(
                        checkpoint["optimizer_G_state_dict"]
                    )
                    self.logger.info("Loaded optimizer state for fine-tuning.")
                if "scaler_G_state_dict" in checkpoint:
                    self.scaler_G.load_state_dict(checkpoint["scaler_G_state_dict"])
                    self.logger.info("Loaded scaler state for fine-tuning.")

                if self.config.get("use_ema") and "ema_state_dict" in checkpoint:
                    self.ema.shadow = checkpoint["ema_state_dict"]
                    self.logger.info("Loaded EMA state for fine-tuning.")

                self.logger.info("Successfully loaded model weights for fine-tuning.")

                self.start_epoch = 0
                self.global_step = 0
                self.best_psnr = checkpoint.get("best_psnr", 0.0)
                self.best_lpips = checkpoint.get("best_lpips", float("inf"))
                self.logger.info(
                    f"Carrying over best metrics from checkpoint: LPIPS={self.best_lpips:.4f}, PSNR={self.best_psnr:.2f}"
                )
                return
            except Exception as e:
                self.logger.error(
                    f"Failed to load fine-tuning checkpoint: {e}. Starting from scratch.",
                    exc_info=True,
                )

        checkpoint_path = self.checkpoint_dir / "latest.pth"
        if not checkpoint_path.exists():
            self.start_epoch = 0
            self.best_psnr = 0.0
            self.logger.info("No checkpoint found, starting from scratch.")
            return

        try:
            checkpoint = torch.load(
                checkpoint_path, map_location=self.device, weights_only=False
            )

            load_model_weights(self.generator, checkpoint["model_state_dict"])
            self.optimizer_G.load_state_dict(checkpoint["optimizer_G_state_dict"])
            self.scaler_G.load_state_dict(checkpoint["scaler_G_state_dict"])

            if self.config.get("use_gan") and "discriminator_state_dict" in checkpoint:
                load_model_weights(
                    self.discriminator, checkpoint["discriminator_state_dict"]
                )
                if "optimizer_D_state_dict" in checkpoint:
                    self.optimizer_D.load_state_dict(
                        checkpoint["optimizer_D_state_dict"]
                    )
                    self.scaler_D.load_state_dict(checkpoint["scaler_D_state_dict"])

            self.start_epoch = checkpoint["epoch"] + 1
            self.global_step = checkpoint.get(
                "global_step", self.start_epoch * len(self.train_loader)
            )
            self.best_psnr = checkpoint.get("best_psnr", 0.0)
            self.best_lpips = checkpoint.get("best_lpips", float("inf"))
            if self.config.get("use_ema") and "ema_state_dict" in checkpoint:
                self.ema.shadow = checkpoint["ema_state_dict"]
                self.logger.info("Loaded EMA state from checkpoint.")

            if "scheduler_state_dict" in checkpoint:
                try:
                    self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

                    self.scheduler.last_epoch = self.global_step

                    self.logger.info(
                        f"Loaded scheduler state. Step disinkronkan ke: {self.global_step}."
                    )
                except Exception as e:
                    self.logger.warning(
                        f"Gagal load scheduler: {e}. Resetting scheduler ke awal (Warmup)."
                    )

            self.logger.info(
                f"Loaded checkpoint from epoch {self.start_epoch - 1}. Resuming training."
            )

        except Exception as e:
            self.logger.error(
                f"Failed to load checkpoint, starting from scratch: {e}", exc_info=True
            )
            self.start_epoch = 0
            self.best_psnr = 0.0
            self.best_lpips = float("inf")
            self.global_step = 0

    def save_checkpoint(self, epoch, is_best=False):
        model_state_dict = (
            self.generator._orig_mod.state_dict()
            if hasattr(self.generator, "_orig_mod")
            else self.generator.state_dict()
        )

        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": model_state_dict,
            "optimizer_G_state_dict": self.optimizer_G.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_G_state_dict": self.scaler_G.state_dict(),
            "best_psnr": self.best_psnr,
            "best_lpips": self.best_lpips,
            "config": self.config,
        }
        if self.config.get("use_gan"):
            checkpoint["discriminator_state_dict"] = self.discriminator.state_dict()
            checkpoint["optimizer_D_state_dict"] = self.optimizer_D.state_dict()
            checkpoint["scaler_D_state_dict"] = self.scaler_D.state_dict()

        if self.config.get("use_ema"):
            checkpoint["ema_state_dict"] = self.ema.shadow

        torch.save(checkpoint, self.checkpoint_dir / "latest.pth")

        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / "best_model.pth")

    def compute_combined_loss(self, pred, target, current_epoch):
        loss_dict = {}
        total_loss = torch.tensor(0.0, device=self.device)

        l1_weight = self.config.get("l1_weight", 1.0)
        if l1_weight > 0:
            l1_loss = self.l1_loss(pred, target)
            total_loss += l1_weight * l1_loss
            loss_dict["l1"] = l1_loss.item()

        lpips_weight = self.config.get("lpips_weight", 0.0)
        if self.lpips_metric and lpips_weight > 0:
            lpips_loss = self.lpips_metric(pred * 2 - 1, target * 2 - 1).mean()
            total_loss += lpips_weight * lpips_loss
            loss_dict["lpips"] = lpips_loss.item()

        fft_weight = self.config.get("fft_weight", 0.0)
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

    def train_epoch(self, epoch):
        check_gpu_temp(self.device, threshold=82, delay=15)

        self.generator.train()
        if self.config.get("use_gan"):
            self.discriminator.train()

        accumulation_steps = self.config.get("accumulation_steps", 1)
        ohem_percent = self._get_current_ohem_percent(epoch)
        if epoch == 0 or self._get_current_ohem_percent(epoch - 1) != ohem_percent:
            self.logger.info(
                f"🔥 OHEM percentage for this epoch set to: {ohem_percent*100}%"
            )

        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch+1}/{self.config['num_epochs']}",
            leave=False,
        )

        self.optimizer_G.zero_grad()
        if self.config.get("use_gan"):
            self.optimizer_D.zero_grad()

        total_loss_G_epoch = 0.0
        total_loss_D_epoch = 0.0

        for batch_idx, (degraded, clean) in enumerate(progress_bar):
            degraded = degraded.to(self.device, non_blocking=True)
            clean = clean.to(self.device, non_blocking=True)

            with autocast(
                device_type=self.device.type, enabled=self.config.get("use_amp", True)
            ):
                restored = self.generator(degraded)

            loss_D = torch.tensor(0.0, device=self.device)
            if self.config.get("use_gan"):
                with autocast(
                    device_type=self.device.type,
                    enabled=self.config.get("use_amp", True),
                ):
                    pred_real = self.discriminator(clean)
                    loss_D_real = self.gan_loss(pred_real, torch.ones_like(pred_real))
                    loss_D_real = loss_D_real / accumulation_steps

                self.scaler_D.scale(loss_D_real).backward()

                with autocast(
                    device_type=self.device.type,
                    enabled=self.config.get("use_amp", True),
                ):
                    pred_fake = self.discriminator(restored.detach())
                    loss_D_fake = self.gan_loss(pred_fake, torch.zeros_like(pred_fake))
                    loss_D_fake = loss_D_fake / accumulation_steps

                self.scaler_D.scale(loss_D_fake).backward()

                loss_D = (loss_D_real + loss_D_fake) * 0.5 * accumulation_steps
                total_loss_D_epoch += loss_D.item()

            with autocast(
                device_type=self.device.type, enabled=self.config.get("use_amp", True)
            ):
                batch_size = restored.shape[0]
                k = max(1, int(batch_size * ohem_percent))

                hard_restored = restored
                hard_clean = clean

                if k < batch_size:
                    with torch.no_grad():
                        l1_loss_per_sample = F.l1_loss(
                            restored, clean, reduction="none"
                        ).mean(dim=[1, 2, 3])

                    _, top_k_indices = torch.topk(l1_loss_per_sample, k=k)

                    hard_restored = restored[top_k_indices]
                    hard_clean = clean[top_k_indices]

                loss_dict = {}
                total_loss_G = torch.tensor(0.0, device=self.device)

                l1_weight = self.config.get("l1_weight", 0.0)
                if l1_weight > 0:
                    loss_G_l1 = self.l1_loss(hard_restored, hard_clean)
                    loss_dict["l1_ohem"] = loss_G_l1.item()
                    total_loss_G += l1_weight * loss_G_l1

                lpips_weight = self.config.get("lpips_weight", 0.0)
                if lpips_weight > 0 and self.lpips_metric:
                    loss_G_lpips = self.lpips_metric(
                        hard_restored * 2 - 1, hard_clean * 2 - 1
                    ).mean()
                    loss_dict["lpips_ohem"] = loss_G_lpips.item()
                    total_loss_G += lpips_weight * loss_G_lpips

                fft_weight = self.config.get("fft_weight", 0.0)
                if fft_weight > 0:
                    pred_fft = torch.fft.fft2(hard_restored, dim=(-2, -1))
                    target_fft = torch.fft.fft2(hard_clean, dim=(-2, -1))
                    loss_G_fft = F.l1_loss(pred_fft.real, target_fft.real) + F.l1_loss(
                        pred_fft.imag, target_fft.imag
                    )
                    loss_dict["fft_ohem"] = loss_G_fft.item()
                    total_loss_G += fft_weight * loss_G_fft

                if self.config.get("use_gan"):
                    pred_gen = self.discriminator(hard_restored)
                    loss_G_gan = self.gan_loss(pred_gen, torch.ones_like(pred_gen))
                    total_loss_G = (
                        total_loss_G + self.config.get("gan_weight", 0.1) * loss_G_gan
                    )
                    loss_dict["gan_ohem"] = loss_G_gan.item()

            total_loss_G_norm = total_loss_G / accumulation_steps

            self.scaler_G.scale(total_loss_G_norm).backward()
            total_loss_G_epoch += total_loss_G.item()

            if (batch_idx + 1) % accumulation_steps == 0:

                if self.config.get("use_gan"):
                    if self.config.get("grad_clip", 0) > 0:
                        self.scaler_D.unscale_(self.optimizer_D)
                        torch.nn.utils.clip_grad_norm_(
                            self.discriminator.parameters(), self.config["grad_clip"]
                        )
                    self.scaler_D.step(self.optimizer_D)
                    self.scaler_D.update()
                    self.optimizer_D.zero_grad()

                if self.config.get("grad_clip", 0) > 0:
                    self.scaler_G.unscale_(self.optimizer_G)
                    torch.nn.utils.clip_grad_norm_(
                        self.generator.parameters(), self.config["grad_clip"]
                    )
                self.scaler_G.step(self.optimizer_G)
                self.scaler_G.update()
                self.optimizer_G.zero_grad()

                if self.config.get("use_ema"):
                    self.ema.update()

            self.scheduler.step()
            self.global_step += 1

            progress_bar.set_postfix(
                {
                    "G_Loss_OHEM": f"{total_loss_G.item():.4f}",
                    "D_Loss": (
                        f"{loss_D.item():.4f}" if self.config.get("use_gan") else "N/A"
                    ),
                    "LR": f"{self.scheduler.get_last_lr()[0]:.2e}",
                }
            )

            if batch_idx % 100 == 0:
                self.writer.add_scalar(
                    "Loss/G_Total_OHEM", total_loss_G.item(), self.global_step
                )
                for name, value in loss_dict.items():
                    self.writer.add_scalar(
                        f"Loss/{name.capitalize()}", value, self.global_step
                    )
                if self.config.get("use_gan"):
                    self.writer.add_scalar(
                        "Loss/Discriminator", loss_D.item(), self.global_step
                    )
                self.writer.add_scalar(
                    "Learning_Rate", self.scheduler.get_last_lr()[0], self.global_step
                )

            if batch_idx % 50 == 0:
                current_lr = self.scheduler.get_last_lr()[0]
                self.logger.debug(
                    f"Batch {batch_idx}/{len(self.train_loader)} | "
                    f"LR: {current_lr:.2e} | "
                    f"Global Step: {self.global_step}"
                )

            if batch_idx % 20 == 0:
                check_gpu_temp(self.device)

        avg_loss_G = total_loss_G_epoch / len(self.train_loader)
        return avg_loss_G

    @torch.no_grad()
    def validate(self, epoch):
        self.generator.eval()

        if self.config.get("use_ema"):
            self.ema.apply_shadow()

        total_psnr = 0
        total_ssim = 0

        train_h = self.config.get("img_height", 256)
        train_w = self.config.get("img_width", 448)

        pbar = tqdm(self.val_loader, desc=f"Validation", leave=False)
        for idx, (degraded, clean) in enumerate(pbar):
            degraded = degraded.to(self.device)
            clean = clean.to(self.device)

            if idx % 10 == 0:
                check_gpu_temp(self.device)

            _, _, h, w = degraded.shape
            use_tiling = (h > train_h * 1.5) or (w > train_w * 1.5)

            with autocast(
                device_type=self.device.type, enabled=self.config.get("use_amp", True)
            ):
                if use_tiling:
                    restored = self._inference_with_tiling(
                        self.generator,
                        degraded,
                        tile_size=(train_w, train_h),
                        overlap=32,
                    )
                else:
                    restored = self.generator(degraded)

            restored = restored.clamp(0, 1)
            clean = clean.clamp(0, 1)

            total_psnr += self.psnr_metric(restored, clean)
            total_ssim += self.ssim_metric(restored, clean)

            del degraded, clean, restored

        torch.cuda.empty_cache()

        if self.config.get("use_ema"):
            self.ema.restore()

        avg_psnr = total_psnr / len(self.val_loader)
        avg_ssim = total_ssim / len(self.val_loader)

        self.writer.add_scalar("Validation/PSNR", avg_psnr, self.global_step)
        self.writer.add_scalar("Validation/SSIM", avg_ssim, self.global_step)

        lpips_score = self.calculate_lpips_on_subset()
        if lpips_score is not None:
            self.writer.add_scalar("Validation/LPIPS", lpips_score, self.global_step)

        torch.cuda.empty_cache()
        self.save_sample_images(epoch)

        return avg_psnr.item(), avg_ssim.item(), lpips_score

    def save_sample_images(self, epoch):
        self.generator.eval()

        try:
            degraded, clean = next(self.val_iter)
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            degraded, clean = next(self.val_iter)

        degraded, clean = (
            degraded.to(self.device),
            clean.to(self.device),
        )

        if self.config.get("use_ema"):
            self.ema.apply_shadow()

        train_h = self.config.get("img_height", 256)
        train_w = self.config.get("img_width", 448)
        _, _, h, w = degraded.shape
        use_tiling = (h > train_h * 1.5) or (w > train_w * 1.5)

        with torch.no_grad():
            with autocast(
                device_type=self.device.type, enabled=self.config.get("use_amp", True)
            ):
                if use_tiling:
                    restored = self._inference_with_tiling(
                        self.generator,
                        degraded,
                        tile_size=(train_w, train_h),
                        overlap=32,
                    )
                    internals = None
                else:
                    restored, internals = self.generator(
                        degraded, return_internals=True
                    )

        max_preview_size = 512
        if h > max_preview_size or w > max_preview_size:
            scale = max_preview_size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            degraded = F.interpolate(
                degraded, size=(new_h, new_w), mode="bilinear", align_corners=False
            )
            restored = F.interpolate(
                restored, size=(new_h, new_w), mode="bilinear", align_corners=False
            )
            clean = F.interpolate(
                clean, size=(new_h, new_w), mode="bilinear", align_corners=False
            )

        grid = make_grid(
            torch.cat([degraded[:1].cpu(), restored[:1].cpu(), clean[:1].cpu()], dim=0),
            nrow=3,
        )
        save_image(grid, self.preview_dir / f"epoch_{epoch+1:04d}.png")
        self.writer.add_image(
            "Validation/Samples (Input | Restored | Ground Truth)",
            grid,
            global_step=epoch,
        )

        if self.config.get("use_ema"):
            self.ema.restore()

        del degraded, restored, clean, grid
        torch.cuda.empty_cache()

    @torch.no_grad()
    def calculate_lpips_on_subset(self, num_batches=4):
        if not self.lpips_metric:
            return None

        self.generator.eval()
        if self.config.get("use_ema"):
            self.ema.apply_shadow()

        train_h = self.config.get("img_height", 256)
        train_w = self.config.get("img_width", 448)

        total_lpips = 0.0
        batches_processed = 0

        try:
            for i, (degraded, clean) in enumerate(self.val_loader):
                if i >= num_batches:
                    break

                degraded, clean = degraded.to(self.device), clean.to(self.device)

                _, _, h, w = degraded.shape
                use_tiling = (h > train_h * 1.5) or (w > train_w * 1.5)

                if use_tiling:
                    restored = self._inference_with_tiling(
                        self.generator,
                        degraded,
                        tile_size=(train_w, train_h),
                        overlap=32,
                    ).clamp(0, 1)
                else:
                    restored = self.generator(degraded).clamp(0, 1)

                if h > 512 or w > 512:
                    scale = 512.0 / max(h, w)
                    new_h, new_w = int(h * scale), int(w * scale)
                    restored_resize = F.interpolate(
                        restored,
                        size=(new_h, new_w),
                        mode="bilinear",
                        align_corners=False,
                    )
                    clean_resize = F.interpolate(
                        clean, size=(new_h, new_w), mode="bilinear", align_corners=False
                    )
                    total_lpips += self.lpips_metric(
                        restored_resize * 2 - 1, clean_resize * 2 - 1
                    ).sum()
                else:
                    total_lpips += self.lpips_metric(
                        restored * 2 - 1, clean * 2 - 1
                    ).sum()

                batches_processed += degraded.size(0)

                del degraded, clean, restored
                if "restored_resize" in locals():
                    del restored_resize, clean_resize
                torch.cuda.empty_cache()

        except RuntimeError as e:
            if "out of memory" in str(e):
                self.logger.warning(
                    f"OOM during LPIPS calculation, skipping remaining batches. Processed {batches_processed}/{num_batches}"
                )
                torch.cuda.empty_cache()
            else:
                raise e

        if self.config.get("use_ema"):
            self.ema.restore()

        return (
            (total_lpips / batches_processed).item() if batches_processed > 0 else None
        )

    def train(self) -> Optional[Tuple[float, float]]:
        self.logger.info("=" * 80)
        self.logger.info("Starting Training Session")
        self.logger.info("=" * 80)

        def fmt_bool(value):
            return "✓" if value else "✗"

        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Model: {self.generator.__class__.__name__}")
        self.logger.info(f"Task Type: {self.config.get('task_type', 'N/A')}")
        self.logger.info("")

        self.logger.info("Training Parameters:")
        self.logger.info(f"  Epochs: {self.config['num_epochs']}")
        self.logger.info(
            f"  Batch Size: {self.config['dataloader_params']['batch_size']}"
        )
        self.logger.info(
            f"  Accumulation Steps: {self.config.get('accumulation_steps', 1)}"
        )
        self.logger.info(f"  Learning Rate: {self.config['learning_rate']:.2e}")
        self.logger.info(f"  Weight Decay: {self.config.get('weight_decay', 1e-4):.2e}")
        self.logger.info(f"  Scheduler: {self.config.get('scheduler', 'onecycle')}")
        self.logger.info(f"  Gradient Clipping: {self.config.get('grad_clip', 0)}")
        self.logger.info("")

        self.logger.info("Model Architecture:")
        self.logger.info(f"  Base Channels: {self.config.get('base_channels', 'N/A')}")
        self.logger.info(f"  Model Size: {self.config.get('model_size', 'N/A')}")
        self.logger.info(
            f"  Enhanced Architecture: {fmt_bool(self.config.get('use_enhanced_architecture', False))}"
        )
        self.logger.info("")

        self.logger.info("Training Techniques:")
        self.logger.info(
            f"  AMP (Mixed Precision): {fmt_bool(self.config.get('use_amp', True))}"
        )
        self.logger.info(
            f"  Channels Last: {fmt_bool(self.config.get('use_channels_last', True))}"
        )
        self.logger.info(
            f"  EMA (Exponential Moving Average): {fmt_bool(self.config.get('use_ema', False))}"
        )
        self.logger.info(
            f"  GAN Training: {fmt_bool(self.config.get('use_gan', False))}"
        )
        self.logger.info(
            f"  Gradient Checkpointing: {fmt_bool(self.config.get('use_checkpointing', False))}"
        )
        self.logger.info("")

        self.logger.info("Loss Configuration:")
        self.logger.info(f"  L1 Loss Weight: {self.config.get('l1_weight', 0)}")
        self.logger.info(f"  LPIPS Loss Weight: {self.config.get('lpips_weight', 0)}")
        self.logger.info(f"  FFT Loss Weight: {self.config.get('fft_weight', 0)}")
        if self.config.get("use_gan"):
            self.logger.info(f"  GAN Loss Weight: {self.config.get('gan_weight', 0)}")
        self.logger.info(
            f"  Advanced Loss: {fmt_bool(self.config.get('use_advanced_loss', False))}"
        )
        self.logger.info(
            f"  Sharpness Loss: {fmt_bool(self.config.get('use_sharpness_loss', False))}"
        )
        self.logger.info("")

        self.logger.info("Data Augmentation:")
        self.logger.info(
            f"  Robust Degradation: {fmt_bool(self.config.get('use_robust_degradation', False))}"
        )
        self.logger.info(
            f"  Geometric Augmentation: {fmt_bool(self.config.get('use_geometric_augmentation', False))}"
        )
        self.logger.info(
            f"  Mosaic Grid Shift: {fmt_bool(self.config.get('use_mosaic_grid_shift', False))}"
        )
        if self.config.get("use_vertical_flip"):
            self.logger.info(
                f"  Vertical Flip: {fmt_bool(self.config.get('use_vertical_flip', False))}"
            )
        self.logger.info("")

        self.logger.info("Image Settings:")
        self.logger.info(
            f"  Image Size: {self.config['img_height']}x{self.config['img_width']}"
        )
        mosaic_range = self.config.get("mosaic_block_size_range", [16, 16])
        self.logger.info(f"  Mosaic Block Size: {mosaic_range[0]}-{mosaic_range[1]}")
        opacity_range = self.config.get("mosaic_opacity_range", [1.0, 1.0])
        self.logger.info(
            f"  Mosaic Opacity: {opacity_range[0]:.1f}-{opacity_range[1]:.1f}"
        )
        self.logger.info("")

        ohem_schedule = self.config.get("ohem_schedule", [])
        if ohem_schedule:
            self.logger.info("OHEM Schedule:")
            for epoch_num, percent in ohem_schedule:
                self.logger.info(f"  Epoch {epoch_num}: {percent*100:.0f}%")
            self.logger.info("")

        self.logger.info("Checkpoint & Early Stopping:")
        self.logger.info(
            f"  Checkpoint Interval: Every {self.config.get('checkpoint_interval_epochs', 1)} epoch(s)"
        )
        patience = self.config.get("early_stopping_patience", -1)
        if patience > 0:
            self.logger.info(f"  Early Stopping Patience: {patience} epochs")
        else:
            self.logger.info(f"  Early Stopping: {fmt_bool(False)}")

        self.logger.info("=" * 80)
        start_time = time.time()
        patience_counter = 0

        for epoch in range(self.start_epoch, self.config["num_epochs"]):
            self.train_epoch(epoch)
            psnr, ssim, lpips_score = self.validate(epoch)

            is_best_psnr = psnr > self.best_psnr
            is_best_lpips = lpips_score is not None and lpips_score < self.best_lpips

            if is_best_psnr:
                self.best_psnr = psnr
                patience_counter = 0

            if is_best_lpips:
                self.best_lpips = lpips_score
                patience_counter = 0
            elif not is_best_psnr:
                patience_counter += 1

            if (
                (epoch + 1) % self.config.get("checkpoint_interval_epochs", 1) == 0
                or is_best_psnr
                or is_best_lpips
            ):
                self.save_checkpoint(epoch, is_best_lpips)

            metrics_str = (
                f"Epoch {epoch+1:03d}/{self.config['num_epochs']} | "
                f"PSNR: {psnr:.2f} dB | SSIM: {ssim:.4f} | "
                f"LPIPS: {lpips_score:.4f} | "
                f"LR: {self.scheduler.get_last_lr()[0]:.2e}"
            )

            markers = []
            if is_best_psnr:
                markers.append(f"★ Best PSNR: {psnr:.2f} dB")
            if is_best_lpips:
                markers.append(f"★ Best LPIPS: {lpips_score:.4f}")

            if markers:
                self.logger.info(metrics_str)
                self.logger.info("  " + " | ".join(markers))
            else:
                self.logger.info(metrics_str)

            if (
                patience_counter >= self.config.get("early_stopping_patience", 25)
                and self.config.get("early_stopping_patience") != -1
            ):
                self.logger.info(
                    f"🛑 Early stopping after {patience_counter} epochs without improvement"
                )
                break

        training_time = (time.time() - start_time) / 3600
        self.logger.info("=" * 80)
        self.logger.info("Training Completed")
        self.logger.info("=" * 80)
        self.logger.info(f"Total Training Time: {training_time:.2f} hours")
        self.logger.info(f"Best PSNR: {self.best_psnr:.2f} dB")
        self.logger.info(f"Best LPIPS: {self.best_lpips:.4f}")
        self.logger.info("=" * 80)

        hparams = {}
        for key, value in self.config.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, (int, float, str, bool)):
                        hparams[f"{key}/{sub_key}"] = sub_value
            elif isinstance(value, (int, float, str, bool)):
                hparams[key] = value

        metrics = {
            "hparam/best_lpips": self.best_lpips,
            "hparam/training_time_hours": training_time,
        }
        self.writer.add_hparams(hparams, metrics, run_name=".")
        self.writer.close()

        return self.best_lpips, training_time

    def _get_current_ohem_percent(self, epoch: int) -> float:
        schedule = self.config.get("ohem_schedule", [])
        if not schedule:
            return self.config.get("ohem_percent", 1.0)

        current_percent = self.config.get("ohem_percent", 1.0)
        for schedule_epoch, percent in sorted(schedule, key=lambda x: x[0]):
            if epoch >= schedule_epoch:
                current_percent = percent
            else:
                break
        return current_percent
