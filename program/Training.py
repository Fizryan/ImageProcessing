# Training.py
# Skrip training yang efisien untuk model inpainting dengan fokus pada optimasi VRAM dan performa

import logging
import time
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import json

import torch
import warnings
import torch.nn as nn
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

from program.Architecture import EfficientUNet, get_model, PatchGANDiscriminator
from program.Utils import check_gpu_temp, load_model_weights

try:
    import lpips
except ImportError:
    lpips = None


class LightPerceptualLoss(nn.Module):
    """Lightweight perceptual loss using a smaller custom feature extractor."""

    def __init__(self, device):
        super().__init__()
        self.feature_extractor = self._build_feature_extractor().to(device).eval()
        self.criterion = nn.L1Loss()

        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def _build_feature_extractor(self):
        """Builds a lightweight CNN for feature extraction."""
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
    """Combines multiple loss functions for high-quality image restoration."""

    def __init__(self, device):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = LightPerceptualLoss(device)

    def fft_loss(self, pred, target):
        """Frequency domain loss to preserve high-frequency details."""
        pred_fft = torch.fft.fft2(pred, dim=(-2, -1))
        target_fft = torch.fft.fft2(target, dim=(-2, -1))
        return F.l1_loss(pred_fft.real, target_fft.real) + F.l1_loss(
            pred_fft.imag, target_fft.imag
        )

    def gradient_loss(self, pred, target):
        """Gradient matching loss for sharp edges."""
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

        # Dynamic weighting strategy
        if current_epoch < 25:  # Focus on structure
            weights = {"l1": 0.6, "perc": 0.2, "fft": 0.1, "grad": 0.1}
        else:  # Focus on perceptual quality and details
            weights = {"l1": 0.2, "perc": 0.4, "fft": 0.2, "grad": 0.2}

        total_loss = sum(weights.get(k, 0) * v for k, v in losses.items())
        loss_dict = {k: v.item() for k, v in losses.items()}
        loss_dict["total"] = total_loss.item()
        return total_loss, loss_dict


class SharpnessOptimizedLoss(nn.Module):
    """Loss function specifically designed for sharpness and detail preservation."""

    def __init__(self, device):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = LightPerceptualLoss(device)

        # Laplacian filter for edge detection
        laplacian_kernel = torch.tensor(
            [[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        self.laplacian_kernel = laplacian_kernel.repeat(3, 1, 1, 1).to(device)

    def edge_aware_loss(self, pred, target):
        """Edge-aware loss that focuses on high-frequency content."""
        pred_edges = F.conv2d(pred, self.laplacian_kernel, padding=1, groups=3)
        target_edges = F.conv2d(target, self.laplacian_kernel, padding=1, groups=3)
        return F.l1_loss(pred_edges, target_edges)

    def frequency_band_loss(self, pred, target, low_freq_ratio=0.3):
        """Separate loss for different frequency bands."""
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


def compute_image_gradient(x):
    """Compute image gradient magnitude."""
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    dx = F.pad(dx, (0, 1, 0, 0))
    dy = F.pad(dy, (0, 0, 0, 1))
    return torch.sqrt(dx**2 + dy**2 + 1e-8)


def create_pixelated_mosaic(
    rgb_tensor: torch.Tensor, block_size: int = 16
) -> torch.Tensor:
    _, h, w = rgb_tensor.shape
    small_tensor = F.interpolate(
        rgb_tensor.unsqueeze(0),
        size=(h // block_size, w // block_size),
        mode="area",
    )
    pixelated_tensor = F.interpolate(small_tensor, size=(h, w), mode="nearest")
    return pixelated_tensor.squeeze(0)


class RestorationDataset(Dataset):
    """Dataset with consistent preprocessing and augmentation."""

    def __init__(
        self,
        clean_dir: Path,
        mask_dir: Path,
        image_size: Tuple[int, int],
        transform=None,
        mosaic_block_size_range: Tuple[int, int] = (16, 16),
        # --- TAMBAHAN BARU ---
        mosaic_opacity_range: Tuple[float, float] = (1.0, 1.0),  # (min, max) opacity
        use_masks=True,
        task_type="demosaic",
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
        # --- TAMBAHAN BARU ---
        self.mosaic_opacity_range = mosaic_opacity_range
        self.use_masks = use_masks
        self.task_type = task_type

    def __len__(self):
        return len(self.clean_paths)

    def __getitem__(self, idx):
        clean_path = self.clean_paths[idx]
        mask_path = self.mask_dir / clean_path.name

        try:
            clean_img = Image.open(clean_path).convert("RGB")

            if self.transform:
                clean_tensor = self.transform(clean_img)
            else:
                clean_tensor = transforms.ToTensor()(clean_img)
                # Ensure correct size if no crop transform is applied
                if clean_tensor.shape[1:] != self.image_size:
                    clean_tensor = transforms.functional.resize(
                        clean_tensor, self.image_size
                    )

            degraded_base_tensor = torch.zeros_like(clean_tensor)
            if self.task_type == "demosaic":
                # 1. Pilih ukuran blok secara acak
                block_size = random.randint(*self.mosaic_block_size_range)
                pixelated_tensor = create_pixelated_mosaic(
                    clean_tensor, block_size=block_size
                )

                # --- BAGIAN BARU: Terapkan Opasitas Acak ---
                # 2. Pilih tingkat opasitas (alpha) secara acak
                opacity = random.uniform(*self.mosaic_opacity_range)

                # 3. Campurkan (blend) gambar mosaik dengan gambar asli
                degraded_base_tensor = (opacity * pixelated_tensor) + (
                    (1 - opacity) * clean_tensor
                )
            elif self.task_type == "inpainting":
                # Untuk inpainting, area yang rusak adalah hitam (nilai 0)
                degraded_base_tensor = torch.zeros_like(clean_tensor)

            # Apply mask if available
            if not self.use_masks:
                input_tensor = degraded_base_tensor
                return input_tensor, clean_tensor

            try:
                mask_img = Image.open(mask_path).convert("L")
                # Apply the same geometric transforms to mask
                mask_tensor = transforms.ToTensor()(mask_img)
                if mask_tensor.shape[1:] != self.image_size:
                    mask_tensor = transforms.functional.resize(
                        mask_tensor,
                        self.image_size,
                        interpolation=transforms.InterpolationMode.NEAREST,
                    )

                # Combine degraded and clean based on mask
                input_tensor = torch.where(
                    mask_tensor > 0.5, degraded_base_tensor, clean_tensor
                )
            except FileNotFoundError:
                # If mask not found, use fully degraded image
                logging.debug(
                    f"Mask not found for {clean_path.name}, using full mosaic."
                )
                input_tensor = degraded_base_tensor

            return input_tensor, clean_tensor

        except Exception as e:
            logging.warning(f"Error processing {clean_path.name}: {e}, skipping.")
            # Return a random other sample to avoid stopping training
            new_idx = (idx + 1) % len(self)
            return self.__getitem__(new_idx)


class ModelEMA:
    """Exponential Moving Average of model weights."""

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
        self.setup_data_loaders()

        self.initialize_model_only()
        if hasattr(torch, "compile") and self.config.get("compile_mode"):
            self.generator = torch.compile(
                self.generator, mode=self.config["compile_mode"]
            )

        self.initialize_optimizers_and_schedulers()
        self.initialize_remaining_components()

        self.load_checkpoint()

    def setup_logging(self):
        log_file = self.config.get("log_file", "Training/training.log")
        Path(log_file).parent.mkdir(exist_ok=True, parents=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)

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
        """Hanya membuat arsitektur model."""
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
        """Membuat optimizer & scheduler SETELAH model dimuat."""
        self.optimizer_G = optim.AdamW(
            self.generator.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config.get("weight_decay", 1e-4),
            betas=(0.9, 0.999),
        )

        scheduler_type = self.config.get("scheduler", "onecycle")
        if scheduler_type == "onecycle":
            self.scheduler = optim.lr_scheduler.OneCycleLR(
                self.optimizer_G,
                max_lr=self.config["learning_rate"],
                total_steps=len(self.train_loader) * self.config["num_epochs"],
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
        """Inisialisasi sisa komponen seperti loss, scaler, metrics, dll."""
        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = LightPerceptualLoss(self.device)

        if self.config.get("use_sharpness_loss"):
            self.criterion = SharpnessOptimizedLoss(self.device)
        elif self.config.get("use_advanced_loss"):
            self.criterion = AdvancedRestorationLoss(self.device)
        else:
            self.criterion = self.compute_combined_loss

        if self.config.get("use_gan"):
            self.gan_loss = nn.BCEWithLogitsLoss()

        self.scaler_G = GradScaler(enabled=self.config.get("use_amp", True))
        self.scaler_D = GradScaler(enabled=self.config.get("use_amp", True))

        if self.config.get("use_ema"):
            self.ema = ModelEMA(
                self.generator, decay=self.config.get("ema_decay", 0.999)
            )

        self.psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        self.ssim_metric = StructuralSimilarityIndexMeasure(
            data_range=1.0,
        ).to(self.device)

        if not hasattr(self, "global_step"):
            self.global_step = 0
        if not hasattr(self, "best_psnr"):
            self.best_psnr = 0.0
        if not hasattr(self, "best_lpips"):
            self.best_lpips = float("inf")

        if lpips:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message="The parameter 'pretrained' is deprecated"
                )
                warnings.filterwarnings("ignore", category=FutureWarning)
                self.lpips_metric = lpips.LPIPS(net="alex").to(self.device).eval()
            for param in self.lpips_metric.parameters():
                param.requires_grad = False
        else:
            self.lpips_metric = None

    def setup_data_loaders(self):
        """Setup with better augmentation."""
        image_size = (self.config["img_height"], self.config["img_width"])
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
                transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
                transforms.ToTensor(),
                transforms.Lambda(
                    lambda x: (
                        torch.clamp(x + torch.randn_like(x) * 0.02, 0, 1)
                        if random.random() < 0.2
                        else x
                    )
                ),
            ]
        )

        val_transform = transforms.Compose(
            [transforms.CenterCrop(image_size), transforms.ToTensor()]
        )

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
        )

        self.val_dataset = RestorationDataset(
            Path(self.config["val_clean_dir"]),
            Path(self.config["val_mask_dir"]),
            image_size,
            transform=val_transform,
            mosaic_block_size_range=self.config.get(
                "mosaic_block_size_range", [16, 16]
            ),
            mosaic_opacity_range=self.config.get("mosaic_opacity_range", [1.0, 1.0]),
            use_masks=self.config.get("use_masks", True),
            task_type=self.config.get("task_type", "demosaic"),
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
            batch_size=self.config.get("val_batch_size", 4),
            shuffle=False,
            num_workers=self.config["dataloader_params"]["num_workers"],
            pin_memory=True,
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
                self.logger.info("Successfully loaded model weights for fine-tuning.")

                self.start_epoch = 0
                self.global_step = 0
                self.best_psnr = 0.0
                self.best_lpips = float("inf")
                return
            except Exception:
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

            self.logger.info(f"Advancing scheduler to step {self.global_step}.")
            for _ in range(self.global_step):
                self.scheduler.step()

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
            checkpoint["optimizer_D_state_dict"] = self.optimizer_D.state_dict()  # noqa
            checkpoint["scaler_D_state_dict"] = self.scaler_D.state_dict()

        if self.config.get("use_ema"):
            checkpoint["ema_state_dict"] = self.ema.shadow

        torch.save(checkpoint, self.checkpoint_dir / "latest.pth")

        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / "best_model.pth")
            self.logger.info(
                f"Saved new best model checkpoint with LPIPS: {self.best_lpips:.4f}"
            )

    def compute_combined_loss(self, pred, target, current_epoch):
        """
        Computes a weighted combination of L1 and LPIPS loss.
        The weights are sourced from the training configuration.
        """
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
        """Loop training yang disederhanakan dengan logika GAN yang benar."""
        self.generator.train()
        if self.config.get("use_gan"):
            self.discriminator.train()

        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch+1}/{self.config['num_epochs']}",
            leave=False,
        )

        for batch_idx, (degraded, clean) in enumerate(progress_bar):
            degraded = degraded.to(self.device, non_blocking=True)
            clean = clean.to(self.device, non_blocking=True)

            with autocast(
                device_type=self.device.type, enabled=self.config.get("use_amp", True)
            ):
                restored = self.generator(degraded)

            loss_D = torch.tensor(0.0)
            if self.config.get("use_gan"):
                self.optimizer_D.zero_grad()

                with autocast(
                    device_type=self.device.type,
                    enabled=self.config.get("use_amp", True),
                ):
                    pred_real = self.discriminator(clean)
                    loss_D_real = self.gan_loss(pred_real, torch.ones_like(pred_real))
                self.scaler_D.scale(loss_D_real).backward()

                with autocast(
                    device_type=self.device.type,
                    enabled=self.config.get("use_amp", True),
                ):
                    pred_fake = self.discriminator(restored.detach())
                    loss_D_fake = self.gan_loss(pred_fake, torch.zeros_like(pred_fake))
                self.scaler_D.scale(loss_D_fake).backward()

                self.scaler_D.step(self.optimizer_D)
                self.scaler_D.update()

                loss_D = (loss_D_real + loss_D_fake) * 0.5

            self.optimizer_G.zero_grad()
            with autocast(
                device_type=self.device.type, enabled=self.config.get("use_amp", True)
            ):
                loss_G_recon, loss_dict = self.criterion(restored, clean, epoch)
                total_loss_G = loss_G_recon

                if self.config.get("use_gan"):
                    pred_gen = self.discriminator(restored)
                    loss_G_gan = self.gan_loss(pred_gen, torch.ones_like(pred_gen))
                    total_loss_G = (
                        total_loss_G + self.config.get("gan_weight", 0.1) * loss_G_gan
                    )
                    loss_dict["gan"] = loss_G_gan.item()

            self.scaler_G.scale(total_loss_G).backward()
            if self.config.get("grad_clip", 0) > 0:
                self.scaler_G.unscale_(self.optimizer_G)
                torch.nn.utils.clip_grad_norm_(
                    self.generator.parameters(), self.config["grad_clip"]
                )
            self.scaler_G.step(self.optimizer_G)
            self.scaler_G.update()

            self.scheduler.step()
            if self.config.get("use_ema"):
                self.ema.update()

            self.global_step += 1

            progress_bar.set_postfix(
                {
                    "G_Loss": f"{total_loss_G.item():.4f}",
                    "D_Loss": (
                        f"{loss_D.item():.4f}" if self.config.get("use_gan") else "N/A"
                    ),
                    "LR": f"{self.scheduler.get_last_lr()[0]:.2e}",
                }
            )

            if batch_idx % 100 == 0:
                self.writer.add_scalar(
                    "Loss/G_Total", total_loss_G.item(), self.global_step
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
                check_gpu_temp(self.device)

        return total_loss_G.item()

    @torch.no_grad()
    def validate(self, epoch):
        self.generator.eval()

        if self.config.get("use_ema"):
            self.ema.apply_shadow()

        total_psnr = 0
        total_ssim = 0

        pbar = tqdm(self.val_loader, desc=f"Validation", leave=False)
        for degraded, clean in pbar:
            degraded = degraded.to(self.device)
            clean = clean.to(self.device)

            with autocast(
                device_type=self.device.type, enabled=self.config.get("use_amp", True)
            ):
                restored = self.generator(degraded)

            restored = restored.clamp(0, 1)
            clean = clean.clamp(0, 1)

            total_psnr += self.psnr_metric(restored, clean)
            total_ssim += self.ssim_metric(restored, clean)

        if self.config.get("use_ema"):
            self.ema.restore()

        avg_psnr = total_psnr / len(self.val_loader)
        avg_ssim = total_ssim / len(self.val_loader)

        self.writer.add_scalar("Validation/PSNR", avg_psnr, self.global_step)
        self.writer.add_scalar("Validation/SSIM", avg_ssim, self.global_step)

        lpips_score = self.calculate_lpips_on_subset()
        if lpips_score is not None:
            self.writer.add_scalar("Validation/LPIPS", lpips_score, self.global_step)

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

        with torch.no_grad():
            with autocast(
                device_type=self.device.type, enabled=self.config.get("use_amp", True)
            ):
                restored, internals = self.generator(degraded, return_internals=True)

        grid = make_grid(
            torch.cat([degraded[:4].cpu(), restored[:4].cpu(), clean[:4].cpu()], dim=0),
            nrow=4,
        )
        save_image(grid, self.preview_dir / f"epoch_{epoch+1:04d}.png")
        self.writer.add_image(
            "Validation/Samples (Input | Restored | Ground Truth)",
            grid,
            global_step=epoch,
        )

        self.log_internal_visualizations(internals, epoch)

        if self.config.get("use_ema"):
            self.ema.restore()

    def log_internal_visualizations(
        self, internals: Dict[str, torch.Tensor], epoch: int
    ):
        """Memproses dan menyimpan visualisasi internal ke TensorBoard."""
        for name, tensor in internals.items():
            if tensor.dim() != 4:
                continue

            tensor_vis = tensor[0:1]

            if tensor_vis.shape[1] == 1:
                feature_map_grid = tensor_vis
            else:
                feature_map_grid = torch.mean(tensor_vis, dim=1, keepdim=True)

            feature_map_grid -= feature_map_grid.min()
            feature_map_grid /= feature_map_grid.max()

            self.writer.add_image(
                f"Internals/{name}",
                feature_map_grid.squeeze(0),
                global_step=epoch,
            )

    @torch.no_grad()
    def calculate_lpips_on_subset(self, num_batches=16):
        if not self.lpips_metric:
            return None

        self.generator.eval()
        if self.config.get("use_ema"):
            self.ema.apply_shadow()

        total_lpips = 0.0
        batches_processed = 0
        for i, (degraded, clean) in enumerate(self.val_loader):
            if i >= num_batches:
                break
            degraded, clean = degraded.to(self.device), clean.to(self.device)
            restored = self.generator(degraded).clamp(0, 1)
            total_lpips += self.lpips_metric(restored * 2 - 1, clean * 2 - 1).sum()
            batches_processed += degraded.size(0)

        if self.config.get("use_ema"):
            self.ema.restore()
        return (
            (total_lpips / batches_processed).item() if batches_processed > 0 else 0.0
        )

    def train(self) -> Optional[Tuple[float, float]]:
        """Main training loop."""
        self.logger.info("Starting training...")
        self.logger.info(f"Using device: {self.device}")
        self.logger.info(f"Config: {json.dumps(self.config, indent=2)}")
        start_time = time.time()
        patience_counter = 0

        for epoch in range(self.start_epoch, self.config["num_epochs"]):
            self.train_epoch(epoch)
            psnr, ssim, lpips_score = self.validate(epoch)

            is_best = psnr > self.best_psnr
            if is_best:
                self.best_psnr = psnr
                patience_counter = 0
                self.logger.info(f"🎯 New best PSNR: {psnr:.2f} dB")

            is_best_lpips = lpips_score is not None and lpips_score < self.best_lpips
            if is_best_lpips:
                self.best_lpips = lpips_score
                self.logger.info(f"🏆 New best LPIPS: {lpips_score:.4f}")
                patience_counter = 0
            elif not is_best:
                patience_counter += 1
                self.logger.info(
                    f"No improvement in LPIPS for {patience_counter} epochs."
                )

            if (epoch + 1) % self.config.get(
                "checkpoint_interval_epochs", 1
            ) == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
            elif is_best_lpips:
                self.save_checkpoint(epoch, True)

            self.logger.info(
                f"Epoch {epoch+1:03d}/{self.config['num_epochs']} | "
                f"Val PSNR: {psnr:.2f} dB | Val SSIM: {ssim:.4f} | "
                f"Val LPIPS: {lpips_score:.4f} | "
                f"Best LPIPS: {self.best_lpips:.4f} | "
                f"LR: {self.scheduler.get_last_lr()[0]:.2e}"
            )

            if (
                patience_counter >= self.config.get("early_stopping_patience", 25)
                and self.config.get("early_stopping_patience") != -1
            ):
                self.logger.info(
                    f"🛑 Early stopping after {patience_counter} epochs without improvement"
                )
                break

        training_time = (time.time() - start_time) / 3600
        self.logger.info(f"Training completed in {training_time:.2f} hours")
        self.logger.info(f"Best Validation LPIPS: {self.best_lpips:.4f}")

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
