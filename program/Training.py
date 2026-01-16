# Training.py

import time
import json
import random
import torch
import warnings
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
from PIL import Image
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import DataLoader, ConcatDataset
from torchmetrics.image import (
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from tqdm.auto import tqdm

from program.Architecture import get_model, PatchGANDiscriminator
from program.Utils import check_gpu_temp, load_model_weights
from program.Losses import (
    AdvancedRestorationLoss,
    SharpnessOptimizedLoss,
    LightPerceptualLoss,
)
from program.Augmentation import RobustDegradation
from program.RestorationDataset import RestorationDataset, RandomScale, ExternalDataset
from program.ModelEMA import ModelEMA
from program.TrainerUtils import TrainerUtils
from program.LoggingSetup import setup_logger, fmt_bool

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

try:
    import lpips
except ImportError:
    lpips = None


class Trainer:
    """Main trainer class for image restoration models."""

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
        """Log active augmentation features."""
        scale_range = self.config.get("scale_augmentation_range", [0.5, 1.0])
        if scale_range[0] < 1.0:
            self.logger.info("✓ Scale Augmentation (Multi-Resolution): ENABLED")
            self.logger.info(
                f"  - Scale Range: {scale_range[0]:.2f} to {scale_range[1]:.2f}"
            )
            self.logger.info(
                f"  - Simulates: {int(scale_range[0]*1080)}p to {int(scale_range[1]*1080)}p"
            )
            self.logger.info("  → Makes model robust to ANY video resolution")
        else:
            self.logger.info("✗ Scale Augmentation: DISABLED (Fixed resolution only)")

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
        if scale_range[0] < 1.0:
            enabled_features.append("Multi-Res Training")
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
        """Setup logging with consistent formatting."""
        log_file = self.config.get("log_file", "Training/training.log")
        self.logger = setup_logger(__name__, log_file=log_file)

        if "tensorboard_log_dir" in self.config and "trial_number" in self.config:
            log_dir = (
                Path(self.config["tensorboard_log_dir"])
                / f"trial_{self.config['trial_number']}"
            )
        else:
            log_dir = Path(self.config["checkpoint_dir"]) / "runs"
        self.writer = SummaryWriter(log_dir=str(log_dir))

    def setup_directories(self):
        """Create necessary directories."""
        self.checkpoint_dir = Path(self.config["checkpoint_dir"])
        self.preview_dir = Path(self.config["preview_dir"])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.preview_dir.mkdir(parents=True, exist_ok=True)

    def initialize_model_only(self):
        """Initialize generator and discriminator models."""
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
        """Initialize optimizers and learning rate schedulers."""
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
        """Initialize loss functions, metrics, and other components."""
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

    def _inference_with_tiling(
        self,
        model: nn.Module,
        image_tensor: torch.Tensor,
        tile_size: Optional[Tuple[int, int]] = None,
        overlap: int = 32,
    ) -> torch.Tensor:
        """Perform tiled inference for large images."""
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

        blend_mask = TrainerUtils.generate_blend_mask((patch_w, patch_h), self.device)

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
        """Setup training and validation data loaders."""
        image_size = (self.config["img_height"], self.config["img_width"])

        train_transform_list = [
            RandomScale(
                scale_range=self.config.get("scale_augmentation_range", (0.5, 1.0)),
                target_crop_size=max(image_size),
            ),
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

        train_mask_dir = self.config.get("train_mask_dir")
        val_mask_dir = self.config.get("val_mask_dir")
        train_external_dir = self.config.get("train_external_dir")
        val_external_dir = self.config.get("val_external_dir")

        train_datasets = []
        val_datasets = []

        if train_mask_dir and Path(train_mask_dir).exists():
            restoration_train_dataset = RestorationDataset(
                Path(self.config["train_clean_dir"]),
                Path(train_mask_dir),
                image_size,
                transform=train_transform,
                mosaic_block_size_range=self.config.get(
                    "mosaic_block_size_range", [16, 16]
                ),
                mosaic_opacity_range=self.config.get(
                    "mosaic_opacity_range", [1.0, 1.0]
                ),
                use_masks=self.config.get("use_masks", True),
                task_type=self.config.get("task_type", "demosaic"),
                keep_original_size=False,
                use_mosaic_grid_shift=self.config.get("use_mosaic_grid_shift", False),
                robust_degradation=dataset_robust_degradation,
            )
            train_datasets.append(restoration_train_dataset)
            self.logger.info(
                f"RestorationDataset (train): {len(restoration_train_dataset)} images with auto-mosaic"
            )
        else:
            self.logger.info(
                "No train_mask_dir provided - skipping auto-mosaic dataset (using external images only)"
            )

        if train_external_dir and Path(train_external_dir).exists():
            external_dataset = ExternalDataset(
                external_dir=Path(train_external_dir),
                clean_dir=Path(self.config["train_clean_dir"]),
                image_size=image_size,
                transform=train_transform,
                keep_original_size=False,
            )
            train_datasets.append(external_dataset)
            self.logger.info(
                f"ExternalDataset (train): {len(external_dataset)} manually degraded images"
            )

        if len(train_datasets) == 0:
            raise ValueError(
                "No training data available. Please provide either train_mask_dir or train_external_dir."
            )
        elif len(train_datasets) == 1:
            self.train_dataset = train_datasets[0]
        else:
            self.train_dataset = ConcatDataset(train_datasets)

        if val_mask_dir and Path(val_mask_dir).exists():
            restoration_val_dataset = RestorationDataset(
                Path(self.config["val_clean_dir"]),
                Path(val_mask_dir),
                image_size,
                transform=None,
                mosaic_block_size_range=self.config.get(
                    "mosaic_block_size_range", [16, 16]
                ),
                mosaic_opacity_range=self.config.get(
                    "mosaic_opacity_range", [1.0, 1.0]
                ),
                use_masks=self.config.get("use_masks", True),
                task_type=self.config.get("task_type", "demosaic"),
                keep_original_size=True,
                use_mosaic_grid_shift=False,
                robust_degradation=None,
            )
            val_datasets.append(restoration_val_dataset)
            self.logger.info(
                f"RestorationDataset (val): {len(restoration_val_dataset)} images with auto-mosaic"
            )
        else:
            self.logger.info(
                "No val_mask_dir provided - skipping auto-mosaic dataset for validation"
            )

        if val_external_dir and Path(val_external_dir).exists():
            val_external_dataset = ExternalDataset(
                external_dir=Path(val_external_dir),
                clean_dir=Path(self.config["val_clean_dir"]),
                image_size=image_size,
                transform=None,
                keep_original_size=True,
            )
            val_datasets.append(val_external_dataset)
            self.logger.info(
                f"ExternalDataset (val): {len(val_external_dataset)} manually degraded images"
            )

        if len(val_datasets) == 0:
            raise ValueError(
                "No validation data available. Please provide either val_mask_dir or val_external_dir."
            )
        elif len(val_datasets) == 1:
            self.val_dataset = val_datasets[0]
        else:
            self.val_dataset = ConcatDataset(val_datasets)

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
        """Load checkpoint if exists."""
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

            checkpoint_epochs = checkpoint.get("config", {}).get(
                "num_epochs", self.config["num_epochs"]
            )
            epochs_changed = checkpoint_epochs != self.config["num_epochs"]

            if "scheduler_state_dict" in checkpoint and not epochs_changed:
                try:
                    self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                    self.scheduler.last_epoch = self.global_step
                    self.logger.info(
                        f"Loaded scheduler state. Step synchronized to: {self.global_step}."
                    )
                except Exception as e:
                    self.logger.warning(
                        f"Failed to load scheduler: {e}. Resetting scheduler to warmup phase."
                    )
            else:
                if epochs_changed:
                    self.logger.warning(
                        f"⚠️  num_epochs changed: {checkpoint_epochs} → {self.config['num_epochs']}"
                    )
                    self.logger.warning(
                        f"⚠️  Scheduler NOT loaded from checkpoint (will start from warmup phase)"
                    )
                    self.logger.warning(
                        f"⚠️  For stable LR fine-tuning, consider using scheduler='cosine_restarts'"
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
        """Save model checkpoint."""
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

    def train_epoch(self, epoch):
        """Train for one epoch."""
        check_gpu_temp(self.device, threshold=82, delay=15)

        self.generator.train()
        if self.config.get("use_gan"):
            self.discriminator.train()

        accumulation_steps = self.config.get("accumulation_steps", 1)
        ohem_percent = TrainerUtils.get_current_ohem_percent(epoch, self.config)
        if (
            epoch == 0
            or TrainerUtils.get_current_ohem_percent(epoch - 1, self.config)
            != ohem_percent
        ):
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

            if batch_idx % 10 == 0:
                check_gpu_temp(self.device)

        avg_loss_G = total_loss_G_epoch / len(self.train_loader)
        return avg_loss_G

    @torch.no_grad()
    def validate(self, epoch):
        """Validate the model."""
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
        """Save sample validation images."""
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
        """Calculate LPIPS score on a subset of validation data."""
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
        """Main training loop."""
        self.logger.info("=" * 80)
        self.logger.info("Starting Training Session")
        self.logger.info("=" * 80)

        def fmt_bool(value):
            return "✓" if value else "✗"

        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Model: {self.generator.__class__.__name__}")
        self.logger.info(f"Task Type: {self.config.get('task_type', 'N/A')}")
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

        self.logger.info("Model Architecture:")
        self.logger.info(f"  Base Channels: {self.config.get('base_channels', 'N/A')}")
        self.logger.info(f"  Model Size: {self.config.get('model_size', 'N/A')}")
        self.logger.info(
            f"  Enhanced Architecture: {fmt_bool(self.config.get('use_enhanced_architecture', False))}"
        )
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

        ohem_schedule = self.config.get("ohem_schedule", [])
        if ohem_schedule:
            self.logger.info("OHEM Schedule:")
            num_epochs = self.config["num_epochs"]
            for epoch_ratio, percent in ohem_schedule:
                actual_epoch = int(epoch_ratio * num_epochs)
                self.logger.info(
                    f"  Epoch {actual_epoch} ({epoch_ratio*100:.1f}% of training): {percent*100:.0f}%"
                )

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
