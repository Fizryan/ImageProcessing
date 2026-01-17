import time
import torch
import warnings
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import DataLoader
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchvision.utils import make_grid
from tqdm.auto import tqdm

from program.LoggingManager import LoggingManager
from program.DirectoryManager import DirectoryManager
from program.Architecture import get_model, PatchGANDiscriminator
from program.DataManager import get_dataloader
from program.Losses import (
    AdvancedRestorationLoss,
    SharpnessOptimizedLoss,
    LightPerceptualLoss,
)
from program.ModelEMA import ModelEMA
from program.TrainerUtils import TrainerUtils
from program.Utils import Utils

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import lpips

    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False

logger = LoggingManager.setup_logging(__name__)


class Trainer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._setup_hardware()
        self._setup_directories()
        self._setup_logging()

        self._setup_data()

        self._setup_model()
        self._setup_optimizer()
        self._setup_losses_and_metrics()

        self.start_epoch = 0
        self.global_step = 0
        self.best_psnr = -float("inf")
        self.best_lpips = float("inf")

        self._load_checkpoint()

        logger.info("Trainer initialized successfully.")

    def _setup_hardware(self):
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")

            if hasattr(torch.backends.cuda, "enable_flash_sdp"):
                torch.backends.cuda.enable_flash_sdp(True)

            logger.info(
                f"Hardware optimization enabled on {torch.cuda.get_device_name(0)}"
            )

    def _setup_directories(self):
        paths_to_create = [
            self.config["checkpoint_dir"],
            self.config.get("tensorboard_log_dir", "logs/tensorboard"),
        ]
        DirectoryManager.setup_directories(paths_to_create)

        self.checkpoint_dir = Path(self.config["checkpoint_dir"])

    def _setup_logging(self):
        log_dir = Path(self.config.get("tensorboard_log_dir", "logs/tensorboard"))
        trial_name = f"trial_{self.config.get('trial_number', 0)}"
        self.writer = SummaryWriter(log_dir=str(log_dir / trial_name))

    def _setup_data(self):
        logger.info("Initializing DataLoaders...")

        train_config = {
            **self.config,
            "clean_dir": self.config["train_clean_dir"],
            "mask_dir": self.config.get("train_mask_dir"),
            "mosaic_block": self.config.get("mosaic_block_size_range", (16, 16)),
            "opacity": self.config.get("mosaic_opacity_range", (1.0, 1.0)),
            "image_size": (self.config["img_height"], self.config["img_width"]),
        }

        self.train_loader = get_dataloader(
            dataset_type="restoration",
            config=train_config,
            batch_size=self.config["dataloader_params"]["batch_size"],
            num_workers=self.config["dataloader_params"]["num_workers"],
            shuffle=True,
        )

        val_config = {
            **self.config,
            "clean_dir": self.config["val_clean_dir"],
            "mask_dir": self.config.get("val_mask_dir"),
            "mosaic_block": (16, 16),
            "opacity": (1.0, 1.0),
            "image_size": (self.config["img_height"], self.config["img_width"]),
        }

        self.val_loader = get_dataloader(
            dataset_type="restoration",
            config=val_config,
            batch_size=1,
            num_workers=2,
            shuffle=False,
        )
        self.val_iter = iter(self.val_loader)

    def _setup_model(self):
        self.generator = get_model(self.config).to(self.device)

        if self.config.get("use_channels_last", True):
            self.generator = self.generator.to(memory_format=torch.channels_last)

        if self.config.get("compile_model", False) and hasattr(torch, "compile"):
            logger.info("Compiling generator model...")
            self.generator = torch.compile(self.generator)

        if self.config.get("use_gan"):
            self.discriminator = PatchGANDiscriminator().to(self.device)
            if self.config.get("use_channels_last", True):
                self.discriminator = self.discriminator.to(
                    memory_format=torch.channels_last
                )

        if self.config.get("use_ema"):
            self.ema = ModelEMA(
                self.generator, decay=self.config.get("ema_decay", 0.999)
            )
        else:
            self.ema = None

    def _setup_optimizer(self):
        self.optimizer_G = optim.AdamW(
            self.generator.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config.get("weight_decay", 1e-4),
            betas=(0.9, 0.999),
        )

        if self.config.get("use_gan"):
            self.optimizer_D = optim.Adam(
                self.discriminator.parameters(),
                lr=self.config.get(
                    "discriminator_lr", self.config["learning_rate"] / 2
                ),
                betas=(0.5, 0.999),
            )

        total_steps = len(self.train_loader) * self.config["num_epochs"]
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer_G,
            max_lr=self.config["learning_rate"],
            total_steps=total_steps,
            pct_start=0.1,
            div_factor=10,
            final_div_factor=100,
        )

        self.scaler_G = GradScaler(enabled=self.config.get("use_amp", True))
        if self.config.get("use_gan"):
            self.scaler_D = GradScaler(enabled=self.config.get("use_amp", True))

    def _setup_losses_and_metrics(self):
        self.l1_loss = nn.L1Loss()

        if self.config.get("use_sharpness_loss"):
            self.criterion = SharpnessOptimizedLoss()
        else:
            self.criterion = AdvancedRestorationLoss()

        self.criterion.to(self.device)

        if self.config.get("use_gan"):
            self.gan_loss = nn.BCEWithLogitsLoss()

        self.psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(
            self.device
        )

        if HAS_LPIPS:
            self.lpips_metric = lpips.LPIPS(net="vgg", verbose=False).to(self.device)
            for param in self.lpips_metric.parameters():
                param.requires_grad = False
        else:
            self.lpips_metric = None

    def _load_checkpoint(self):
        ckpt_path = self.checkpoint_dir / "latest.pth"
        if not ckpt_path.exists():
            return

        try:
            checkpoint = torch.load(
                ckpt_path, map_location=self.device, weights_only=False
            )

            state_dict = checkpoint["model_state_dict"]
            new_state_dict = {}
            for k, v in state_dict.items():
                new_key = k.replace("_orig_mod.", "")
                new_state_dict[new_key] = v
            self.generator.load_state_dict(new_state_dict)

            self.optimizer_G.load_state_dict(checkpoint["optimizer_G_state_dict"])
            self.scaler_G.load_state_dict(checkpoint["scaler_G_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

            if self.config.get("use_gan") and "discriminator_state_dict" in checkpoint:
                self.discriminator.load_state_dict(
                    checkpoint["discriminator_state_dict"]
                )
                self.optimizer_D.load_state_dict(checkpoint["optimizer_D_state_dict"])
                self.scaler_D.load_state_dict(checkpoint["scaler_D_state_dict"])

            if self.ema and "ema_state_dict" in checkpoint:
                self.ema.load_state_dict(checkpoint["ema_state_dict"])

            self.start_epoch = checkpoint["epoch"] + 1
            self.global_step = checkpoint.get("global_step", 0)
            self.best_psnr = checkpoint.get("best_psnr", 0.0)
            self.best_lpips = checkpoint.get("best_lpips", float("inf"))

            logger.info(f"Resuming training from Epoch {self.start_epoch}")

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}. Starting from scratch.")

    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        model_state = (
            self.generator._orig_mod.state_dict()
            if hasattr(self.generator, "_orig_mod")
            else self.generator.state_dict()
        )

        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": model_state,
            "optimizer_G_state_dict": self.optimizer_G.state_dict(),
            "scaler_G_state_dict": self.scaler_G.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_psnr": self.best_psnr,
            "best_lpips": self.best_lpips,
            "config": self.config,
        }

        if self.config.get("use_gan"):
            checkpoint["discriminator_state_dict"] = self.discriminator.state_dict()
            checkpoint["optimizer_D_state_dict"] = self.optimizer_D.state_dict()
            checkpoint["scaler_D_state_dict"] = self.scaler_D.state_dict()

        if self.ema:
            checkpoint["ema_state_dict"] = self.ema.state_dict()

        torch.save(checkpoint, self.checkpoint_dir / "latest.pth")

        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / "best_model.pth")
            logger.info(f"New Best Model Saved! (PSNR: {self.best_psnr:.2f})")

    def train_epoch(self, epoch: int):
        self.generator.train()
        if self.config.get("use_gan"):
            self.discriminator.train()

        Utils.check_gpu_temp(self.device)

        ohem_percent = TrainerUtils.get_current_ohem_percent(epoch, self.config)

        pbar = tqdm(
            self.train_loader,
            desc=f"Ep {epoch+1}/{self.config['num_epochs']}",
            leave=False,
        )

        total_g_loss = 0.0

        for batch_idx, (degraded, clean) in enumerate(pbar):
            degraded, clean = degraded.to(self.device, non_blocking=True), clean.to(
                self.device, non_blocking=True
            )

            loss_D_val = 0.0
            if self.config.get("use_gan"):
                self.optimizer_D.zero_grad(set_to_none=True)

                with autocast(
                    device_type=self.device.type,
                    enabled=self.config.get("use_amp", True),
                ):
                    restored_d = self.generator(degraded).detach()

                    pred_real = self.discriminator(clean)
                    loss_real = self.gan_loss(pred_real, torch.ones_like(pred_real))

                    pred_fake = self.discriminator(restored_d)
                    loss_fake = self.gan_loss(pred_fake, torch.zeros_like(pred_fake))

                    loss_D = (loss_real + loss_fake) * 0.5

                self.scaler_D.scale(loss_D).backward()
                self.scaler_D.step(self.optimizer_D)
                self.scaler_D.update()
                loss_D_val = loss_D.item()

            self.optimizer_G.zero_grad(set_to_none=True)

            with autocast(
                device_type=self.device.type, enabled=self.config.get("use_amp", True)
            ):
                restored = self.generator(degraded)

                if ohem_percent < 1.0:
                    l1_pixel = F.l1_loss(restored, clean, reduction="none").mean(
                        dim=[1, 2, 3]
                    )
                    k = max(1, int(restored.size(0) * ohem_percent))
                    _, top_idx = torch.topk(l1_pixel, k)

                    restored_ohem = restored[top_idx]
                    clean_ohem = clean[top_idx]
                else:
                    restored_ohem = restored
                    clean_ohem = clean

                loss_G, loss_dict = self.criterion(restored_ohem, clean_ohem, epoch)

                if self.config.get("use_gan"):
                    pred_g = self.discriminator(restored_ohem)
                    loss_g_gan = self.gan_loss(pred_g, torch.ones_like(pred_g))
                    loss_G += self.config.get("gan_weight", 0.01) * loss_g_gan
                    loss_dict["gan"] = loss_g_gan.item()

            self.scaler_G.scale(loss_G).backward()

            if self.config.get("grad_clip", 0) > 0:
                self.scaler_G.unscale_(self.optimizer_G)
                torch.nn.utils.clip_grad_norm_(
                    self.generator.parameters(), self.config["grad_clip"]
                )

            self.scaler_G.step(self.optimizer_G)
            self.scaler_G.update()

            self.scheduler.step()
            if self.ema:
                self.ema.update()

            total_g_loss += loss_G.item()
            self.global_step += 1

            if batch_idx % 50 == 0:
                pbar.set_postfix(
                    {
                        "G_Loss": f"{loss_G.item():.4f}",
                        "D_Loss": f"{loss_D_val:.4f}",
                        "VRAM": Utils.get_vram_usage(self.device),
                    }
                )

                self.writer.add_scalar("Loss/G_Total", loss_G.item(), self.global_step)

                for loss_name, loss_value in loss_dict.items():
                    if loss_name != "total":
                        self.writer.add_scalar(
                            f"Loss_Components/{loss_name}", loss_value, self.global_step
                        )

                if self.config.get("use_gan"):
                    self.writer.add_scalar("Loss/D_Total", loss_D_val, self.global_step)
                    self.writer.add_scalar(
                        "Loss_Components/gan_g",
                        loss_dict.get("gan", 0),
                        self.global_step,
                    )

                self.writer.add_scalar(
                    "Training/LR", self.scheduler.get_last_lr()[0], self.global_step
                )

                if ohem_percent < 1.0:
                    self.writer.add_scalar(
                        "Training/OHEM_Percent", ohem_percent, self.global_step
                    )

                if batch_idx % 200 == 0 and self.device.type == "cuda":
                    vram_used = torch.cuda.memory_allocated(self.device) / 1024**3
                    vram_reserved = torch.cuda.memory_reserved(self.device) / 1024**3
                    self.writer.add_scalar(
                        "System/VRAM_Used_GB", vram_used, self.global_step
                    )
                    self.writer.add_scalar(
                        "System/VRAM_Reserved_GB", vram_reserved, self.global_step
                    )

        avg_epoch_loss = total_g_loss / len(self.train_loader)
        self.writer.add_scalar("Epoch/Train_Loss_Avg", avg_epoch_loss, epoch)

    def _inference_tiled(self, img: torch.Tensor) -> torch.Tensor:
        patch_h, patch_w = 256, 448
        overlap = 32

        b, c, h, w = img.shape

        if h <= patch_h and w <= patch_w:
            return self.generator(img)

        stride_h = patch_h - overlap
        stride_w = patch_w - overlap

        blend_mask = TrainerUtils.generate_blend_mask((patch_w, patch_h), self.device)

        pad_h = (stride_h - (h - patch_h) % stride_h) % stride_h
        pad_w = (stride_w - (w - patch_w) % stride_w) % stride_w

        img_pad = F.pad(img, (0, pad_w, 0, pad_h), mode="reflect")
        out_pad = torch.zeros_like(img_pad)
        count_pad = torch.zeros_like(img_pad)

        H_pad, W_pad = img_pad.shape[2:]

        for y in range(0, H_pad - patch_h + 1, stride_h):
            for x in range(0, W_pad - patch_w + 1, stride_w):
                patch = img_pad[:, :, y : y + patch_h, x : x + patch_w]
                with torch.no_grad():
                    patch_out = self.generator(patch)
                out_pad[:, :, y : y + patch_h, x : x + patch_w] += (
                    patch_out * blend_mask
                )
                count_pad[:, :, y : y + patch_h, x : x + patch_w] += blend_mask

        output_full = out_pad / torch.where(
            count_pad == 0, torch.ones_like(count_pad), count_pad
        )
        return output_full[:, :, :h, :w]

    @torch.no_grad()
    def _calculate_lpips_subset(self, num_batches: int = 4) -> Optional[float]:
        if not HAS_LPIPS or self.lpips_metric is None:
            return None

        torch.cuda.empty_cache()

        total_lpips = 0.0
        count = 0

        loader_iter = iter(self.val_loader)

        for _ in range(num_batches):
            try:
                degraded, clean = next(loader_iter)
            except StopIteration:
                break

            degraded, clean = degraded.to(self.device), clean.to(self.device)

            with autocast(
                device_type=self.device.type, enabled=self.config.get("use_amp", True)
            ):
                if degraded.shape[2] > 512 or degraded.shape[3] > 512:
                    scale = 0.5
                    degraded = F.interpolate(
                        degraded, scale_factor=scale, mode="bilinear"
                    )
                    clean = F.interpolate(clean, scale_factor=scale, mode="bilinear")

                restored = self.generator(degraded).clamp(0, 1)

                val = self.lpips_metric(restored * 2 - 1, clean * 2 - 1).mean()

            total_lpips += val.item()
            count += 1

        return total_lpips / count if count > 0 else None

    @torch.no_grad()
    def validate(self, epoch: int) -> Tuple[float, float, Optional[float]]:
        self.generator.eval()
        if self.ema:
            self.ema.apply_shadow()

        total_psnr = 0.0
        total_ssim = 0.0

        pbar = tqdm(self.val_loader, desc="Validation", leave=False)
        for degraded, clean in pbar:
            degraded, clean = degraded.to(self.device), clean.to(self.device)

            with autocast(
                device_type=self.device.type, enabled=self.config.get("use_amp", True)
            ):
                if degraded.shape[2] > 600 or degraded.shape[3] > 600:
                    restored = self._inference_tiled(degraded)
                else:
                    restored = self.generator(degraded)

            restored = restored.clamp(0, 1)

            total_psnr += self.psnr_metric(restored, clean).item()
            total_ssim += self.ssim_metric(restored, clean).item()

        avg_psnr = total_psnr / len(self.val_loader)
        avg_ssim = total_ssim / len(self.val_loader)

        subset_n = self.config.get("lpips_subset_batches", 4)
        avg_lpips = self._calculate_lpips_subset(num_batches=subset_n)

        self.writer.add_scalar("Val/PSNR", avg_psnr, self.global_step)
        self.writer.add_scalar("Val/SSIM", avg_ssim, self.global_step)
        if avg_lpips is not None:
            self.writer.add_scalar("Val/LPIPS", avg_lpips, self.global_step)

        self._save_preview_to_tensorboard(epoch)

        if self.ema:
            self.ema.restore()

        return avg_psnr, avg_ssim, avg_lpips

    def _save_preview_to_tensorboard(self, epoch: int):
        try:
            degraded, clean = next(self.val_iter)
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            degraded, clean = next(self.val_iter)

        degraded, clean = degraded.to(self.device), clean.to(self.device)

        with torch.no_grad():
            with autocast(
                device_type=self.device.type, enabled=self.config.get("use_amp", True)
            ):
                restored = self.generator(degraded).clamp(0, 1)

        if restored.shape[3] > 512:
            restored = F.interpolate(restored, scale_factor=0.5, mode="bilinear")
            degraded = F.interpolate(degraded, scale_factor=0.5, mode="bilinear")
            clean = F.interpolate(clean, scale_factor=0.5, mode="bilinear")

        grid = make_grid(
            torch.cat([degraded, restored, clean], dim=0), nrow=degraded.size(0)
        )

        self.writer.add_image("Validation/Preview_Epoch", grid, global_step=epoch)

    def run(self):
        logger.info("Starting Training Loop...")

        try:
            for epoch in range(self.start_epoch, self.config["num_epochs"]):
                self.train_epoch(epoch)

                psnr, ssim, lpips_val = self.validate(epoch)

                lpips_str = f"{lpips_val:.4f}" if lpips_val is not None else "N/A"

                logger.info(
                    f"Epoch {epoch+1} Summary | PSNR: {psnr:.2f}dB | SSIM: {ssim:.4f} | LPIPS: {lpips_str}"
                )

                is_best_psnr = psnr > self.best_psnr
                is_best_lpips = lpips_val is not None and lpips_val < self.best_lpips
                is_best = is_best_psnr or is_best_lpips

                if is_best_psnr:
                    self.best_psnr = psnr
                    logger.info(f"  → New Best PSNR: {psnr:.2f}dB")

                if is_best_lpips:
                    self.best_lpips = lpips_val
                    logger.info(f"  → New Best LPIPS: {lpips_val:.4f}")

                if (epoch + 1) % self.config.get("checkpoint_freq", 1) == 0 or is_best:
                    self._save_checkpoint(epoch, is_best)

        except KeyboardInterrupt:
            logger.warning(
                "Training interrupted by user. Saving emergency checkpoint..."
            )
            self._save_checkpoint(epoch, is_best=False)

        logger.info("Training Finished.")
        self.writer.close()
