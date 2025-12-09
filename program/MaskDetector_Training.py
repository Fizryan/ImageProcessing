# MaskDetector_Training.py
# Training script for the mosaic detection model.

import logging
import time
from pathlib import Path
from typing import Tuple, Dict, Any
import json
import random

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image, ImageDraw
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from tqdm.auto import tqdm

from program.Architecture import SOTARestorationUNet
from program.Utils import check_gpu_temp, load_model_weights


def apply_mosaic(img: Image.Image, block_size: int) -> Image.Image:
    """Applies a pixelation/mosaic effect to an image."""
    small_img = img.resize(
        (max(1, img.width // block_size), max(1, img.height // block_size)),
        Image.Resampling.NEAREST,
    )
    return small_img.resize(img.size, Image.Resampling.NEAREST)


class DetectorDataset(Dataset):
    def __init__(
        self,
        clean_dir: Path,
        mask_dir: Path,
        image_size: Tuple[int, int],
        mosaic_block_size_range: Tuple[int, int],
    ):
        self.image_size = image_size
        self.clean_paths = sorted(
            [
                p
                for p in clean_dir.iterdir()
                if p.suffix.lower() in [".png", ".jpg", ".jpeg"]
            ]
        )
        self.mask_dir = mask_dir
        self.mosaic_block_size_range = mosaic_block_size_range
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.clean_paths)

    def __getitem__(self, idx):
        clean_path = self.clean_paths[idx]
        mask_path = self.mask_dir / clean_path.name

        try:
            clean_img = (
                Image.open(clean_path).convert("RGB").resize(self.image_size[::-1])
            )

            # 1. Load the mask (the target)
            try:
                target_mask = (
                    Image.open(mask_path).convert("L").resize(self.image_size[::-1])
                )
            except FileNotFoundError:
                logging.debug(
                    f"Mask not found for {clean_path.name}, creating empty mask."
                )
                target_mask = Image.new("L", self.image_size[::-1], 0)

            # 2. Create a mosaic version of the clean image
            block_size = random.randint(*self.mosaic_block_size_range)
            mosaic_img = apply_mosaic(clean_img, block_size)

            # 3. Composite to create the input image using the loaded mask
            input_img = Image.composite(mosaic_img, clean_img, target_mask)

            # 4. Convert to tensors
            input_tensor = self.to_tensor(input_img)  # Range [0, 1]
            target_tensor = self.to_tensor(target_mask)  # Range [0, 1]

            return input_tensor, target_tensor

        except Exception as e:
            logging.error(f"Error loading {clean_path.name}: {e}")
            return self.__getitem__(torch.randint(0, len(self), (1,)).item())


class MaskDetectorTrainer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.setup_logging()

        torch.backends.cudnn.benchmark = True
        self.setup_directories()
        self.setup_data_loaders()
        self.initialize_components()
        self.load_checkpoint()

    def setup_logging(self):
        log_file = self.config.get("log_file", "Training/detector.log")
        Path(log_file).parent.mkdir(exist_ok=True, parents=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )
        self.logger = logging.getLogger(__name__)
        self.writer = SummaryWriter(
            log_dir=str(Path(self.config["checkpoint_dir"]) / "runs")
        )

    def setup_directories(self):
        self.checkpoint_dir = Path(self.config["checkpoint_dir"])
        self.preview_dir = Path(self.config["preview_dir"])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.preview_dir.mkdir(parents=True, exist_ok=True)

    def initialize_components(self):
        self.model = EfficientUNet(
            in_channels=3,
            out_channels=1,
            base_channels=self.config.get("base_channels", 16),
            use_checkpointing=self.config.get("use_checkpointing", True),
            use_global_residual=False,  # Detector model should not use global residual
        ).to(self.device)

        # Reconfigure output for detector: single channel, no Tanh activation
        self.model.out_conv = nn.Conv2d(
            self.config.get("base_channels", 16), 1, 3, padding=1
        )
        self.model.final_act = nn.Identity()

        if self.config.get("use_channels_last", True):
            self.model = self.model.to(memory_format=torch.channels_last)

        if hasattr(torch, "compile") and self.config.get("compile_mode"):
            self.model = torch.compile(self.model, mode=self.config["compile_mode"])

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config.get("weight_decay", 1e-5),
        )

        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.config["learning_rate"],
            steps_per_epoch=len(self.train_loader),
            epochs=self.config["num_epochs"],
            **self.config.get("onecycle_params", {}),
        )

        self.criterion = nn.BCEWithLogitsLoss()
        self.use_amp = self.config.get("use_amp", True) and self.device.type == "cuda"
        self.scaler = GradScaler(enabled=self.use_amp)
        self.global_step = 0
        self.best_loss = float("inf")

    def setup_data_loaders(self):
        image_size = (self.config["img_height"], self.config["img_width"])
        train_dataset = DetectorDataset(
            Path(self.config["train_clean_dir"]),
            Path(self.config["train_mask_dir"]),
            image_size,
            self.config.get("mosaic_block_size_range", [16, 16]),
        )
        self.train_loader = DataLoader(
            train_dataset, **self.config["dataloader_params"]
        )

        val_dataset = DetectorDataset(
            Path(self.config["val_clean_dir"]),
            Path(self.config["val_mask_dir"]),
            image_size,
            self.config.get("mosaic_block_size_range", [16, 16]),
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.get("val_batch_size", 4),
            shuffle=False,
            num_workers=2,
        )
        self.val_iter = iter(self.val_loader)

    def load_checkpoint(self):
        checkpoint_path = self.checkpoint_dir / "latest.pth"
        if not checkpoint_path.exists():
            self.start_epoch = 0
            self.logger.info("No checkpoint found, starting from scratch.")
            return

        try:
            checkpoint = torch.load(
                checkpoint_path, map_location=self.device, weights_only=False
            )
            load_model_weights(self.model, checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            scheduler_state = checkpoint["scheduler_state_dict"]
            if "total_steps" in scheduler_state:
                del scheduler_state["total_steps"]
            if "_schedule_phases" in scheduler_state:
                del scheduler_state["_schedule_phases"]
            self.scheduler.load_state_dict(scheduler_state)

            self.start_epoch = checkpoint["epoch"] + 1
            self.global_step = checkpoint.get("global_step", 0)
            self.best_loss = checkpoint.get("best_loss", float("inf"))
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
            self.logger.info(
                f"Loaded checkpoint from epoch {self.start_epoch - 1}. Resuming."
            )
        except Exception as e:
            self.logger.error(
                f"Failed to load checkpoint, starting from scratch: {e}", exc_info=True
            )
            self.start_epoch = 0
            self.best_loss = float("inf")

    def save_checkpoint(self, epoch, is_best=False):
        model_state_dict = (
            self.model._orig_mod.state_dict()
            if hasattr(self.model, "_orig_mod")
            else self.model.state_dict()
        )
        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_loss": self.best_loss,
            "config": self.config,
        }
        torch.save(checkpoint, self.checkpoint_dir / "latest.pth")
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / "best_detector_model.pth")
            self.logger.info(
                f"Saved new best detector model checkpoint with validation loss: {self.best_loss:.4f}"
            )

    def train_epoch(self, epoch):
        self.model.train()
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch+1}/{self.config['num_epochs']}",
            leave=False,
        )

        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(self.device, non_blocking=True), targets.to(
                self.device, non_blocking=True
            )
            if self.config.get("use_channels_last", True):
                inputs = inputs.to(memory_format=torch.channels_last)

            self.optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=self.device.type, enabled=self.use_amp):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

            self.scaler.scale(loss).backward()
            if self.config.get("grad_clip", 0) > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config["grad_clip"]
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            self.global_step += 1

            if self.global_step % 50 == 0:
                self.writer.add_scalar("Loss/Train", loss.item(), self.global_step)
                self.writer.add_scalar(
                    "Learning_Rate", self.scheduler.get_last_lr()[0], self.global_step
                )
                check_gpu_temp(self.device)

            progress_bar.set_postfix(
                {
                    "Loss": f"{loss.item():.4f}",
                    "LR": f"{self.scheduler.get_last_lr()[0]:.2e}",
                }
            )

    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()
        total_loss = 0
        pbar = tqdm(self.val_loader, desc="Validation", leave=False)
        for inputs, targets in pbar:
            inputs, targets = inputs.to(self.device, non_blocking=True), targets.to(
                self.device, non_blocking=True
            )
            if self.config.get("use_channels_last", True):
                inputs = inputs.to(memory_format=torch.channels_last)

            with autocast(device_type=self.device.type, enabled=self.use_amp):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
            total_loss += loss.item()

        avg_loss = total_loss / len(self.val_loader)
        self.writer.add_scalar("Loss/Validation", avg_loss, self.global_step)
        self.save_sample_images(epoch)
        return avg_loss

    def save_sample_images(self, epoch):
        self.model.eval()
        try:
            inputs, targets = next(self.val_iter)
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs, targets = next(self.val_iter)

        inputs, targets = inputs.to(self.device), targets.to(self.device)

        with torch.no_grad(), autocast(
            device_type=self.device.type, enabled=self.use_amp
        ):
            outputs = self.model(inputs)
            predicted_masks = torch.sigmoid(outputs)

        inputs_vis = inputs.clamp(0, 1)

        predicted_masks_vis = predicted_masks.repeat(1, 3, 1, 1)
        targets_vis = targets.repeat(1, 3, 1, 1)

        grid = make_grid(
            torch.cat(
                [inputs_vis[:4], predicted_masks_vis[:4], targets_vis[:4]], dim=0
            ),
            nrow=4,
        )
        save_image(grid, self.preview_dir / f"epoch_{epoch+1:04d}.png")
        self.writer.add_image(
            "Validation/Samples (Input | Pred | GT)", grid, global_step=epoch
        )

    def train(self):
        self.logger.info("Starting mosaic detector model training...")
        self.logger.info(f"Using device: {self.device}")
        self.logger.info(f"Config: {json.dumps(self.config, indent=2)}")
        start_time = time.time()

        for epoch in range(self.start_epoch, self.config["num_epochs"]):
            self.train_epoch(epoch)
            val_loss = self.validate(epoch)

            is_best = val_loss < self.best_loss
            if is_best:
                self.best_loss = val_loss

            if (epoch + 1) % self.config.get(
                "checkpoint_interval_epochs", 1
            ) == 0 or is_best:
                self.save_checkpoint(epoch, is_best)

            self.logger.info(
                f"Epoch {epoch+1}/{self.config['num_epochs']} | "
                f"Val Loss: {val_loss:.4f} | Best Loss: {self.best_loss:.4f}"
            )

        training_time = (time.time() - start_time) / 3600
        self.logger.info(f"Training completed in {training_time:.2f} hours")
        self.logger.info(f"Best Validation Loss: {self.best_loss:.4f}")

        hparams = {}
        for key, value in self.config.items():
            if isinstance(value, (int, float, str, bool)):
                hparams[key] = value
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, (int, float, str, bool)):
                        hparams[f"{key}/{sub_key}"] = sub_value

        metrics = {
            "hparam/best_loss": self.best_loss,
            "hparam/training_time_hours": training_time,
        }
        self.writer.add_hparams(hparams, metrics, run_name=".")

        self.writer.close()
