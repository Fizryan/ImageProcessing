# Training.py
# This module contains the training logic for the image restoration model.

import logging
import math
import time
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchvision import transforms
import torchvision.transforms.v2 as T
from torchvision.utils import make_grid, save_image
import numpy as np
from tqdm.auto import tqdm

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
    logging.warning("Matplotlib not found. LR Finder plotting will be disabled.")

try:
    import lpips
except ImportError:
    lpips = None
    logging.warning("LPIPS library not found. Enhanced validation will be disabled.")

try:
    from GPUtil import getGPUs
except ImportError:
    getGPUs = None
    logging.warning("GPUtil not available. GPU temp monitoring disabled.")

from program.Architecture import (
    SuperResolutionNet,
    UNetLite,
    PerceptualLoss,
    EdgeLoss,
    MultiScaleLoss,
    FrequencyLoss,
    PatchDiscriminator,
)


class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {
            name: param.clone().detach() for name, param in model.named_parameters()
        }
        self.num_updates = 0
        self.backup = {}

    def update(self):
        self.num_updates += 1
        decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (
                    decay * self.shadow[name].data + (1 - decay) * param.data
                )

    def apply_shadow(self):
        self.backup = {
            name: param.clone().detach()
            for name, param in self.model.named_parameters()
        }
        for name, param in self.model.named_parameters():
            param.data.copy_(self.shadow[name].data)

    def restore(self):
        if not self.backup:
            return
        for name, param in self.model.named_parameters():
            param.data.copy_(self.backup[name].data)
        self.backup = {}


class LRFinder:
    def __init__(self, model, optimizer, criterion, device, train_loader):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.train_loader = train_loader
        self.history = {"lr": [], "loss": []}
        self.best_loss = float("inf")

        self.model_state = model.state_dict()
        self.optimizer_state = optimizer.state_dict()

    def range_test(
        self,
        start_lr=1e-7,
        end_lr=1,
        num_iter=100,
        smooth_f=0.05,
        diverge_th=5,
    ):
        self.history = {"lr": [], "loss": []}
        self.best_loss = float("inf")
        lr_scheduler = np.linspace(start_lr, end_lr, num_iter)
        avg_loss = 0.0
        loader_iter = iter(self.train_loader)

        self.model.train()
        pbar = tqdm(total=num_iter, desc="LR Finder")

        for i, lr in enumerate(lr_scheduler):
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            try:
                inputs, targets, _ = next(loader_iter)
            except StopIteration:
                loader_iter = iter(self.train_loader)
                inputs, targets, _ = next(loader_iter)

            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs, _, _, _ = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            avg_loss = smooth_f * loss.item() + (1 - smooth_f) * avg_loss
            smoothed_loss = avg_loss / (1 - (1 - smooth_f) ** (i + 1))

            self.history["lr"].append(lr)
            self.history["loss"].append(smoothed_loss)

            if smoothed_loss < self.best_loss:
                self.best_loss = smoothed_loss

            if i > 10 and smoothed_loss > diverge_th * self.best_loss:
                logging.info("Loss diverging, stopping early.")
                break

            pbar.update(1)
            pbar.set_postfix(loss=f"{smoothed_loss:.4e}", lr=f"{lr:.4e}")

        pbar.close()
        logging.info("LR Finder test finished. Restoring model and optimizer state.")
        self.model.load_state_dict(self.model_state)
        self.optimizer.load_state_dict(self.optimizer_state)

    def plot(self, skip_start=10, skip_end=5, log_lr=True, save_path=None):
        if plt is None:
            logging.error("Matplotlib is required for plotting the LR range test.")
            return

        lrs = self.history["lr"][skip_start:-skip_end]
        losses = self.history["loss"][skip_start:-skip_end]

        fig, ax = plt.subplots()
        ax.plot(lrs, losses)
        if log_lr:
            ax.set_xscale("log")
        ax.set_xlabel("Learning Rate")
        ax.set_ylabel("Loss")
        ax.grid(True, which="both", ls="--")

        if save_path:
            fig.savefig(save_path)
            logging.info(f"LR Finder plot saved to {save_path}")
        else:
            plt.show()

        plt.close(fig)

        try:
            min_grad_idx = (np.gradient(np.array(losses))).argmin()
            suggested_lr = lrs[min_grad_idx]
            logging.info(f"==> Suggested LR (steepest gradient): {suggested_lr:.2e}")
        except (ValueError, IndexError) as e:
            logging.warning(f"Could not suggest a learning rate: {e}")


class CombinedRestorationDataset(Dataset):
    def __init__(
        self,
        clean_dir: Path,
        task_dirs: dict,
        image_size: Tuple[int, int],
        cache_limit: int = 200,
    ):
        self.image_size = image_size
        self.cache = {}
        self.cache_limit = cache_limit
        self.transform = T.Compose(
            [
                T.Resize(image_size, antialias=True),
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
            ]
        )
        self.mask_transform = T.Compose(
            [
                T.Resize(image_size, interpolation=T.InterpolationMode.NEAREST),
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
            ]
        )

        clean_paths = sorted(
            p
            for p in clean_dir.iterdir()
            if p.suffix.lower() in [".png", ".jpg", ".jpeg"]
        )
        self.task_list = []

        corruption_tasks = {
            k: v for k, v in task_dirs.items() if k not in ["clean", "mask"]
        }
        mask_dir = Path(task_dirs["mask"]) if "mask" in task_dirs else None

        for clean_path in tqdm(clean_paths, desc="Verifying Dataset Tasks"):
            for task_name, corrupted_dir_str in corruption_tasks.items():
                corrupted_path = Path(corrupted_dir_str) / clean_path.name
                if not corrupted_path.exists():
                    logging.warning(
                        f"Missing corrupted file for task '{task_name}': {corrupted_path}. Skipping."
                    )
                    continue

                task_item = {
                    "clean_path": clean_path,
                    "task": task_name,
                    "corrupted_path": corrupted_path,
                }

                if task_name == "inpainting":
                    if not mask_dir or not mask_dir.is_dir():
                        logging.warning(
                            "Inpainting task found but no valid mask directory provided. Skipping."
                        )
                        continue
                    mask_path = mask_dir / clean_path.name
                    if not mask_path.exists():
                        logging.warning(
                            f"Missing mask file for inpainting: {mask_path}. Skipping."
                        )
                        continue
                    task_item["mask_path"] = mask_path
                self.task_list.append(task_item)

        logging.info(
            f"Dataset initialized with {len(self.task_list)} tasks. "
            f"Cache limit: {cache_limit} items."
        )

    def __len__(self):
        return len(self.task_list)

    def __getitem__(self, idx):
        task_info = self.task_list[idx]
        cache_key = task_info["corrupted_path"]

        if cache_key in self.cache:
            return self.cache[cache_key]

        try:
            clean_tensor = self.transform(
                Image.open(task_info["clean_path"]).convert("RGB")
            )
            corrupted_tensor = self.transform(
                Image.open(task_info["corrupted_path"]).convert("RGB")
            )

            if task_info["task"] == "inpainting":
                mask_path = task_info.get("mask_path")
                if not mask_path:
                    raise FileNotFoundError(
                        f"No mask path found for inpainting task: {task_info['corrupted_path']}"
                    )
                mask_pil = Image.open(mask_path).convert("L")
                mask_tensor = self.mask_transform(mask_pil)
                input_mask = (mask_tensor > 0.5).float()
            else:
                input_mask = torch.zeros(1, *self.image_size)

            loss_mask = (
                input_mask.clone()
                if task_info["task"] == "inpainting"
                else torch.ones(1, *self.image_size)
            )

            model_input = torch.cat([corrupted_tensor, input_mask], dim=0)
            result = (
                (model_input * 2.0 - 1.0),
                (clean_tensor * 2.0 - 1.0),
                loss_mask,
            )

            if len(self.cache) < self.cache_limit:
                self.cache[cache_key] = result

            return result

        except Exception as e:
            logging.error(f"Error loading {task_info['corrupted_path']}: {e}")
            return (
                torch.zeros(4, *self.image_size),
                torch.zeros(3, *self.image_size),
                torch.ones(1, *self.image_size),
            )


class TestVisualizerDataset(Dataset):
    def __init__(self, image_dir: Path, mask_dir: Path, image_size: Tuple[int, int]):
        self.image_paths = sorted(
            [
                p
                for p in image_dir.iterdir()
                if p.suffix.lower() in [".png", ".jpg", ".jpeg"]
            ]
        )
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.transform = T.Compose(
            [
                T.Resize(image_size, antialias=True),
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
            ]
        )
        self.mask_transform = T.Compose(
            [
                T.Resize(image_size, interpolation=T.InterpolationMode.NEAREST),
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_dir / image_path.name

        if not mask_path.exists():
            found = False
            for ext in [".png", ".jpg", ".jpeg"]:
                alt_mask_path = self.mask_dir / (image_path.stem + ext)
                if alt_mask_path.exists():
                    mask_path = alt_mask_path
                    found = True
                    break
            if not found:
                return None

        try:
            source_pil = Image.open(image_path).convert("RGB")
            mask_pil = Image.open(mask_path).convert("L")

            source_tensor = self.transform(source_pil)
            mask_tensor = self.mask_transform(mask_pil)

            input_mask = (mask_tensor > 0.5).float()
            model_input = torch.cat([source_tensor, input_mask], dim=0)

            return model_input * 2.0 - 1.0
        except Exception as e:
            logging.error(f"Error loading test image {image_path}: {e}")
            return None


class SRDataset(Dataset):
    def __init__(
        self, hr_dir: Path, lr_patch_size: Tuple[int, int], upscale_factor: int
    ):
        self.hr_paths = sorted(
            p for p in hr_dir.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]
        )
        self.lr_patch_size = lr_patch_size
        self.hr_patch_size = (
            lr_patch_size[0] * upscale_factor,
            lr_patch_size[1] * upscale_factor,
        )

        self.lr_transform = T.Compose(
            [
                T.Resize(
                    self.lr_patch_size,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                T.ToTensor(),
            ]
        )
        self.hr_transform = T.ToTensor()

    def __len__(self):
        return len(self.hr_paths)

    def __getitem__(self, idx):
        hr_img = Image.open(self.hr_paths[idx]).convert("RGB")
        i, j, h, w = transforms.RandomCrop.get_params(
            hr_img, output_size=self.hr_patch_size
        )
        hr_patch = transforms.functional.crop(hr_img, i, j, h, w)
        return (self.lr_transform(hr_patch) * 2 - 1), (
            self.hr_transform(hr_patch) * 2 - 1
        )


class Trainer:
    def __init__(self, config: dict):
        self.config = config
        self.training_type = config.get("training_mode")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logging.getLogger(self.__class__.__name__)
        self.use_channels_last = (
            self.config.get("use_channels_last", False) and self.device.type == "cuda"
        )

        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

        self._setup_directories()
        self._initialize_training_components()
        self.setup_progressive_training()
        self._load_checkpoint()
        self._check_gpu_temp()

    def setup_progressive_training(self):
        self.use_progressive_training = self.config.get(
            "use_progressive_training", False
        )
        if self.use_progressive_training:
            self.progressive_phases = self.config["progressive_phases"]
            self.current_phase_idx = -1
            self.logger.info("Progressive training is enabled.")
            self._update_progressive_phase(0)

    def _setup_directories(self):
        self.checkpoint_dir = Path(self.config["checkpoint_dir"])
        self.preview_dir = Path(self.config["preview_dir"])
        self.tensorboard_dir = self.checkpoint_dir.parent / "runs"

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.preview_dir.mkdir(parents=True, exist_ok=True)
        self.tensorboard_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(
            "Checkpoint, preview, and TensorBoard directories initialized."
        )

    def _verify_directories(self, dirs_to_check=None):
        if dirs_to_check is None:
            dirs_to_check = self.config["data_dirs"]

        try:
            for name, path_or_dict in dirs_to_check.items():
                if isinstance(path_or_dict, dict):
                    self._verify_directories(path_or_dict)
                else:
                    path = Path(path_or_dict)
                    if not path.is_dir():
                        raise FileNotFoundError(f"Directory '{name}' not found: {path}")
                    if not any(path.iterdir()):
                        self.logger.warning(f"Directory '{name}' is empty: {path}")
        except Exception as e:
            self.logger.error(f"Error verifying directories: {e}")
            raise

    def _initialize_training_components(self):
        self.logger.info(f"Using device: {self.device}")
        self.writer = SummaryWriter(log_dir=self.tensorboard_dir)

        if self.training_type == "restoration":
            self.logger.info("Initializing restoration model...")
            self._verify_directories()
            self.logger.info("All data directories verified.")
            image_size = (self.config["img_height"], self.config["img_width"])

            dataset = CombinedRestorationDataset(
                clean_dir=Path(self.config["data_dirs"]["train"]["clean"]),
                task_dirs=self.config["data_dirs"]["train"],
                image_size=image_size,
                cache_limit=self.config.get("cache_limit", 200),
            )
            self.model = UNetLite(
                in_channels=4,
                out_channels=3,
                base_channels=self.config.get("base_channels", 16),
                use_checkpointing=self.config.get("use_checkpointing", True),
            ).to(self.device)

            if self.use_channels_last:
                self.model.to(memory_format=torch.channels_last)
                self.logger.info("Model converted to channels_last memory format.")
            val_dataloader_params = self.config["dataloader_params"].copy()
            val_dataloader_params["shuffle"] = False

            val_dataset = CombinedRestorationDataset(
                clean_dir=Path(self.config["data_dirs"]["validation"]["clean"]),
                task_dirs=self.config["data_dirs"]["validation"],
                image_size=image_size,
                cache_limit=self.config.get("cache_limit", 50),
            )
            self.val_dataloader = DataLoader(val_dataset, **val_dataloader_params)

            test_img_dir = Path(self.config["sample_images"])
            test_mask_dir = Path(self.config["sample_masks"])
            if test_img_dir.is_dir() and test_mask_dir.is_dir():
                self.logger.info("Initializing test dataloader for visualization.")
                test_dataset = TestVisualizerDataset(
                    image_dir=test_img_dir,
                    mask_dir=test_mask_dir,
                    image_size=image_size,
                )
                if len(test_dataset) > 0:
                    test_dl_params = self.config["dataloader_params"].copy()
                    test_dl_params["shuffle"] = False
                    test_dl_params["batch_size"] = min(
                        4, test_dl_params.get("batch_size", 4)
                    )
                    self.test_dataloader = DataLoader(test_dataset, **test_dl_params)
                else:
                    self.test_dataloader = None
                    self.logger.warning("Test dataset is empty.")
            else:
                self.test_dataloader = None
                self.logger.warning("Samples directories not found for test viz.")

        elif self.training_type == "super_resolution":
            self.logger.info("Initializing super-resolution model...")
            self._verify_directories({"hr_data_dir": self.config["hr_data_dir"]})
            self.logger.info("All data directories verified.")

            dataset = SRDataset(
                hr_dir=Path(self.config["hr_data_dir"]),
                lr_patch_size=(
                    self.config["lr_patch_height"],
                    self.config["lr_patch_width"],
                ),
                upscale_factor=self.config["upscale_factor"],
            )
            self.model = SuperResolutionNet(
                num_res_blocks=self.config.get("num_res_blocks", 16),
                upscale_factor=self.config["upscale_factor"],
            ).to(self.device)
            # Note: Validation dataloader for SR is not implemented
            self.val_dataloader = None

        else:
            raise ValueError(f"Unknown training type: {self.training_type}")

        self.dataloader = DataLoader(dataset, **self.config["dataloader_params"])
        self.logger.info(
            f"Model initialized with "
            f"{sum(p.numel() for p in self.model.parameters()):,} trainable parameters."
        )

        if hasattr(torch, "compile"):
            self.model = torch.compile(
                self.model, mode=self.config.get("compile_mode", "reduce-overhead")
            )
            self.logger.info("Model compiled with torch.compile()")

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config.get("weight_decay", 1e-3),
        )

        self.use_gan = self.config.get("use_gan", False)
        if self.use_gan:
            self.logger.info("GAN training is enabled.")
            self.discriminator = PatchDiscriminator(in_channels=3).to(self.device)
            if self.use_channels_last:
                self.discriminator.to(memory_format=torch.channels_last)
            self.optimizer_D = optim.AdamW(
                self.discriminator.parameters(),
                lr=self.config.get("discriminator_lr", self.config["learning_rate"]),
                betas=(0.5, 0.999),
                weight_decay=self.config.get("weight_decay", 1e-3),
            )
            self.criterion_gan = nn.BCEWithLogitsLoss()
            if hasattr(torch, "compile"):
                self.discriminator = torch.compile(
                    self.discriminator,
                    mode=self.config.get("compile_mode", "reduce-overhead"),
                )
                self.logger.info("Discriminator compiled with torch.compile()")
        else:
            self.discriminator = None
            self.optimizer_D = None
            self.criterion_gan = None

        scheduler_type = self.config.get("scheduler", "plateau")
        if scheduler_type == "onecycle":
            onecycle_cfg = self.config.get("onecycle_params", {})
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.config["learning_rate"],
                steps_per_epoch=len(self.dataloader),
                epochs=self.config["num_epochs"],
                pct_start=onecycle_cfg.get("pct_start", 0.3),
                div_factor=onecycle_cfg.get("div_factor", 25),
                final_div_factor=onecycle_cfg.get("final_div_factor", 1e4),
                three_phase=onecycle_cfg.get("three_phase", False),
                anneal_strategy=onecycle_cfg.get("anneal_strategy", "cos"),
            )
            self.logger.info("Using OneCycleLR scheduler.")
            self.logger.info(
                f"Scheduler initialized. Total steps: {self.scheduler.total_steps}"
            )
        else:
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, mode="max", factor=0.5, patience=5
            )
            self.logger.info("Using ReduceLROnPlateau scheduler.")
        self.criterion_l1 = nn.L1Loss()
        self.criterion_perceptual = PerceptualLoss().to(self.device)
        self.criterion_edge = EdgeLoss(self.device)
        self.criterion_multiscale = MultiScaleLoss().to(self.device)
        self.criterion_frequency = FrequencyLoss().to(self.device)
        self.criterion_ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(
            self.device
        )
        if lpips:
            import warnings

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=UserWarning,
                    message="The parameter 'pretrained' is deprecated",
                )
                warnings.filterwarnings(
                    "ignore",
                    category=UserWarning,
                    message="Arguments other than a weight enum or `None` for 'weights' are deprecated",
                )
                warnings.filterwarnings(
                    "ignore",
                    category=FutureWarning,
                    message="You are using `torch.load` with `weights_only=False`",
                )
                self.lpips_model = lpips.LPIPS(net="vgg").to(self.device).eval()
        else:
            self.lpips_model = None

        self.use_amp = self.config.get("use_amp", True) and self.device.type == "cuda"
        self.amp_dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16
        )
        self.scaler = GradScaler(
            enabled=self.use_amp, init_scale=2.0**16, growth_interval=1000
        )

        self.logger.info(f"AMP {'enabled' if self.use_amp else 'disabled'}.")

        self.ema = None
        if self.config.get("use_ema", False):
            self.ema = ModelEMA(self.model, decay=self.config.get("ema_decay", 0.999))
            self.logger.info(
                f"Exponential Moving Average (EMA) enabled with decay {self.ema.decay}."
            )

    def _update_progressive_phase(self, new_phase_idx):
        if not self.use_progressive_training or new_phase_idx == self.current_phase_idx:
            return

        self.current_phase_idx = new_phase_idx
        phase_config = self.progressive_phases[self.current_phase_idx]
        new_size = phase_config["size"]
        lr_mult = phase_config["lr_mult"]

        for dl in [self.dataloader, self.val_dataloader]:
            if dl and hasattr(dl.dataset, "image_size"):
                dl.dataset.image_size = new_size
                dl.dataset.transform.transforms[0] = T.Resize(new_size, antialias=True)
                dl.dataset.mask_transform.transforms[0] = T.Resize(
                    new_size, interpolation=T.InterpolationMode.NEAREST
                )
                dl.dataset.cache.clear()

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.config["learning_rate"] * lr_mult

        self.logger.info(
            f"Progressing to phase {self.current_phase_idx + 1}: size={new_size}, lr_mult={lr_mult}"
        )

    def _load_checkpoint(self):
        self.start_epoch = 0
        self.best_ssim = 0.0
        checkpoint_path = self.checkpoint_dir / "last_checkpoint.pth"

        if not checkpoint_path.exists():
            self.logger.info("No checkpoint found, starting fresh.")
            return

        try:
            checkpoint = torch.load(
                checkpoint_path, map_location=self.device, weights_only=False
            )
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            if self.use_amp:
                self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

            if self.use_gan and "discriminator_state_dict" in checkpoint:
                self.discriminator.load_state_dict(
                    checkpoint["discriminator_state_dict"]
                )
                if "optimizer_D_state_dict" in checkpoint:
                    self.optimizer_D.load_state_dict(
                        checkpoint["optimizer_D_state_dict"]
                    )

            checkpoint_config_lr = checkpoint.get("config_learning_rate")
            current_config_lr = self.config["learning_rate"]
            reset_scheduler = False

            if checkpoint_config_lr is not None:
                if not math.isclose(
                    checkpoint_config_lr, current_config_lr, rel_tol=1e-9
                ):
                    self.logger.warning(
                        f"Config LR changed from {checkpoint_config_lr:.2e} to {current_config_lr:.2e}. "
                        "Resetting scheduler and updating optimizer."
                    )
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = current_config_lr
                    reset_scheduler = True
            else:
                self.logger.warning(
                    "Loading an old checkpoint format. Scheduler will be reset. "
                    "Optimizer LR is updated to the current config value."
                )
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = current_config_lr
                reset_scheduler = True

            if not reset_scheduler and "scheduler_state_dict" in checkpoint:
                try:
                    self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                    self.logger.info("Resumed scheduler state from checkpoint.")
                except Exception as e:
                    self.logger.warning(
                        f"Could not load scheduler state: {e}. Reinitializing."
                    )
                    reset_scheduler = True
            else:
                self.logger.info("Scheduler state is being re-initialized.")

            self.start_epoch = checkpoint["epoch"] + 1
            self.best_ssim = checkpoint.get("best_ssim", 0.0)
            if self.ema and "ema_state_dict" in checkpoint:
                self.ema.shadow = checkpoint["ema_state_dict"]
                self.logger.info("Resumed EMA state from checkpoint.")
            self.logger.info(
                f"Resumed from epoch {self.start_epoch} "
                f"(best SSIM: {self.best_ssim:.5f})"
            )

        except Exception as e:
            self.logger.error(f"Error loading checkpoint: {e}. Starting fresh.")

    def _denorm_for_viz(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=-1.0)
        return (tensor.clamp(-1, 1) + 1) / 2

    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_ssim": self.best_ssim,
            "config_learning_rate": self.config["learning_rate"],
        }
        if self.ema:
            state["ema_state_dict"] = self.ema.shadow
        if self.use_gan:
            state["discriminator_state_dict"] = self.discriminator.state_dict()
            state["optimizer_D_state_dict"] = self.optimizer_D.state_dict()

        torch.save(state, self.checkpoint_dir / "last_checkpoint.pth")
        torch.save(self.model.state_dict(), self.checkpoint_dir / "model.pth")

        if is_best:
            if self.ema:
                self.ema.apply_shadow()
                torch.save(
                    self.model.state_dict(), self.checkpoint_dir / "best_model.pth"
                )
                self.ema.restore()
            else:
                torch.save(
                    self.model.state_dict(), self.checkpoint_dir / "best_model.pth"
                )
            self.logger.info(f"New best model saved (SSIM: {self.best_ssim:.5f})")

    def adaptive_loss_weights(self, epoch):
        p_weight = max(
            0.05, self.config.get("perceptual_weight", 0.1) * (0.95 ** (epoch / 10))
        )
        ssim_weight = self.config.get("ssim_weight", 0.1)
        edge_weight = self.config.get("edge_weight", 0.0)
        multiscale_weight = self.config.get("multiscale_loss_weight", 0.0)
        frequency_weight = self.config.get("frequency_loss_weight", 0.0)
        return (
            p_weight,
            ssim_weight,
            edge_weight,
            multiscale_weight,
            frequency_weight,
        )

    def adaptive_gradient_clipping(self, model, max_norm_base=1.0, epoch=0):
        factor = max(0.1, 1.0 - (epoch / self.config["num_epochs"]) * 0.9)
        max_norm = max_norm_base * factor
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        if self.use_gan:
            self.discriminator.train()
        total_loss_epoch = 0.0
        accum_steps = self.config.get("grad_accum_steps", 1)
        (
            p_weight,
            ssim_weight,
            edge_weight,
            multiscale_weight,
            frequency_weight,
        ) = self.adaptive_loss_weights(epoch)
        gan_weight = self.config.get("gan_weight", 0.0)

        pbar = tqdm(
            self.dataloader,
            desc=f"Epoch {epoch+1}/{self.config['num_epochs']}",
            dynamic_ncols=True,
            leave=False,
        )

        for i, (inputs, targets, masks) in enumerate(pbar):
            if self.device.type == "cuda" and hasattr(
                torch.compiler, "cudagraph_mark_step_begin"
            ):
                torch.compiler.cudagraph_mark_step_begin()

            inputs, targets, masks = (
                inputs.to(self.device, non_blocking=True),
                targets.to(self.device, non_blocking=True),
                masks.to(self.device, non_blocking=True),
            )
            if self.use_channels_last:
                inputs = inputs.to(memory_format=torch.channels_last)
                targets = targets.to(memory_format=torch.channels_last)

            if self.use_gan:
                with torch.no_grad():
                    with autocast(device_type=self.device.type, enabled=self.use_amp):
                        fake_raw, *_ = self.model(inputs)
                        fake_composite = fake_raw * masks + targets * (1 - masks)

                with autocast(device_type=self.device.type, enabled=self.use_amp):
                    pred_real = self.discriminator(targets)
                    loss_D_real = self.criterion_gan(
                        pred_real, torch.ones_like(pred_real)
                    )

                loss_D_real_scaled = loss_D_real / accum_steps
                self.scaler.scale(loss_D_real_scaled).backward()

                with autocast(device_type=self.device.type, enabled=self.use_amp):
                    pred_fake = self.discriminator(fake_composite.detach())
                    loss_D_fake = self.criterion_gan(
                        pred_fake, torch.zeros_like(pred_fake)
                    )

                loss_D_fake_scaled = loss_D_fake / accum_steps
                self.scaler.scale(loss_D_fake_scaled).backward()
                del pred_fake, fake_raw, fake_composite

                loss_D = (loss_D_real + loss_D_fake) * 0.5

            with autocast(device_type=self.device.type, enabled=self.use_amp):
                if not self.use_gan:
                    final_out_raw, aux_out1, aux_out2, aux_out3 = self.model(inputs)
                else:
                    final_out_raw, aux_out1, aux_out2, aux_out3 = self.model(inputs)

                final_out_raw = torch.clamp(final_out_raw, -1.0, 1.0)
                final_outputs = final_out_raw * masks + targets * (1 - masks)

                loss_l1_main = self.criterion_l1(final_outputs, targets)
                loss_p_main = self.criterion_perceptual(final_outputs, targets)
                loss_edge = self.criterion_edge(final_outputs, targets)
                loss_multiscale = self.criterion_multiscale(final_outputs, targets)
                loss_freq = self.criterion_frequency(final_outputs, targets)

                final_outputs_denorm = (final_outputs.clamp(-1, 1) + 1) / 2
                targets_denorm = (targets.clamp(-1, 1) + 1) / 2
                loss_ssim_main = 1.0 - self.criterion_ssim(
                    final_outputs_denorm, targets_denorm
                )

                main_loss = (
                    loss_l1_main
                    + p_weight * loss_p_main
                    + ssim_weight * loss_ssim_main
                    + edge_weight * loss_edge
                    + multiscale_weight * loss_multiscale
                    + frequency_weight * loss_freq
                )

                aux_losses = []  #
                aux_outputs = [aux_out1, aux_out2, aux_out3]
                aux_weights = [0.1, 0.2, 0.3]

                for aux_idx, aux_out in enumerate(aux_outputs):
                    target_down = F.interpolate(
                        targets, size=aux_out.shape[2:], mode="area"
                    )
                    mask_down = F.interpolate(
                        masks, size=aux_out.shape[2:], mode="nearest"
                    )

                    final_aux_out = aux_out * mask_down + target_down * (1 - mask_down)

                    aux_losses.append(self.criterion_l1(final_aux_out, target_down))

                total_aux_loss = sum(w * l for w, l in zip(aux_weights, aux_losses))

                loss_G_gan = 0.0
                if self.use_gan:
                    pred_fake_for_G = self.discriminator(final_out_raw)
                    loss_G_gan = self.criterion_gan(
                        pred_fake_for_G, torch.ones_like(pred_fake_for_G)
                    )

                total_loss_batch = main_loss + total_aux_loss + gan_weight * loss_G_gan

            if not torch.isfinite(total_loss_batch):
                if (i + 1) % accum_steps == 0 or (i + 1) == len(self.dataloader):
                    self.optimizer.zero_grad(set_to_none=True)

                    if i == 0:
                        self._save_enhanced_preview(
                            inputs, final_out_raw, targets, epoch
                        )
                        self._cleanup_previews(keep=10)
                continue

            step = epoch * len(self.dataloader) + i
            self.writer.add_scalar("Loss/train_batch", total_loss_batch.item(), step)
            self.writer.add_scalar("Loss/train_l1_main", loss_l1_main.item(), step)
            self.writer.add_scalar(
                "Loss/train_perceptual_main", loss_p_main.item(), step
            )
            self.writer.add_scalar("Loss/train_ssim_main", loss_ssim_main.item(), step)
            self.writer.add_scalar("Loss/train_edge", loss_edge.item(), step)
            self.writer.add_scalar(
                "Loss/train_multiscale", loss_multiscale.item(), step
            )
            self.writer.add_scalar("Loss/train_frequency", loss_freq.item(), step)
            if self.use_gan:
                self.writer.add_scalar("Loss/train_G_gan", loss_G_gan.item(), step)
                self.writer.add_scalar("Loss/train_D_total", loss_D.item(), step)
                self.writer.add_scalar("Loss/train_D_real", loss_D_real.item(), step)
                self.writer.add_scalar("Loss/train_D_fake", loss_D_fake.item(), step)
            loss_to_backward = total_loss_batch / accum_steps
            self.scaler.scale(loss_to_backward).backward()

            if (i + 1) % accum_steps == 0 or (i + 1) == len(self.dataloader):
                old_scale = self.scaler.get_scale()

                if self.use_gan:
                    self.scaler.unscale_(self.optimizer_D)
                    self.adaptive_gradient_clipping(self.discriminator, epoch=epoch)
                    self.scaler.step(self.optimizer_D)
                    self.optimizer_D.zero_grad(set_to_none=True)

                self.scaler.unscale_(self.optimizer)
                self.adaptive_gradient_clipping(self.model, epoch=epoch)
                self.scaler.step(self.optimizer)
                self.optimizer.zero_grad(set_to_none=True)

                self.scaler.update()

                if self.scaler.get_scale() >= old_scale:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR):
                        self.scheduler.step()
                        self.writer.add_scalar(
                            "LR/learning_rate_step",
                            self.optimizer.param_groups[0]["lr"],
                            step,
                        )

                if self.ema:
                    self.ema.update()

            total_loss_epoch += total_loss_batch.item()
            pbar.set_postfix(
                loss=f"{total_loss_batch.item():.5f}",
                l1=f"{loss_l1_main.item():.5f}",
                p=f"{loss_p_main.item():.5f}",
            )

            del (
                total_loss_batch,
                loss_to_backward,
                main_loss,
                total_aux_loss,
                loss_multiscale,
                loss_freq,
            )
            if self.use_gan:
                del loss_G_gan, loss_D

            if i == 0:
                self._save_enhanced_preview(inputs, final_out_raw, targets, epoch)
                self._cleanup_previews(keep=10)

            interval = self.config.get("checkpoint_interval_steps", 0)
            if interval > 0 and step > 0 and step % interval == 0:
                self.logger.info(f"Saving granular checkpoint at step {step}")
                self._save_checkpoint(epoch)

            if i % 20 == 0:
                self._check_gpu_temp()

        avg_epoch_loss = total_loss_epoch / len(self.dataloader)
        self.writer.add_scalar("Loss/train_epoch", avg_epoch_loss, epoch)
        self.writer.add_scalar(
            "LR/learning_rate", self.optimizer.param_groups[0]["lr"], epoch
        )
        return avg_epoch_loss

    def _validate_epoch(self, epoch: int) -> dict:
        if not self.val_dataloader:
            self.logger.warning("No validation dataloader found, skipping validation.")
            return {"ssim": 0.0, "psnr": 0.0, "lpips": 0.0}

        if self.ema:
            self.ema.apply_shadow()

        self.model.eval()
        metrics = {"ssim": 0.0, "psnr": 0.0, "lpips": 0.0}

        pbar = tqdm(
            self.val_dataloader,
            desc=f"Validating Epoch {epoch+1}",
            dynamic_ncols=True,
            leave=False,
        )

        with torch.no_grad():
            for i, (inputs, targets, masks) in enumerate(pbar):
                inputs, targets, masks = (
                    inputs.to(self.device, non_blocking=True),
                    targets.to(self.device, non_blocking=True),
                    masks.to(self.device, non_blocking=True),
                )
                if self.use_channels_last:
                    inputs = inputs.to(memory_format=torch.channels_last)
                    targets = targets.to(memory_format=torch.channels_last)

                with autocast(device_type=self.device.type, enabled=self.use_amp):
                    final_out_raw, *_ = self.model(inputs)
                    final_out_raw = torch.clamp(final_out_raw, -1.0, 1.0)
                    final_outputs = final_out_raw * masks + targets * (1 - masks)

                final_outputs_denorm = (final_outputs.clamp(-1, 1) + 1) / 2
                targets_denorm = (targets.clamp(-1, 1) + 1) / 2

                ssim_val = self.criterion_ssim(
                    final_outputs_denorm.float(), targets_denorm.float()
                )
                mse = F.mse_loss(final_outputs_denorm, targets_denorm)

                if not torch.isfinite(ssim_val):
                    continue

                metrics["ssim"] += ssim_val.item()
                if mse > 0:
                    metrics["psnr"] += 10 * torch.log10(1.0 / mse).item()

                if self.lpips_model:
                    metrics["lpips"] += (
                        self.lpips_model(final_outputs, targets).mean().item()
                    )

                pbar.set_postfix(
                    val_ssim=f"{ssim_val.item():.4f}",
                    val_psnr=f"{metrics['psnr'] / (i+1):.2f}",
                )

                if i == 0:
                    input_grid = make_grid(self._denorm_for_viz(inputs[:, :3].cpu()))
                    output_grid = make_grid(self._denorm_for_viz(final_outputs.cpu()))
                    target_grid = make_grid(self._denorm_for_viz(targets.cpu()))
                    self.writer.add_image("Previews/val_input", input_grid, epoch)
                    self.writer.add_image("Previews/val_output", output_grid, epoch)
                    self.writer.add_image("Previews/val_target", target_grid, epoch)

        for k in metrics:
            metrics[k] /= len(self.val_dataloader)
            self.writer.add_scalar(f"Val/{k.upper()}", metrics[k], epoch)

        if self.ema:
            self.ema.restore()

        return metrics

    def _test_on_samples(self, epoch: int):
        if not hasattr(self, "test_dataloader") or not self.test_dataloader:
            return

        if self.ema:
            self.ema.apply_shadow()

        self.model.eval()
        with torch.no_grad():
            data = next(iter(self.test_dataloader), None)
            if data is None:
                if self.ema:
                    self.ema.restore()
                return

            inputs = data
            inputs = inputs.to(self.device, non_blocking=True)
            if self.use_channels_last:
                inputs = inputs.to(memory_format=torch.channels_last)

            with autocast(device_type=self.device.type, enabled=self.use_amp):
                final_out_raw, *_ = self.model(inputs)

            mask = inputs[:, 3:4, :, :]
            corrupted_image = inputs[:, :3, :, :]
            final_outputs = final_out_raw * mask + corrupted_image * (1 - mask)

            combined_batch = torch.cat([corrupted_image, final_outputs], dim=0)
            denormalized_batch = self._denorm_for_viz(combined_batch.cpu())

            grid = make_grid(
                denormalized_batch,
                nrow=inputs.size(0),
            )
            self.writer.add_image("Previews/test_samples_comparison", grid, epoch)

        if self.ema:
            self.ema.restore()

    def train(self):
        self.logger.info(f"Starting training on {self.device}...")
        start_time = time.time()

        for epoch in range(self.start_epoch, self.config["num_epochs"]):
            if self.use_progressive_training:
                total_phase_epochs = sum(
                    p["epochs"]
                    for p in self.progressive_phases[: self.current_phase_idx + 1]
                )
                if (
                    epoch >= total_phase_epochs
                    and self.current_phase_idx < len(self.progressive_phases) - 1
                ):
                    self._update_progressive_phase(self.current_phase_idx + 1)

            epoch_loss = self._train_epoch(epoch)
            val_metrics = self._validate_epoch(epoch)
            self._test_on_samples(epoch)

            lr = self.optimizer.param_groups[0]["lr"]
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_metrics["ssim"])

            is_best = val_metrics["ssim"] > self.best_ssim
            if is_best:
                self.best_ssim = val_metrics["ssim"]

            self._save_checkpoint(epoch, is_best)
            self.logger.info(
                f"Epoch {epoch+1:03d} | Loss: {epoch_loss:.4f} | SSIM: {val_metrics['ssim']:.4f} | PSNR: {val_metrics['psnr']:.2f} | LPIPS: {val_metrics['lpips']:.4f} | Best SSIM: {self.best_ssim:.4f} | LR: {lr:.2e}"
            )
            self._check_gpu_temp()

        self.writer.close()
        self.logger.info(
            f"Training completed in {(time.time() - start_time)/60:.2f} minutes."
        )

    def _save_enhanced_preview(self, inputs, outputs, targets, epoch, batch_idx=0):
        preview_dir = self.preview_dir

        with torch.no_grad():
            input_preview = self._denorm_for_viz(inputs[batch_idx, :3].cpu())
            output_preview = self._denorm_for_viz(outputs[batch_idx].cpu())
            target_preview = self._denorm_for_viz(targets[batch_idx].cpu())

            error_map = torch.abs(output_preview - target_preview)
            error_map = (error_map - error_map.min()) / (
                error_map.max() - error_map.min() + 1e-8
            )

            if batch_idx == 0:
                psnr = 10 * torch.log10(
                    1.0 / F.mse_loss(output_preview, target_preview)
                )
                ssim = self.criterion_ssim(
                    output_preview.unsqueeze(0).float(),
                    target_preview.unsqueeze(0).float(),
                )

                self.writer.add_scalar("Preview/PSNR", psnr.item(), epoch)
                self.writer.add_scalar("Preview/SSIM", ssim.item(), epoch)

            if input_preview.shape != output_preview.shape:
                input_preview = F.interpolate(
                    input_preview.unsqueeze(0),
                    size=output_preview.shape[1:],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze(0)

            grid = make_grid(
                [input_preview, output_preview, target_preview, error_map], nrow=4
            )
            save_image(grid, preview_dir / f"preview_{epoch+1:04d}.png")

    def _cleanup_previews(self, keep=10):
        preview_dir = self.preview_dir
        previews = sorted(
            [
                p
                for p in preview_dir.glob("preview_*.png")
                if p.stem.replace("preview_", "").isdigit()
            ],
            key=lambda p: int(p.stem.replace("preview_", "")),
        )

        if len(previews) > keep:
            for f in previews[:-keep]:
                try:
                    f.unlink()
                except OSError as e:
                    self.logger.error(f"Error deleting preview: {f.name}: {e}")

    def _check_gpu_temp(self, threshold=85, delay=15):
        if not getGPUs or self.device.type != "cuda":
            return

        try:
            gpu = getGPUs()[0]
            temperature = gpu.temperature + 2.0
            if temperature >= threshold:
                time.sleep(delay)
        except Exception as e:
            self.logger.error(f"GPU temp check failed: {e}")
