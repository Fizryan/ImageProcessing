# Train.py
# This module implements the training pipeline for the image restoration model using PyTorch.

import logging
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from torch.optim.lr_scheduler import ReduceLROnPlateau
from PIL import Image
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
import time

from program.Architecture import AdvancedUNet

try:
    import GPUtil
except ImportError:
    GPUtil = None
    logging.warning("GPUtil not found. GPU temperature monitoring is disabled. Install with 'pip install gputil'")

class CombinedRestorationDataset(Dataset):
    def __init__(self, clean_dir: Path, noise_dir: Path, mosaic_dir: Path, inpainting_dir: Path, image_size: Tuple[int, int]):
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor()
        ])

        clean_image_paths = sorted([p for p in clean_dir.iterdir() if p.suffix.lower() in ['.png', '.jpg', '.jpeg']])
        self.task_list = []

        for path in clean_image_paths:
            self.task_list.append({'clean_path': path, 'task': 'noise', 'corrupted_path': noise_dir / path.name})
            self.task_list.append({'clean_path': path, 'task': 'mosaic', 'corrupted_path': mosaic_dir / path.name})
            self.task_list.append({'clean_path': path, 'task': 'inpainting', 'corrupted_path': inpainting_dir / path.name})
        logging.info(f"Dataset initialized with {len(self.task_list)} total tasks.")

    def __len__(self):
        return len(self.task_list)

    def __getitem__(self, idx):
        task_info = self.task_list[idx]
        try:
            clean_tensor = self.transform(Image.open(task_info['clean_path']).convert('RGB'))
            corrupted_tensor = self.transform(Image.open(task_info['corrupted_path']).convert('RGB'))
        except FileNotFoundError as e:
            logging.error(f"File not found: {e}. Returning zeros.")
            return torch.zeros(4, *self.image_size), torch.zeros(3, *self.image_size)
        
        mask = (corrupted_tensor.sum(dim=0) == 0).float().unsqueeze(0) if task_info['task'] == 'inpainting' else torch.zeros(1, *self.image_size)
        model_input = torch.cat([corrupted_tensor, mask], dim=0)
        return (model_input * 2.0 - 1.0), (clean_tensor * 2.0 - 1.0)

class Trainer:
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logging.getLogger(__name__)
        self._setup_helpers()
        self._load_checkpoint()
        self._prepare_directory()
        self._check_directories()
        if GPUtil:
            self._check_gpu_temp()

    def _prepare_directory(self):
        Path(self.config['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
        Path(self.config['preview_dir']).mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Checkpoint and preview directories prepared.")

    def _check_directories(self):
        for key, path in self.config['data_dirs'].items():
            if not Path(path).is_dir():
                raise FileNotFoundError(f"Directory for {key} not found: {path}")
        self.logger.info("All required data directories are present.")

    def _setup_helpers(self):
        self.logger.info(f"Using device: {self.device}")
        image_size = (self.config['img_height'], self.config['img_width'])
        dataset = CombinedRestorationDataset(
            clean_dir=Path(self.config['data_dirs']['clean']),
            noise_dir=Path(self.config['data_dirs']['noise']),
            mosaic_dir=Path(self.config['data_dirs']['mosaic']),
            inpainting_dir=Path(self.config['data_dirs']['inpainting']),
            image_size=image_size
        )
        self.dataloader = DataLoader(dataset, **self.config['dataloader_params'])
        model = AdvancedUNet(in_channels=4, out_channels=3)
        self.logger.info(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} trainable parameters.")

        if hasattr(torch, 'compile'):
            self.model = torch.compile(model, mode=self.config.get('compile_mode', 'default')).to(self.device)
            self.logger.info(f"Model compiled with mode: '{self.config.get('compile_mode', 'default')}'")
        else:
            self.model = model.to(self.device)
            self.logger.warning("torch.compile not available.")
        
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.config['learning_rate'])
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', factor=0.5, patience=3)
        self.criterion = nn.L1Loss()
        self.use_amp = 'cuda' in self.device.type
        self.scaler = GradScaler(enabled=self.use_amp)

    def _load_checkpoint(self):
        self.start_epoch = 0
        self.best_loss = float('inf')
        chkpt_path = Path(self.config['checkpoint_dir']) / 'last_checkpoint.pth'
        
        if chkpt_path.exists():
            self.logger.info(f"Loading checkpoint from {chkpt_path}...")
            checkpoint = torch.load(chkpt_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            if self.use_amp and 'scaler_state_dict' in checkpoint:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

            if 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                self.logger.info("Scheduler state loaded from checkpoint.")

            self.start_epoch = checkpoint.get('epoch', 0) + 1
            self.best_loss = checkpoint.get('best_loss', float('inf'))
            self.logger.info(f"Resuming from epoch {self.start_epoch}, best loss: {self.best_loss:.5f}")

    def _save_checkpoint(self, epoch, is_best=False):
        chkpt_dir = Path(self.config['checkpoint_dir'])
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'scaler_state_dict': self.scaler.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
        }
        torch.save(state, chkpt_dir / 'last_checkpoint.pth')
        if is_best:
            torch.save(self.model.state_dict(), chkpt_dir / 'best_model.pth')
            self.logger.info(f"New best model saved with loss {self.best_loss:.5f}")

    def _run_one_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        pbar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{self.config['num_epochs']}", leave=False, ncols=100)
        for i, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)
            
            with autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
            
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            running_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.5f}"})
            if i == 0: 
                self._save_preview(inputs, outputs, targets, epoch)
                self._cleanup_previews(keep=10)
        return running_loss / len(self.dataloader)

    def train(self):
        self.logger.info("Starting training...")
        for epoch in range(self.start_epoch, self.config['num_epochs']):
            epoch_loss = self._run_one_epoch(epoch)
            current_lr = self.optimizer.param_groups[0]['lr']
            self.logger.info(f"Epoch {epoch+1} complete. Average Loss: {epoch_loss:.5f} | Current LR: {current_lr:.1e}")
            self.scheduler.step(epoch_loss)
            
            is_best = epoch_loss < self.best_loss
            if is_best:
                self.best_loss = epoch_loss
            
            self._save_checkpoint(epoch, is_best=is_best)
            self._check_gpu_temp()
        self.logger.info("Training finished.")

    def _save_preview(self, inputs, outputs, clean_targets, epoch):
        def denorm(t): return (t.clamp(-1, 1) + 1) / 2
        preview_dir = Path(self.config['preview_dir']); preview_dir.mkdir(exist_ok=True)
        with torch.no_grad():
            grid = make_grid([denorm(inputs[0, :3, :, :].cpu()), denorm(outputs[0].cpu()), denorm(clean_targets[0].cpu())], nrow=3)
            save_image(grid, preview_dir / f'epoch_{epoch+1:04d}.png')

    def _cleanup_previews(self, keep=10):
        preview_dir = Path(self.config['preview_dir'])
        previews = sorted(
            [p for p in preview_dir.glob("epoch_*.png") if p.stem.replace("epoch_", "").isdigit()],
            key=lambda p: int(p.stem.replace("epoch_", ""))
        )
        if len(previews) > keep:
            for f in previews[:-keep]:
                try: f.unlink(); self.logger.debug(f"Deleted old preview: {f.name}")
                except OSError as e: self.logger.error(f"Error deleting preview file {f.name}: {e}")

    def _check_gpu_temp(self):
        if GPUtil is None: return
        try:
            gpu = GPUtil.getGPUs()[0]
            if gpu.temperature > self.config['max_gpu_temp']:
                self.logger.warning(f"GPU temp at {gpu.temperature}°C. Pausing for 30s...")
                time.sleep(30)
        except Exception: pass