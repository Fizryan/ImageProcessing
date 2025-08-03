# Training.py
# This module contains the training logic for the image restoration model.

import logging
from pathlib import Path
from typing import Tuple
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from tqdm.auto import tqdm
import os
from PIL import Image

try:
    from GPUtil import getGPUs
except ImportError:
    getGPUs = None
    logging.warning("GPUtil not available. GPU temp monitoring disabled.")

from program.Architecture import AdvancedUNet

class CombinedRestorationDataset(Dataset):
    def __init__(
        self,
        clean_dir: Path,
        noise_dir: Path,
        mosaic_dir: Path,
        inpainting_dir: Path,
        image_size: Tuple[int, int],
        cache_limit: int = 100
    ):
        self.image_size = image_size
        self.cache = {}
        self.cache_limit = cache_limit
        self.transform = transforms.Compose([
            transforms.Resize(image_size, antialias=True),
            transforms.ToTensor()
        ])

        clean_paths = sorted(p for p in clean_dir.iterdir() if p.suffix.lower() in ['.png', '.jpg', '.jpeg'])
        self.task_list = []
        
        for path in clean_paths:
            self.task_list.extend([
                {'clean_path': path, 'task': 'noise', 'corrupted_path': noise_dir/path.name},
                {'clean_path': path, 'task': 'mosaic', 'corrupted_path': mosaic_dir/path.name},
                {'clean_path': path, 'task': 'inpainting', 'corrupted_path': inpainting_dir/path.name}
            ])
        
        logging.info(f"Dataset initialized with {len(self.task_list)} tasks. Cache limit: {cache_limit} items.")

    def __len__(self):
        return len(self.task_list)

    def __getitem__(self, idx):
        task_info = self.task_list[idx]
        cache_key = task_info['corrupted_path']
        
        if cache_key in self.cache:
            return self.cache[cache_key]
            
        try:
            clean_tensor = self.transform(Image.open(task_info['clean_path']).convert('RGB'))
            corrupted_tensor = self.transform(Image.open(task_info['corrupted_path']).convert('RGB'))
            
            mask = (corrupted_tensor.sum(dim=0) == 0).float().unsqueeze(0) if task_info['task'] == 'inpainting' else torch.zeros(1, *self.image_size)
            
            model_input = torch.cat([corrupted_tensor, mask], dim=0)
            
            result = (model_input * 2.0 - 1.0, clean_tensor * 2.0 - 1.0)
            
            if len(self.cache) < self.cache_limit:
                self.cache[cache_key] = result
            
            return result
            
        except Exception as e:
            logging.error(f"Error loading {task_info['corrupted_path']}: {e}")
            return torch.zeros(4, *self.image_size), torch.zeros(3, *self.image_size)

class Trainer:
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logging.getLogger(self.__class__.__name__)
        
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        self._setup_directories()
        self._verify_directories()
        self._initialize_training_components()
        self._load_checkpoint()
        self._check_gpu_temp()
        self.train_summary = {
            'Best Loss': self.best_loss,
            'Last Loss': 0.0,
            'Learning Rate': self.config['learning_rate'],
            'Epochs': 0,
            'Time': 0.0
        }
        self.logger.info(f"Trainer initialized with config: {self.config}")
        
    def _setup_directories(self):
        Path(self.config['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
        Path(self.config['preview_dir']).mkdir(parents=True, exist_ok=True)
        self.logger.info("Checkpoint and preview directories initialized.")

    def _verify_directories(self):
        for name, path_str in self.config['data_dirs'].items():
            path = Path(path_str)
            if not path.is_dir():
                raise FileNotFoundError(f"Directory '{name}' not found: {path}")
            if not any(path.iterdir()):
                self.logger.warning(f"Directory '{name}' is empty: {path}")
        self.logger.info("All data directories verified.")

    def _initialize_training_components(self):
        self.logger.info(f"Using device: {self.device}")
        image_size = (self.config['img_height'], self.config['img_width'])
        
        dataset = CombinedRestorationDataset(
            clean_dir=Path(self.config['data_dirs']['clean']),
            noise_dir=Path(self.config['data_dirs']['noise']),
            mosaic_dir=Path(self.config['data_dirs']['mosaic']),
            inpainting_dir=Path(self.config['data_dirs']['inpainting']),
            image_size=image_size,
            cache_limit=self.config.get('cache_limit', 200)
        )
        
        self.dataloader = DataLoader(
            dataset,
            batch_size=self.config['dataloader_params']['batch_size'],
            shuffle=True,
            num_workers=min(os.cpu_count(), self.config.get('num_workers', 4)),
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        )
        
        model = AdvancedUNet(in_channels=4, out_channels=3).to(self.device)
        self.logger.info(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} trainable parameters.")

        if hasattr(torch, 'compile'):
            self.model = torch.compile(model, mode=self.config.get('compile_mode', 'reduce-overhead'))
            self.logger.info("Model compiled with torch.compile()")
        
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.config['learning_rate'], weight_decay=1e-4)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=3)
        self.criterion = nn.L1Loss()
        self.use_amp = self.device.type == 'cuda'
        self.scaler = GradScaler(enabled=self.use_amp)
        self.logger.info(f"AMP {'enabled' if self.use_amp else 'disabled'}.")

    def _load_checkpoint(self):
        self.start_epoch = 0
        self.best_loss = float('inf')
        checkpoint_path = Path(self.config['checkpoint_dir']) / 'last_checkpoint.pth'
        
        if not checkpoint_path.exists():
            self.logger.info("No checkpoint found, starting fresh.")
            return
            
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if self.use_amp: self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.start_epoch = checkpoint['epoch'] + 1
            self.best_loss = checkpoint['best_loss']
            self.logger.info(f"Resumed from epoch {self.start_epoch} (best loss: {self.best_loss:.5f})")
        except Exception as e:
            self.logger.error(f"Error loading checkpoint: {e}. Starting fresh.")

    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        state = {
            'epoch': epoch, 'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(), 'scaler_state_dict': self.scaler.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(), 'best_loss': self.best_loss,
        }
        torch.save(state, Path(self.config['checkpoint_dir']) / 'last_checkpoint.pth')
        if is_best:
            torch.save(self.model.state_dict(), Path(self.config['checkpoint_dir']) / 'best_model.pth')
            self.logger.info(f"New best model saved (loss: {self.best_loss:.5f})")

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        accum_steps = self.config.get('grad_accum_steps', 1)
        
        pbar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{self.config['num_epochs']}", dynamic_ncols=True, leave=False)
        
        for i, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)
            
            with autocast(device_type=self.device.type, enabled=self.use_amp):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets) / accum_steps
            
            self.scaler.scale(loss).backward()
            
            if (i + 1) % accum_steps == 0 or (i + 1) == len(self.dataloader):
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
            
            total_loss += loss.item() * accum_steps
            pbar.set_postfix(loss=f"{loss.item() * accum_steps:.5f}")
            
            if i == 0:
                self._save_preview(inputs, outputs, targets, epoch)
                self._cleanup_previews(keep=10)
            
            if i % 5 == 0:
                self._check_gpu_temp()
        
        return total_loss / len(self.dataloader)

    def train(self):
        try:
            self.logger.info(f"Starting training on {self.device}...")
            start_time = time.time()
            self.logger.info(f"Current Best Loss: {self.best_loss:.5f}")
            
            for epoch in range(self.start_epoch, self.config['num_epochs']):
                self.train_summary['Epochs'] += 1
                epoch_loss = self._train_epoch(epoch)
                lr = self.optimizer.param_groups[0]['lr']
                self.train_summary['Learning Rate'] = lr
                self.scheduler.step(epoch_loss)
                
                is_best = epoch_loss < self.best_loss
                if is_best: 
                    self.best_loss = epoch_loss
                    self.train_summary['Best Loss'] = self.best_loss
                
                self.train_summary['Last Loss'] = epoch_loss
                self.train_summary['Time'] += time.time() - start_time
                self._save_checkpoint(epoch, is_best)
                
                self.logger.info(f"Epoch {epoch+1:03d} | Loss: {epoch_loss:.5f} | LR: {lr:.2e}")
                self._check_gpu_temp()
            
            self.logger.info(f"Training completed in {(time.time() - start_time)/60:.2f} minutes.")
        except KeyboardInterrupt:
            self.logger.info(f"Training Summary: Best Loss: {self.train_summary['Best Loss']:.5f}, Last Loss: {self.train_summary['Last Loss']:.5f}, Learning Rate: {self.train_summary['Learning Rate']:.2e}, Epochs: {self.train_summary['Epochs']}, Time: {self.train_summary['Time']:.2f} seconds")

    def _save_preview(self, inputs, outputs, targets, epoch):
        preview_dir = Path(self.config['preview_dir'])
        with torch.no_grad():
            denorm = lambda x: (x.clamp(-1, 1) + 1) / 2
            grid = make_grid([denorm(inputs[0,:3].cpu()), denorm(outputs[0].cpu()), denorm(targets[0].cpu())], nrow=3)
            save_image(grid, preview_dir / f'preview_{epoch+1:04d}.png')

    def _cleanup_previews(self, keep=10):
        preview_dir = Path(self.config['preview_dir'])
        previews = sorted(
            [p for p in preview_dir.glob("preview_*.png") if p.stem.replace("preview_", "").isdigit()],
            key=lambda p: int(p.stem.replace("preview_", ""))
        )
        if len(previews) > keep:
            for f in previews[:-keep]:
                try: f.unlink()
                except OSError as e: self.logger.error(f"Error deleting preview: {f.name}: {e}")

    def _check_gpu_temp(self, threshold=85, delay=15):
        if not getGPUs or self.device.type != 'cuda': return
        try:
            gpu = getGPUs()[0]
            temperature = gpu.temperature + 2.0
            if temperature > threshold:
                self.logger.warning(f"GPU temp >{threshold}°C ({temperature}°C). Throttling for {delay}s.")
                time.sleep(delay)
        except Exception as e:
            self.logger.error(f"GPU temp check failed: {e}")