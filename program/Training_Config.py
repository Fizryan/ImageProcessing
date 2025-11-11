# Training_Config.py
# This module contains the configuration settings for the training pipeline of the image restoration model.

TRAINING_CONFIG = {
    # --- General ---
    "task_type": "inpainting",  # 'demosaic', 'inpainting', or 'deblur'
    "model_size": "efficient",
    "checkpoint_dir": "Training/checkpoints",
    "preview_dir": "Training/previews",
    "log_file": "Training/checkpoints/Training.log",
    "num_epochs": 200,
    "checkpoint_interval_epochs": 5,
    "sample_interval_epochs": 1,
    # --- Model & Architecture ---
    "base_channels": 32,
    "mosaic_block_size_range": [5, 15],
    "mosaic_opacity_range": [1.0, 1.0],
    # --- Data ---
    "train_clean_dir": "dataset/train/clean_images",  # Source for training
    "val_clean_dir": "dataset/validation/clean_images",  # Source for validation
    "train_mask_dir": "dataset/train/mask_images",  # Source for training masks
    "val_mask_dir": "dataset/validation/mask_images",  # Source for validation masks
    "img_height": 256,
    "img_width": 448,
    "dataloader_params": {
        "batch_size": 4,
        "shuffle": True,
        "num_workers": 4,
        "pin_memory": True,
        "persistent_workers": True,
    },
    "val_batch_size": 4,
    # --- Optimizer & Scheduler ---
    "learning_rate": 1.19e-07,
    "weight_decay": 3.32e-07,
    "scheduler": "onecycle",  # 'onecycle' or 'cosine_restarts'
    "onecycle_params": {
        "pct_start": 0.30610497251099344,
        "div_factor": 20,
        "final_div_factor": 1e4,
    },
    "cosine_restarts_params": {
        "T_0": 10,
        "eta_min": 1e-6,
    },
    # --- Performance ---
    "use_amp": True,
    "use_channels_last": True,
    "compile_mode": "reduce-overhead",
    "use_checkpointing": True,
    "grad_clip": 1.680313,
    "accumulation_steps": 4,
    "ohem_percent": 1.0,  # Starting OHEM percentage
    "ohem_schedule": [(0, 1.0), (25, 0.75), (75, 0.5)],  # (epoch, percent)
    "early_stopping_patience": -1,
    "use_advanced_loss": False,
    "use_ema": True,
    "use_sharpness_loss": False,
    "use_gan": True,  # Set to True to enable GAN training
    "gan_weight": 0.069420,
    "discriminator_lr": 2.23e-04,
    # Loss weights from Optuna
    "l1_weight": 1.906870,
    "lpips_weight": 0.637318,
    "fft_weight": 0.122248,  # Weight for FFT loss
    # Other advanced features
    "use_enhanced_architecture": True,  # Set to True to use DetailPreservationUNet
}
