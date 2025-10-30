# Training_Config.py
# This module contains the configuration settings for the training pipeline of the image restoration model.

TRAINING_CONFIG = {
    # --- General ---
    "task_type": "demosaic",  # 'demosaic', 'inpainting', or 'deblur'
    "model_size": "efficient",
    "checkpoint_dir": "Training/checkpoints",
    "preview_dir": "Training/previews",
    "log_file": "Training/checkpoints/Training.log",
    "num_epochs": 250,
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
    "learning_rate": 0.0004930967808936792,
    "weight_decay": 0.0008499301888996408,
    "scheduler": "cosine_restarts",  # 'onecycle' or 'cosine_restarts'
    "onecycle_params": {
        "pct_start": 0.3771202692439005,
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
    "grad_clip": 1.3632420407003598,  # From Optuna
    "early_stopping_patience": -1,
    # --- Advanced Features ---
    "use_advanced_loss": False,
    "use_ema": True,
    "ema_decay": 0.999,
    "use_sharpness_loss": False,  # Set to True to enable the most advanced loss
    "use_gan": True,  # Set to True to enable GAN training
    "gan_weight": 0.016057043250212413,
    "discriminator_lr": 0.0009464549369640666,
    # Loss weights from Optuna
    "l1_weight": 1.1128178555956723,
    "lpips_weight": 0.8397145131281629,
    "fft_weight": 0.05522052764990448,  # Bobot untuk FFT loss
    # Other advanced features
    "use_enhanced_architecture": True,  # Set to True to use DetailPreservationUNet
}
