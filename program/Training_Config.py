# Training_Config.py
# This module contains the configuration settings for the training pipeline of the image restoration model.

TRAINING_CONFIG = {
    # --- General ---
    "task_type": "demosaic",  # 'demosaic', 'inpainting'
    "model_size": "efficient",
    "checkpoint_dir": "Training/checkpoints",
    "finetune_checkpoint_path": "Training/checkpoints/best_model.pth",
    "preview_dir": "Training/previews",
    "log_file": "Training/checkpoints/Training.log",
    "num_epochs": 50,
    "checkpoint_interval_epochs": 5,
    "sample_interval_epochs": 1,
    # --- Model & Architecture ---
    "base_channels": 48,
    "mosaic_block_size_range": [20, 60],
    "mosaic_opacity_range": [1.0, 1.0],
    # --- Data ---
    "train_clean_dir": "dataset/train/clean_images",  # Source for training
    "val_clean_dir": "dataset/validation/clean_images",  # Source for validation
    "train_mask_dir": "dataset/train/mask_images",  # Source for training masks
    "val_mask_dir": "dataset/validation/mask_images",  # Source for validation masks
    "img_height": 256,
    "img_width": 256,
    "dataloader_params": {
        "batch_size": 2,
        "shuffle": True,
        "num_workers": 4,
        "pin_memory": True,
        "persistent_workers": True,
    },
    "val_batch_size": 4,
    # --- Optimizer & Scheduler ---
    "learning_rate": 5e-5,
    "weight_decay": 1e-4,
    "scheduler": "cosine_restarts",  # 'onecycle' or 'cosine_restarts'
    "onecycle_params": {
        "pct_start": 0.3,
        "div_factor": 20,
        "final_div_factor": 1e4,
    },
    "cosine_restarts_params": {
        "T_0": 10,
        "eta_min": 1e-7,
    },
    # --- Performance ---
    "use_amp": True,
    "use_channels_last": True,
    "compile_mode": None,  # "reduce-overhead",
    "use_checkpointing": True,
    "grad_clip": 0.5,
    "accumulation_steps": 4,
    "ohem_percent": 1.0,  # Starting OHEM percentage
    "ohem_schedule": [
        # (0.0, 1.0),
        # (0.125, 0.8),
        # (0.25, 0.5),
    ],  # (epoch_ratio, percent)
    "early_stopping_patience": -1,
    "use_advanced_loss": True,
    "use_ema": True,
    "use_sharpness_loss": False,
    "use_gan": True,  # Set to True to enable GAN training
    "gan_weight": 0.2,
    "discriminator_lr": 5e-5,
    # Loss weights from Optuna
    "l1_weight": 0.1,
    "lpips_weight": 1.0,
    "fft_weight": 0.1,  # Weight for FFT loss
    # Other advanced features
    "use_enhanced_architecture": True,  # Set to True to use DetailPreservationUNet
    # --- Advanced Data Augmentation ---
    "use_robust_degradation": False,  # Simulate real-world defects (blur, noise, JPEG)
    "robust_degradation_prob": 0.0,  # Probability of applying robust degradation
    "robust_degradation_config": {
        "blur_prob": 0.0,  # Probability of Gaussian blur
        "noise_prob": 0.0,  # Probability of Gaussian noise
        "jpeg_prob": 0.0,  # Probability of JPEG compression artifacts
        "noise_std_range": [0.01, 0.05],  # Noise standard deviation range
        "jpeg_scale_range": [0.5, 0.9],  # JPEG compression scale factor range
    },
    "use_geometric_augmentation": True,  # Enable rotation/flip augmentations
    "use_vertical_flip": True,  # Add vertical flip for geometric invariance
    "use_mosaic_grid_shift": True,  # Shift mosaic grid randomly (prevents overfitting)
    "scale_augmentation_range": [
        0.75,
        1.0,
    ],  # Multi-resolution training: 0.5=480p, 0.75=720p, 1.0=1080p
}
