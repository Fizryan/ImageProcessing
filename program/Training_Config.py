# Training_Config.py

TRAINING_CONFIG = {
    # --- General ---
    "task_type": "demosaic",
    "model_size": "efficient",
    "checkpoint_dir": "Training/checkpoints",
    "finetune_checkpoint_path": None,
    "preview_dir": "Training/previews",
    "log_file": "Training/checkpoints/Training.log",
    "num_epochs": 200,
    "checkpoint_interval_epochs": 5,
    "sample_interval_epochs": 1,
    # --- Model & Architecture ---
    "base_channels": 48,
    "mosaic_block_size_range": [20, 60],
    "mosaic_opacity_range": [1.0, 1.0],
    # --- Data ---
    "train_clean_dir": "dataset/train/clean_images",
    "val_clean_dir": "dataset/validation/clean_images",
    "train_mask_dir": "dataset/train/mask_images",
    "val_mask_dir": "dataset/validation/mask_images",
    "train_external_dir": None,  # "dataset/train/external_images",
    "val_external_dir": None,  # "dataset/validation/external_images",
    "img_height": 256,
    "img_width": 256,
    # --- DataLoader ---
    "dataloader_params": {
        "batch_size": 4,
        "shuffle": True,
        "num_workers": 4,
        "pin_memory": True,
        "persistent_workers": True,
    },
    "val_batch_size": 4,
    # --- Optimizer & Scheduler ---
    "learning_rate": 2e-4,
    "weight_decay": 1e-4,
    "scheduler": "onecycle",
    "onecycle_params": {
        "pct_start": 0.1,
        "div_factor": 25,
        "final_div_factor": 1e4,
    },
    # --- Performance ---
    "use_amp": True,
    "use_channels_last": True,
    "compile_mode": None,
    "use_checkpointing": True,
    "grad_clip": 0.5,
    "accumulation_steps": 4,
    "ohem_percent": 1.0,
    "ohem_schedule": [
        (0.0, 1.0),
        (0.25, 0.8),
        (0.5, 0.5),
    ],
    "early_stopping_patience": 20,
    "use_advanced_loss": True,
    "use_ema": True,
    # --- GAN Config ---
    "use_gan": True,
    "gan_weight": 0.02,
    "discriminator_lr": 1e-4,
    # --- Loss Weights ---
    "l1_weight": 1.0,
    "lpips_weight": 0.5,
    "fft_weight": 0.05,
    "use_sharpness_loss": False,
    "use_enhanced_architecture": True,
    "use_robust_degradation": False,
    "robust_degradation_prob": 0.5,
    "robust_degradation_config": {
        "blur_prob": 0.3,
        "noise_prob": 0.3,
        "jpeg_prob": 0.3,
        "noise_std_range": [0.01, 0.03],
        "jpeg_scale_range": [0.7, 0.95],
    },
    "use_geometric_augmentation": True,
    "use_vertical_flip": True,
    "use_mosaic_grid_shift": True,
    "scale_augmentation_range": [0.5, 1.0],
}
