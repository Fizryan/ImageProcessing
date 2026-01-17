import os

VERSION = "V1"
BASE_DIR = "database"

VER_PATH = f"/{VERSION}" if VERSION else ""

DATASET_ROOT = f"{BASE_DIR}/dataset"
MODELS_ROOT = f"{BASE_DIR}{VER_PATH}/models"
RESULTS_ROOT = f"{BASE_DIR}{VER_PATH}/results"

MAIN_PATHS = {
    "dataset": DATASET_ROOT,
    "train_dir": f"{DATASET_ROOT}/train",
    "val_dir": f"{DATASET_ROOT}/val",
    "models": MODELS_ROOT,
    "tensorboard": f"{RESULTS_ROOT}/tensorboard",
    "logs": f"logs",
}

USE_MASK = True

DIRECTORIES_CONFIG = {
    "checkpoint_dir": f"{MAIN_PATHS['models']}/checkpoints",
    "tensorboard_log_dir": MAIN_PATHS["tensorboard"],
    "log_file": f"{MAIN_PATHS['logs']}/app.log",
    "train_clean_dir": f"{MAIN_PATHS['train_dir']}/clean_image",
    "train_mask_dir": f"{MAIN_PATHS['train_dir']}/mask_image" if USE_MASK else None,
    "train_external_dir": None,
    "val_clean_dir": f"{MAIN_PATHS['val_dir']}/clean_image",
    "val_mask_dir": f"{MAIN_PATHS['val_dir']}/mask_image" if USE_MASK else None,
    "val_external_dir": None,
}

HYPERPARAMETERS = {
    "trial_number": 1,
    "task_type": "demosaic",
    "num_epochs": 200,
    "checkpoint_freq": 5,
    "model_size": "efficient",
    "base_channels": 48,
    "in_channels": 3,
    "out_channels": 3,
    "use_enhanced_architecture": True,
    "img_height": 256,
    "img_width": 256,
    "dataloader_params": {
        "batch_size": 4,
        "num_workers": 4,
        "persistent_workers": True,
        "prefetch_factor": 4,
    },
    "learning_rate": 3e-4,
    "weight_decay": 1e-4,
    "scheduler": "onecycle",
    "grad_clip": 1.0,
    "accumulation_steps": 2,
    "use_amp": True,
    "use_channels_last": True,
    "use_checkpointing": False,
    "use_ema": True,
    "ema_decay": 0.9995,
    "compile_model": False,
    "ohem_percent": 1.0,
    "ohem_schedule": [
        (0.0, 1.00),
        (0.3, 0.75),
        (0.6, 0.50),
    ],
    "use_gan": True,
    "gan_weight": 0.02,
    "discriminator_lr": 1e-4,
    "l1_weight": 1.0,
    "lpips_weight": 0.3,
    "fft_weight": 0.05,
    "use_sharpness_loss": False,
    "use_advanced_loss": True,
    "mosaic_block_size_range": [20, 60],
    "mosaic_opacity_range": [0.8, 1.0],
    "scale_augmentation_range": [0.5, 1.0],
    "use_geometric_augmentation": True,
    "use_mosaic_grid_shift": True,
    "use_robust_degradation": False,
    "robust_degradation_prob": 0.5,
    "robust_degradation_config": {
        "blur_prob": 0.3,
        "noise_prob": 0.3,
        "jpeg_prob": 0.3,
    },
    "lpips_subset_batches": 4,
}

MAIN_CONFIG = {
    **DIRECTORIES_CONFIG,
    **HYPERPARAMETERS,
    "version": VERSION,
    "base_dir": BASE_DIR,
}
