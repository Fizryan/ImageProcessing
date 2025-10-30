# MaskDetector_Training_Config.py
# Configuration for training the mosaic area detection model.

DETECTOR_TRAINING_CONFIG = {
    # --- General ---
    "model_size": "efficient_detector",
    "checkpoint_dir": "Training/detector_checkpoints",
    "preview_dir": "Training/detector_previews",
    "log_file": "Training/detector_checkpoints/Training.log",
    "num_epochs": 100,
    "checkpoint_interval_epochs": 5,
    "sample_interval_epochs": 1,
    # --- Model & Architecture ---
    "base_channels": 16,
    "use_checkpointing": True,
    "mosaic_block_size_range": [5, 30],
    # --- Data ---
    "train_clean_dir": "dataset/train/clean_images",
    "val_clean_dir": "dataset/validation/clean_images",
    "train_mask_dir": "dataset/train/mask_images",
    "val_mask_dir": "dataset/validation/mask_images",
    "img_height": 256,
    "img_width": 448,
    "dataloader_params": {
        "batch_size": 4,
        "shuffle": True,
        "num_workers": 4,
        "pin_memory": True,
        "persistent_workers": True,
    },
    "val_batch_size": 8,
    # --- Optimizer & Scheduler ---
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "scheduler": "onecycle",
    "onecycle_params": {"pct_start": 0.3, "div_factor": 25},
    # --- Performance ---
    "use_amp": True,
    "use_channels_last": True,
    "compile_mode": "reduce-overhead",
    "grad_clip": 1.0,
}
