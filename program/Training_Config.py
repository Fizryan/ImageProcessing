# Training_Config.py
# This module contains the configuration settings for the training pipeline of the image restoration model.

TRAINING_MODE = "restoration"  # 'restoration' or 'super_resolution'
MODEL_SIZE = "lite"

COMMON_CONFIG = {
    "checkpoint_dir": "Training/checkpoints",
    "preview_dir": "Training/previews",
    "num_epochs": 200,
    "scheduler": "onecycle",  # 'onecycle' or 'plateau'
    "compile_mode": "reduce-overhead",  # max-autotune
    "learning_rate": 1e-3,  # For plateau, this is initial LR. For onecycle, this is max_lr.
    "discriminator_lr": 1e-4,
    "perceptual_weight": 0.8,  # range 0.0 - 1.0
    "ssim_weight": 0.2,  # range 0.0 - 1.0
    "multiscale_loss_weight": 0.5,  # range 0.0 - 1.0
    "frequency_loss_weight": 0.1,  # range 0.0 - 1.0
    "gan_weight": 0.1,  # range 0.0 - 1.0
    "edge_weight": 0.15,  # range 0.0 - 1.0
    "weight_decay": 1e-2,
    "use_amp": True,
    "use_channels_last": True,
    "use_gan": True,
    "use_checkpointing": True,
    "use_ema": True,
    "ema_decay": 0.999,
    "dataloader_params": {
        "shuffle": True,
        "num_workers": 4,
        "pin_memory": True,
        "persistent_workers": True,
    },
    "sample_images": "Samples/images",
    "sample_masks": "Samples/masks",
    "use_progressive_training": False,
}

TRAINING_CONFIG = {}

if TRAINING_MODE == "restoration":
    RESTORATION_CONFIG = {
        "training_mode": "restoration",
        "model_size": MODEL_SIZE,
        "data_dirs": {
            "train": {
                "clean": "dataset/train/clean_images",
                # 'noise': 'dataset/train/noisy_images',
                # 'mosaic': 'dataset/train/mosaic_images',
                "inpainting": "dataset/train/inpainting_images",
                "mask": "dataset/train/mask_images",
                # 'blur': 'dataset/train/blurry_images'
            },
            "validation": {
                "clean": "dataset/validation/clean_images",
                # 'noise': 'dataset/validation/noisy_images',
                # 'mosaic': 'dataset/validation/mosaic_images',
                "inpainting": "dataset/validation/inpainting_images",
                "mask": "dataset/validation/mask_images",
                # 'blur': 'dataset/validation/blurry_images'
            },
        },
        "img_height": 256,
        "img_width": 448,
        "dataloader_params": {"batch_size": 1},
        "base_channels": 10,  # Further reduced to save memory
        "grad_accum_steps": 4,  # Increase accumulation to maintain effective batch size
        "checkpoint_interval_steps": 5000,
        "onecycle_params": {
            "pct_start": 0.3,
            "div_factor": 25,
            "final_div_factor": 1e4,
            "three_phase": False,
            "anneal_strategy": "cos",
        },
        "progressive_phases": [
            {
                "epochs": 50,
                "size": (128, 224),
                "lr_mult": 1.0,
            },
            {
                "epochs": 150,
                "size": (256, 448),
                "lr_mult": 0.5,
            },
        ],
    }
    TRAINING_CONFIG = {**COMMON_CONFIG, **RESTORATION_CONFIG}
    TRAINING_CONFIG["dataloader_params"] = {
        **COMMON_CONFIG["dataloader_params"],
        **RESTORATION_CONFIG["dataloader_params"],
    }

elif TRAINING_MODE == "super_resolution":
    SR_CONFIG = {
        "training_mode": "super_resolution",
        "hr_data_dir": "dataset/resized_images",
        "lr_patch_height": 128,
        "lr_patch_width": 128,
        "upscale_factor": 4,
        "num_res_blocks": 16,
        "dataloader_params": {"batch_size": 2},
    }
    TRAINING_CONFIG = {**COMMON_CONFIG, **SR_CONFIG}
    TRAINING_CONFIG["dataloader_params"] = {
        **COMMON_CONFIG["dataloader_params"],
        **SR_CONFIG["dataloader_params"],
    }
