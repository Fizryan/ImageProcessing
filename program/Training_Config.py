# Training_Config.py
# This module contains the configuration settings for the training pipeline of the image restoration model.

TRAINING_CONFIG = {
    'data_dirs': {
        'clean': 'dataset/clean_images',
        'noise': 'dataset/noisy_images',
        'mosaic': 'dataset/mosaic_images',
        'inpainting': 'dataset/grayscale_images'
    },
    'checkpoint_dir': 'Training/checkpoints',
    'preview_dir': 'Training/previews',
    'img_height': 384,
    'img_width': 256,
    'learning_rate': 2e-5, # 1e-4 or 2e-4 or 2e-5 for AdamW
    'num_epochs': 1000,
    'compile_mode': 'reduce-overhead', # Options: 'default', 'reduce-overhead', 'max-autotune'
    'max_gpu_temp': 85,
    'dataloader_params': {
        'batch_size': 2, # Adjust based on GPU memory
        'shuffle': True,
        'num_workers': 4,
        'pin_memory': True,
        'persistent_workers': True
    }
}