# main.py
# ImageProcessing Pipeline Main Script

import logging
import logging.config
import time
from typing import Any, Callable
from pathlib import Path

from program.Logging_Config import LOGGING_CONFIG
from program.Training_Config import TRAINING_CONFIG

from program.CollectingImage import CollectingImage
from program.Resizing import Resizer
from program.Noise import NoiseGenerator
from program.Grayscaling import Grayscaler
from program.Mosaic import MosaicGenerator
from program.Training import Trainer
from program.Inference import ImageRestorer

logger = logging.getLogger(__name__)

def _prepare_directory():
    Path("logs").mkdir(parents=True, exist_ok=True)
    Path("Samples").mkdir(parents=True, exist_ok=True)

def get_user_input(prompt: str, default: Any, target_type: Callable = str) -> Any:
    while True:
        try:
            user_input = input(f"{prompt} (default: {default}): ").strip()
            if not user_input:
                return default
            return target_type(user_input)
        except ValueError:
            logger.error(f"Invalid input. Please enter a valid {target_type.__name__}.")

def handle_download():
    logger.info("Starting image download process...")
    count = get_user_input("Number of images to download", 20, int)
    collector = CollectingImage(count=count, max_workers=10)
    collector.download_images()

def handle_resize():
    logger.info("Starting image resizing process...")
    input_dir = get_user_input("Input directory", "dataset/clean_images")
    output_dir = get_user_input("Output directory", "dataset/resized_images")
    width = get_user_input("Target width", 384, int)
    height = get_user_input("Target height", 512, int)
    
    resizer = Resizer(input_dir=input_dir, output_dir=output_dir, width=width, height=height)
    resizer.process_images()

def handle_noise():
    logger.info("Starting noise generation process...")
    input_dir = get_user_input("Input directory", "dataset/resized_images")
    output_dir = get_user_input("Output directory", "dataset/noisy_images")
    noise_level = get_user_input("Noise level (0.0 to 1.0)", 0.1, float)

    noiser = NoiseGenerator(input_dir=input_dir, noise_dir=output_dir, noise_level=noise_level)
    noiser.generate_noisy_images()

def handle_grayscale():
    logger.info("Starting grayscale conversion process...")
    input_dir = get_user_input("Input directory", "dataset/resized_images")
    output_dir = get_user_input("Output directory", "dataset/grayscale_images")
    output_format = get_user_input("Output format (e.g., PNG, JPEG, or leave empty to keep original)", "PNG")
    
    grayscaler = Grayscaler(input_dir=input_dir, output_dir=output_dir, output_format=output_format or None)
    grayscaler.process_images()

def handle_mosaic():
    logger.info("Starting mosaic image generation process...")
    input_dir = get_user_input("Input directory", "dataset/resized_images")
    output_dir = get_user_input("Output directory", "dataset/mosaic_images")
    block_size = get_user_input("Block size for mosaic effect", 25, int)
    mosaic_gen = MosaicGenerator(input_dir=input_dir, output_dir=output_dir, block_size=block_size)
    mosaic_gen.generate_mosaic_images()

def handle_training():
    logger.info("Starting model training process...")
    trainer = Trainer(config=TRAINING_CONFIG)
    trainer.train()

def handle_inference():
    logger.info("Starting image restoration process...")

    model_path = get_user_input("Path to the trained model", "Training/checkpoints/best_model.pth")
    img_height = get_user_input("Image height for restoration", 512, int)
    img_width = get_user_input("Image width for restoration", 384, int)
    restorer = ImageRestorer(model_path=model_path, img_height=img_height, img_width=img_width)

    input_path = get_user_input("Path to the input image", "Samples/sample_1.png")
    output_path = get_user_input("Path to save the restored image", "Results/restored_image.png")
    task_type = get_user_input("Task type (noise, mosaic, inpainting)", "noise", str)
    restored_image = restorer.restore_image_from_path(input_path=input_path, output_path=output_path, task_type=task_type)
    logger.info(f"Inference completed.")

def display_menu():
    print("\n" + "="*28)
    print("      IMAGE PROCESSING")
    print("="*28)
    print("  1. Download New Images")
    print("  2. Resize Images")
    print("  3. Add Noise to Images")
    print("  4. Convert Images to Grayscale")
    print("  5. Generate Mosaic Images")
    print("  6. Train Model")
    print("  7. Run Inference (Image Restoration)")
    print("  0. Exit")
    print("="*28)

def main():
    _prepare_directory()
    logging.config.dictConfig(LOGGING_CONFIG)
    logger.info("Image Processing Pipeline started.")
    actions = {
        "1": handle_download,
        "2": handle_resize,
        "3": handle_noise,
        "4": handle_grayscale,
        "5": handle_mosaic,
        "6": handle_training,
        "7": handle_inference
    }

    while True:
        display_menu()
        choice = input("Enter your choice (0-7): ").strip()

        if choice == '0':
            logger.info("Exiting program. Goodbye!")
            break
        
        action = actions.get(choice)
        if action:
            try:
                action()
                logger.info(f"Task '{action.__name__}' completed successfully.")
            except Exception as e:
                logger.error(f"An unexpected error occurred during the task: {e}", exc_info=True)
        else:
            logger.warning("Invalid choice. Please try again.")
        
        time.sleep(2)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n")
        logger.info("Program interrupted by user. Exiting...")
    except Exception as e:
        print("\n")
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)