# main.py
# ImageProcessing Pipeline Main Script

import logging
import logging.config
import time
import queue
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler
from typing import Any, Callable, List
from pathlib import Path
from tqdm import tqdm

import torch
from PIL import Image
from torchvision import transforms

from program.Logging_Config import LOGGING_CONFIG
from program.Training_Config import TRAINING_CONFIG

from program.CollectingImage import CollectingImage
from program.Resizing import Resizer
from program.Noise import NoiseGenerator
from program.Grayscaling import Grayscaler
from program.Mosaic import MosaicGenerator
from program.MaskGenerator import MaskGenerator
from program.BlurGenerator import BlurGenerator, apply_blur
from program.Training import Trainer
from program.Inference import ImageRestorer
from program.MaskDetector_Training import MaskDetectorTrainer
from program.MaskDetector_Training_Config import DETECTOR_TRAINING_CONFIG

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
    height = get_user_input("Target height", 256, int)
    width = get_user_input("Target width", 448, int)
    resizer = Resizer(
        input_dir=input_dir, output_dir=output_dir, width=width, height=height
    )
    resizer.process_images()


def handle_noise():
    logger.info("Starting noise generation process...")
    input_dir = get_user_input("Input directory", "dataset/resized_images")
    mask_dir = get_user_input("Mask directory", "dataset/mask_images")
    output_dir = get_user_input("Output directory", "dataset/noisy_images")
    noise_level = get_user_input("Noise level (0.0 to 1.0)", 0.1, float)
    overwrite = (
        get_user_input("Overwrite existing files? (yes/no)", "no", str).lower() == "yes"
    )
    noiser = NoiseGenerator(
        input_dir=input_dir,
        mask_dir=mask_dir,
        noise_dir=output_dir,
        noise_level=noise_level,
        overwrite=overwrite,
    )
    noiser.generate_noisy_images()


def handle_grayscale():
    logger.info("Starting grayscale conversion process...")
    input_dir = get_user_input("Input directory", "dataset/resized_images")
    output_dir = get_user_input("Output directory", "dataset/grayscale_images")
    output_format = get_user_input(
        "Output format (e.g., PNG, JPEG, or leave empty to keep original)", "PNG"
    )
    overwrite = (
        get_user_input("Overwrite existing files? (yes/no)", "no", str).lower() == "yes"
    )
    grayscaler = Grayscaler(
        input_dir=input_dir,
        output_dir=output_dir,
        output_format=output_format or None,
        overwrite=overwrite,
    )
    grayscaler.process_images()


def handle_mosaic():
    logger.info("Starting mosaic image generation process...")
    input_dir = get_user_input("Input directory", "dataset/resized_images")
    mask_dir = get_user_input("Mask directory", "dataset/mask_images")
    output_dir = get_user_input("Output directory", "dataset/mosaic_images")
    block_size = get_user_input("Block size for mosaic effect", 25, int)
    overwrite = (
        get_user_input("Overwrite existing files? (yes/no)", "no", str).lower() == "yes"
    )
    mosaic_gen = MosaicGenerator(
        input_dir=input_dir,
        mask_dir=mask_dir,
        output_dir=output_dir,
        block_size=block_size,
        overwrite=overwrite,
    )
    mosaic_gen.generate_mosaic_images()


def handle_blur():
    logger.info("Starting blur generation process...")
    input_dir = get_user_input("Input directory", "dataset/resized_images")
    output_dir = get_user_input("Output directory", "dataset/blurry_images")
    blur_radius_range = get_user_input(
        "Blur radius range (min, max)", (1.0, 3.0), tuple
    )
    overwrite = (
        get_user_input("Overwrite existing files? (yes/no)", "no", str).lower() == "yes"
    )
    blur_gen = BlurGenerator(
        input_dir=input_dir,
        output_dir=output_dir,
        blur_radius_range=blur_radius_range,
        overwrite=overwrite,
    )
    blur_gen.generate_blurry_images()


def handle_training():
    logger.info("Starting model training process...")
    trainer = Trainer(config=TRAINING_CONFIG)
    trainer.train()


def handle_inpainting_training():
    """Handles training for the inpainting task with a dedicated config."""
    logger.info("Starting model training process for Inpainting...")

    from copy import deepcopy

    inpainting_config = deepcopy(TRAINING_CONFIG)

    inpainting_config["task_type"] = "inpainting"
    inpainting_config["checkpoint_dir"] = "Training/inpainting_checkpoints"
    inpainting_config["preview_dir"] = "Training/inpainting_previews"
    inpainting_config["log_file"] = "Training/inpainting_checkpoints/Training.log"

    trainer = Trainer(config=inpainting_config)
    trainer.train()


def handle_generate_detector_data():
    logger.info("Starting mosaic detector data generation...")
    clean_dir = get_user_input(
        "Input directory (resized clean images)", "dataset/resized_images"
    )
    output_dir = get_user_input(
        "Output directory for detector data", "dataset/detector_data"
    )
    block_size = get_user_input("Mosaic block size", 16, int)
    overwrite = (
        get_user_input("Overwrite existing files? (yes/no)", "no", str).lower() == "yes"
    )

    generator = MaskGenerator(
        clean_dir=clean_dir,
        output_dir=output_dir,
        block_size=block_size,
        overwrite=overwrite,
    )
    generator.generate_data()


def handle_train_detector():
    logger.info("Starting mosaic detector model training...")
    trainer = MaskDetectorTrainer(config=DETECTOR_TRAINING_CONFIG)
    trainer.train()


def handle_demosaic_inference():
    logger.info("Starting image restoration (Demosaic/Inpainting) process...")
    model_path = get_user_input(
        "Path to the trained restoration model", "Training/checkpoints/best_model.pth"
    )
    if not Path(model_path).exists():
        logger.error(f"Model file not found: {model_path}")
        return

    try:
        restorer = ImageRestorer(model_path=model_path)
    except Exception as e:
        logger.error(f"Failed to initialize the restorer: {e}", exc_info=True)
        return

    input_path_str = get_user_input(
        "Path to input image or directory", "Samples/test_images"
    )
    input_path = Path(input_path_str)
    if not input_path.exists():
        logger.error(f"Input path does not exist: {input_path}")
        return

    iterations = get_user_input("Number of iterations (for better results)", 1, int)
    if iterations > 1:
        logger.info(
            "Using iterative refinement. This may take longer but can improve quality."
        )

    use_tta = (
        get_user_input(
            "Use Test-Time Augmentation (TTA) for better quality? (yes/no)", "yes", str
        ).lower()
        == "yes"
    )
    final_blend_alpha = get_user_input(
        "Final blend with original (0.0 to 1.0, 0=off)", 0.0, float
    )

    if input_path.is_file():
        output_path_str = get_user_input(
            "Path to save the restored image", f"Results/{input_path.stem}_restored.png"
        )
        output_path = Path(output_path_str)
        restorer.restore_image_from_path(
            input_path,
            output_path,
            iterations=iterations,
            use_tta=use_tta,
            final_blend_alpha=final_blend_alpha,
        )

    elif input_path.is_dir():
        output_dir_str = get_user_input(
            "Path to save the restored images", "Results/test_images"
        )
        output_dir = Path(output_dir_str)

        restorer.process_directory(
            input_dir=input_path,
            output_dir=output_dir,
            iterations=iterations,
            use_tta=use_tta,
            final_blend_alpha=final_blend_alpha,
        )
    else:
        logger.error(f"Input path is not a valid file or directory: {input_path}")
        return

    logger.info("Inference completed.")


def handle_detector_inference():
    logger.info("Starting mosaic mask detection process...")
    model_path = get_user_input(
        "Path to the trained detector model",
        "Training/detector_checkpoints/best_detector_model.pth",
    )
    if not Path(model_path).exists():
        logger.error(f"Model file not found: {model_path}")
        return

    try:
        detector = ImageRestorer(model_path=model_path, is_detector=True)
        logger.info("Detector model loaded into inference engine.")
    except Exception as e:
        logger.error(f"Failed to initialize the detector: {e}", exc_info=True)
        return

    input_path_str = get_user_input(
        "Path to input image or directory", "Samples/test_images"
    )
    input_path = Path(input_path_str)
    if not input_path.exists():
        logger.error(f"Input path does not exist: {input_path}")
        return

    output_dir_str = get_user_input(
        "Path to save the detected masks", "Results/test_masks"
    )
    output_dir = Path(output_dir_str)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_files: List[Path] = (
        [input_path] if input_path.is_file() else list(input_path.glob("*.*"))
    )

    for img_path in tqdm(image_files, desc="Detecting masks"):
        try:
            with Image.open(img_path) as img:
                mask_pil = detector.detect_mask(img)
                mask_pil.save(output_dir / f"{img_path.stem}_mask.png")
        except Exception as e:
            logger.error(f"Failed to process {img_path.name}: {e}")

    logger.info("Mask detection completed.")


def handle_full_pipeline():
    logger.info("Starting full pipeline: Detect Mask -> Demosaic")

    detector_model_path = get_user_input(
        "Path to the trained detector model",
        "Training/detector_checkpoints/best_detector_model.pth",
    )
    if not Path(detector_model_path).exists():
        logger.error(f"Detector model file not found: {detector_model_path}")
        return
    try:
        detector = ImageRestorer(model_path=detector_model_path, is_detector=True)
    except Exception as e:
        logger.error(f"Failed to initialize the detector: {e}", exc_info=True)
        return

    demosaic_model_path = get_user_input(
        "Path to the trained demosaic model", "Training/checkpoints/best_model.pth"
    )
    if not Path(demosaic_model_path).exists():
        logger.error(f"Demosaic model file not found: {demosaic_model_path}")
        return
    try:
        restorer = ImageRestorer(model_path=demosaic_model_path)
    except Exception as e:
        logger.error(f"Failed to initialize the restorer: {e}", exc_info=True)
        return

    input_path_str = get_user_input(
        "Path to input image or directory", "dataset/inference"
    )
    input_path = Path(input_path_str)
    if not input_path.exists():
        logger.error(f"Input path does not exist: {input_path}")
        return

    output_dir_str = get_user_input(
        "Path to save the final restored images", "Results/full_pipeline"
    )
    output_dir = Path(output_dir_str)
    output_dir.mkdir(parents=True, exist_ok=True)

    iterations = get_user_input("Demosaic iterations", 1, int)
    use_tta = (
        get_user_input("Use TTA for demosaic? (yes/no)", "no", str).lower() == "yes"
    )

    image_files: List[Path] = (
        [input_path] if input_path.is_file() else list(input_path.glob("*.*"))
    )

    for img_path in tqdm(image_files, desc="Running Full Pipeline"):
        try:
            with Image.open(img_path) as original_img:
                original_img = original_img.convert("RGB")

                logger.debug(f"Detecting mask for {img_path.name}...")
                detected_mask_pil = detector.detect_mask(original_img)

                logger.debug(f"Restoring image {img_path.name}...")
                restored_image = restorer.restore_image(
                    original_img,
                    iterations=iterations,
                    use_tta=use_tta,
                    mask_pil=detected_mask_pil,
                )

                output_path = output_dir / f"{img_path.stem}_restored.png"
                restored_image.save(output_path)

        except Exception as e:
            logger.error(f"Failed to process {img_path.name} in pipeline: {e}")

    logger.info("Full pipeline completed.")


def handle_inference():
    logger.info("Starting image restoration process...")

    inference_type = get_user_input(
        "Choose inference type [1: Demosaic, 2: Detect Mosaic Mask, 3: Full Pipeline]",
        "1",
        str,
    )

    if inference_type == "1":
        handle_demosaic_inference()
    elif inference_type == "2":
        handle_detector_inference()
    elif inference_type == "3":
        handle_full_pipeline()
    else:
        logger.warning("Invalid choice. Please select 1, 2, or 3.")
        return


def display_menu():
    print("\n" + "=" * 28)
    print("      IMAGE PROCESSING      ")
    print("=" * 28)
    print("--- Data Preparation ---")
    print(" 1. Download New Images")
    print(" 2. Resize Images")
    print(" 3. Generate Mosaic Detector Data")
    print(" 4. Generate Mosaic Images")
    print(" 5. Generate Blurry Images")
    print(" 6. Add Noise to Images")
    print(" 7. Convert to Grayscale")
    print("")
    print("--- Training ---")
    print(" 8. Train Demosaic Model")
    print(" 9. Train Inpainting Model")
    print("10. Train Mosaic Detector Model")
    print("")
    print("--- Inference ---")
    print("11. Run Inference (Menu)")
    print("12. Run Full Pipeline (Detect & Demosaic)")
    print("")
    print(" 0. Exit")
    print("=" * 28)


def main():
    _prepare_directory()
    logging.config.dictConfig(LOGGING_CONFIG)

    log_queue = queue.Queue(-1)
    root_logger = logging.getLogger()

    original_handlers = []
    for handler in root_logger.handlers[:]:
        if isinstance(handler, (logging.FileHandler, RotatingFileHandler)):
            original_handlers.append(handler)
            root_logger.removeHandler(handler)

    queue_handler = QueueHandler(log_queue)
    root_logger.addHandler(queue_handler)

    listener = QueueListener(log_queue, *original_handlers, respect_handler_level=True)
    listener.start()

    logger.info("Image Processing Pipeline started with process-safe logging.")

    actions = {
        "1": ("Download New Images", handle_download),
        "2": ("Resize Images", handle_resize),
        "3": ("Generate Mosaic Detector Data", handle_generate_detector_data),
        "4": ("Generate Mosaic Images", handle_mosaic),
        "5": ("Generate Blurry Images", handle_blur),
        "6": ("Add Noise to Images", handle_noise),
        "7": ("Convert to Grayscale", handle_grayscale),
        "8": ("Train Demosaic Model", handle_training),
        "9": ("Train Inpainting Model", handle_inpainting_training),
        "10": ("Train Mosaic Detector Model", handle_train_detector),
        "11": ("Run Inference (Menu)", handle_inference),
        "12": ("Run Full Pipeline (Detect & Demosaic)", handle_full_pipeline),
    }

    start_time = time.time()
    try:
        while True:
            display_menu()
            choice = input(f"Enter your choice (0-{len(actions)}): ").strip()

            if choice == "0":
                logger.info("Exiting program. Goodbye!")
                break

            action_tuple = actions.get(choice)
            if action_tuple:
                action_name, action_func = action_tuple
                try:
                    logger.info(f"--- Executing: {action_name} ---")
                    action_func()
                    logger.info(f"Task '{action_name}' completed successfully.")
                except Exception as e:
                    logger.error(
                        f"An unexpected error occurred during the task: {e}",
                        exc_info=True,
                    )
            else:
                logger.warning("Invalid choice. Please try again.")

            time.sleep(1)
    finally:
        logger.info(
            f"Program completed in {(time.time() - start_time)/60:.2f} minutes."
        )
        listener.stop()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n")
        logger.info(f"Program interrupted by user. Exiting...")
    except Exception as e:
        print("\n")
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
