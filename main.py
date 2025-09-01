# main.py
# ImageProcessing Pipeline Main Script

import logging
import logging.config
import time
import queue
from logging.handlers import QueueHandler, QueueListener
from typing import Any, Callable
from pathlib import Path
from tqdm import tqdm

from program.Logging_Config import LOGGING_CONFIG
from program.Training_Config import TRAINING_CONFIG

from program.CollectingImage import CollectingImage
from program.Resizing import Resizer
from program.Noise import NoiseGenerator
from program.Grayscaling import Grayscaler
from program.Mosaic import MosaicGenerator
from program.BlurGenerator import BlurGenerator
from program.Training import Trainer, LRFinder, CombinedRestorationDataset
from program.Architecture import UNetLite
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
    height = get_user_input("Target height", 256, int)
    width = get_user_input("Target width", 448, int)
    resizer = Resizer(
        input_dir=input_dir, output_dir=output_dir, width=width, height=height
    )
    resizer.process_images()


def handle_noise():
    logger.info("Starting noise generation process...")
    input_dir = get_user_input("Input directory", "dataset/re_augmented_images")
    mask_dir = get_user_input("Mask directory", "dataset/mask_images")
    output_dir = get_user_input("Output directory", "dataset/noisy_images")
    noise_level = get_user_input("Noise level (0.0 to 1.0)", 0.1, float)
    overwrite = (
        get_user_input("Overwrite existing files? (yes/no)", "no", str).lower() == "yes"
    )
    overwrite = True if overwrite else False
    noiser = NoiseGenerator(
        input_dir=input_dir,
        noise_dir=output_dir,
        mask_dir=mask_dir,
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
    overwrite = True if overwrite else False
    grayscaler = Grayscaler(
        input_dir=input_dir,
        output_dir=output_dir,
        output_format=output_format or None,
        overwrite=overwrite,
    )
    grayscaler.process_images()


def handle_mosaic():
    logger.info("Starting mosaic image generation process...")
    input_dir = get_user_input("Input directory", "dataset/re_augmented_images")
    mask_dir = get_user_input("Mask directory", "dataset/mask_images")
    output_dir = get_user_input("Output directory", "dataset/mosaic_images")
    block_size = get_user_input("Block size for mosaic effect", 25, int)
    overwrite = (
        get_user_input("Overwrite existing files? (yes/no)", "no", str).lower() == "yes"
    )
    overwrite = True if overwrite else False
    mosaic_gen = MosaicGenerator(
        input_dir=input_dir,
        output_dir=output_dir,
        mask_dir=mask_dir,
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
    overwrite = True if overwrite else False
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


def handle_lr_finder():
    logger.info("Starting Learning Rate Finder...")
    if TRAINING_CONFIG.get("training_mode") != "restoration":
        logger.error("LR Finder is only implemented for 'restoration' mode.")
        return

    try:
        import torch
        import torch.optim as optim
        from torch.utils.data import DataLoader

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        image_size = (
            TRAINING_CONFIG["img_height"],
            TRAINING_CONFIG["img_width"],
        )
        dataset = CombinedRestorationDataset(
            clean_dir=Path(TRAINING_CONFIG["data_dirs"]["train"]["clean"]),
            task_dirs=TRAINING_CONFIG["data_dirs"]["train"],
            image_size=image_size,
            cache_limit=10,
        )
        dataloader = DataLoader(dataset, **TRAINING_CONFIG["dataloader_params"])
        model = UNetLite(
            in_channels=4,
            out_channels=3,
            base_channels=TRAINING_CONFIG.get("base_channels", 16),
        ).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=1e-7)
        criterion = torch.nn.L1Loss()

        lr_finder = LRFinder(model, optimizer, criterion, device, dataloader)
        lr_finder.range_test(end_lr=1, num_iter=200)
        lr_finder.plot(save_path="Training/lr_finder_plot.png")
    except Exception as e:
        logger.error(f"Failed to run LR Finder: {e}", exc_info=True)


def handle_inference():
    logger.info("Starting image restoration process...")
    model_path = get_user_input(
        "Path to the trained model", "Training/checkpoints/best_model.pth"
    )
    if not Path(model_path).exists():
        logger.error(f"Model file not found: {model_path}")
        return

    img_height = get_user_input(
        "Image height model was trained on", TRAINING_CONFIG.get("img_height", 256), int
    )
    img_width = get_user_input(
        "Image width model was trained on", TRAINING_CONFIG.get("img_width", 448), int
    )
    base_channels = get_user_input(
        "Model base channels", TRAINING_CONFIG.get("base_channels", 12), int
    )
    restorer = ImageRestorer(
        model_path=model_path,
        img_height=img_height,
        img_width=img_width,
        base_channels=base_channels,
    )

    input_path_str = get_user_input(
        "Path to input image or directory", "Samples/images"
    )
    input_path = Path(input_path_str)
    if not input_path.exists():
        logger.error(f"Input path does not exist: {input_path}")
        return

    task_type = get_user_input(
        "Task type (noise, mosaic, inpainting, blur)", "inpainting", str
    )
    iterations = get_user_input("Number of iterations (for bettler results)", 5, int)

    use_tta = (
        get_user_input(
            "Use Test-Time Augmentation (TTA) for better quality? (yes/no)", "yes", str
        ).lower()
        == "yes"
    )

    final_blend_alpha = 0.0
    use_poisson = False
    use_multi_scale = False
    use_edge_aware = False
    adaptive_iterations = False

    if task_type == "inpainting":
        strategy = get_user_input(
            "Choose strategy [1: Iterative, 2: Multi-Scale, 3: Adaptive]", "1", str
        )

        if strategy == "2":
            use_multi_scale = True
            logger.info("Using Multi-Scale Inpainting strategy.")
        elif strategy == "3":
            adaptive_iterations = True
            logger.info(
                f"Using Adaptive Inpainting strategy with max {iterations} iterations."
            )
        else:
            logger.info(
                f"Using standard Iterative Inpainting with {iterations} iterations."
            )
            use_poisson = (
                get_user_input(
                    "Use Poisson Blending for seamless results? (yes/no)", "yes", str
                ).lower()
                == "yes"
            )

        use_edge_aware = (
            get_user_input(
                "Use Edge-Aware mask refinement? (yes/no)", "no", str
            ).lower()
            == "yes"
        )
    else:
        final_blend_alpha = get_user_input(
            "Final blend with original (0.0 to 1.0, 0=off)", 0.0, float
        )

    if input_path.is_file():
        output_path_str = get_user_input(
            "Path to save the restored image", f"Results/{input_path.stem}_restored.png"
        )
        output_path = Path(output_path_str)
        mask_path = None
        if task_type == "inpainting":
            mask_path_str = get_user_input(
                "Path to the mask image", f"Samples/masks/{input_path.name}"
            )
            mask_path = Path(mask_path_str)
            if not mask_path.exists():
                logger.error(f"Mask path does not exist: {mask_path}")
                return
        restorer.restore_image_from_path(
            input_path,
            output_path,
            task_type,
            mask_path=mask_path,
            iterations=iterations,
            use_tta=use_tta,
            final_blend_alpha=final_blend_alpha,
            use_poisson_blending=use_poisson,
            use_multi_scale=use_multi_scale,
            use_edge_aware=use_edge_aware,
            adaptive_iterations=adaptive_iterations,
        )

    elif input_path.is_dir():
        output_dir_str = get_user_input("Path to save the restored images", "Results")
        output_dir = Path(output_dir_str)
        mask_dir = None
        if task_type == "inpainting":
            mask_dir_str = get_user_input(
                "Path to the directory of masks (must have same filenames)",
                "Samples/masks",
            )
            mask_dir = Path(mask_dir_str)
            if not mask_dir.exists() or not mask_dir.is_dir():
                logger.error(
                    f"Mask directory does not exist or is not a directory: {mask_dir}"
                )
                return

        restorer.process_directory(
            input_dir=input_path,
            output_dir=output_dir,
            task_type=task_type,
            mask_dir=mask_dir,
            iterations=iterations,
            use_tta=use_tta,
            final_blend_alpha=final_blend_alpha,
            use_poisson_blending=use_poisson,
            use_multi_scale=use_multi_scale,
            use_edge_aware=use_edge_aware,
            adaptive_iterations=adaptive_iterations,
        )
    else:
        logger.error(f"Input path is not a valid file or directory: {input_path}")
        return

    logger.info("Inference completed.")


def handle_dataset():
    logger.info("Starting dataset preparation process...")
    input_dir = get_user_input("Input directory for dataset", "dataset")
    output_dir = get_user_input("Output directory for dataset", "dataset")

    input_user = (
        get_user_input("download new images? (yes/no)", "no", str).lower() == "yes"
    )
    if input_user:
        logger.info("Starting image download process...")
        count = get_user_input("Number of images to download", 20, int)
        collector = CollectingImage(
            count=count, max_workers=5, save_path=output_dir + "/clean_images"
        )
        collector.download_images()

    overwrite = (
        get_user_input("Overwrite existing files? (yes/no)", "no", str).lower() == "yes"
    )
    overwrite = True if overwrite else False

    logger.info("Starting image resizing process...")
    height = get_user_input("Target height", 256, int)
    width = get_user_input("Target width", 448, int)
    resizer = Resizer(
        input_dir=input_dir + "/clean_images",
        output_dir=output_dir + "/resized_images",
        width=width,
        height=height,
        overwrite=overwrite,
    )
    resizer.process_images()

    logger.info("Starting noise generation process...")
    noise_level = get_user_input("Noise level (0.0 to 1.0)", 0.1, float)
    noiser = NoiseGenerator(
        input_dir=input_dir + "/resized_images",
        noise_dir=output_dir + "/noisy_images",
        noise_level=noise_level,
        overwrite=overwrite,
    )
    noiser.generate_noisy_images()

    logger.info("Starting mosaic image generation process...")
    block_size = get_user_input("Block size for mosaic effect (1 to 100)", 5, int)
    mosaic_gen = MosaicGenerator(
        input_dir=input_dir + "/resized_images",
        output_dir=output_dir + "/mosaic_images",
        block_size=block_size,
        overwrite=overwrite,
    )
    mosaic_gen.generate_mosaic_images()

    logger.info("Starting grayscale conversion process...")
    output_format = get_user_input(
        "Output format (e.g., PNG, JPEG, or leave empty to keep original)", "PNG"
    )
    grayscaler = Grayscaler(
        input_dir=input_dir + "/resized_images",
        output_dir=output_dir + "/grayscale_images",
        output_format=output_format or None,
        overwrite=overwrite,
    )
    grayscaler.process_images()

    logger.info("Starting blur generation process...")
    blur_radius_range = get_user_input(
        "Blur radius range (min, max)", (1.0, 3.0), tuple
    )
    blur_gen = BlurGenerator(
        input_dir=input_dir + "/resized_images",
        output_dir=output_dir + "/blurry_images",
        blur_radius_range=blur_radius_range,
        overwrite=overwrite,
    )
    blur_gen.generate_blurry_images()


def display_menu():
    print("\n" + "=" * 28)
    print("      IMAGE PROCESSING")
    print("=" * 28)
    print("  1. Download New Images")
    print("  2. Resize Images")
    print("  3. Add Noise to Images")
    print("  4. Convert Images to Grayscale")
    print("  5. Generate Mosaic Images")
    print("  6. Generate Blurry Images")
    print("  7. Train Model")
    print("  8. Run Inference (Image Restoration)")
    print("  9. Prepare Dataset")
    print(" 10. Find Optimal Learning Rate")
    print("  0. Exit")
    print("=" * 28)


def main():
    _prepare_directory()
    logging.config.dictConfig(LOGGING_CONFIG)

    log_queue = queue.Queue(-1)
    root_logger = logging.getLogger()

    original_handlers = []
    for handler in root_logger.handlers[:]:
        if isinstance(
            handler, (logging.FileHandler, logging.handlers.RotatingFileHandler)
        ):
            original_handlers.append(handler)
            root_logger.removeHandler(handler)

    queue_handler = QueueHandler(log_queue)
    root_logger.addHandler(queue_handler)

    listener = QueueListener(log_queue, *original_handlers, respect_handler_level=True)
    listener.start()

    logger.info("Image Processing Pipeline started with process-safe logging.")

    actions = {
        "1": handle_download,
        "2": handle_resize,
        "3": handle_noise,
        "4": handle_grayscale,
        "5": handle_mosaic,
        "6": handle_blur,
        "7": handle_training,
        "8": handle_inference,
        "9": handle_dataset,
        "10": handle_lr_finder,
    }

    try:
        while True:
            display_menu()
            choice = input(f"Enter your choice (0-{len(actions)}): ").strip()

            if choice == "0":
                logger.info("Exiting program. Goodbye!")
                break

            action = actions.get(choice)
            if action:
                try:
                    action()
                    logger.info(f"Task '{action.__name__}' completed successfully.")
                except Exception as e:
                    logger.error(
                        f"An unexpected error occurred during the task: {e}",
                        exc_info=True,
                    )
            else:
                logger.warning("Invalid choice. Please try again.")

            time.sleep(1)
    finally:
        listener.stop()


if __name__ == "__main__":
    try:
        start_time = time.time()
        main()
        logger.info(
            f"Program completed in {(time.time() - start_time)/60:.2f} minutes."
        )
    except KeyboardInterrupt:
        print("\n")
        logger.info(
            f"Program interrupted by user. Exiting... {(time.time() - start_time)/60:.2f} minutes."
        )
    except Exception as e:
        print("\n")
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
