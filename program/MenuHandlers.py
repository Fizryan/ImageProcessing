# MenuHandlers.py
# Handles menu actions for the main application

from pathlib import Path
from program.LoggingSetup import setup_logger
from program.InputValidator import InputValidator
from program.DirectoryManager import DirectoryManager
from program.Resizing import Resizer
from program.Training import Trainer
from program.Inference import ImageRestorer
from program.Inference_Video import VideoRestorer
from program.Training_Config import TRAINING_CONFIG

logger = setup_logger(__name__)


class MenuHandlers:
    """Handles all menu actions for the application."""

    @staticmethod
    def handle_resize():
        """Handle image resizing process."""
        logger.info("Starting image resizing process...")

        input_dir = InputValidator.get_path_input(
            "Input directory", "dataset/clean_images", must_exist=True
        )
        output_dir = InputValidator.get_user_input(
            "Output directory", "dataset/resized_images"
        )

        height = InputValidator.get_numeric_range("Target height", 64, 2048, 256, int)
        width = InputValidator.get_numeric_range("Target width", 64, 2048, 448, int)

        DirectoryManager.ensure_output_directory(output_dir)

        resizer = Resizer(
            input_dir=str(input_dir),
            output_dir=output_dir,
            width=width,
            height=height,
        )
        resizer.process_images()

    @staticmethod
    def handle_training():
        """Handle training process."""
        logger.info("Starting training process...")
        trainer = Trainer(TRAINING_CONFIG)
        trainer.train()

    @staticmethod
    def handle_inpainting_training():
        """Handle inpainting training process."""
        logger.info("Starting inpainting training...")
        config = TRAINING_CONFIG.copy()
        config.update(
            {
                "clean_images_dir": "dataset/train/clean_images",
                "mask_images_dir": "dataset/train/mask_images",
                "val_clean_dir": "dataset/validation/clean_images",
                "val_mask_dir": "dataset/validation/mask_images",
                "task_type": "inpainting",
                "checkpoint_dir": "Training/Inpainting/checkpoints",
                "preview_dir": "Training/Inpainting/previews",
                "log_file": "Training/Inpainting/training.log",
            }
        )

        trainer = Trainer(config)
        trainer.train()

    @staticmethod
    def handle_image_restoration():
        """Handle image restoration process."""
        logger.info("Starting image restoration...")

        model_path = InputValidator.get_path_input(
            "Model path",
            "Training/checkpoints/best_model.pth",
            must_exist=True,
            is_directory=False,
        )
        input_path = InputValidator.get_path_input(
            "Input path (file or directory)", "Samples/test_images", must_exist=True
        )
        output_path = InputValidator.get_user_input(
            "Output directory", "Results/test_images"
        )

        iterations = InputValidator.get_numeric_range(
            "Number of iterations", 1, 5, 1, int
        )
        use_tta = InputValidator.confirm_action(
            "Use Test-Time Augmentation (TTA)? (May cause blur)", default=False
        )

        DirectoryManager.ensure_output_directory(output_path)

        restorer = ImageRestorer(
            model_path=str(model_path),
            img_height=256,
            img_width=448,
            base_channels=16,
        )

        if input_path.is_file():
            output_file = Path(output_path) / f"restored_{input_path.name}"
            restorer.restore_image_from_path(
                input_path=input_path,
                output_path=output_file,
                iterations=iterations,
                use_tta=use_tta,
            )
        else:
            restorer.process_directory(
                input_dir=input_path,
                output_dir=output_path,
                iterations=iterations,
                use_tta=use_tta,
            )

    @staticmethod
    def handle_video_restoration():
        """Handle video restoration process."""
        logger.info("Starting video restoration...")

        model_path = InputValidator.get_path_input(
            "Model path",
            "Training/checkpoints/best_model.pth",
            must_exist=True,
            is_directory=False,
        )
        input_video = InputValidator.get_path_input(
            "Input video path",
            "dataset/test_full_1.mp4",
            must_exist=True,
            is_directory=False,
        )
        output_video = InputValidator.get_user_input(
            "Output video path", "Results/restored_video.mp4"
        )

        merge_audio = InputValidator.confirm_action(
            "Merge audio from original video?", default=True
        )

        show_preview = InputValidator.confirm_action(
            "Show live preview during processing?", default=True
        )

        DirectoryManager.ensure_output_directory(Path(output_video).parent)

        restorer = VideoRestorer(
            model_path=str(model_path),
            tile_size=(448, 256),
            overlap=32,
            use_amp=True,
        )

        restorer.process_video(
            input_path=str(input_video),
            output_path=output_video,
            show_preview=show_preview,
            merge_audio=merge_audio,
        )
