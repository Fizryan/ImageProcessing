import sys
import argparse
from pathlib import Path
from typing import Optional

from program.LoggingManager import LoggingManager
from program.DirectoryManager import DirectoryManager
from program.Utils import Utils
from program.ProgramConfig import MAIN_CONFIG, DIRECTORIES_CONFIG
from program.Trainer import Trainer
from program.Inference import ImageRestorer

logger = LoggingManager.setup_logging("MainSystem", log_file_path="logs/app.log")


def parse_args():
    parser = argparse.ArgumentParser(description="Image Restoration Pipeline")
    parser.add_argument("--mode", choices=["train", "inference"], help="Mode operasi")
    parser.add_argument("--input", type=str, help="Input folder (Inference only)")
    parser.add_argument("--output", type=str, help="Output folder (Inference only)")
    parser.add_argument("--model", type=str, help="Path to .pth model (Inference only)")
    parser.add_argument(
        "--tta", action="store_true", help="Enable Test-Time Augmentation"
    )
    return parser.parse_args()


def run_training_mode():
    logger.info(">>> MODE: TRAINING SELECTED")
    logger.info(f"Initializing Trainer (Version: {MAIN_CONFIG['version']})...")

    trainer = Trainer(config=MAIN_CONFIG)
    trainer.run()


def run_inference_mode(
    input_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    model_path: Optional[str] = None,
    use_tta: bool = False,
):
    logger.info(">>> MODE: INFERENCE SELECTED")

    if not model_path:
        default_model = Path(DIRECTORIES_CONFIG["checkpoint_dir"]) / "best_model.pth"
        if default_model.exists():
            model_path = str(default_model)
            logger.info(
                f"No model path provided. Using default best model: {model_path}"
            )
        else:
            model_path = input("Enter model path (.pth): ").strip()

    if not Path(model_path).exists():
        logger.error(f"Model file not found: {model_path}")
        return

    if not input_dir:
        input_dir = input("Enter Input Folder Path (Degraded Images): ").strip()
    if not output_dir:
        output_dir = input("Enter Output Folder Path (Save Result): ").strip()

    if not Path(input_dir).exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return

    try:
        restorer = ImageRestorer(model_path=model_path)
        restorer.process_directory(
            input_dir=input_dir,
            output_dir=output_dir,
            use_tta=use_tta,
        )
    except Exception as e:
        logger.error(f"Inference failed: {e}")


def main():
    logger.info("--- Application Started ---")

    try:
        logger.info("Initializing Project Structure...")
        DirectoryManager.setup_directories(config=DIRECTORIES_CONFIG)

        if Utils.get_gpu_load() is not None:
            logger.info("GPU Detected & Ready.")
            Utils.get_gpu_info()

        args = parse_args()

        if args.mode == "train":
            run_training_mode()

        elif args.mode == "inference":
            run_inference_mode(args.input, args.output, args.model, args.tta)

        else:
            print("\n" + "=" * 40)
            print("   IMAGE RESTORATION AI - MAIN MENU")
            print("=" * 40)
            print("1. Train New Model")
            print("2. Run Inference (Restore Images)")
            print("=" * 40)

            choice = input("Enter choice (1/2): ").strip()

            if choice == "1":
                run_training_mode()
            elif choice == "2":
                tta_input = (
                    input("Use TTA (High Quality, Slower)? (y/n): ").lower().strip()
                )
                use_tta = tta_input == "y"
                run_inference_mode(use_tta=use_tta)
            else:
                logger.warning("Invalid choice selected.")

    except KeyboardInterrupt:
        logger.warning("\nApplication stopped by User (KeyboardInterrupt).")
        sys.exit(0)

    except Exception as e:
        logger.critical(
            "CRITICAL: Uncaught exception caused application shutdown.", exc_info=True
        )
        sys.exit(1)

    finally:
        logger.info("--- Application Shutdown ---")


if __name__ == "__main__":
    main()
