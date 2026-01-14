# main.py

import sys
from program.LoggingSetup import setup_logger
from program.DirectoryManager import DirectoryManager
from program.InputValidator import InputValidator
from program.MenuHandlers import MenuHandlers

logger = setup_logger(__name__, log_file="logs/app.log")


def display_menu():
    """Display the main menu options."""
    print("\n" + "=" * 70)
    print(" " * 18 + "IMAGE PROCESSING PIPELINE")
    print("=" * 70)
    print("📂  Data Preparation")
    print("    1. Resize Images")
    print("    2. Add Mosaic/Degradation")
    print("🎯  Training")
    print("    3. Train Demosaic Model")
    print("    4. Train Inpainting Model")
    print("🔮  Inference")
    print("    5. Image Restoration (Single/Batch)")
    print("    6. Video Restoration")
    print("\n👋  Exit")
    print("    0. Exit")
    print("=" * 70)


def main():
    """Main application entry point."""
    logger.info("=" * 70)
    logger.info("Starting Image Processing Pipeline Application")
    logger.info("=" * 70)

    try:
        DirectoryManager.setup_base_directories()
        logger.info("✓ Base directories initialized successfully")
    except Exception as e:
        logger.error(f"✗ Failed to setup directories: {e}")
        sys.exit(1)

    menu_actions = {
        "1": ("Resize Images", MenuHandlers.handle_resize),
        "2": ("Add Mosaic/Degradation", MenuHandlers.handle_add_mosaic),
        "3": ("Train Demosaic Model", MenuHandlers.handle_training),
        "4": ("Train Inpainting Model", MenuHandlers.handle_inpainting_training),
        "5": ("Image Restoration", MenuHandlers.handle_image_restoration),
        "6": ("Video Restoration", MenuHandlers.handle_video_restoration),
    }

    while True:
        try:
            display_menu()
            choice = input("👉 Select an option: ").strip()

            if choice == "0":
                logger.info("=" * 70)
                logger.info("Application terminated by user")
                logger.info("=" * 70)
                print("\n✓ Goodbye!\n")
                break

            if choice in menu_actions:
                action_name, action_func = menu_actions[choice]
                logger.info("-" * 70)
                logger.info(f"User selected: {action_name}")
                logger.info("-" * 70)

                try:
                    action_func()
                    logger.info("=" * 70)
                    logger.info(f"✓ {action_name} completed successfully")
                    logger.info("=" * 70)
                except KeyboardInterrupt:
                    logger.warning("-" * 70)
                    logger.warning(f"⚠ {action_name} cancelled by user")
                    logger.warning("-" * 70)
                    print("\n⚠ Operation cancelled\n")
                except Exception as e:
                    logger.error("=" * 70)
                    logger.error(f"✗ {action_name} failed: {e}")
                    logger.error("=" * 70)
                    print(f"\n✗ Error: {e}\n")
            else:
                print("\n⚠ Invalid choice. Please try again.\n")

        except KeyboardInterrupt:
            logger.info("=" * 70)
            logger.info("Application interrupted by user")
            logger.info("=" * 70)
            print("\n\n✓ Goodbye!\n")
            break
        except Exception as e:
            logger.error("=" * 70)
            logger.error(f"✗ Unexpected error: {e}")
            logger.error("=" * 70)
            print(f"\n✗ Unexpected error: {e}\n")


if __name__ == "__main__":
    main()
