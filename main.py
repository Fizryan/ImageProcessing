import sys
from program.LoggingManager import LoggingManager
from program.DirectoryManager import DirectoryManager
from program.Utils import Utils
from program.ProgramConfig import MAIN_CONFIG, DIRECTORIES_CONFIG
from program.Trainer import Trainer


def main():
    logger = LoggingManager.setup_logging(__name__, log_file_path="logs/system.log")
    logger.info("--- Application Started ---")

    try:
        logger.info("Initializing Project Structure...")
        DirectoryManager.setup_directories(config=DIRECTORIES_CONFIG)

        logger.info("Checking Hardware Resources...")
        Utils.get_gpu_info()

        logger.info(f"Initializing Trainer (Version: {MAIN_CONFIG['version']})...")
        trainer = Trainer(config=MAIN_CONFIG)

        trainer.run()

    except KeyboardInterrupt:
        logger.warning("Application stopped by User (KeyboardInterrupt).")
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
