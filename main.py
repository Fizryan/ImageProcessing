from program.LoggingManager import LoggingManager
from program.DirectoryManager import DirectoryManager
from program.Utils import Utils
from program.ProgramConfig import DIRECTORIES_CONFIG


def main():
    logger = LoggingManager.setup_logging(__name__)
    logger.info("--- Application started ---")
    try:
        DirectoryManager.setup_directories(config=DIRECTORIES_CONFIG)
    except Exception as e:
        logger.critical("Uncaught exception caused application shutdown", exc_info=e)
        raise e


if __name__ == "__main__":
    main()
