from pathlib import Path
from typing import Optional, List, Any, Union

from program.LoggingManager import LoggingManager

logger = LoggingManager.setup_logging(__name__)


class DirectoryManager:
    def __init__(self):
        raise RuntimeError("DirectoryManager is a static class")

    @staticmethod
    def setup_directories(config: Union[dict[str, Any], List[str]]):
        if not config:
            logger.warning("No config provided")
            return

        if isinstance(config, list):
            for path_str in config:
                if path_str is None:
                    continue
                path = Path(path_str)
                if path.suffix:
                    DirectoryManager.create_dir(path.parent)
                else:
                    DirectoryManager.create_dir(path)
            return

        file_keys = {"log_file"}

        for key, path_str in config.items():
            if path_str is None:
                logger.info(f"Skipping '{key}' as it is None")
                continue

            path: Path = Path(path_str)

            if key in file_keys or path.suffix:
                DirectoryManager.create_dir(path.parent)
            else:
                DirectoryManager.create_dir(path)

    @staticmethod
    def create_dir(path: Path) -> None:
        if not path.exists():
            logger.info(f"Creating directory '{path}'")
            try:
                path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Directory '{path}' created successfully")
            except Exception as e:
                logger.error(f"Error creating directory '{path}': {e}")
        else:
            logger.info(f"Directory '{path}' already exists")
