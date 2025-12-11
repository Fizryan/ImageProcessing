# DirectoryManager.py
# Manages directory setup and validation for the application

from pathlib import Path
from program.LoggingSetup import setup_logger

logger = setup_logger(__name__)


class DirectoryManager:
    """Handles directory creation and validation."""

    @staticmethod
    def setup_base_directories():
        """Create base directories required by the application."""
        directories = ["logs", "Samples", "Results", "Training"]

        for dir_name in directories:
            path = Path(dir_name)
            path.mkdir(parents=True, exist_ok=True)

        logger.info("Base directories initialized successfully")

    @staticmethod
    def validate_directory(path: str, create_if_missing: bool = False) -> Path:
        """
        Validate a directory path.

        Args:
            path: Directory path to validate
            create_if_missing: If True, create directory if it doesn't exist

        Returns:
            Path object if valid

        Raises:
            FileNotFoundError: If directory doesn't exist and create_if_missing=False
        """
        dir_path = Path(path).resolve()

        if not dir_path.exists():
            if create_if_missing:
                dir_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created directory: {dir_path}")
            else:
                raise FileNotFoundError(f"Directory not found: {dir_path}")

        if not dir_path.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {dir_path}")

        return dir_path

    @staticmethod
    def ensure_output_directory(output_path: str) -> Path:
        """
        Ensure output directory exists, create if needed.

        Args:
            output_path: Output directory path

        Returns:
            Path object
        """
        path = Path(output_path).resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path
