# InputValidator.py
# Input validation utilities for user interactions

from typing import Any, Callable, List, Union
from pathlib import Path
from program.LoggingSetup import setup_logger

logger = setup_logger(__name__)


class InputValidator:
    """Handles user input validation and prompts."""

    @staticmethod
    def get_user_input(
        prompt: str,
        default: Any,
        target_type: Callable = str,
        choices: List[Any] = None,
    ) -> Any:
        """
        Get validated user input with default value.

        Args:
            prompt: Prompt message to display
            default: Default value if user just presses Enter
            target_type: Type to convert input to (str, int, float, bool)
            choices: Optional list of valid choices

        Returns:
            Validated user input
        """
        while True:
            try:
                user_input = input(f"{prompt} [{default}]: ").strip()

                if not user_input:
                    return default

                if target_type == bool:
                    return user_input.lower() in ["y", "yes", "true", "1"]

                converted_value = target_type(user_input)

                if choices is not None and converted_value not in choices:
                    logger.warning(f"Invalid choice. Please select from: {choices}")
                    continue

                return converted_value

            except ValueError:
                logger.error(f"Invalid input. Expected {target_type.__name__}")
            except KeyboardInterrupt:
                logger.info("\nOperation cancelled by user")
                raise

    @staticmethod
    def get_path_input(
        prompt: str, default: str, must_exist: bool = False, is_directory: bool = True
    ) -> Path:
        """
        Get validated path input from user.

        Args:
            prompt: Prompt message
            default: Default path
            must_exist: If True, path must already exist
            is_directory: If True, validate as directory, else as file

        Returns:
            Validated Path object
        """
        while True:
            try:
                path_str = InputValidator.get_user_input(prompt, default, str)
                path = Path(path_str).resolve()

                if must_exist and not path.exists():
                    logger.error(f"Path does not exist: {path}")
                    continue

                if must_exist and is_directory and not path.is_dir():
                    logger.error(f"Path is not a directory: {path}")
                    continue

                if must_exist and not is_directory and not path.is_file():
                    logger.error(f"Path is not a file: {path}")
                    continue

                return path

            except Exception as e:
                logger.error(f"Invalid path: {e}")

    @staticmethod
    def confirm_action(message: str, default: bool = False) -> bool:
        """
        Ask user for confirmation.

        Args:
            message: Confirmation message
            default: Default response

        Returns:
            True if user confirms, False otherwise
        """
        default_str = "Y/n" if default else "y/N"
        response = input(f"{message} [{default_str}]: ").strip().lower()

        if not response:
            return default

        return response in ["y", "yes"]

    @staticmethod
    def get_numeric_range(
        prompt: str,
        min_value: Union[int, float],
        max_value: Union[int, float],
        default: Union[int, float],
        value_type: type = int,
    ) -> Union[int, float]:
        """
        Get numeric input within a specified range.

        Args:
            prompt: Prompt message
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            default: Default value
            value_type: int or float

        Returns:
            Validated numeric value
        """
        while True:
            try:
                value = InputValidator.get_user_input(
                    f"{prompt} ({min_value}-{max_value})", default, value_type
                )

                if min_value <= value <= max_value:
                    return value
                else:
                    logger.warning(f"Value must be between {min_value} and {max_value}")

            except Exception as e:
                logger.error(f"Invalid input: {e}")
