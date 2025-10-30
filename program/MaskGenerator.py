# MaskGenerator.py
# This module generates a dataset for training a mosaic detection model.
# It creates images with random mosaic patches and their corresponding binary masks.

import logging
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Tuple, List

import numpy as np
from PIL import Image, ImageDraw, UnidentifiedImageError
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def create_random_mask(width: int, height: int) -> Image.Image:
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    num_shapes = random.randint(3, 8)

    for _ in range(num_shapes):
        shape_type = random.choice(["rectangle", "ellipse"])
        left = random.randint(0, width - 50)
        top = random.randint(0, height - 50)
        right = random.randint(left + 40, width)
        bottom = random.randint(top + 40, height)

        if shape_type == "rectangle":
            draw.rectangle([left, top, right, bottom], fill=255)
        elif shape_type == "ellipse":
            draw.ellipse([left, top, right, bottom], fill=255)

    return mask


def apply_mosaic(img: Image.Image, block_size: int) -> Image.Image:
    small_img = img.resize(
        (max(1, img.width // block_size), max(1, img.height // block_size)),
        Image.Resampling.NEAREST,
    )
    return small_img.resize(img.size, Image.Resampling.NEAREST)


def process_detector_data_worker(
    input_path: Path,
    input_save_dir: Path,
    target_save_dir: Path,
    block_size: int,
    max_retries: int,
    overwrite: bool,
) -> Tuple[str, str]:
    input_save_path = input_save_dir / input_path.name
    target_save_path = target_save_dir / input_path.name

    if (input_save_path.exists() or target_save_path.exists()) and not overwrite:
        return "SKIPPED", f"File already exists for {input_path.name}"

    for attempt in range(max_retries):
        try:
            with Image.open(input_path) as img:
                img = img.convert("RGB")
                mask = create_random_mask(img.width, img.height)
                mosaic_img = apply_mosaic(img, block_size)

                composite_img = Image.composite(mosaic_img, img, mask)

                composite_img.save(input_save_path)
                mask.save(target_save_path)

            return "SUCCESS", f"Generated data for {input_path.name}"
        except (UnidentifiedImageError, IOError) as e:
            return "ERROR", f"Corrupt or unreadable image: {input_path.name} ({e})"
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            return (
                "ERROR",
                f"Failed to process {input_path.name} after {max_retries} attempts: {e}",
            )


class MaskGenerator:
    def __init__(
        self,
        clean_dir="dataset/resized_images",
        output_dir="dataset/detector_data",
        block_size=16,
        max_workers=None,
        max_retries=3,
        overwrite=False,
        val_split=0.15,
    ):
        self.clean_dir = Path(clean_dir)
        if not self.clean_dir.is_dir():
            raise FileNotFoundError(
                f"Clean image directory not found: {self.clean_dir}"
            )

        self.output_dir = Path(output_dir)
        self.train_input_save_dir = self.output_dir / "train" / "input"
        self.train_target_save_dir = self.output_dir / "train" / "target"
        self.val_input_save_dir = self.output_dir / "val" / "input"
        self.val_target_save_dir = self.output_dir / "val" / "target"

        self.block_size = block_size
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.overwrite = overwrite
        self.logger = logging.getLogger(self.__class__.__name__)

        if not 0 < val_split < 1:
            raise ValueError("val_split must be between 0 and 1.")
        self.val_split = val_split

        self._prepare_directories()

    def _prepare_directories(self):
        self.train_input_save_dir.mkdir(parents=True, exist_ok=True)
        self.train_target_save_dir.mkdir(parents=True, exist_ok=True)
        self.val_input_save_dir.mkdir(parents=True, exist_ok=True)
        self.val_target_save_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Output directories prepared at {self.output_dir}")

    def _run_generation(
        self, image_files: List[Path], input_dir: Path, target_dir: Path, desc: str
    ):
        counters = {"SUCCESS": 0, "SKIPPED": 0, "ERROR": 0}
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            tasks = {
                executor.submit(
                    process_detector_data_worker,
                    path,
                    input_dir,
                    target_dir,
                    self.block_size,
                    self.max_retries,
                    self.overwrite,
                ): path.name
                for path in image_files
            }

            pbar = tqdm(as_completed(tasks), total=len(tasks), desc=desc, ncols=100)
            for future in pbar:
                status, message = future.result()
                counters[status] += 1
                if status == "ERROR":
                    self.logger.error(message)
                else:
                    self.logger.debug(message)
        return counters

    def generate_data(self):
        allowed_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
        image_files = [
            p
            for p in self.clean_dir.iterdir()
            if p.suffix.lower() in allowed_extensions
        ]

        if not image_files:
            self.logger.warning("No clean images found in the input directory.")
            return

        self.logger.info(f"Found {len(image_files)} clean images to process.")

        train_files, val_files = train_test_split(
            image_files, test_size=self.val_split, random_state=42
        )
        self.logger.info(
            f"Splitting data: {len(train_files)} for training, {len(val_files)} for validation."
        )

        self.logger.info("Generating training data...")
        train_counters = self._run_generation(
            train_files,
            self.train_input_save_dir,
            self.train_target_save_dir,
            "Generating Train Data",
        )
        self.logger.info(
            f"Training data generation complete: {train_counters['SUCCESS']} succeeded, {train_counters['SKIPPED']} skipped, {train_counters['ERROR']} errors."
        )

        self.logger.info("Generating validation data...")
        val_counters = self._run_generation(
            val_files,
            self.val_input_save_dir,
            self.val_target_save_dir,
            "Generating Val Data",
        )
        self.logger.info(
            f"Validation data generation complete: {val_counters['SUCCESS']} succeeded, {val_counters['SKIPPED']} skipped, {val_counters['ERROR']} errors."
        )
