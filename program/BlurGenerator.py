# BlurGenerator.py
# This module generates blurry images from a set of clean images by applying a Gaussian blur effect.

import logging
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Tuple

from PIL import Image, ImageFilter, UnidentifiedImageError
from tqdm import tqdm

def apply_blur(img: Image.Image, radius: float) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=radius))

def process_blur_worker(input_path: Path, output_dir: Path, blur_radius_range: Tuple[float, float],
        max_retries: int, overwrite: bool) -> Tuple[str, Path | str]:
    output_path = output_dir / input_path.name
    
    if output_path.exists() and not overwrite:
        return "SKIPPED", output_path

    blur_radius = random.uniform(*blur_radius_range)

    for attempt in range(max_retries):
        try:
            with Image.open(input_path) as img:
                img.verify()

            with Image.open(input_path) as img:
                blurry_img = apply_blur(img, blur_radius)
                blurry_img.save(output_path)
            
            return "SUCCESS", output_path
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            return "ERROR", f"Failed to process {input_path.name} after {max_retries} attempts: {e}"

class BlurGenerator:
    def __init__(self, input_dir="dataset/clean_images", output_dir="dataset/blurry_images", 
            blur_radius_range=(1.0, 3.0), max_workers=4, max_retries=3, overwrite=False):
        
        self.input_dir = Path(input_dir)
        if not self.input_dir.is_dir():
            raise FileNotFoundError(f"Input directory not found: {self.input_dir}")
            
        self.output_dir = Path(output_dir)
        self.blur_radius_range = blur_radius_range
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.overwrite = overwrite
        self.logger = logging.getLogger(self.__class__.__name__)

        if not (isinstance(blur_radius_range, tuple) and len(blur_radius_range) == 2 and blur_radius_range[0] < blur_radius_range[1]):
            raise ValueError("blur_radius_range must be a tuple of two numbers (min, max).")
            
        self._prepare_directory()

    def _prepare_directory(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Output directory prepared at {self.output_dir}")

    def _verify_images(self, image_files: list) -> list:
        self.logger.info("Verifying all images before processing...")
        valid_files = []
        corrupt_files = 0
        pbar = tqdm(image_files, desc="Verifying Images", ncols=100)
        for path in pbar:
            try:
                with Image.open(path) as img:
                    img.verify()
                valid_files.append(path)
            except (UnidentifiedImageError, IOError):
                self.logger.error(f"Corrupt or unreadable image detected and will be skipped: {path.name}")
                corrupt_files += 1
        if corrupt_files > 0:
            self.logger.warning(f"Total corrupt images skipped: {corrupt_files}")
        return valid_files

    def generate_blurry_images(self):
        allowed_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
        image_files = [p for p in self.input_dir.iterdir() if p.suffix.lower() in allowed_extensions]
        
        if not image_files:
            self.logger.warning("No images found in the input directory.")
            return

        valid_image_files = self._verify_images(image_files)

        if not valid_image_files:
            self.logger.warning("No valid images left to process after verification.")
            return
            
        self.logger.info(f"Starting to process {len(valid_image_files)} valid images.")
        counters = {"SUCCESS": 0, "SKIPPED": 0, "ERROR": 0}
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            tasks = {
                executor.submit(process_blur_worker, path, self.output_dir, self.blur_radius_range, self.max_retries, self.overwrite): path.name
                for path in valid_image_files
            }

            pbar = tqdm(as_completed(tasks), total=len(valid_image_files), desc="Generating Blurry Images", ncols=100)
            for future in pbar:
                try:
                    status, result = future.result()
                    counters[status] += 1
                    if status == "ERROR":
                        self.logger.error(result)
                    else:
                        self.logger.debug(f"{status}: {result.name}")
                except Exception as e:
                    counters["ERROR"] += 1
                    self.logger.error(f"An unhandled exception occurred in worker: {e}")

        self.logger.info("Finished generating all blurry images.")
        for status, count in counters.items():
            self.logger.info(f"{status}: {count}")