# Mosaic.py
# This module generates mosaic images from a set of input images by applying a pixelation effect.

import logging
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from typing import Tuple
import random

def _create_full_mosaic(img: Image.Image, block_size: int) -> Image.Image:
    small_img = img.resize(
        (max(1, img.width // block_size), max(1, img.height // block_size)),
        Image.Resampling.NEAREST
    )
    return small_img.resize(img.size, Image.Resampling.NEAREST)

def apply_mosaic(img: Image.Image, block_size: int) -> Image.Image:
    output_img = img.copy()
    
    full_mosaic_img = _create_full_mosaic(img, block_size)
    
    width, height = img.size
    
    num_regions = random.randint(1, 3)
    
    for _ in range(num_regions):
        region_w = random.randint(int(width * 0.2), int(width * 0.6))
        region_h = random.randint(int(height * 0.2), int(height * 0.6))
        
        offset_x = random.randint(0, width - region_w)
        offset_y = random.randint(0, height - region_h)
        
        box = (offset_x, offset_y, offset_x + region_w, offset_y + region_h)
        
        mosaic_region = full_mosaic_img.crop(box)
        
        output_img.paste(mosaic_region, box)
        
    return output_img

def process_mosaic_worker(input_path: Path, output_dir: Path, block_size: int, 
        max_retries: int, overwrite: bool) -> Tuple[str, Path | str]:
    output_path = output_dir / input_path.name
    
    if output_path.exists() and not overwrite:
        return "SKIPPED", output_path

    for attempt in range(max_retries):
        try:
            with Image.open(input_path) as img:
                mosaic_img = apply_mosaic(img, block_size)
                mosaic_img.save(output_path)
            return "SUCCESS", output_path
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            return "ERROR", f"Failed to process {input_path.name} after {max_retries} attempts: {e}"

class MosaicGenerator:
    def __init__(self, input_dir="dataset/clean_images", output_dir="dataset/mosaic_images", 
                 block_size=25, max_workers=None, max_retries=3, overwrite=False):
        
        self.input_dir = Path(input_dir)
        if not self.input_dir.is_dir():
            raise FileNotFoundError(f"Input directory not found: {self.input_dir}")
            
        self.output_dir = Path(output_dir)
        self.block_size = block_size
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.overwrite = overwrite
        self.logger = logging.getLogger(__name__)

        if not isinstance(block_size, int) or block_size < 1:
            raise ValueError("block_size must be a positive integer (>= 1).")
            
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

    def generate_mosaic_images(self):
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
                executor.submit(process_mosaic_worker, path, self.output_dir, self.block_size, self.max_retries, self.overwrite): path.name
                for path in valid_image_files
            }

            pbar = tqdm(as_completed(tasks), total=len(valid_image_files), desc="Generating Mosaics", ncols=100)
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
                    self.logger.error(f"Unhandled exception: {e}")

        self.logger.info(f"Processing complete: {counters['SUCCESS']} succeeded, "
                        f"{counters['SKIPPED']} skipped, {counters['ERROR']} errors.")
        self.logger.info("Mosaic generation completed successfully.")