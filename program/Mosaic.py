# Mosaic.py
# This module generates mosaic images from a set of input images by applying a pixelation effect.

import logging
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from typing import Tuple, List

def apply_mosaic_with_mask(img: Image.Image, mask: Image.Image, block_size: int) -> Image.Image:
    try:
        if img.size != mask.size:
            mask = mask.resize(img.size, Image.Resampling.NEAREST)

        small_img = img.resize(
            (max(1, img.width // block_size), max(1, img.height // block_size)),
            Image.Resampling.NEAREST
        )
        full_mosaic_img = small_img.resize(img.size, Image.Resampling.NEAREST)

        return Image.composite(full_mosaic_img, img, mask.convert('L'))

    except Exception as e:
        raise RuntimeError(f"Failed to apply mosaic with mask: {e}")

def process_mosaic_worker(input_path: Path, output_dir: Path, mask_path: Path, block_size: int,
        max_retries: int, overwrite: bool) -> Tuple[str, Path | str]:
    output_path = output_dir / input_path.name
    
    if output_path.exists() and not overwrite:
        return "SKIPPED", output_path

    if not mask_path.exists():
        return "ERROR", f"Mask file not found for {input_path.name}: {mask_path}"

    for attempt in range(max_retries):
        try:
            with Image.open(input_path) as img, Image.open(mask_path) as mask:
                mosaic_img = apply_mosaic_with_mask(img, mask, block_size)
                mosaic_img.save(output_path)
            return "SUCCESS", output_path
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            return "ERROR", f"Failed to process {input_path.name} after {max_retries} attempts: {e}"

class MosaicGenerator:
    def __init__(self, input_dir="dataset/clean_images", mask_dir="dataset/masks", output_dir="dataset/mosaic_images",
            block_size=25, max_workers=None, max_retries=3, overwrite=False):
        
        self.input_dir = Path(input_dir)
        if not self.input_dir.is_dir():
            raise FileNotFoundError(f"Input directory not found: {self.input_dir}")

        self.mask_dir = Path(mask_dir)
        if not self.mask_dir.is_dir():
            raise FileNotFoundError(f"Mask directory not found: {self.mask_dir}")
            
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

    def _verify_image_pairs(self, image_files: List[Path]) -> List[Tuple[Path, Path]]:
        self.logger.info("Verifying all image and mask pairs before processing...")
        valid_pairs = []
        corrupt_files = 0
        pbar = tqdm(image_files, desc="Verifying Pairs", ncols=100)
        for path in pbar:
            mask_path = self.mask_dir / path.name
            if not mask_path.exists():
                self.logger.warning(f"Skipping {path.name}: No corresponding mask file found in {self.mask_dir}")
                continue
            try:
                with Image.open(path) as img, Image.open(mask_path) as mask:
                    img.verify()
                    mask.verify()
                valid_pairs.append((path, mask_path))
            except (UnidentifiedImageError, IOError):
                self.logger.error(f"Corrupt or unreadable file pair detected and will be skipped: {path.name} and {mask_path.name}")
                corrupt_files += 1
        
        if corrupt_files > 0:
            self.logger.warning(f"Total corrupt image pairs skipped: {corrupt_files}")
        return valid_pairs

    def generate_mosaic_images(self):
        allowed_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    
        image_files = [p for p in self.input_dir.iterdir() 
                       if p.suffix.lower() in allowed_extensions]
        
        if not image_files:
            self.logger.warning("No images found in the input directory.")
            return

        valid_image_pairs = self._verify_image_pairs(image_files)

        if not valid_image_pairs:
            self.logger.warning("No valid image-mask pairs left to process after verification.")
            return
            
        self.logger.info(f"Starting to process {len(valid_image_pairs)} valid image-mask pairs.")

        counters = {"SUCCESS": 0, "SKIPPED": 0, "ERROR": 0}
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            tasks = {
                executor.submit(
                    process_mosaic_worker, 
                    img_path, 
                    self.output_dir, 
                    mask_path, 
                    self.block_size, 
                    self.max_retries, 
                    self.overwrite
                ): img_path.name
                for img_path, mask_path in valid_image_pairs
            }

            pbar = tqdm(as_completed(tasks), total=len(valid_image_pairs), desc="Generating Mosaics", ncols=100)
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
