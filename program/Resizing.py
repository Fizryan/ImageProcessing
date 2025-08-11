# Resizer.py
# This script resizes images from a specified input directory and saves them to an output directory.

import logging
from PIL import Image, ImageOps, UnidentifiedImageError
from tqdm import tqdm
from pathlib import Path
from typing import Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

def resize_image_worker(input_path, output_path, size, resample_filter, output_format, max_retries, overwrite) -> Tuple[str, str]:
    if output_path.exists() and not overwrite:
        return "SKIPPED", f"File already exists: {output_path.name}"
    
    for attempt in range(max_retries):
        try:
            with Image.open(input_path) as img:
                img.load()
                resized_img = Resizer._process_and_resize(img, size, resample_filter)
                resized_img.save(output_path, format=output_format)
            
            return "SUCCESS", f"Resized {input_path.name} to {output_path.name}"

        except (UnidentifiedImageError, IOError) as e:
            return "ERROR", f"Corrupt or unreadable image: {input_path.name} ({e})"
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            return "ERROR", f"Failed to process {input_path.name} after {max_retries} attempts: {e}"

class Resizer:
    def __init__(self, input_dir="dataset/clean_images", output_dir="dataset/resized_images", 
            width=384, height=512, output_format="PNG", max_workers=None, 
            max_retries=3, overwrite=False):
        
        self.input_dir = Path(input_dir).resolve()
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory {self.input_dir} does not exist.")
        self.output_dir = Path(output_dir).resolve()
        self.size = (width, height)
        self.output_format = output_format
        self.resample_filter = Image.Resampling.LANCZOS
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.overwrite = overwrite
        self.logger = logging.getLogger(__name__)

        if not (isinstance(width, int) and width > 0 and isinstance(height, int) and height > 0):
            self.logger.error("Width and height must be positive integers.")
            raise ValueError("Width and height must be positive integers.")
            
        self._prepare_directory()
    
    def _prepare_directory(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Output directory prepared at {self.output_dir}")
    
    @staticmethod
    def _process_and_resize(img: Image.Image, size: tuple, resample_filter) -> Image.Image:
        processed_img = ImageOps.exif_transpose(img)
        processed_img = processed_img.convert("RGB")
        return processed_img.resize(size, resample_filter)

    def process_images(self):
        allowed_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
        image_files = [p for p in self.input_dir.iterdir() if p.suffix.lower() in allowed_extensions]
        
        if not image_files:
            self.logger.warning("No images found in the input directory.")
            return

        self.logger.info(f"Found {len(image_files)} images to process.")

        counters = {"SUCCESS": 0, "SKIPPED": 0, "ERROR": 0}
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            tasks = {}
            for path in image_files:
                output_filename = f"{path.stem}.{self.output_format.lower()}"
                output_path = self.output_dir / output_filename
                
                future = executor.submit(resize_image_worker, path, output_path, self.size, 
                    self.resample_filter, self.output_format, 
                    self.max_retries, self.overwrite)
                tasks[future] = path.name

            pbar = tqdm(as_completed(tasks), total=len(image_files), desc="Resizing Images", ncols=100)
            for future in pbar:
                status, message = future.result()
                counters[status] += 1
                
                if status == "ERROR":
                    self.logger.error(message)
                else:
                    self.logger.debug(message)

        self.logger.info(f"Processing complete: {counters['SUCCESS']} succeeded, "
                        f"{counters['SKIPPED']} skipped, {counters['ERROR']} errors.")