# Grayscaling.py
# This module converts images to grayscale from a specified input directory and saves them to an output directory.

import logging
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from pathlib import Path
from typing import Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from typing import Optional

def grayscale_image_worker(input_path, output_dir, output_format, max_retries, overwrite) -> Tuple[str, str]:
    if output_format:
        output_filename = f"{input_path.stem}.{output_format.lower()}"
    else:
        output_filename = input_path.name
    
    output_path = output_dir / output_filename
    
    if output_path.exists() and not overwrite:
        return "SKIPPED", f"File already exists: {output_path.name}"

    for attempt in range(max_retries):
        try:
            with Image.open(input_path) as img:
                img.load()
                grayscale_img = img.convert("L")
                save_format = "JPEG" if output_format and output_format.lower() == 'jpg' else output_format
                grayscale_img.save(output_path, format=save_format)
            
            return "SUCCESS", f"Converted {input_path.name} to {output_path.name}"

        except (UnidentifiedImageError, IOError) as e:
            return "ERROR", f"Corrupt or unreadable image: {input_path.name} ({e})"
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            return "ERROR", f"Failed to process {input_path.name} after {max_retries} attempts: {e}"

class Grayscaler:
    def __init__(self, input_dir="dataset/clean_images", output_dir="dataset/grayscale_images", 
            output_format: Optional[str] = None, max_workers=None, max_retries=3, overwrite=False):
        
        self.input_dir = Path(input_dir).resolve()
        if not self.input_dir.is_dir():
            raise FileNotFoundError(f"Input directory not found: {self.input_dir}")
            
        self.output_dir = Path(output_dir).resolve()
        self.output_format = output_format
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.overwrite = overwrite
        self.logger = logging.getLogger(__name__)

        if self.output_format:
            self.output_format = Image.registered_extensions().get(f".{self.output_format.lower()}", self.output_format.upper())
            if self.output_format not in Image.SAVE:
                raise ValueError(f"Output format '{self.output_format}' is not supported by Pillow.")
        
        self._prepare_directory()
    
    def _prepare_directory(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Output directory prepared at {self.output_dir}")
    
    def process_images(self):
        allowed_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
        image_files = [p for p in self.input_dir.iterdir() if p.suffix.lower() in allowed_extensions]
        
        if not image_files:
            self.logger.warning("No images found in the input directory.")
            return

        self.logger.info(f"Found {len(image_files)} images to convert.")
        counters = {"SUCCESS": 0, "SKIPPED": 0, "ERROR": 0}
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            tasks = {
                executor.submit(grayscale_image_worker, path, self.output_dir, self.output_format, self.max_retries, self.overwrite): path.name
                for path in image_files
            }

            pbar = tqdm(as_completed(tasks), total=len(image_files), desc="Converting to Grayscale", ncols=100)
            for future in pbar:
                status, message = future.result()
                counters[status] += 1
                
                if status == "ERROR": self.logger.error(message)
                else: self.logger.debug(message)
        
        self.logger.info(f"Processing complete: {counters['SUCCESS']} succeeded, "
                        f"{counters['SKIPPED']} skipped, {counters['ERROR']} errors.")