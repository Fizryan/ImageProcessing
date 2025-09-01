# Noise.py
# This module generates noisy images from a set of clean images by applying Gaussian or Salt-and-Pepper noise.

import sys
import logging
import numpy as np
import random
import hashlib
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple, List

def process_image_worker(image_path: Path, mask_path: Path, output_dir: Path, noise_type: str, 
        noise_level: float, seed: int, max_retries=3, overwrite=False) -> Tuple[str, str]:
    output_path = output_dir / image_path.name
    if output_path.exists() and not overwrite:
        return "SKIPPED", f"File already exists: {output_path.name}"

    if not mask_path.exists():
        return "ERROR", f"Mask file not found for {image_path.name}: {mask_path}"

    if noise_type == 'random':
        random.seed(seed)
        noise_type = random.choice(["gaussian", "salt_pepper"])

    for attempt in range(max_retries):
        try:
            with Image.open(image_path) as img, Image.open(mask_path) as mask:
                img.verify()
                mask.verify()

            with Image.open(image_path) as img, Image.open(mask_path) as mask:
                if img.size != mask.size:
                    mask = mask.resize(img.size, Image.Resampling.NEAREST)
                mask_array = np.array(mask.convert("L"))
                
                original_array = np.array(img.convert("RGB"))
                noisy_array_full = np.array(img.convert("RGB"))

                if noise_type == "gaussian":
                    noisy_array_full = NoiseGenerator._add_gaussian_noise(original_array, noise_level)
                elif noise_type == "salt_pepper":
                    noisy_array_full = NoiseGenerator._add_salt_and_pepper_noise(original_array, noise_level)
                else:
                    return "ERROR", f"Unknown noise type for {image_path.name}"

                apply_noise_mask = mask_array > 128
                final_array = np.where(np.expand_dims(apply_noise_mask, axis=-1), noisy_array_full, original_array)

                noisy_image = Image.fromarray(final_array)
                noisy_image.save(output_path)
                return "SUCCESS", f"Generated {noise_type} noise for {output_path.name}"

        except (UnidentifiedImageError, IOError):
            return "ERROR", f"Corrupt or unreadable image file: {image_path.name}"
        except Exception as e:
            if attempt < max_retries - 1:
                continue
            return "ERROR", f"Failed to process {image_path.name} after {max_retries} attempts: {e}"

class NoiseGenerator:
    def __init__(self, input_dir="dataset/clean_images", mask_dir="dataset/masks", noise_dir="dataset/noisy_images", 
                 noise_level=0.1, noise_type='random', max_workers=None, max_retries=3, overwrite=False):
        
        self.input_dir = Path(input_dir)
        if not self.input_dir.is_dir():
            raise FileNotFoundError(f"Input directory not found: {self.input_dir}")

        self.mask_dir = Path(mask_dir)
        if not self.mask_dir.is_dir():
            raise FileNotFoundError(f"Mask directory not found: {self.mask_dir}")
            
        self.noise_dir = Path(noise_dir)
        self.noise_level = noise_level
        self.noise_type = noise_type
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.overwrite = overwrite
        self.logger = logging.getLogger(__name__)

        if not 0 <= self.noise_level <= 1:
            self.logger.error("Noise level must be between 0 and 1.")
            raise ValueError("Noise level must be between 0 and 1.")
            
        self._prepare_directory()

    def _prepare_directory(self):
        self.noise_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Directory prepared at {self.noise_dir}")
    
    def _verify_image_pairs(self, image_files: List[Path]) -> List[Tuple[Path, Path]]:
        self.logger.info("Verifying all image and mask pairs before processing...")
        valid_pairs = []
        pbar = tqdm(image_files, desc="Verifying Pairs", ncols=100)
        for path in pbar:
            mask_path = self.mask_dir / path.name
            if not mask_path.exists():
                self.logger.warning(f"Skipping {path.name}: No corresponding mask file found in {self.mask_dir}")
                continue
            valid_pairs.append((path, mask_path))
        return valid_pairs

    @staticmethod
    def _add_gaussian_noise(image_array, noise_level):
        std_dev = noise_level * 255
        noise = np.random.normal(0, std_dev, image_array.shape)
        noisy_array = np.clip(image_array + noise, 0, 255)
        return noisy_array.astype(np.uint8)

    @staticmethod
    def _add_salt_and_pepper_noise(image_array, noise_level):
        noisy_array = image_array.copy()
        num_pixels_to_affect = int(noise_level * image_array.shape[0] * image_array.shape[1])
        
        salt_coords_y = np.random.randint(0, image_array.shape[0], num_pixels_to_affect // 2)
        salt_coords_x = np.random.randint(0, image_array.shape[1], num_pixels_to_affect // 2)
        noisy_array[salt_coords_y, salt_coords_x] = [255, 255, 255]

        pepper_coords_y = np.random.randint(0, image_array.shape[0], num_pixels_to_affect // 2)
        pepper_coords_x = np.random.randint(0, image_array.shape[1], num_pixels_to_affect // 2)
        noisy_array[pepper_coords_y, pepper_coords_x] = [0, 0, 0]
        
        return noisy_array

    def generate_noisy_images(self):
        allowed_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
        image_files = [p for p in self.input_dir.iterdir() if p.suffix.lower() in allowed_extensions]
        
        if not image_files:
            self.logger.warning("No images found in the input directory.")
            return

        valid_image_pairs = self._verify_image_pairs(image_files)

        if not valid_image_pairs:
            self.logger.warning("No valid image-mask pairs left to process after verification.")
            return
            
        self.logger.info(f"Starting to process {len(valid_image_pairs)} valid image-mask pairs.")

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            tasks = {}
            for img_path, mask_path in valid_image_pairs:
                file_hash = hashlib.sha256(img_path.name.encode()).hexdigest()
                seed = int(file_hash, 16) % (2**32)
                
                future = executor.submit(process_image_worker, img_path, mask_path, self.noise_dir, self.noise_type, 
                                         self.noise_level, seed, self.max_retries, self.overwrite)
                tasks[future] = img_path.name

            pbar = tqdm(as_completed(tasks), total=len(tasks), desc="Generating Noisy Images", ncols=100, disable=not sys.stdout.isatty())
            for future in pbar:
                status, message = future.result()
                if status == "ERROR":
                    self.logger.error(message)
                else:
                    self.logger.debug(message)

        self.logger.info("Finished generating all noisy images.")