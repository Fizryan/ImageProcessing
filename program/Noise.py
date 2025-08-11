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

def process_image_worker(image_path, output_dir, noise_type, noise_level, seed, max_retries=3, overwrite=False):
    if noise_type == 'random':
        random.seed(seed)
        noise_type = random.choice(["gaussian", "salt_pepper"])

    for attempt in range(max_retries):
        try:
            with Image.open(image_path) as img:
                img.verify()
            
            with Image.open(image_path) as img:
                image_array = np.array(img.convert("RGB"))

            if noise_type == "gaussian":
                noisy_array = NoiseGenerator._add_gaussian_noise(image_array, noise_level)
            elif noise_type == "salt_pepper":
                noisy_array = NoiseGenerator._add_salt_and_pepper_noise(image_array, noise_level)
            else:
                return "ERROR", f"Unknown noise type for {image_path.name}"

            noisy_image = Image.fromarray(noisy_array)
            output_path = output_dir / image_path.name
            if output_path.exists() and not overwrite:
                return "SKIPPED", f"File already exists: {output_path.name}"
            
            noisy_image.save(output_path)
            return "SUCCESS", f"Generated {noise_type} noise for {output_path.name}"

        except (UnidentifiedImageError, IOError):
            return "ERROR", f"Corrupt or unreadable image file: {image_path.name}"
        except Exception as e:
            if attempt < max_retries - 1:
                continue
            return "ERROR", f"Failed to process {image_path.name} after {max_retries} attempts: {e}"

class NoiseGenerator:
    def __init__(self, input_dir="dataset/clean_images", noise_dir="dataset/noisy_images", 
                 noise_level=0.1, noise_type='random', max_workers=None, max_retries=3, overwrite=False):
        self.input_dir = Path(input_dir)
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
        allowed_extensions = {".png", ".jpg", ".jpeg"}
        image_files = [p for p in self.input_dir.iterdir() if p.suffix.lower() in allowed_extensions]
        self.logger.info(f"Found {len(image_files)} images to process.")

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            tasks = {}
            for path in image_files:
                file_hash = hashlib.sha256(path.name.encode()).hexdigest()
                seed = int(file_hash, 16) % (2**32)
                
                future = executor.submit(process_image_worker, path, self.noise_dir, self.noise_type, self.noise_level, seed, self.max_retries, self.overwrite)
                tasks[future] = path.name

            pbar = tqdm(as_completed(tasks), total=len(image_files), desc="Generating Noisy Images", ncols=100, disable=not sys.stdout.isatty())
            for future in pbar:
                status, message = future.result()
                if status == "ERROR":
                    self.logger.error(message)
                else:
                    self.logger.debug(message)

        self.logger.info("Finished generating all noisy images.")