# CollectingImage.py
# This script collects images from a specified URL and saves them to a local directory.

import logging
import requests
import time
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

class CollectingImage:
    def __init__(self, image_url="https://picsum.photos", save_path="dataset/clean_images", 
                 count=10, width=3060, height=4080, max_workers=4, max_retries=3):
        self.image_url = image_url
        self.save_path = Path(save_path)
        self.count = count
        self.width = width
        self.height = height
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; ImageDownloader/1.0; +https://example.com)"})
        self.logger = logging.getLogger(__name__)
        self._prepare_directory()
        self.start_index = self._get_start_index()

    def _prepare_directory(self):
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Directory prepared at {self.save_path}")

    def _get_start_index(self):
        existing_files = self.save_path.glob("image_*.png")
        max_index = 0
        for f in existing_files:
            try:
                index = int(f.stem.split('_')[1])
                if index > max_index:
                    max_index = index
            except (ValueError, IndexError):
                continue
        return max_index

    def _get_image(self, index):
        url = f"{self.image_url}/{self.width}/{self.height}?random={index}"
        file_path = self.save_path / f"image_{index}.png"
        try:
            response = self.session.get(url, timeout=20) 
            response.raise_for_status()
            with open(file_path, 'wb') as file:
                file.write(response.content)
            return True, f"Image {index} downloaded successfully."
        except requests.RequestException as e:
            return False, f"Failed to download image_{index}: {e}"

    def _download_batch(self, indices):
        results = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self._get_image, idx): idx for idx in indices}
            for future in tqdm(as_completed(futures), total=len(indices), desc="Downloading", ncols=100, leave=False):
                success, message = future.result()
                index = futures[future]
                results[index] = success
                if success:
                    self.logger.info(message)
                else:
                    self.logger.warning(message)
        return results

    def download_images(self):
        self.logger.info(f"Starting download of {self.count} images using up to {self.max_workers} workers.")

        indices_to_download = [self.start_index + i + 1 for i in range(self.count)]
        successful_downloads = 0
        
        pbar = tqdm(total=self.count, desc="Overall Progress", ncols=100)
        
        for attempt in range(self.max_retries + 1):
            if not indices_to_download:
                self.logger.info("All images have been downloaded.")
                break

            self.logger.info(f"Attempt {attempt + 1}/{self.max_retries + 1} - Downloading {len(indices_to_download)} images.")
            
            batch_results = self._download_batch(indices_to_download)
            
            newly_failed = []
            for index, success in batch_results.items():
                if success:
                    successful_downloads += 1
                    pbar.update(1)
                else:
                    newly_failed.append(index)
            
            indices_to_download = newly_failed
            
            if indices_to_download and attempt < self.max_retries:
                self.logger.warning(f"Retrying {len(indices_to_download)} failed images in 2 seconds...")
                time.sleep(2)
        
        pbar.close()

        if indices_to_download:
            self.logger.error(f"Operation finished. Only {successful_downloads}/{self.count} images were downloaded.")
            self.logger.error(f"Failed image indices: {indices_to_download}")
        else:
            self.logger.info(f"Successfully downloaded all {self.count} images.")