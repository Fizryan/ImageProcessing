# CollectingImage.py
# This script collects images from a specified URL and saves them to a local directory.

import logging
import requests
import time
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

class CollectingImage:
    def __init__(self, image_url="https://picsum.photos", save_path="../dataset/clean_images", 
                 count=10, width=3060, height=4080, max_workers=5, max_retries=3):
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
            for future in tqdm(as_completed(futures), total=len(indices), desc="Downloading", ncols=100):
                success, message = future.result()
                index = futures[future]
                results[index] = success
                if success:
                    self.logger.info(message)
                else:
                    self.logger.warning(message)
        return results

    def download_images(self):
        self.logger.info(f"Starting threaded download of {self.count} images using {self.max_workers} workers.")

        all_indices = [self.start_index + i + 1 for i in range(self.count)]
        retry_count = 0
        failed_indices = all_indices

        while retry_count <= self.max_retries and failed_indices:
            self.logger.info(f"Attempt {retry_count + 1}/{self.max_retries + 1} - Downloading {len(failed_indices)} images.")
            results = self._download_batch(failed_indices)
            failed_indices = [idx for idx, success in results.items() if not success]
            retry_count += 1
            if failed_indices and retry_count <= self.max_retries:
                self.logger.warning(f"Retrying {len(failed_indices)} failed images...")
                time.sleep(2)

        successful = self.count - len(failed_indices)
        if failed_indices:
            self.logger.error(f"Only {successful}/{self.count} images downloaded after {self.max_retries} retries.")
            self.logger.error(f"Failed image indices: {failed_indices}")
            time.sleep(1)
        else:
            self.logger.info(f"Successfully downloaded all {self.count} images to {self.save_path}.")