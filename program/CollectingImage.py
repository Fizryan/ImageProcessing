# CollectingImage.py
# This script collects images from a specified URL and saves them to a local directory.

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image
from tqdm import tqdm

class CollectingImage:
    def __init__(self, save_path="dataset/clean_images", count=10, width=3060, height=4080, 
            max_workers=5, max_retries=3, source="pexels", pexels_query="nature", resize=True):
        
        self.save_path = Path(save_path)
        self.count = count
        self.width = width
        self.height = height
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.source = source.lower()
        self.pexels_query = pexels_query
        self.resize = resize
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.pexels_api_key = os.getenv("PEXELS_API_KEY")

        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; ImageDownloader/1.0)"})

        if self.source == "pexels" and not self.pexels_api_key:
            raise ValueError("Pexels source selected, but PEXELS_API_KEY environment variable is not set.")
        
        self._prepare_directory()
        self.start_index = self._get_start_index()

    def _prepare_directory(self):
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Directory prepared at {self.save_path}")

    def _get_start_index(self):
        existing_files = list(self.save_path.glob("image_*.png"))
        if not existing_files:
            return 0
        try:
            return max(int(f.stem.split('_')[1]) for f in existing_files)
        except (ValueError, IndexError):
            return 0

    def _fetch_pexels_urls(self) -> list:
        if not self.pexels_api_key: return []
        self.logger.info(f"Fetching {self.count} image URLs from Pexels for query: '{self.pexels_query}'...")
        urls = []
        page = 1
        per_page = 80
        headers = {"Authorization": self.pexels_api_key}
        while len(urls) < self.count:
            try:
                search_url = f"https://api.pexels.com/v1/search?query={self.pexels_query}&per_page={per_page}&page={page}"
                response = self.session.get(search_url, headers=headers, timeout=20)
                response.raise_for_status()
                data = response.json()
                page_urls = [photo['src']['original'] for photo in data.get("photos", [])]
                if not page_urls:
                    self.logger.warning("No more photos found on Pexels for this query.")
                    break
                urls.extend(page_urls)
                page += 1
            except requests.RequestException as e:
                self.logger.error(f"Failed to fetch URLs from Pexels: {e}")
                break
        return urls[:self.count]

    def _download_and_process_url(self, url: str, index: int):
        file_path = self.save_path / f"image_{index}.png"
        try:
            response = self.session.get(url, timeout=60)
            response.raise_for_status()
            
            img = Image.open(BytesIO(response.content)).convert("RGB")

            if img.width > img.height:
                self.logger.debug(f"Rotating landscape image {index} to portrait.")
                img = img.rotate(90, expand=True)

            if self.resize:
                img = img.resize((self.width, self.height), Image.Resampling.LANCZOS)

            img.save(file_path, format='PNG')
            return True, f"Image {index} downloaded successfully."
        except Exception as e:
            return False, f"Failed to process image {index} from {url}: {e}"

    def download_images(self):
        self.logger.info(f"Starting download of {self.count} images from '{self.source}'.")
        if self.source == 'pexels':
            image_urls = self._fetch_pexels_urls()
            if not image_urls:
                self.logger.error("Could not fetch any image URLs from Pexels. Aborting.")
                return
            tasks_to_run = [(url, self.start_index + i + 1) for i, url in enumerate(image_urls)]
        elif self.source == 'picsum':
            base_url = "https://picsum.photos"
            tasks_to_run = [
                (f"{base_url}/{self.width}/{self.height}?random={self.start_index + i + 1}", self.start_index + i + 1)
                for i in range(self.count)
            ]
        else:
            self.logger.error(f"Unknown source: '{self.source}'. Please use 'pexels' or 'picsum'.")
            return
        successful_downloads = 0
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self._download_and_process_url, url, index): index for url, index in tasks_to_run}
            pbar = tqdm(as_completed(futures), total=len(tasks_to_run), desc="Downloading Images", ncols=100)
            for future in pbar:
                success, message = future.result()
                if success:
                    successful_downloads += 1
                    self.logger.debug(message)
                else:
                    self.logger.warning(message)
        self.logger.info(f"Download complete: {successful_downloads}/{self.count} images were successful.")