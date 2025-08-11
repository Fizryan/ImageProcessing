# Inference.py
# This module contains the inference logic for the image restoration model.

import logging
from pathlib import Path
from typing import Literal

import torch
from PIL import Image, ImageOps
from torch.amp import autocast
from torchvision import transforms
from tqdm import tqdm
from program.Architecture import AdvancedUNet

class ImageRestorer:
    def __init__(self, model_path: str, img_height: int = 384, img_width: int = 256):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logging.getLogger(self.__class__.__name__)
        self.target_size = (img_width, img_height)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        self.model = self._load_model(model_path)
        self.logger.info(f"ImageRestorer initialized on device: {self.device}")

    def _load_model(self, model_path: str) -> torch.nn.Module:
        self.logger.info(f"Loading model from {model_path}...")
        model = AdvancedUNet(in_channels=4, out_channels=3)
        try:
            state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
            if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
                state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
        except Exception as e:
            self.logger.error(f"Failed to load model weights: {e}", exc_info=True)
            raise
        if hasattr(torch, 'compile'):
            model = torch.compile(model, mode='reduce-overhead')
        model = model.to(self.device)
        model.eval()
        return model

    @staticmethod
    def _resize_with_padding(img: Image.Image, target_size: tuple):
        original_aspect = img.width / img.height
        target_aspect = target_size[0] / target_size[1]
        if original_aspect > target_aspect:
            new_w, new_h = target_size[0], int(target_size[0] / original_aspect)
        else:
            new_w, new_h = int(target_size[1] * original_aspect), target_size[1]
        resized_img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        padded_img = Image.new("RGB", target_size, (0, 0, 0))
        paste_x, paste_y = (target_size[0] - new_w) // 2, (target_size[1] - new_h) // 2
        paste_box = (paste_x, paste_y, paste_x + new_w, paste_y + new_h)
        padded_img.paste(resized_img, (paste_x, paste_y))
        return padded_img, paste_box

    def _prepare_tensor(self, img_pil: Image.Image, task_type: Literal['noise', 'mosaic', 'inpainting']):
        padded_pil, paste_box = self._resize_with_padding(img_pil, self.target_size)
        img_tensor = transforms.ToTensor()(padded_pil)
        mask_size = (self.target_size[1], self.target_size[0])
        mask = (img_tensor.sum(dim=0) == 0).float().unsqueeze(0) if task_type == 'inpainting' else torch.zeros(1, *mask_size)
        model_input = torch.cat([img_tensor, mask], dim=0)
        model_input = model_input * 2.0 - 1.0
        return model_input.unsqueeze(0).to(self.device), paste_box

    def restore_image(self, image_pil: Image.Image, task_type: Literal['noise', 'mosaic', 'inpainting'] = 'noise'):
        image_pil = ImageOps.exif_transpose(image_pil)
        original_size = image_pil.size
        with torch.no_grad():
            input_tensor, paste_box = self._prepare_tensor(image_pil.convert("RGB"), task_type)
            with autocast(device_type=self.device.type):
                output_tensor = self.model(input_tensor)
            output_tensor = (output_tensor.squeeze(0).cpu().clamp(-1, 1) + 1) / 2
            output_pil_padded = transforms.ToPILImage()(output_tensor)
        output_pil_cropped = output_pil_padded.crop(paste_box)
        return output_pil_cropped.resize(original_size, Image.Resampling.LANCZOS)

    def restore_image_from_path(self, input_path: Path, output_path: Path, task_type: Literal['noise', 'mosaic', 'inpainting'] = 'noise'):
        if not input_path.exists():
            self.logger.error(f"Input file not found: {input_path}")
            return False
        try:
            with Image.open(input_path) as img:
                restored_image = self.restore_image(img, task_type)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                restored_image.save(output_path)
                self.logger.debug(f"Successfully restored '{input_path.name}' and saved to '{output_path.name}'")
            return True
        except Exception as e:
            self.logger.error(f"Failed to process {input_path.name}: {e}", exc_info=True)
            return False

    def process_directory(self, input_dir: str | Path, output_dir: str | Path, 
            task_type: Literal['noise', 'mosaic', 'inpainting'] = 'noise'):
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        if not input_dir.is_dir():
            self.logger.error(f"Input path is not a valid directory: {input_dir}")
            return
            
        output_dir.mkdir(parents=True, exist_ok=True)
        
        allowed_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
        image_files = [p for p in input_dir.iterdir() if p.suffix.lower() in allowed_extensions]
        
        if not image_files:
            self.logger.warning(f"No images found in the input directory: {input_dir}")
            return
            
        self.logger.info(f"Found {len(image_files)} images to process from '{input_dir.name}'.")
        
        success_count = 0
        pbar = tqdm(image_files, desc=f"Restoring {task_type} images", ncols=100)
        for path in pbar:
            output_path = output_dir / path.name
            if self.restore_image_from_path(path, output_path, task_type):
                success_count += 1
        
        self.logger.info(f"Processing complete. {success_count}/{len(image_files)} images restored successfully.")