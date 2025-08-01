# Inference.py
# ImageProcessing Inference Module

import logging
from pathlib import Path
from typing import Literal, Tuple

import torch
from PIL import Image
from torch.amp import autocast
from torchvision import transforms
from torchvision.utils import save_image

from program.Architecture import AdvancedUNet

class ImageRestorer:
    def __init__(self, model_path="Training/checkpoints/best_model.pth", img_height=512, img_width=384):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logging.getLogger(__name__)

        self.image_size = (img_height, img_width)
        self.model = self._load_model(model_path)

        self.output_dir_path = "Results"
        
        self.logger.info(f"ImageRestorer initialized on device: {self.device}")

    def _prepare_directory(self):
        Path(self.output_dir_path).mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Output directory prepared at {self.output_dir_path}")

    def _load_model(self, model_path: str) -> torch.nn.Module:
        self.logger.info(f"Loading model from {model_path}...")
        
        model = AdvancedUNet(in_channels=4, out_channels=3)
        try:
            state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
            
            if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
                self.logger.info("Compiled model detected. Cleaning state_dict keys...")
                cleaned_state_dict = {
                    key.replace('_orig_mod.', ''): value 
                    for key, value in state_dict.items()
                }
                state_dict = cleaned_state_dict

            model.load_state_dict(state_dict)
        except Exception as e:
            self.logger.error(f"Failed to load model weights: {e}")
            raise
            
        if hasattr(torch, 'compile'):
            model = torch.compile(model, mode='reduce-overhead')
            self.logger.info("Model compiled for faster inference.")
            
        model = model.to(self.device)
        model.eval()
        return model

    def _prepare_tensor(self, img_pil: Image.Image, 
            task_type: Literal['noise', 'mosaic', 'inpainting']) -> torch.Tensor:
        transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor()
        ])
        img_tensor = transform(img_pil)

        if task_type == 'inpainting':
            mask = (img_tensor.sum(dim=0) == 0).float().unsqueeze(0)
        else:
            mask = torch.zeros(1, *self.image_size)

        model_input = torch.cat([img_tensor, mask], dim=0)
        model_input = model_input * 2.0 - 1.0
        
        return model_input.unsqueeze(0).to(self.device)

    def restore_image(self, image_pil: Image.Image, 
            task_type: Literal['noise', 'mosaic', 'inpainting'] = 'noise') -> Image.Image:

        if not isinstance(image_pil, Image.Image):
            raise TypeError("Input must be a PIL.Image object.")

        original_size = image_pil.size
        
        with torch.no_grad():
            input_tensor = self._prepare_tensor(image_pil.convert("RGB"), task_type)
            
            with autocast(device_type=self.device.type):
                output_tensor = self.model(input_tensor)

            output_tensor = (output_tensor.squeeze(0).cpu().clamp(-1, 1) + 1) / 2
            output_pil = transforms.ToPILImage()(output_tensor)
        
        return output_pil.resize(original_size, Image.Resampling.LANCZOS)

    def restore_image_from_path(self, input_path: str | Path, output_path= str | Path, 
            task_type: Literal['noise', 'mosaic', 'inpainting'] = 'noise'):
        self.logger.info(f"Restoring image from {input_path} with task type '{task_type}'...")
        input_p = Path(input_path)
        output_p = Path(output_path)
        
        if not input_p.exists():
            self.logger.error(f"Input file not found: {input_p}")
            return

        try:
            with Image.open(input_p) as img:
                restored_image = self.restore_image(img, task_type)
                
                output_p.parent.mkdir(parents=True, exist_ok=True)
                
                restored_image.save(output_p)
                self.logger.debug(f"Successfully restored '{input_p.name}' and saved to '{output_p.name}'")
        except Exception as e:
            self.logger.error(f"Failed to process {input_p.name}: {e}", exc_info=True)