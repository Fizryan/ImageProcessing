# Inference.py
# This module contains the inference logic for the image restoration model.

import logging
from pathlib import Path
from typing import Literal, Optional, Tuple, Union, List
import time
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageEnhance, ImageChops, Image
from torch.amp import autocast
from torchvision import transforms
from tqdm import tqdm

from program.Architecture import SOTARestorationUNet
from program.Utils import load_model_weights


class ImageRestorer:
    def __init__(
        self,
        model_path: str,
        model_size: str = "efficient",
        img_height: int = 256,
        img_width: int = 448,
        base_channels: int = 16,
        is_detector: bool = False,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logging.getLogger(self.__class__.__name__)
        self.target_size = (img_width, img_height)
        self.base_channels = base_channels
        self.model_size = model_size
        self.is_detector = is_detector
        self.config = {}

        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        self.model = self._load_model(model_path)

    def _load_model(self, model_path: str) -> nn.Module:
        self.logger.info(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

        if "config" in checkpoint and isinstance(checkpoint["config"], dict):
            self.logger.info(
                "Found config in checkpoint. Initializing model from saved config."
            )
            self.config = checkpoint["config"]
            base_channels = self.config.get("base_channels", self.base_channels)
            self.model_size = self.config.get("model_size", self.model_size)
            self.target_size = (
                self.config.get("img_width", self.target_size[0]),
                self.config.get("img_height", self.target_size[1]),
            )
            mosaic_config = self.config.get(
                "mosaic_block_size_range"
            ) or self.config.get("mosaic_block_size", 16)
            self.mosaic_block_size_info = mosaic_config

            self.logger.info(
                f"Loaded parameters from checkpoint: "
                f"patch_size={self.target_size}, "
                f"base_channels={base_channels}, "
                f"mosaic_block_size_info={self.mosaic_block_size_info}"
            )
        else:
            self.logger.warning(
                "No config found in checkpoint. Using provided/default parameters."
            )
            base_channels = self.base_channels
            self.mosaic_block_size = 16

        out_channels = 1 if self.is_detector else 3

        use_global_residual = not self.is_detector

        self.logger.info(
            f"Attempting to load model with architecture: '{self.model_size}'"
        )
        model: nn.Module = EfficientUNet(
            in_channels=3,
            out_channels=out_channels,
            base_channels=base_channels,
            use_global_residual=use_global_residual,
        )

        if self.is_detector:
            if hasattr(model, "final_conv"):
                num_final_ch = model.final_conv[0].in_channels
                model.final_conv[-1] = nn.Conv2d(num_final_ch, 1, 3, padding=1)
                model.final_act = nn.Identity()
                self.logger.info("Reconfigured model output layer for detector mode.")

        try:
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            load_model_weights(model, state_dict)
        except Exception as e:
            self.logger.error(f"Failed to load model weights: {e}", exc_info=True)
            raise

        model = model.to(self.device)
        model.eval()

        if hasattr(torch, "compile") and torch.cuda.is_available():
            try:
                model = torch.compile(model, mode="reduce-overhead")
                self.logger.info("Model compiled with torch.compile")
                return model
            except Exception as e:
                self.logger.warning(f"Model compilation failed: {e}")

        return model

    @staticmethod
    def _resize_with_padding(
        img: Image.Image, target_size: tuple
    ) -> Tuple[Image.Image, tuple]:
        original_aspect = img.width / img.height
        target_aspect = target_size[0] / target_size[1]

        if original_aspect > target_aspect:
            new_w = target_size[0]
            new_h = int(target_size[0] / original_aspect)
        else:
            new_h = target_size[1]
            new_w = int(target_size[1] * original_aspect)

        resized_img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        if img.mode == "L":
            fill_color = 0
        elif img.mode == "RGBA":
            fill_color = (0, 0, 0, 0)
        else:
            fill_color = (0, 0, 0)

        padded_img = Image.new(img.mode, target_size, fill_color)
        paste_x = (target_size[0] - new_w) // 2
        paste_y = (target_size[1] - new_h) // 2
        paste_box = (paste_x, paste_y, paste_x + new_w, paste_y + new_h)
        padded_img.paste(resized_img, (paste_x, paste_y))

        return padded_img, paste_box

    @staticmethod
    def _generate_blend_mask(patch_size: Tuple[int, int], device: torch.device):
        patch_w, patch_h = patch_size
        hann_h = torch.hann_window(patch_h * 2, periodic=False, device=device)[:patch_h]
        hann_w = torch.hann_window(patch_w * 2, periodic=False, device=device)[:patch_w]
        blend_mask = hann_h.unsqueeze(1) * hann_w.unsqueeze(0)
        return blend_mask.view(1, 1, patch_h, patch_w)

    def _run_sliding_window(
        self,
        image_pil: Image.Image,
        tile_size: Optional[Tuple[int, int]] = None,
        overlap: int = 32,
    ) -> Image.Image:
        with torch.no_grad():
            img_tensor = transforms.ToTensor()(image_pil).unsqueeze(0).to(self.device)
            b, c, h, w = img_tensor.shape

            if tile_size is None:
                patch_w, patch_h = self.target_size
            else:
                patch_w, patch_h = tile_size

            if patch_w <= 0 or patch_h <= 0:
                raise ValueError("Invalid tile_size specified for sliding window.")

            if overlap and overlap > 0:
                stride_w = max(1, patch_w - overlap)
                stride_h = max(1, patch_h - overlap)
            else:
                stride_h = max(1, patch_h // 2)
                stride_w = max(1, patch_w // 2)

            pad_h = (stride_h - (h - patch_h) % stride_h) % stride_h
            pad_w = (stride_w - (w - patch_w) % stride_w) % stride_w
            padded_tensor = F.pad(img_tensor, (0, pad_w, 0, pad_h), "reflect")
            _, _, padded_h, padded_w = padded_tensor.shape

            # Initialize accumulation buffers with float32 for high precision
            # Even when using FP16 inference, accumulator must be float32 to avoid rounding errors
            result_accumulator = torch.zeros(
                (b, c, padded_h, padded_w),
                dtype=torch.float32,
                device=self.device,
            )
            divisor = torch.zeros(
                (b, c, padded_h, padded_w),
                dtype=torch.float32,
                device=self.device,
            )

            # Generate blending mask (must be float32 for accurate blending)
            blend_mask = self._generate_blend_mask((patch_w, patch_h), self.device)
            blend_mask = blend_mask.float()  # Force float32

            patches = []
            patch_coords = []
            for y in range(0, padded_h - patch_h + 1, stride_h):
                for x in range(0, padded_w - patch_w + 1, stride_w):
                    patch = padded_tensor[:, :, y : y + patch_h, x : x + patch_w]
                    patches.append(patch)
                    patch_coords.append((y, x))

            batch_size = self.config.get("val_batch_size", 4)
            results = []
            for i in tqdm(
                range(0, len(patches), batch_size),
                desc="Processing Patches",
                leave=False,
            ):
                batch_patches = torch.cat(patches[i : i + batch_size], dim=0)
                if self.config.get("use_channels_last", True):
                    batch_patches = batch_patches.to(memory_format=torch.channels_last)

                with autocast(
                    device_type=self.device.type,
                    enabled=self.config.get("use_amp", True),
                ):
                    output_batch = self.model(batch_patches)
                results.extend([p.cpu() for p in output_batch])

            for i, (y, x) in enumerate(patch_coords):
                patch_result = results[i].to(self.device).unsqueeze(0)

                patch_result = patch_result.float()

                result_accumulator[:, :, y : y + patch_h, x : x + patch_w] += (
                    patch_result * blend_mask
                )
                divisor[:, :, y : y + patch_h, x : x + patch_w] += blend_mask

            divisor = torch.where(divisor == 0, torch.ones_like(divisor), divisor)

            final_tensor = (result_accumulator / divisor).clamp(0, 1)

            final_tensor_cropped = final_tensor[:, :, :h, :w]

            return transforms.ToPILImage()(final_tensor_cropped.squeeze(0).cpu())

    def _apply_tta(
        self,
        image_pil: Image.Image,
        tile_size: Optional[Tuple[int, int]] = None,
        overlap: int = 32,
    ) -> Image.Image:
        self.logger.info("Applying Test-Time Augmentation (TTA)...")
        result = self._run_sliding_window(
            image_pil, tile_size=tile_size, overlap=overlap
        )

        flipped_img = image_pil.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        flipped_result = self._run_sliding_window(
            flipped_img, tile_size=tile_size, overlap=overlap
        )
        unflipped_result = flipped_result.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        vflipped_img = image_pil.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        vflipped_result = self._run_sliding_window(
            vflipped_img, tile_size=tile_size, overlap=overlap
        )
        unvflipped_result = vflipped_result.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

        rotated_img = image_pil.rotate(90, expand=True)
        rotated_result = self._run_sliding_window(
            rotated_img, tile_size=tile_size, overlap=overlap
        )
        unrotated_result = rotated_result.rotate(-90, expand=True)

        if result.size != unflipped_result.size:
            unflipped_result = unflipped_result.resize(
                result.size, Image.Resampling.LANCZOS
            )
        if result.size != unvflipped_result.size:
            unvflipped_result = unvflipped_result.resize(
                result.size, Image.Resampling.LANCZOS
            )
        if result.size != unrotated_result.size:
            unrotated_result = unrotated_result.resize(
                result.size, Image.Resampling.LANCZOS
            )

        result_arr = np.array(result, dtype=np.float32)
        unflipped_arr = np.array(unflipped_result, dtype=np.float32)
        unvflipped_arr = np.array(unvflipped_result, dtype=np.float32)
        unrotated_arr = np.array(unrotated_result, dtype=np.float32)

        avg_arr = (result_arr + unflipped_arr + unvflipped_arr + unrotated_arr) / 4.0
        return Image.fromarray(avg_arr.astype(np.uint8))

    def _postprocess_image(
        self, image: Image.Image, original_size: tuple
    ) -> Image.Image:
        if image.size != original_size:
            image = image.resize(original_size, Image.Resampling.LANCZOS)

        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.1)
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.05)

        return image

    def restore_image(
        self,
        image_pil: Image.Image,
        iterations: int = 1,
        use_tta: bool = True,
        final_blend_alpha: float = 0.0,
        tile_size: Optional[Tuple[int, int]] = None,
        overlap: int = 32,
        mask_pil: Optional[Image.Image] = None,
    ) -> Image.Image:
        start_time = time.time()

        image_pil = ImageOps.exif_transpose(image_pil).convert("RGB")
        original_size = image_pil.size

        current_image = image_pil
        pbar = tqdm(
            range(iterations), desc="Restoration Iterations", leave=False, ncols=100
        )
        for i in pbar:
            if use_tta:
                current_image = self._apply_tta(
                    current_image, tile_size=tile_size, overlap=overlap
                )
            else:
                current_image = self._run_sliding_window(
                    current_image, tile_size=tile_size, overlap=overlap
                )
        result = current_image

        if final_blend_alpha > 0.0:
            self.logger.debug(
                f"Blending final result with original image (alpha={final_blend_alpha})"
            )
            resized_original = image_pil.resize(result.size, Image.Resampling.LANCZOS)
            result = Image.blend(result, resized_original, alpha=final_blend_alpha)

        result = self._postprocess_image(result, original_size)

        if mask_pil:
            self.logger.info("Applying detected mask to composite final image.")
            mask_pil = mask_pil.resize(result.size, Image.Resampling.NEAREST).convert(
                "L"
            )
            result = Image.composite(result, image_pil, mask_pil)

        processing_time = time.time() - start_time
        self.logger.info(f"Image restored in {processing_time:.2f} seconds.")

        return result

    def detect_mask(self, image_pil: Image.Image) -> Image.Image:
        if not self.is_detector:
            raise RuntimeError(
                "This ImageRestorer instance is not configured as a detector."
            )

        with torch.no_grad():
            padded_img, paste_box = self._resize_with_padding(
                image_pil.convert("RGB"), self.target_size
            )
            input_tensor = (
                transforms.ToTensor()(padded_img).unsqueeze(0).to(self.device)
            )

            with autocast(device_type=self.device.type):
                output_logits = self.model(input_tensor)

            mask_tensor = torch.sigmoid(output_logits).squeeze(0).cpu()
            mask_pil_padded = transforms.ToPILImage()(mask_tensor)
            mask_pil_cropped = mask_pil_padded.crop(paste_box)

            return mask_pil_cropped.resize(image_pil.size, Image.Resampling.LANCZOS)

    def restore_image_from_path(
        self,
        input_path: Path,
        output_path: Path,
        iterations: int = 1,
        use_tta: bool = True,
        final_blend_alpha: float = 0.0,
        tile_size: Optional[Tuple[int, int]] = None,
        overlap: int = 32,
    ) -> bool:

        if not input_path.exists():
            self.logger.error(f"Input file not found: {input_path}")
            return False

        try:
            with Image.open(input_path) as img:
                restored_image = self.restore_image(
                    img,
                    iterations=iterations,
                    use_tta=use_tta,
                    final_blend_alpha=final_blend_alpha,
                    tile_size=tile_size,
                    overlap=overlap,
                )

                output_path.parent.mkdir(parents=True, exist_ok=True)

                if output_path.suffix.lower() in [".jpg", ".jpeg"]:
                    restored_image.save(output_path, "JPEG", quality=95, optimize=True)
                else:
                    restored_image.save(output_path, optimize=True)

            return True

        except Exception as e:
            self.logger.error(
                f"Failed to process {input_path.name}: {e}", exc_info=True
            )
            return False

    def process_directory(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        iterations: int = 1,
        use_tta: bool = True,
        final_blend_alpha: float = 0.0,
        tile_size: Optional[Tuple[int, int]] = None,
        overlap: int = 32,
    ) -> None:
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        if not input_dir.is_dir():
            return

        output_dir.mkdir(parents=True, exist_ok=True)

        allowed_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff"}
        image_files = [
            p for p in input_dir.iterdir() if p.suffix.lower() in allowed_extensions
        ]

        if not image_files:
            return

        success_count = 0
        pbar = tqdm(image_files, desc=f"Demosaicing images", ncols=100, leave=False)

        for path in pbar:
            output_path = output_dir / path.name

            if self.restore_image_from_path(
                path,
                output_path,
                iterations,
                use_tta,
                final_blend_alpha,
                tile_size,
                overlap,
            ):
                success_count += 1

            pbar.set_postfix({"success": f"{success_count}/{len(image_files)}"})
