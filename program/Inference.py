import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Literal, Optional, Tuple, Union, List, Dict
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
from torch.amp import autocast
from torchvision import transforms
from tqdm.auto import tqdm

from program.Architecture import SOTARestorationUNet
from program.LoggingManager import LoggingManager
from program.Utils import Utils
from program.TrainerUtils import TrainerUtils

logger = LoggingManager.setup_logging(__name__)


class ImageRestorer:
    def __init__(
        self,
        model_path: Union[str, Path],
        device: Optional[str] = None,
        force_cpu: bool = False,
    ):
        if force_cpu:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.model_path = Path(model_path)
        self.config: Dict = {}
        self.model = self._load_model(self.model_path)

        logger.info(f"ImageRestorer initialized on {self.device}")

    def _load_model(self, model_path: Path) -> nn.Module:
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        logger.info(f"Loading checkpoint: {model_path.name}")

        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

        if "model_state_dict" in checkpoint:
            if "ema_state_dict" in checkpoint:
                logger.info("Using EMA weights (Best Quality).")
                state_dict = checkpoint["ema_state_dict"]
            else:
                state_dict = checkpoint["model_state_dict"]

            self.config = checkpoint.get("config", {})
        else:
            state_dict = checkpoint
            self.config = {}

        base_channels = self.config.get("base_channels")
        if base_channels is None:
            for k, v in state_dict.items():
                if "intro.weight" in k:
                    base_channels = v.shape[0]
                    break
            base_channels = base_channels or 32

        logger.info(f"Initializing Model (Base Channels: {base_channels})...")

        model = SOTARestorationUNet(
            in_channels=self.config.get("in_channels", 3),
            out_channels=self.config.get("out_channels", 3),
            base_channels=base_channels,
            use_global_residual=self.config.get("use_global_residual", True),
            use_checkpointing=False,
        )

        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace("module.", "").replace("_orig_mod.", "")
            new_state_dict[name] = v

        try:
            model.load_state_dict(new_state_dict, strict=True)
        except RuntimeError as e:
            logger.warning(f"Strict loading failed, trying loose loading: {e}")
            model.load_state_dict(new_state_dict, strict=False)

        model = model.to(self.device)
        model.eval()

        if self.device.type == "cuda" and hasattr(torch, "compile"):
            try:
                model = torch.compile(model, mode="reduce-overhead")
                logger.info("Model compiled successfully.")
            except Exception:
                pass

        return model

    def _inference_tiled(
        self,
        img_tensor: torch.Tensor,
        tile_size: Tuple[int, int] = (448, 256),
        overlap: int = 32,
    ) -> torch.Tensor:
        b, c, h, w = img_tensor.shape
        patch_w, patch_h = tile_size

        if h <= patch_h and w <= patch_w:
            return self.model(img_tensor)

        stride_h = patch_h - overlap
        stride_w = patch_w - overlap

        pad_h = (stride_h - (h - patch_h) % stride_h) % stride_h
        pad_w = (stride_w - (w - patch_w) % stride_w) % stride_w

        img_pad = F.pad(img_tensor, (0, pad_w, 0, pad_h), mode="reflect")
        output_pad = torch.zeros_like(img_pad)
        weight_pad = torch.zeros_like(img_pad)

        H_pad, W_pad = img_pad.shape[2:]

        blend_mask = TrainerUtils.generate_blend_mask((patch_w, patch_h), self.device)

        for y in range(0, H_pad - patch_h + 1, stride_h):
            for x in range(0, W_pad - patch_w + 1, stride_w):
                patch = img_pad[:, :, y : y + patch_h, x : x + patch_w]

                with autocast(device_type=self.device.type):
                    patch_out = self.model(patch)

                output_pad[:, :, y : y + patch_h, x : x + patch_w] += (
                    patch_out * blend_mask
                )
                weight_pad[:, :, y : y + patch_h, x : x + patch_w] += blend_mask

        output_full = output_pad / torch.where(
            weight_pad == 0, torch.ones_like(weight_pad), weight_pad
        )

        return output_full[:, :, :h, :w]

    def _apply_tta(
        self, img_tensor: torch.Tensor, tile_size: Tuple[int, int]
    ) -> torch.Tensor:
        results = []

        results.append(self._inference_tiled(img_tensor, tile_size))

        img_hflip = torch.flip(img_tensor, [3])
        out_hflip = self._inference_tiled(img_hflip, tile_size)
        results.append(torch.flip(out_hflip, [3]))

        img_vflip = torch.flip(img_tensor, [2])
        out_vflip = self._inference_tiled(img_vflip, tile_size)
        results.append(torch.flip(out_vflip, [2]))

        img_rot = torch.rot90(img_tensor, 1, [2, 3])
        tile_size_rot = (tile_size[1], tile_size[0])
        out_rot = self._inference_tiled(img_rot, tile_size_rot)
        results.append(torch.rot90(out_rot, -1, [2, 3]))

        return torch.stack(results).mean(dim=0)

    def restore_image(
        self,
        image_pil: Image.Image,
        use_tta: bool = False,
        tile_size: Tuple[int, int] = (448, 256),
        post_process: bool = False,
    ) -> Image.Image:
        original_size = image_pil.size

        img_tensor = transforms.ToTensor()(image_pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if use_tta:
                out_tensor = self._apply_tta(img_tensor, tile_size)
            else:
                out_tensor = self._inference_tiled(img_tensor, tile_size)

        out_tensor = out_tensor.squeeze(0).clamp(0, 1).cpu()
        restored_img = transforms.ToPILImage()(out_tensor)

        if post_process:
            restored_img = restored_img.filter(
                ImageFilter.UnsharpMask(radius=0.5, percent=50, threshold=2)
            )

        return restored_img

    def process_directory(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        use_tta: bool = False,
    ):
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
        files = [p for p in input_path.iterdir() if p.suffix.lower() in extensions]

        if not files:
            logger.warning(f"No images found in {input_path}")
            return

        logger.info(f"Processing {len(files)} images...")

        pbar = tqdm(files, desc="Restoring")
        for file_path in pbar:
            try:
                img = Image.open(file_path).convert("RGB")

                restored = self.restore_image(img, use_tta=use_tta)

                save_path = output_path / file_path.name

                if save_path.suffix.lower() in [".jpg", ".jpeg"]:
                    restored.save(save_path, quality=95, optimize=True)
                else:
                    restored.save(save_path, optimize=True)

                if self.device.type == "cuda":
                    Utils.check_gpu_temp(self.device)

            except Exception as e:
                logger.error(f"Failed to process {file_path.name}: {e}")

        logger.info("Batch processing completed.")
