# Inference.py
# This module contains the inference logic for the image restoration model.

import logging
from pathlib import Path
from typing import Literal, Optional, Tuple, Union, List
import time
import math

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageEnhance, ImageChops
from torch.amp import autocast
from torchvision import transforms
from tqdm import tqdm

from program.Architecture import UNetLite

try:
    from GPUtil import getGPUs
    import cv2
except ImportError:
    getGPUs = None
    cv2 = None
    logging.warning("GPUtil or OpenCV not available. Some features disabled.")


class ImageRestorer:
    def __init__(
        self,
        model_path: str,
        img_height: int = 256,
        img_width: int = 448,
        base_channels: int = 16,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logging.getLogger(self.__class__.__name__)
        self.target_size = (img_width, img_height)
        self.base_channels = base_channels

        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        self.model = self._load_model(model_path)

    def _load_model(self, model_path: str) -> torch.nn.Module:
        model = UNetLite(
            in_channels=4, out_channels=3, base_channels=self.base_channels
        )

        try:
            state_dict = torch.load(model_path, map_location="cpu", weights_only=True)

            if any(key.startswith("_orig_mod.") for key in state_dict.keys()):
                state_dict = {
                    k.replace("_orig_mod.", ""): v for k, v in state_dict.items()
                }
            elif any(key.startswith("module.") for key in state_dict.keys()):
                state_dict = {
                    k.replace("module.", ""): v for k, v in state_dict.items()
                }

            model.load_state_dict(state_dict)
            # self.logger.info("Model weights loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load model weights: {e}", exc_info=True)
            raise

        if hasattr(torch, "compile") and torch.cuda.is_available():
            try:
                model = torch.compile(model, mode="reduce-overhead")
                # self.logger.info("Model compiled with torch.compile")
            except Exception as e:
                self.logger.warning(f"Model compilation failed: {e}")

        model = model.to(self.device)
        model.eval()
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

    def _prepare_tensor(
        self,
        img_pil: Image.Image,
        task_type: str,
        mask_pil: Optional[Image.Image] = None,
    ) -> Tuple[torch.Tensor, tuple, torch.Tensor, torch.Tensor]:
        padded_pil, paste_box = self._resize_with_padding(img_pil, self.target_size)
        img_tensor = transforms.ToTensor()(padded_pil)
        mask_size = (self.target_size[1], self.target_size[0])

        if task_type == "inpainting":
            if mask_pil is not None:
                padded_mask_pil, _ = self._resize_with_padding(
                    mask_pil.convert("L"), self.target_size
                )
                mask_tensor = transforms.ToTensor()(padded_mask_pil)
                mask_tensor = (mask_tensor > 0.5).float()
            else:
                if img_pil.mode == "RGBA":
                    alpha = img_pil.split()[-1]
                    padded_mask_pil, _ = self._resize_with_padding(
                        alpha, self.target_size
                    )
                    mask_tensor = transforms.ToTensor()(padded_mask_pil)
                    mask_tensor = 1 - (mask_tensor > 0.5).float()
                else:
                    self.logger.warning(
                        "No mask provided for inpainting and no alpha channel found, using zeros"
                    )
                    mask_tensor = torch.zeros(1, *mask_size)

            input_img_tensor = img_tensor
        else:
            mask_tensor = torch.zeros(1, *mask_size)
            input_img_tensor = img_tensor

        model_input = torch.cat([input_img_tensor, mask_tensor], dim=0)
        model_input = model_input * 2.0 - 1.0

        return (
            model_input.unsqueeze(0).to(self.device),
            paste_box,
            img_tensor,
            mask_tensor,
        )

    def _run_model_once(
        self,
        image_pil: Image.Image,
        task_type: str,
        mask_pil: Optional[Image.Image] = None,
    ) -> Image.Image:
        with torch.no_grad():
            input_tensor, paste_box, original_padded_tensor, mask_tensor = (
                self._prepare_tensor(image_pil, task_type, mask_pil)
            )

            with autocast(device_type=self.device.type):
                output_tensor, *_ = self.model(input_tensor)

            output_tensor = (output_tensor.squeeze(0).cpu().clamp(-1, 1) + 1) / 2

            if task_type == "inpainting":
                final_tensor = (
                    original_padded_tensor * (1 - mask_tensor)
                    + output_tensor * mask_tensor
                )
            else:
                final_tensor = output_tensor

            output_pil_padded = transforms.ToPILImage()(final_tensor)

            output_pil_cropped = output_pil_padded.crop(paste_box)

            return output_pil_cropped

    def _poisson_blend(
        self, src_img: Image.Image, dst_img: Image.Image, mask_img: Image.Image
    ) -> Image.Image:
        try:
            import cv2

            src_arr = np.array(src_img.convert("RGB"))
            dst_arr = np.array(dst_img.convert("RGB"))
            mask_arr = np.array(mask_img.convert("L"))

            contours, _ = cv2.findContours(
                mask_arr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                raise RuntimeError("No contours found in mask")

            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            center = (x + w // 2, y + h // 2)

            result = cv2.seamlessClone(
                src_arr, dst_arr, mask_arr, center, cv2.NORMAL_CLONE
            )

            return Image.fromarray(result)

        except (ImportError, RuntimeError, Exception) as e:
            self.logger.warning(f"Poisson blending failed: {e}, using simple blending")
            mask_arr = np.array(mask_img.convert("L")) / 255.0
            mask_arr = np.stack([mask_arr] * 3, axis=-1)

            src_arr = np.array(src_img.convert("RGB")).astype(np.float32)
            dst_arr = np.array(dst_img.convert("RGB")).astype(np.float32)

            result_arr = dst_arr * (1 - mask_arr) + src_arr * mask_arr
            return Image.fromarray(result_arr.astype(np.uint8))

    def _apply_tta(
        self,
        image_pil: Image.Image,
        task_type: str,
        mask_pil: Optional[Image.Image] = None,
    ) -> Image.Image:
        result = self._run_model_once(image_pil, task_type, mask_pil)

        flipped_img = image_pil.transpose(Image.FLIP_LEFT_RIGHT)
        flipped_mask = mask_pil.transpose(Image.FLIP_LEFT_RIGHT) if mask_pil else None
        flipped_result = self._run_model_once(flipped_img, task_type, flipped_mask)
        unflipped_result = flipped_result.transpose(Image.FLIP_LEFT_RIGHT)

        vflipped_img = image_pil.transpose(Image.FLIP_TOP_BOTTOM)
        vflipped_mask = mask_pil.transpose(Image.FLIP_TOP_BOTTOM) if mask_pil else None
        vflipped_result = self._run_model_once(vflipped_img, task_type, vflipped_mask)
        unvflipped_result = vflipped_result.transpose(Image.FLIP_TOP_BOTTOM)

        rotated_img = image_pil.rotate(90, expand=True)
        rotated_mask = mask_pil.rotate(90, expand=True) if mask_pil else None
        rotated_result = self._run_model_once(rotated_img, task_type, rotated_mask)
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
        self, image: Image.Image, original_size: tuple, task_type: str
    ) -> Image.Image:
        if image.size != original_size:
            image = image.resize(original_size, Image.Resampling.LANCZOS)

        if task_type == "noise":
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.1)
        elif task_type == "blur":
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.05)
        elif task_type == "mosaic":
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(1.1)

        return image

    def _multi_scale_inpainting(
        self,
        image_pil: Image.Image,
        mask_pil: Image.Image,
        scales: List[float] = [0.5, 0.75, 1.0, 1.25, 1.5],
    ) -> Image.Image:

        original_size = image_pil.size
        results = []

        for scale in scales:
            new_width = int(original_size[0] * scale)
            new_height = int(original_size[1] * scale)
            new_size = (new_width, new_height)

            scaled_image = image_pil.resize(new_size, Image.Resampling.LANCZOS)
            scaled_mask = mask_pil.resize(new_size, Image.Resampling.LANCZOS)

            with torch.no_grad():
                input_tensor, paste_box, original_padded_tensor, mask_tensor = (
                    self._prepare_tensor(scaled_image, "inpainting", scaled_mask)
                )

                with autocast(device_type=self.device.type):
                    output_tensor, *_ = self.model(input_tensor)

                output_tensor_denorm = (
                    output_tensor.squeeze(0).cpu().clamp(-1, 1) + 1
                ) / 2

                final_tensor = (
                    original_padded_tensor * (1 - mask_tensor)
                    + output_tensor_denorm * mask_tensor
                )
                final_pil = transforms.ToPILImage()(final_tensor)

                cropped_pil = final_pil.crop(paste_box)
                resized_pil = cropped_pil.resize(
                    original_size, Image.Resampling.LANCZOS
                )

                results.append(np.array(resized_pil, dtype=np.float32))

        if results:
            avg_result = np.mean(results, axis=0).astype(np.uint8)
            return Image.fromarray(avg_result)

        return image_pil

    def _refine_mask_edges(
        self, mask_pil: Image.Image, image_pil: Image.Image
    ) -> Image.Image:
        if cv2 is None:
            return mask_pil

        try:
            mask_np = np.array(mask_pil.convert("L"))
            image_np = np.array(image_pil.convert("RGB"))

            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 100, 200)

            kernel = np.ones((3, 3), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)

            mask_np[edges > 0] = 0

            return Image.fromarray(mask_np)
        except Exception as e:
            self.logger.warning(f"Mask edge refinement failed: {e}")
            return mask_pil

    def _adaptive_inpainting(
        self,
        image_pil: Image.Image,
        mask_pil: Image.Image,
        max_iterations: int = 50,
        change_threshold: float = 0.001,
    ) -> Image.Image:

        current_image = image_pil
        original_padded_pil, paste_box = self._resize_with_padding(
            image_pil, self.target_size
        )
        padded_mask_pil, _ = self._resize_with_padding(
            mask_pil.convert("L"), self.target_size
        )

        for iteration in range(max_iterations):
            with torch.no_grad():
                input_tensor, _, original_padded_tensor, mask_tensor = (
                    self._prepare_tensor(current_image, "inpainting", mask_pil)
                )

                with autocast(device_type=self.device.type):
                    output_tensor, *_ = self.model(input_tensor)

                output_tensor_denorm = (
                    output_tensor.squeeze(0).cpu().clamp(-1, 1) + 1
                ) / 2

                src_pil = transforms.ToPILImage()(output_tensor_denorm)

                enhancer = ImageEnhance.Contrast(src_pil)
                src_pil = enhancer.enhance(1.1)

                if cv2 is not None:
                    try:
                        blended_padded_pil = self._poisson_blend(
                            src_pil, original_padded_pil, padded_mask_pil
                        )
                    except Exception as e:
                        self.logger.warning(
                            f"Poisson blending failed: {e}, using simple blending"
                        )
                        blended_tensor = (
                            original_padded_tensor * (1 - mask_tensor)
                            + output_tensor_denorm * mask_tensor
                        )
                        blended_padded_pil = transforms.ToPILImage()(blended_tensor)
                else:
                    blended_tensor = (
                        original_padded_tensor * (1 - mask_tensor)
                        + output_tensor_denorm * mask_tensor
                    )
                    blended_padded_pil = transforms.ToPILImage()(blended_tensor)

                previous_image_arr = np.array(current_image, dtype=np.float32)

                unpadded_pil = blended_padded_pil.crop(paste_box)
                current_image = unpadded_pil.resize(
                    image_pil.size, Image.Resampling.LANCZOS
                )

                if iteration % 5 == 0:
                    enhancer = ImageEnhance.Sharpness(current_image)
                    current_image = enhancer.enhance(1.2)
                    enhancer = ImageEnhance.Contrast(current_image)
                    current_image = enhancer.enhance(1.1)

                curr_image_arr = np.array(current_image, dtype=np.float32)
                change = np.mean(np.abs(previous_image_arr - curr_image_arr))

                if change < change_threshold and iteration > 20:
                    self.logger.info(
                        f"Stopping early at iteration {iteration}, change: {change:.6f}"
                    )
                    break

        self.logger.info(f"Adaptive Inpainting finished after {iteration+1} iterations")
        return current_image

    def _run_iterative_inpainting(
        self,
        image_pil: Image.Image,
        mask_pil: Image.Image,
        iterations: int,
    ):
        original_size = image_pil.size
        original_padded_pil, paste_box = self._resize_with_padding(
            image_pil, self.target_size
        )
        padded_mask_pil, _ = self._resize_with_padding(
            mask_pil.convert("L"), self.target_size
        )
        current_padded_pil = original_padded_pil.copy()

        pbar = tqdm(
            range(iterations),
            desc=f"Iterative inpainting restoration",
            ncols=100,
            leave=False,
        )

        for i in pbar:
            with torch.no_grad():
                img_tensor = transforms.ToTensor()(current_padded_pil).to(self.device)
                mask_tensor = transforms.ToTensor()(padded_mask_pil).to(self.device)
                mask_tensor = (mask_tensor > 0.5).float()

                input_img_tensor = img_tensor
                model_input = torch.cat([input_img_tensor, mask_tensor], dim=0)
                model_input = (model_input * 2.0 - 1.0).unsqueeze(0)

                outputs = []
                with autocast(device_type=self.device.type):
                    final_output, *_ = self.model(model_input)
                    outputs.append(final_output.clone())

                    input_flipped = torch.flip(model_input, dims=[-1])
                    final_output_flipped, *_ = self.model(input_flipped)
                    outputs.append(torch.flip(final_output_flipped, dims=[-1]).clone())

                    input_vflipped = torch.flip(model_input, dims=[-2])
                    final_output_vflipped, *_ = self.model(input_vflipped)
                    outputs.append(torch.flip(final_output_vflipped, dims=[-2]).clone())

                    input_hvflipped = torch.flip(model_input, dims=[-1, -2])
                    final_output_hvflipped, *_ = self.model(input_hvflipped)
                    outputs.append(
                        torch.flip(final_output_hvflipped, dims=[-1, -2]).clone()
                    )

                avg_output = torch.mean(torch.stack(outputs, dim=0), dim=0).squeeze(0)

                output_tensor_denorm = ((avg_output.cpu().clamp(-1, 1) + 1) / 2).float()

                mask_tensor_bin = (mask_tensor > 0.5).float().cpu()
                dst_tensor = transforms.ToTensor()(current_padded_pil).cpu()
                blended_tensor = (
                    dst_tensor * (1 - mask_tensor_bin)
                    + output_tensor_denorm * mask_tensor_bin
                )

                blended_padded_pil = transforms.ToPILImage()(blended_tensor)
                current_padded_pil = blended_padded_pil

            pbar.set_postfix({"iteration": i + 1})

        pbar.close()

        unpadded_pil = current_padded_pil.crop(paste_box)
        return unpadded_pil.resize(original_size, Image.Resampling.LANCZOS)

    def _run_basic_restoration(
        self,
        image_pil: Image.Image,
        task_type: str,
        mask_pil: Image.Image,
        iterations: int,
        use_tta: bool,
    ):
        current_image_pil = image_pil
        pbar = tqdm(
            range(iterations),
            desc=f"Iterative {task_type} restoration",
            ncols=100,
            leave=False,
        )

        for i in pbar:
            if use_tta:
                restored_pil = self._apply_tta(current_image_pil, task_type, mask_pil)
            else:
                restored_pil = self._run_model_once(
                    current_image_pil, task_type, mask_pil
                )
            current_image_pil = restored_pil

            pbar.set_postfix({"iteration": i + 1})

        pbar.close()
        return current_image_pil

    def restore_image(
        self,
        image_pil: Image.Image,
        task_type: Literal["noise", "mosaic", "inpainting", "blur"] = "noise",
        mask_pil: Optional[Image.Image] = None,
        iterations: int = 1,
        use_tta: bool = True,
        final_blend_alpha: float = 0.0,
        use_poisson_blending: bool = True,
        use_multi_scale: bool = False,
        use_edge_aware: bool = True,
        adaptive_iterations: bool = False,
    ) -> Image.Image:

        start_time = time.time()

        if image_pil.mode == "RGBA" and task_type == "inpainting" and mask_pil is None:
            r, g, b, alpha = image_pil.split()
            mask_pil = alpha
            image_pil = Image.merge("RGB", (r, g, b))
        else:
            image_pil = ImageOps.exif_transpose(image_pil).convert("RGB")

        original_size = image_pil.size

        if task_type == "inpainting":
            if mask_pil is None:
                self.logger.error("Inpainting requires a mask image")
                return image_pil

            if use_edge_aware and cv2 is not None:
                mask_pil = self._refine_mask_edges(mask_pil, image_pil)

            if adaptive_iterations:
                result = self._adaptive_inpainting(
                    image_pil, mask_pil, max_iterations=iterations
                )
            elif use_multi_scale:
                result = self._multi_scale_inpainting(image_pil, mask_pil)
            else:
                result = self._run_iterative_inpainting(image_pil, mask_pil, iterations)

            if final_blend_alpha > 0.0:
                self.logger.debug(
                    f"Blending final result with original image (alpha={final_blend_alpha})"
                )
                result = Image.blend(result, image_pil, alpha=final_blend_alpha)

        else:
            result = self._run_basic_restoration(
                image_pil, task_type, mask_pil, iterations, use_tta
            )

            if final_blend_alpha > 0.0:
                self.logger.debug(
                    f"Blending final result with original image (alpha={final_blend_alpha})"
                )
                resized_original = image_pil.resize(
                    result.size, Image.Resampling.LANCZOS
                )
                result = Image.blend(result, resized_original, alpha=final_blend_alpha)

            result = self._postprocess_image(result, original_size, task_type)

        processing_time = time.time() - start_time
        # self.logger.info(f"Image restored in {processing_time:.2f} seconds")

        return result

    def restore_image_from_path(
        self,
        input_path: Path,
        output_path: Path,
        task_type: Literal["noise", "mosaic", "inpainting", "blur"] = "noise",
        mask_path: Optional[Path] = None,
        iterations: int = 1,
        use_tta: bool = True,
        final_blend_alpha: float = 0.0,
        use_poisson_blending: bool = True,
        use_multi_scale: bool = False,
        use_edge_aware: bool = True,
        adaptive_iterations: bool = False,
    ) -> bool:

        if not input_path.exists():
            self.logger.error(f"Input file not found: {input_path}")
            return False

        try:
            mask_pil = None
            if task_type == "inpainting" and mask_path:
                if not mask_path.exists():
                    self.logger.error(f"Mask file not found: {mask_path}")
                    return False
                mask_pil = Image.open(mask_path).convert("L")

            with Image.open(input_path) as img:
                restored_image = self.restore_image(
                    img,
                    task_type,
                    mask_pil=mask_pil,
                    iterations=iterations,
                    use_tta=use_tta,
                    final_blend_alpha=final_blend_alpha,
                    use_poisson_blending=use_poisson_blending,
                    use_multi_scale=use_multi_scale,
                    use_edge_aware=use_edge_aware,
                    adaptive_iterations=adaptive_iterations,
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
        task_type: Literal["noise", "mosaic", "inpainting", "blur"] = "noise",
        mask_dir: Optional[Union[str, Path]] = None,
        iterations: int = 1,
        use_tta: bool = True,
        final_blend_alpha: float = 0.0,
        use_poisson_blending: bool = True,
        use_multi_scale: bool = False,
        use_edge_aware: bool = True,
        adaptive_iterations: bool = False,
    ) -> None:

        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        if not input_dir.is_dir():
            # self.logger.error(f"Input path is not a valid directory: {input_dir}")
            return

        mask_dir = Path(mask_dir) if mask_dir else None
        if task_type == "inpainting" and mask_dir and not mask_dir.is_dir():
            # self.logger.error(f"Mask directory is not a valid directory: {mask_dir}")
            return

        output_dir.mkdir(parents=True, exist_ok=True)

        allowed_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff"}
        image_files = [
            p for p in input_dir.iterdir() if p.suffix.lower() in allowed_extensions
        ]

        if not image_files:
            # self.logger.warning(f"No images found in the input directory: {input_dir}")
            return

        # self.logger.info(f"Found {len(image_files)} images to process from '{input_dir.name}'")

        success_count = 0
        pbar = tqdm(
            image_files, desc=f"Restoring {task_type} images", ncols=100, leave=False
        )

        for path in pbar:
            output_path = output_dir / path.name
            mask_path = None

            if task_type == "inpainting" and mask_dir:
                mask_path = mask_dir / path.name
                if not mask_path.exists():
                    for ext in allowed_extensions:
                        alt_mask_path = mask_dir / (path.stem + ext)
                        if alt_mask_path.exists():
                            mask_path = alt_mask_path
                            break

                    if not mask_path or not mask_path.exists():
                        # self.logger.warning(f"Mask not found for {path.name} in {mask_dir}. Skipping image.")
                        continue

            if self.restore_image_from_path(
                path,
                output_path,
                task_type,
                mask_path,
                iterations,
                use_tta,
                final_blend_alpha,
                use_poisson_blending,
                use_multi_scale,
                use_edge_aware,
                adaptive_iterations,
            ):
                success_count += 1

            pbar.set_postfix({"success": f"{success_count}/{len(image_files)}"})

        pbar.close()

        # self.logger.info(f"Processing complete. {success_count}/{len(image_files)} images restored successfully")

    def _check_gpu_temp(self, threshold: float = 85, delay: int = 15):
        if not getGPUs or self.device.type != "cuda":
            return

        try:
            gpu = getGPUs()[0]
            temperature = gpu.temperature

            if temperature >= threshold:
                self.logger.warning(
                    f"GPU temperature high: {temperature}°C. Cooling down for {delay} seconds."
                )
                time.sleep(delay)
        except Exception as e:
            self.logger.error(f"GPU temperature check failed: {e}")
