# Inference_Video.py
# Streaming video inference with tiling for full-resolution video restoration

import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast
from torchvision import transforms
from tqdm import tqdm

from program.Architecture import SOTARestorationUNet, get_model
from program.Utils import load_model_weights, check_gpu_temp

try:
    import ffmpeg
except ImportError:
    ffmpeg = None


class VideoRestorer:
    def __init__(
        self,
        model_path: str,
        tile_size: Tuple[int, int] = (448, 256),
        overlap: int = 32,
        use_amp: bool = True,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logging.getLogger(self.__class__.__name__)
        self.tile_size = tile_size
        self.overlap = overlap
        self.use_amp = use_amp

        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        self.model = self._load_model(model_path)

    def _load_model(self, model_path: str) -> torch.nn.Module:
        self.logger.info(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

        # Extract config if available
        if "config" in checkpoint:
            config = checkpoint["config"]
            model_class = get_model(config)
            model_params = {
                "in_channels": 3,
                "out_channels": 3,
                "base_channels": config.get("base_channels", 32),
                "use_checkpointing": False,  # Disable checkpointing for inference
                "use_global_residual": True,
            }
            model = model_class(**model_params)
        else:
            # Fallback to default architecture
            self.logger.warning(
                "Config not found in checkpoint, using default EfficientUNet"
            )
            model = EfficientUNet(
                in_channels=3,
                out_channels=3,
                base_channels=32,
                use_checkpointing=False,
                use_global_residual=True,
            )

        # Load weights
        if "ema_state_dict" in checkpoint:
            self.logger.info("Loading EMA weights")
            load_model_weights(model, checkpoint["ema_state_dict"])
        elif "model_state_dict" in checkpoint:
            load_model_weights(model, checkpoint["model_state_dict"])
        else:
            self.logger.warning("No recognized state dict found, loading directly")
            load_model_weights(model, checkpoint)

        model = model.to(self.device)
        model.eval()

        # Convert to FP16 if using AMP
        if self.use_amp:
            model = model.half()

        self.logger.info(f"Model loaded successfully on {self.device}")
        return model

    @staticmethod
    def _generate_blend_mask(
        patch_size: Tuple[int, int], device: torch.device
    ) -> torch.Tensor:
        """Generate Hann window blending mask for seamless tile stitching"""
        patch_w, patch_h = patch_size
        hann_h = torch.hann_window(patch_h * 2, periodic=False, device=device)[:patch_h]
        hann_w = torch.hann_window(patch_w * 2, periodic=False, device=device)[:patch_w]
        blend_mask = hann_h.unsqueeze(1) * hann_w.unsqueeze(0)
        return blend_mask.view(1, 1, patch_h, patch_w)

    @torch.no_grad()
    def _tiled_inference(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Process large images using tiling strategy.
        Same logic as Training.py validation to ensure consistency.
        """
        b, c, h, w = image_tensor.shape
        patch_w, patch_h = self.tile_size

        # If image is smaller than tile size, process directly
        if h <= patch_h and w <= patch_w:
            if self.use_amp:
                with autocast(device_type="cuda", dtype=torch.float16):
                    return self.model(image_tensor)
            return self.model(image_tensor)

        # Calculate stride
        stride_w = max(1, patch_w - self.overlap)
        stride_h = max(1, patch_h - self.overlap)

        # Pad image to fit tile size
        pad_h = (stride_h - (h - patch_h) % stride_h) % stride_h
        pad_w = (stride_w - (w - patch_w) % stride_w) % stride_w
        padded_tensor = F.pad(image_tensor, (0, pad_w, 0, pad_h), "reflect")
        _, _, padded_h, padded_w = padded_tensor.shape

        # Initialize accumulation buffers with float32 for high precision
        # Even when using FP16 inference, accumulator must be float32 to avoid rounding errors
        result_accumulator = torch.zeros(
            (b, c, padded_h, padded_w),
            dtype=torch.float32,
            device=image_tensor.device,
        )
        divisor = torch.zeros(
            (b, c, padded_h, padded_w),
            dtype=torch.float32,
            device=image_tensor.device,
        )

        # Generate blending mask (must be float32 for accurate blending)
        blend_mask = self._generate_blend_mask((patch_w, patch_h), image_tensor.device)
        blend_mask = blend_mask.float()  # Force float32

        # Process tiles
        for y in range(0, padded_h - patch_h + 1, stride_h):
            for x in range(0, padded_w - patch_w + 1, stride_w):
                # Extract patch
                patch = padded_tensor[:, :, y : y + patch_h, x : x + patch_w]

                # Inference (can use FP16 for speed)
                if self.use_amp:
                    with autocast(device_type="cuda", dtype=torch.float16):
                        patch_result = self.model(patch)
                else:
                    patch_result = self.model(patch)

                # CRITICAL: Convert result to float32 before accumulation
                # This prevents rounding errors at tile boundaries
                patch_result = patch_result.float()

                # Accumulate with blending
                result_accumulator[:, :, y : y + patch_h, x : x + patch_w] += (
                    patch_result * blend_mask
                )
                divisor[:, :, y : y + patch_h, x : x + patch_w] += blend_mask

        # Normalize
        divisor = torch.where(divisor == 0, torch.ones_like(divisor), divisor)
        final_tensor = result_accumulator / divisor

        # Crop back to original size
        return final_tensor[:, :, :h, :w].clamp(0, 1)

    def _merge_audio_with_ffmpeg(
        self, video_no_audio: Path, original_video: Path, output_final: Path
    ) -> bool:
        """
        Merge audio from original video into restored video using ffmpeg-python.
        Returns True if successful, False otherwise.
        """
        if ffmpeg is None:
            self.logger.warning(
                "ffmpeg-python not installed. Skipping audio merge. Install with: pip install ffmpeg-python"
            )
            return False

        try:
            self.logger.info("Merging audio from original video...")

            # Get video and audio streams
            video_stream = ffmpeg.input(str(video_no_audio)).video
            audio_stream = ffmpeg.input(str(original_video)).audio

            # Merge and output
            (
                ffmpeg.output(
                    video_stream,
                    audio_stream,
                    str(output_final),
                    vcodec="libx264",
                    acodec="aac",
                    shortest=None,
                )
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True, quiet=True)
            )

            self.logger.info(f"✅ Audio merged successfully: {output_final}")

            # Remove the temporary video without audio
            if video_no_audio.exists():
                video_no_audio.unlink()
                self.logger.info(f"Removed temporary file: {video_no_audio}")

            return True

        except ffmpeg.Error as e:
            stderr = e.stderr.decode() if e.stderr else "No error details"
            self.logger.error(f"FFmpeg error during audio merge: {stderr}")
            return False
        except Exception as e:
            self.logger.error(f"Failed to merge audio: {e}")
            return False

    def process_video(
        self,
        input_path: str,
        output_path: str,
        show_preview: bool = False,
        merge_audio: bool = True,
    ) -> bool:
        """
        Process video frame-by-frame with streaming to avoid memory issues.
        Only holds 1 frame in memory at a time.
        Automatically merges audio from original video if merge_audio=True.
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        if not input_path.exists():
            self.logger.error(f"Input video not found: {input_path}")
            return False

        # If audio merge is requested, create temp file first
        if merge_audio and ffmpeg is not None:
            temp_output = (
                output_path.parent / f"{output_path.stem}_temp{output_path.suffix}"
            )
            final_output = output_path
            actual_output = temp_output
        else:
            actual_output = output_path
            final_output = output_path

        # Open video
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            self.logger.error("Failed to open video")
            return False

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.logger.info(
            f"Input: {width}x{height} @ {fps:.2f}fps | {total_frames} frames"
        )
        self.logger.info(f"Output: {final_output}")

        # Prepare output writer
        actual_output.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(actual_output), fourcc, fps, (width, height))

        if not out.isOpened():
            self.logger.error("Failed to create output video writer")
            cap.release()
            return False

        # Transform
        to_tensor = transforms.ToTensor()

        # Process frames
        success_count = 0
        with tqdm(total=total_frames, desc="Processing Video", unit="frame") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                try:
                    # Preprocess: BGR -> RGB -> Tensor
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    input_tensor = to_tensor(frame_rgb).unsqueeze(0).to(self.device)

                    if self.use_amp:
                        input_tensor = input_tensor.half()

                    # Inference with tiling
                    restored_tensor = self._tiled_inference(input_tensor)

                    # Postprocess: Tensor -> Numpy -> BGR
                    restored_img = (
                        restored_tensor.squeeze(0)
                        .permute(1, 2, 0)
                        .float()
                        .cpu()
                        .numpy()
                    )
                    restored_img = (restored_img * 255).astype(np.uint8)
                    restored_bgr = cv2.cvtColor(restored_img, cv2.COLOR_RGB2BGR)

                    # Write frame
                    out.write(restored_bgr)
                    success_count += 1

                    # Check GPU temperature every 30 frames to prevent overheating
                    if success_count % 30 == 0:
                        check_gpu_temp(self.device, threshold=85, delay=15)

                    # Optional preview (skip if GUI not available)
                    if show_preview:
                        try:
                            preview = cv2.resize(restored_bgr, (960, 540))
                            cv2.imshow("Restored Preview", preview)
                            if cv2.waitKey(1) & 0xFF == ord("q"):
                                self.logger.info("Preview interrupted by user")
                                break
                        except cv2.error:
                            # OpenCV GUI not available (headless environment)
                            if success_count == 1:
                                self.logger.warning(
                                    "⚠️  Preview not available (headless environment). Continuing without preview..."
                                )
                            show_preview = False  # Disable further attempts

                except Exception as e:
                    self.logger.error(f"Error processing frame {success_count}: {e}")
                    continue

                pbar.update(1)

        # Cleanup
        cap.release()
        out.release()
        if show_preview:
            cv2.destroyAllWindows()

        self.logger.info(f"Completed! Processed {success_count}/{total_frames} frames")

        # Merge audio if requested and restoration was successful
        if success_count > 0:
            if merge_audio and ffmpeg is not None:
                merge_success = self._merge_audio_with_ffmpeg(
                    actual_output, input_path, final_output
                )
                if not merge_success:
                    self.logger.warning(
                        "Audio merge failed. Video saved without audio at: "
                        f"{actual_output}"
                    )
                    # Rename temp file to final if merge failed
                    if actual_output != final_output:
                        actual_output.rename(final_output)
            elif merge_audio and ffmpeg is None:
                self.logger.warning(
                    "⚠️  ffmpeg-python not installed. Video saved without audio."
                )
                self.logger.info("Install with: pip install ffmpeg-python")
                self.logger.info(
                    "Manual audio merge command:\n"
                    f"   ffmpeg -i {final_output} -i {input_path} -c copy -map 0:v:0 -map 1:a:0 {final_output.stem}_with_audio.mp4"
                )
            else:
                self.logger.info(f"✅ Video saved (no audio merge): {final_output}")

        return success_count > 0
