# Inference_Video.py
# Streaming video inference with tiling for full-resolution video restoration

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
from program.LoggingSetup import setup_logger

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
        self.logger = setup_logger(self.__class__.__name__)
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

        state_dict = None
        if "ema_state_dict" in checkpoint:
            state_dict = checkpoint["ema_state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        base_channels = 32
        if "intro.weight" in state_dict:
            base_channels = state_dict["intro.weight"].shape[0]
            self.logger.info(
                f"Detected base_channels={base_channels} from checkpoint weights"
            )

        if "config" in checkpoint:
            config = checkpoint["config"]
            model_class = get_model(config)
            model_params = {
                "in_channels": 3,
                "out_channels": 3,
                "base_channels": base_channels,
                "use_checkpointing": False,
                "use_global_residual": True,
            }
            model = model_class(**model_params)
        else:
            self.logger.warning(
                "Config not found in checkpoint, using SOTARestorationUNet"
            )
            model = SOTARestorationUNet(
                in_channels=3,
                out_channels=3,
                base_channels=base_channels,
                use_checkpointing=False,
                use_global_residual=True,
            )

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

        if h <= patch_h and w <= patch_w:
            if self.use_amp:
                with autocast(device_type="cuda", dtype=torch.float16):
                    return self.model(image_tensor)
            return self.model(image_tensor)

        stride_w = max(1, patch_w - self.overlap)
        stride_h = max(1, patch_h - self.overlap)

        pad_h = (stride_h - (h - patch_h) % stride_h) % stride_h
        pad_w = (stride_w - (w - patch_w) % stride_w) % stride_w
        padded_tensor = F.pad(image_tensor, (0, pad_w, 0, pad_h), "reflect")
        _, _, padded_h, padded_w = padded_tensor.shape

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

        blend_mask = self._generate_blend_mask((patch_w, patch_h), image_tensor.device)
        blend_mask = blend_mask.float()

        for y in range(0, padded_h - patch_h + 1, stride_h):
            for x in range(0, padded_w - patch_w + 1, stride_w):
                patch = padded_tensor[:, :, y : y + patch_h, x : x + patch_w]

                if self.use_amp:
                    with autocast(device_type="cuda", dtype=torch.float16):
                        patch_result = self.model(patch)
                else:
                    patch_result = self.model(patch)

                patch_result = patch_result.float()

                result_accumulator[:, :, y : y + patch_h, x : x + patch_w] += (
                    patch_result * blend_mask
                )
                divisor[:, :, y : y + patch_h, x : x + patch_w] += blend_mask

        divisor = torch.where(divisor == 0, torch.ones_like(divisor), divisor)
        final_tensor = result_accumulator / divisor

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

            video_stream = ffmpeg.input(str(video_no_audio)).video
            audio_stream = ffmpeg.input(str(original_video)).audio

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

        if merge_audio and ffmpeg is not None:
            temp_output = (
                output_path.parent / f"{output_path.stem}_temp{output_path.suffix}"
            )
            final_output = output_path
            actual_output = temp_output
        else:
            actual_output = output_path
            final_output = output_path

        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            self.logger.error("Failed to open video")
            return False

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.logger.info(
            f"Input: {width}x{height} @ {fps:.2f}fps | {total_frames} frames"
        )
        self.logger.info(f"Output: {final_output}")

        actual_output.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(actual_output), fourcc, fps, (width, height))

        if not out.isOpened():
            self.logger.error("Failed to create output video writer")
            cap.release()
            return False

        to_tensor = transforms.ToTensor()

        success_count = 0
        with tqdm(total=total_frames, desc="Processing Video", unit="frame") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                try:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    input_tensor = to_tensor(frame_rgb).unsqueeze(0).to(self.device)

                    if self.use_amp:
                        input_tensor = input_tensor.half()

                    restored_tensor = self._tiled_inference(input_tensor)

                    restored_img = (
                        restored_tensor.squeeze(0)
                        .permute(1, 2, 0)
                        .float()
                        .cpu()
                        .numpy()
                    )
                    restored_img = (restored_img * 255).astype(np.uint8)
                    restored_bgr = cv2.cvtColor(restored_img, cv2.COLOR_RGB2BGR)

                    out.write(restored_bgr)
                    success_count += 1

                    if success_count % 10 == 0:
                        check_gpu_temp(self.device)

                    if show_preview:
                        try:
                            preview = cv2.resize(restored_bgr, (960, 540))
                            cv2.imshow(
                                "Video Restoration - Live Preview (Press 'q' to stop)",
                                preview,
                            )
                            if cv2.waitKey(1) & 0xFF == ord("q"):
                                self.logger.info(
                                    "⏸️  Preview interrupted by user (Press 'q')"
                                )
                                break
                        except cv2.error as e:
                            if success_count == 1:
                                self.logger.warning(
                                    f"⚠️  Preview not available: {e}\n"
                                    "   This can happen in headless environments (SSH/WSL without X11).\n"
                                    "   Video processing continues without preview..."
                                )
                            show_preview = False

                except Exception as e:
                    self.logger.error(f"Error processing frame {success_count}: {e}")
                    continue

                pbar.update(1)

        cap.release()
        out.release()
        if show_preview:
            cv2.destroyAllWindows()

        self.logger.info(f"Completed! Processed {success_count}/{total_frames} frames")

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
