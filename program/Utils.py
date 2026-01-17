import time
import torch
from typing import Optional, List, Union

from program.LoggingManager import LoggingManager

logger = LoggingManager.setup_logging(__name__)

try:
    import GPUtil

    HAS_GPUTIL = True
except ImportError:
    HAS_GPUTIL = False
    logger.warning(
        "GPUtil module not found. GPU monitoring/throttling will be disabled."
    )


class Utils:
    def __init__(self):
        raise RuntimeError("Utils is a static class")

    @staticmethod
    def _get_target_gpu_id(device: torch.device) -> int:
        if device.type != "cuda":
            return -1
        return device.index if device.index is not None else 0

    @staticmethod
    def get_gpu_info(device: Optional[torch.device] = None) -> None:
        if not HAS_GPUTIL:
            return

        try:
            gpus = GPUtil.getGPUs()
            target_id = Utils._get_target_gpu_id(device) if device else -1

            for gpu in gpus:
                if target_id != -1 and gpu.id != target_id:
                    continue

                logger.info(f"--- GPU {gpu.id}: {gpu.name} ---")
                logger.info(
                    f"   Load: {gpu.load * 100:.1f}% | Temp: {gpu.temperature}°C"
                )
                logger.info(
                    f"   VRAM: {gpu.memoryUsed}MB / {gpu.memoryTotal}MB ({gpu.memoryUtil * 100:.1f}%)"
                )

        except Exception as e:
            logger.error(f"Failed to retrieve GPU Info: {e}")

    @staticmethod
    def get_gpu_load(device: Optional[torch.device] = None) -> Optional[float]:
        if not HAS_GPUTIL:
            return None

        try:
            gpus = GPUtil.getGPUs()
            target_id = Utils._get_target_gpu_id(device) if device else -1

            for gpu in gpus:
                if target_id != -1 and gpu.id != target_id:
                    continue

                return gpu.load

        except Exception:
            pass
        return None

    @staticmethod
    def get_vram_usage(device: torch.device) -> str:
        if not HAS_GPUTIL or device.type != "cuda":
            return "N/A"

        try:
            target_id = Utils._get_target_gpu_id(device)
            target_gpu = next((g for g in GPUtil.getGPUs() if g.id == target_id), None)

            if target_gpu:
                return f"{int(target_gpu.memoryUsed)}/{int(target_gpu.memoryTotal)} MB"
        except Exception:
            pass
        return "N/A"

    @staticmethod
    def check_gpu_temp(
        device: torch.device, threshold: float = 80.0, delay: int = 30
    ) -> None:
        if not HAS_GPUTIL or device.type != "cuda":
            return

        try:
            target_id = Utils._get_target_gpu_id(device)
            target_gpu = next((g for g in GPUtil.getGPUs() if g.id == target_id), None)

            if not target_gpu:
                return

            temp = target_gpu.temperature

            if temp >= threshold + 5:
                logger.warning(
                    f"CRITICAL GPU TEMP: {temp}°C (Limit: {threshold + 5}°C). "
                    f"Throttling execution for {delay * 2}s to cool down..."
                )
                time.sleep(delay * 2)
                logger.info(f"Resuming execution. Temp check passed.")

            elif temp >= threshold:
                logger.warning(
                    f"HIGH GPU TEMP: {temp}°C (Limit: {threshold}°C). "
                    f"Pausing for {delay}s..."
                )
                time.sleep(delay)

        except Exception as e:
            logger.error(f"Error monitoring GPU temp: {e}")
