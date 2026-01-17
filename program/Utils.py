import time
import torch
from program.LoggingManager import LoggingManager

try:
    from GPUtil import getGPUs
except ImportError:
    getGPUs = None

logger = LoggingManager.setup_logging(__name__)


class Utils:
    def __init__(self):
        raise RuntimeError("Utils is a static class")

    @staticmethod
    def get_gpu_info():
        if getGPUs:
            try:
                gpus = getGPUs()
                for gpu in gpus:
                    logger.info(f"GPU: {gpu.name}")
                    logger.info(f"Load: {gpu.load * 100}%")
                    logger.info(f"Memory: {gpu.memoryUsed}MB / {gpu.memoryTotal}MB")
                    logger.info(f"Temperature: {gpu.temperature}°C")
                    logger.info(f"GPU Utilization: {gpu.gpuUtil * 100}%")
                    logger.info(f"Memory Utilization: {gpu.memoryUtil * 100}%")
            except Exception as e:
                logger.error(f"Error getting GPU info: {e}")

    @staticmethod
    def get_gpu_load():
        if getGPUs:
            try:
                gpu = getGPUs()[0]
                return gpu.load
            except Exception as e:
                logger.error(f"Error getting GPU load: {e}")
                return 0.0

    @staticmethod
    def get_gpu_memory_usage():
        if getGPUs:
            try:
                gpu = getGPUs()[0]
                return gpu.memoryUsed
            except Exception as e:
                logger.error(f"Error getting GPU memory usage: {e}")
                return 0.0

    @staticmethod
    def check_gpu_temp(device: torch.device, threshold: float = 84, delay: int = 15):
        if not getGPUs or device.type != "cuda":
            return

        try:
            gpu = getGPUs()[0]
            temp: float = gpu.temperature
            if temp >= threshold + 4:
                time.sleep(delay * 2)
            elif temp >= threshold:
                time.sleep(delay)
        except Exception as e:
            logger.error(f"Error checking GPU temperature: {e}")
