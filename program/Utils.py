# program/Utils.py
# This module provides shared utility functions for the application.

import logging
import time
import torch

try:
    from GPUtil import getGPUs
except ImportError:
    getGPUs = None
    logging.warning("GPUtil not available. GPU temp monitoring disabled.")

logger = logging.getLogger(__name__)


def check_gpu_temp(device, threshold: float = 85, delay: int = 15):
    """Checks GPU temperature and pauses if it exceeds a threshold."""
    if not getGPUs or device.type != "cuda":
        return

    try:
        gpu = getGPUs()[0]
        temperature = gpu.temperature
        if temperature >= threshold:
            # logger.warning(
            #     f"GPU temperature high: {temperature}°C. Cooling down for {delay} seconds."
            # )
            time.sleep(delay)
    except Exception as e:
        logger.error(f"GPU temperature check failed: {e}")


def load_model_weights(model: torch.nn.Module, state_dict: dict):
    """
    Loads a state_dict into a model, handling compiled vs. non-compiled model mismatches
    and DataParallel/DDP prefixes.
    """
    is_model_compiled = hasattr(model, "_orig_mod")
    is_ckpt_compiled = any(key.startswith("_orig_mod.") for key in state_dict.keys())

    if is_model_compiled and not is_ckpt_compiled:
        logger.debug(
            "Adding '_orig_mod.' prefix to checkpoint keys for compiled model."
        )
        state_dict = {f"_orig_mod.{k}": v for k, v in state_dict.items()}
    elif not is_model_compiled and is_ckpt_compiled:
        logger.debug(
            "Removing '_orig_mod.' prefix from checkpoint keys for non-compiled model."
        )
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    if any(key.startswith("module.") for key in state_dict.keys()):
        logger.debug("Removing 'module.' prefix from checkpoint keys.")
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=True)
    logger.info("Model weights loaded successfully.")
