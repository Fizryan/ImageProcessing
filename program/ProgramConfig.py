VERSION = ""
BASE_DIR = "database"

VER_PATH = f"/{VERSION}" if VERSION else ""

DATASET_ROOT = f"{BASE_DIR}/dataset"
MODELS_ROOT = f"{BASE_DIR}{VER_PATH}/models"

MAIN_PATHS = {
    "dataset": DATASET_ROOT,
    "train_dir": f"{DATASET_ROOT}/train",
    "val_dir": f"{DATASET_ROOT}/val",
    "models": MODELS_ROOT,
    "tensorboard": f"{BASE_DIR}{VER_PATH}/tensorboard",
    "logs": f"{BASE_DIR}{VER_PATH}/logs",
}

USE_MASK = False

DIRECTORIES_CONFIG = {
    "checkpoints_dir": f"{MAIN_PATHS['models']}/checkpoints",
    "train_clean_dir": f"{MAIN_PATHS['train_dir']}/clean_image",
    "train_mask_dir": f"{MAIN_PATHS['train_dir']}/mask_image" if USE_MASK else None,
    "val_clean_dir": f"{MAIN_PATHS['val_dir']}/clean_image",
    "val_mask_dir": f"{MAIN_PATHS['val_dir']}/mask_image" if USE_MASK else None,
}
