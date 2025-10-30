# optimize.py

import sys
import logging
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Tuple

import optuna
import torch

from program.Training import Trainer
from program.Training_Config import TRAINING_CONFIG

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
PRETRAINED_MODEL_PATH = "Training/checkpoints/best_model.pth"
IS_FINETUNE_MODE = Path(PRETRAINED_MODEL_PATH).exists()

if IS_FINETUNE_MODE:
    STUDY_NAME = "finetune_hpo"
    HPO_MAIN_DIR = Path("Training/optuna_finetune")
    logging.info(
        "🚀 Training Mode: Fine Tuning from Model Checkpoint (Path:%s)",
        PRETRAINED_MODEL_PATH,
    )
else:
    STUDY_NAME = "scratch_hpo"
    HPO_MAIN_DIR = Path("Training/optuna_from_scratch")
    logging.info("🚀 Training Mode: FROM SCRATCH")


def objective(trial: optuna.trial.Trial) -> Tuple[float, float]:
    config = deepcopy(TRAINING_CONFIG)

    if IS_FINETUNE_MODE:
        logging.info("Using pre-trained model for fine-tuning.")
        config["finetune_checkpoint_path"] = PRETRAINED_MODEL_PATH

        config["learning_rate"] = trial.suggest_float(
            "learning_rate", 1e-8, 5e-5, log=True
        )
        config["weight_decay"] = trial.suggest_float(
            "weight_decay", 1e-7, 1e-4, log=True
        )

        scheduler_type = trial.suggest_categorical(
            "scheduler", ["onecycle", "cosine_restarts"]
        )
        config["scheduler"] = scheduler_type

        config["l1_weight"] = trial.suggest_float("l1_weight", 1.0, 2.0)
        config["lpips_weight"] = trial.suggest_float("lpips_weight", 0.5, 1.5)
        config["fft_weight"] = trial.suggest_float("fft_weight", 0.05, 0.3, log=True)

        config["num_epochs"] = 30
        config["early_stopping_patience"] = 7
    else:
        logging.info("Using scratch training.")
        config["finetune_checkpoint_path"] = None

        config["learning_rate"] = trial.suggest_float(
            "learning_rate", 1e-4, 5e-4, log=True
        )
        config["weight_decay"] = trial.suggest_float(
            "weight_decay", 1e-6, 1e-3, log=True
        )
        config["base_channels"] = trial.suggest_categorical(
            "base_channels", [16, 24, 32]
        )
        config["l1_weight"] = trial.suggest_float("l1_weight", 0.7, 1.5)
        config["lpips_weight"] = trial.suggest_float("lpips_weight", 0.3, 1.0)
        config["fft_weight"] = trial.suggest_float("fft_weight", 0.05, 0.3, log=True)
        config["onecycle_params"]["pct_start"] = trial.suggest_float(
            "pct_start", 0.1, 0.4
        )

        config["num_epochs"] = 50
        config["early_stopping_patience"] = 10

    config["grad_clip"] = trial.suggest_float("grad_clip", 0.5, 2.0)

    # --- GAN Parameters ---
    use_gan = True  # trial.suggest_categorical("use_gan", [True, False])
    config["use_gan"] = use_gan
    if use_gan:
        config["gan_weight"] = trial.suggest_float("gan_weight", 0.01, 0.5, log=True)
        config["discriminator_lr"] = trial.suggest_float(
            "discriminator_lr", 1e-5, 1e-3, log=True
        )

    config["dataloader_params"]["batch_size"] = trial.suggest_categorical(
        "batch_size", [1, 2, 4]
    )
    config["use_sharpness_loss"] = False
    config["use_advanced_loss"] = False
    config["trial_number"] = trial.number

    trial_dir = HPO_MAIN_DIR / f"trial_{trial.number}"
    config["checkpoint_dir"] = str(trial_dir / "checkpoints")
    config["preview_dir"] = str(trial_dir / "previews")
    config["log_file"] = str(trial_dir / "training.log")
    config["tensorboard_log_dir"] = str(HPO_MAIN_DIR / "tensorboard_logs")

    try:
        logging.info(
            "--- Starting Trial %d (Mode: %s) ---",
            trial.number,
            "Fine-Tune" if IS_FINETUNE_MODE else "From Scratch",
        )
        trainer = Trainer(config)
        result = trainer.train()

        if result is None:
            logging.warning(
                "Trial %d did not return a valid result. Pruning.",
                trial.number,
            )
            raise optuna.exceptions.TrialPruned()

        best_lpips, training_time = result
        logging.info(
            f"Trial {trial.number} finished. LPIPS: {best_lpips:.4f}, Time: {training_time:.2f} hrs."
        )
        shutil.rmtree(config["preview_dir"])

    except KeyboardInterrupt:
        logging.info("Trial %d interrupted by user.", trial.number)
        if trial_dir.exists():
            shutil.rmtree(trial_dir)
            shutil.rmtree(HPO_MAIN_DIR / "tensorboard_logs" / f"trial_{trial.number}")
        raise

    except torch.cuda.OutOfMemoryError:
        logging.error("Trial %d run out of memory.", trial.number, exc_info=True)
        if trial_dir.exists():
            shutil.rmtree(trial_dir)
            shutil.rmtree(HPO_MAIN_DIR / "tensorboard_logs" / f"trial_{trial.number}")
        raise
    except Exception:
        logging.error("Trial %d failed with an exception.", trial.number, exc_info=True)
        if trial_dir.exists():
            shutil.rmtree(trial_dir)
            shutil.rmtree(HPO_MAIN_DIR / "tensorboard_logs" / f"trial_{trial.number}")
        raise optuna.exceptions.TrialPruned()

    return best_lpips, training_time


if __name__ == "__main__":
    study = optuna.create_study(
        directions=["minimize", "minimize"],  # Minimize LPIPS and Training Time
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.MedianPruner(),
        study_name=STUDY_NAME,
        storage="sqlite:///hpo_study.db",
        load_if_exists=True,
    )

    try:
        study.optimize(objective, n_trials=20)
    except KeyboardInterrupt:
        logging.info("\n🛑 Optimisasi dihentikan oleh pengguna.")
    except torch.cuda.OutOfMemoryError:
        logging.info("\n🛑 Optimisasi dihentikan cuda out of memory.")

    finished_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]
    logging.info("Jumlah trial yang selesai: %d", len(finished_trials))

    try:
        best_trials = study.best_trials
        logging.info("🏆 Found %d best trials (Pareto front):", len(best_trials))
        for i, trial in enumerate(best_trials):
            logging.info("  --- Best Trial %d (Number: %d) ---", i + 1, trial.number)
            # Objectives: [LPIPS, Training Time]
            logging.info(
                "    Values: LPIPS=%.4f, Time=%.2f hrs",
                trial.values[0],
                trial.values[1],
            )
            logging.info("    Parameters:")
            for key, value in trial.params.items():
                logging.info("      %s: %s", key, value)

    except ValueError:
        logging.warning("Tidak ada trial yang berhasil diselesaikan.")
