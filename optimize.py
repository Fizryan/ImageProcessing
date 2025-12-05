# optimize.py

import sys
import logging
import shutil
import time
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
    N_TRIALS_TO_RUN = 100
else:
    STUDY_NAME = "scratch_hpo"
    HPO_MAIN_DIR = Path("Training/optuna_from_scratch")
    logging.info("🚀 Training Mode: FROM SCRATCH")
    N_TRIALS_TO_RUN = 100


def objective(trial: optuna.trial.Trial) -> float:
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

    # --- Optimizer Parameters ---

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
        "batch_size", [1, 2, 4, 8]
    )
    config["use_sharpness_loss"] = False
    config["use_advanced_loss"] = False
    config["trial_number"] = trial.number

    MAX_OOM_ATTEMPTS = 3
    current_attempt = 0

    while current_attempt < MAX_OOM_ATTEMPTS:
        current_attempt += 1

        trial_dir = HPO_MAIN_DIR / f"trial_{trial.number}"
        config["checkpoint_dir"] = str(trial_dir / "checkpoints")
        config["preview_dir"] = str(trial_dir / "previews")
        config["log_file"] = str(trial_dir / "training.log")
        config["tensorboard_log_dir"] = str(HPO_MAIN_DIR / "tensorboard_logs")

        tb_log_path = HPO_MAIN_DIR / "tensorboard_logs" / f"trial_{trial.number}"

        try:
            logging.info(
                "--- Starting Trial %d (Attempt %d/%d) (Mode: %s) ---",
                trial.number,
                current_attempt,
                MAX_OOM_ATTEMPTS,
                "Fine-Tune" if IS_FINETUNE_MODE else "From Scratch",
            )

            trainer = Trainer(config)
            result = trainer.train()

            if result is None:
                logging.warning(
                    "Trial %d (Attempt %d) did not return a valid result. Pruning.",
                    trial.number,
                    current_attempt,
                )
                raise optuna.exceptions.TrialPruned()

            best_lpips, training_time = result
            logging.info(
                f"Trial {trial.number} finished. LPIPS: {best_lpips:.4f}, Time: {training_time:.2f} hrs."
            )
            if Path(config["preview_dir"]).exists():
                shutil.rmtree(config["preview_dir"])

            return best_lpips

        except KeyboardInterrupt:
            logging.info("Trial %d interrupted by user.", trial.number)
            if trial_dir.exists():
                shutil.rmtree(trial_dir)
            if tb_log_path.exists():
                shutil.rmtree(tb_log_path)
            raise

        except torch.cuda.OutOfMemoryError:
            logging.error(
                "Trial %d (Attempt %d/%d) run out of memory.",
                trial.number,
                current_attempt,
                MAX_OOM_ATTEMPTS,
                exc_info=True,
            )

            if trial_dir.exists():
                shutil.rmtree(trial_dir)
            if tb_log_path.exists():
                shutil.rmtree(tb_log_path)

            if current_attempt < MAX_OOM_ATTEMPTS:
                logging.warning("Retrying in 15s...")
                time.sleep(15)
            else:
                logging.error(
                    "Trial %d failed OOM after %d attempts. PRUNING this trial.",
                    trial.number,
                    MAX_OOM_ATTEMPTS,
                )
                raise optuna.exceptions.TrialPruned()

        except Exception:
            logging.error(
                "Trial %d (Attempt %d) failed with an exception.",
                trial.number,
                current_attempt,
                exc_info=True,
            )
            if trial_dir.exists():
                shutil.rmtree(trial_dir)
            if tb_log_path.exists():
                shutil.rmtree(tb_log_path)
            raise optuna.exceptions.TrialPruned()

    raise optuna.exceptions.TrialPruned()


if __name__ == "__main__":
    study = optuna.create_study(
        direction="minimize",  # Minimize LPIPS
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.MedianPruner(),
        study_name=STUDY_NAME,
        storage="sqlite:///hpo_study.db",
        load_if_exists=True,
    )

    MAX_PRUNED_LIMIT = 3

    pruned_count = len(
        [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    )
    logging.info(f"Resuming study. Found {pruned_count} previously pruned trials.")

    total_trials_run_so_far = len(study.trials)

    try:
        while total_trials_run_so_far < N_TRIALS_TO_RUN:

            if pruned_count >= MAX_PRUNED_LIMIT:
                logging.error(
                    f"🛑 Optimization stopped: Prune limit ({MAX_PRUNED_LIMIT}) reached."
                )
                break

            current_trial_number_display = total_trials_run_so_far + 1
            logging.info(
                f"--- Starting Trial {current_trial_number_display}/{N_TRIALS_TO_RUN} (Pruned count: {pruned_count}/{MAX_PRUNED_LIMIT}) ---"
            )

            try:
                study.optimize(objective, n_trials=1)

                last_trial = study.trials[-1]
                if last_trial.state == optuna.trial.TrialState.PRUNED:
                    logging.warning(
                        f"Trial {last_trial.number} pruned by MedianPruner."
                    )
                    pruned_count += 1
                elif last_trial.state == optuna.trial.TrialState.FAIL:
                    logging.warning(f"Trial {last_trial.number} failed.")

            except optuna.exceptions.TrialPruned:
                logging.warning(
                    f"Trial PRUNED (OOM 3x). Prune count: {pruned_count + 1}"
                )
                pruned_count += 1

            total_trials_run_so_far = len(study.trials)

    except KeyboardInterrupt:
        logging.info("\n🛑 Optimization stopped by user.")
    except Exception as e:
        logging.error(f"\n🛑 Optimization stopped due to an error: {e}", exc_info=True)

    logging.info("Optimization finished.")

    finished_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]
    logging.info("Number of finished trials: %d", len(finished_trials))

    try:
        best_trial = study.best_trial
        logging.info("🏆 Best trial so far (Number: %d):", best_trial.number)
        logging.info("    Value (LPIPS): %.4f", best_trial.value)
        logging.info("    Parameters:")
        for key, value in best_trial.params.items():
            logging.info("      %s: %s", key, value)

    except ValueError:
        logging.warning("No trials were completed successfully.")
