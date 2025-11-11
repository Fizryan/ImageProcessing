# check_best.py
# Script to display all completed trials from Optuna study sorted by best LPIPS

import logging
from pathlib import Path
from typing import List, Dict, Any

import optuna
from tabulate import tabulate

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_study(study_name: str) -> optuna.Study:
    """Load Optuna study from SQLite database."""
    try:
        study = optuna.load_study(
            study_name=study_name,
            storage="sqlite:///hpo_study.db",
        )
        logger.info(f"✅ Loaded study: {study_name}")
        return study
    except KeyError:
        logger.error(f"❌ Study '{study_name}' not found in database.")
        return None
    except Exception as e:
        logger.error(f"❌ Error loading study: {e}")
        return None


def get_completed_trials(study: optuna.Study) -> List[optuna.trial.FrozenTrial]:
    """Get all completed trials from the study."""
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    return completed


def display_trials_sorted_by_lpips(trials: List[optuna.trial.FrozenTrial]) -> None:
    """Display trials sorted by LPIPS (lowest first)."""
    if not trials:
        logger.warning("⚠️  No completed trials found.")
        return

    sorted_trials = sorted(
        trials, key=lambda t: t.values[0] if t.values else float("inf")
    )

    print("\n" + "=" * 120)
    print("🏆 ALL COMPLETED TRIALS - SORTED BY LPIPS (LOWEST TO HIGHEST)")
    print("=" * 120 + "\n")

    table_data = []
    for rank, trial in enumerate(sorted_trials, 1):
        lpips = trial.values[0] if len(trial.values) > 0 else "N/A"
        training_time = trial.values[1] if len(trial.values) > 1 else "N/A"

        lpips_str = f"{lpips:.6f}" if isinstance(lpips, float) else str(lpips)
        time_str = (
            f"{training_time:.2f} hrs"
            if isinstance(training_time, float)
            else str(training_time)
        )

        table_data.append([rank, trial.number, lpips_str, time_str, len(trial.params)])

    headers = ["Rank", "Trial #", "LPIPS ↓", "Time (hrs)", "Params"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

    print("\n" + "=" * 120)
    print("📊 DETAILED CONFIGURATION - TOP 10 BEST TRIALS BY LPIPS")
    print("=" * 120 + "\n")

    for rank, trial in enumerate(sorted_trials[:10], 1):
        lpips = trial.values[0] if len(trial.values) > 0 else "N/A"
        training_time = trial.values[1] if len(trial.values) > 1 else "N/A"

        print(f"\n{'─' * 120}")
        print(f"🥇 RANK #{rank} | TRIAL #{trial.number}")
        print(f"{'─' * 120}")
        print(
            f"📈 LPIPS:         {lpips:.6f}"
            if isinstance(lpips, float)
            else f"📈 LPIPS:         {lpips}"
        )
        print(
            f"⏱️  Training Time: {training_time:.2f} hours"
            if isinstance(training_time, float)
            else f"⏱️  Training Time: {training_time}"
        )
        print(f"\n📋 PARAMETERS:")

        sorted_params = sorted(trial.params.items())
        for param_name, param_value in sorted_params:
            if isinstance(param_value, float):
                if param_value < 0.001:
                    param_str = f"{param_value:.2e}"
                else:
                    param_str = f"{param_value:.6f}"
            else:
                param_str = str(param_value)

            print(f"   • {param_name:25s} : {param_str}")

    print(f"\n{'=' * 120}\n")


def display_best_trial_details(study: optuna.Study) -> None:
    """Display details of the absolute best trial (lowest LPIPS)."""
    try:
        completed = get_completed_trials(study)
        if not completed:
            logger.warning("⚠️  No completed trials to show best trial.")
            return

        best_trial = min(
            completed, key=lambda t: t.values[0] if t.values else float("inf")
        )

        print("🏆 ABSOLUTE BEST TRIAL (LOWEST LPIPS)")

        print(f"Trial Number:     {best_trial.number}")
        print(
            f"LPIPS:            {best_trial.values[0]:.6f}"
            if best_trial.values
            else "LPIPS: N/A"
        )
        print(
            f"Training Time:    {best_trial.values[1]:.2f} hours"
            if len(best_trial.values) > 1
            else "Training Time: N/A"
        )

        print("\n📋 BEST CONFIGURATION:")
        sorted_params = sorted(best_trial.params.items())
        for param_name, param_value in sorted_params:
            if isinstance(param_value, float):
                if param_value < 0.001:
                    param_str = f"{param_value:.2e}"
                else:
                    param_str = f"{param_value:.6f}"
            else:
                param_str = str(param_value)
            print(f"   • {param_name:25s} : {param_str}")

    except Exception as e:
        logger.error(f"❌ Error displaying best trial: {e}")


def display_study_statistics(study: optuna.Study) -> None:
    """Display overall study statistics."""
    total_trials = len(study.trials)
    completed = len(
        [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    )
    pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
    failed = len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])

    print("\n" + "=" * 120)
    print("📊 STUDY STATISTICS")
    print("=" * 120)
    print(f"Study Name:       {study.study_name}")
    print(f"Total Trials:     {total_trials}")
    print(f"Completed:        {completed} ✅")
    print(f"Pruned:           {pruned} ⚠️")
    print(f"Failed:           {failed} ❌")
    print("=" * 120 + "\n")


def main():
    """Main function to display trial configurations."""
    print("OPTUNA STUDY ANALYZER - BEST TRIALS BY LPIPS")

    pretrained_model_path = Path("Training/checkpoints/best_model.pth")
    if pretrained_model_path.exists():
        study_name = "finetune_hpo"
        print(f"\n📌 Mode: Fine-tuning (Study: {study_name})")
    else:
        study_name = "scratch_hpo"
        print(f"\n📌 Mode: From Scratch (Study: {study_name})")

    study = load_study(study_name)
    if study is None:
        return

    display_study_statistics(study)

    completed_trials = get_completed_trials(study)

    if not completed_trials:
        logger.warning("⚠️  No completed trials found in the study.")
        return

    display_trials_sorted_by_lpips(completed_trials)

    display_best_trial_details(study)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user.")
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
