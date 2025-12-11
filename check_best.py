# check_best.py
import os
import subprocess
import sys


def main():
    """
    Checks for optuna-dashboard and provides instructions to run it.
    """
    db_file = "hpo_study.db"

    print("=" * 60)
    print("📊 Optuna Study Dashboard Launcher")
    print("=" * 60)

    if not os.path.exists(db_file):
        print(f"❌ Error: Database file '{db_file}' not found in this directory.")
        print("Please run the optimization script ('optimize.py') first to create it.")
        return

    try:
        import optuna_dashboard

        print(f"✅ 'optuna-dashboard' is installed.")
        print(f"🚀 Launching dashboard for '{db_file}'...")
        print("   Open your browser to http://127.0.0.1:8080")
        print("   Press Ctrl+C in the terminal to stop the dashboard.")
        print("-" * 60)

        command = ["optuna-dashboard", f"sqlite:///{db_file}"]
        subprocess.run(command)

    except ImportError:
        print("⚠️ 'optuna-dashboard' is not installed.")
        print("\nTo visualize your study, please install and run the dashboard:")
        print("\n1. Install the package:")
        print("   pip install optuna-dashboard")
        print("\n2. Run the dashboard from your terminal:")
        print(f"   optuna-dashboard sqlite:///{db_file}")
        print("\n3. Open your web browser to the address shown in the terminal.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDashboard stopped by user. Goodbye!")
        sys.exit(0)
