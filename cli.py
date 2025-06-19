import argparse
import subprocess
import sys
from pathlib import Path


def run_script(script_name: str) -> None:
    """Runs this script in a separate process."""
    script_path = Path(__file__).resolve().parent / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"No script found: {script_path}")
    subprocess.run([sys.executable, str(script_path)], check=True)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Utility to start stages of data processing"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("collect", help="Data collection from Prometheus")
    subparsers.add_parser("preprocess", help="Preparation of collected data")
    subparsers.add_parser("train", help="Model learning")
    subparsers.add_parser("detect", help="Running a realtime detector")

    args = parser.parse_args()

    if args.command == "collect":
        run_script("data_collector.py")
    elif args.command == "preprocess":
        run_script("preprocess_data.py")
    elif args.command == "train":
        run_script("train_autoencoder.py")
    elif args.command == "detect":
        run_script("realtime_detector.py")


if __name__ == "__main__":
    main()
