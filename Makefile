# Makefile To manage a Python project using uv

# --- Variables--
# Use uv for all environment and packet operations
UV = uv
PYTHON_IN_VENV = .venv/bin/python

# The default goal is to show the help
.DEFAULT_GOAL := help

# `.PHONY` Declare goals that are not files.
.PHONY: help setup-uv install sync lock update collect preprocess train detect build clean

# --- Development teams ---

help:
	@echo "Makefile Prometheus-anomaly-detection-lstm project"
	@echo "--------------------------------------------------------"
	@echo "Available commands:"
	@echo "\n  make setup-uv      - Checks and installs uv if it is missing."
	@echo "  make install       - (Primary settings) Establishes uv, creates environment, and sets dependencies."
	@echo "  make sync          - (Quick update) Synchronizes the environment with the lock file."
	@echo "  make lock          - Updates the lock file (requirements.lock.txt) based on pyproject.toml."
	@echo "  make update        - Updates the lock file and immediately synchronizes the environment."
	@echo "\n--- Workflow teams ---"
	@echo "  make collect       - It starts collecting data from Prometheus."
	@echo "  make preprocess    - Starts pre-processing of collected data."
	@echo "  make train         - Starts model training."
	@echo "  make detect        - Starts a real-time anomaly detector."
	@echo "\n--- Assembly and cleaning ---"
	@echo "  make build         - Collects package distributions (wheel and sdist)."
	@echo "  make clean         - Removes assembly artifacts, caches, and virtual environments."

# Inspection and installation uv
setup-uv:
	@echo "⬇️  Verification and installation of uv. ."
	@if ! command -v uv &> /dev/null; then \
		echo "uv Not found. Installation. . ."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
		echo "‼️  Important: Restart your terminal or execute'source $$HOME/.cargo/env', I want the uv command to be available."; \
	else \
		echo "✅ uv It's already set."; \
	fi

# Creating an environment if there is none and setting dependencies
install: setup-uv .venv/pyvenv.cfg requirements.lock.txt
	@echo "📦 Installation of dependencies in the environment. . ."
	$(UV) pip sync requirements.lock.txt
	@echo "✅ The environment is ready and the dependencies are set. Activate it: source.venv/bin/activate"

# The explicit creation of a virtual environment
# This purpose is used as a dependency for 'install'
.venv/pyvenv.cfg:
	@echo "🐍 Creating a virtual environment .venv. ."
	$(UV) venv

# Quick synchronization with the lock file
sync:
	@echo "🔄 Synchronization with requirements.lock.txt ."
	$(UV) pip sync requirements.lock.txt
	@echo "✅ Synchronization complete."

# Update the lock file after changing pyproject.toml
lock:
	@echo "🔒 Update requirements.lock.txt from pyproject.toml. . ."
	$(UV) pip compile pyproject.toml --extra dev -o requirements.lock.txt
	@echo "✅ Lock-File updated."

# Combination of ‘lock’ and ‘sync’ for a full update
update: lock sync

# --- Workflow teams (using cli.py)

collect:
	@echo "📊 Start data collection. . ."
	$(UV) run python cli.py collect

preprocess:
	@echo "🛠️  Starting data preprocessing. . ."
	$(UV) run python cli.py preprocess

train:
	@echo "🎓 Start model training. . ."
	$(UV) run python cli.py train

detect:
	@echo "📡 Running a real-time detector. . ."
	$(UV) run python cli.py detect

# --- Assembly and cleaning ---

# Assembly of distributions
build:
	@echo "📦 Package assembly. . ."
	$(UV) run python -m build

# Delete all generated files
clean:
	@echo "🧹 Clean up the project. . ."
	rm -rf .venv
	rm -rf dist
	rm -rf build
	rm -rf .pytest_cache
	rm -rf *.egg-info
	find . -type d -name "__pycache__" -exec rm -r {} +
	echo "✅ Cleanup complete."

