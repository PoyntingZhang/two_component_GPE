#!/usr/bin/env bash
# Optional per-run virtual environment. Configure environment.sh first.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source ./environment.sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -r requirements-gpu.txt
mkdir -p logs
python -m pip freeze > logs/environment.txt
python -m gpe.main config/run.yaml --check
