#!/usr/bin/env bash
# Run on an allocated GPU node, or in a Colab GPU runtime.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source ./environment.sh
mkdir -p logs
python -u -m gpe.main config/run.yaml "$@" 2>&1 | tee -a logs/run.log
