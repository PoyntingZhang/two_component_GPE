#!/usr/bin/env bash
# Per-run environment settings. Load the target machine's Python/CUDA modules
# here if required, using its current documentation. This file is sourced by
# both setup_environment.sh and run.sh.
#
# On Colab, use the existing Python/CUDA environment; no modules are needed.
# On CSD3, choose the modules for the allocated GPU partition before setup.
if [[ -f .venv/bin/activate ]]; then
    source .venv/bin/activate
fi
