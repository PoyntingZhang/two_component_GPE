#!/usr/bin/env bash
# Double-click on macOS. The first launch installs the small local GUI dependencies.
set -e
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ ! -x .venv/bin/python ]]; then
    python3 -m venv .venv
fi
if ! .venv/bin/python -c 'import yaml, numpy, matplotlib' >/dev/null 2>&1; then
    .venv/bin/python -m pip install -r gui_periphery/requirements.txt
fi
exec .venv/bin/python -m gui_periphery.app
