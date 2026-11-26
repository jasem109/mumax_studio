#!/usr/bin/env sh
# mumax Studio launcher for Linux / macOS.
cd "$(dirname "$0")" || exit 1
PY=${PYTHON:-python3}
command -v "$PY" >/dev/null 2>&1 || { echo "Python 3.9+ is required"; exit 1; }
"$PY" -c "import numpy, scipy, PIL" 2>/dev/null || "$PY" -m pip install --user -r requirements.txt
exec "$PY" -m mumaxstudio gui "$@"
