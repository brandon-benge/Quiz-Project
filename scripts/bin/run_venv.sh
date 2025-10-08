#!/usr/bin/env bash
set -euo pipefail
# Simple venv runner for Quiz-Project
# Usage examples:
#   ./scripts/bin/run_venv.sh pip install -r requirements.txt
#   ./scripts/bin/run_venv.sh python scripts/quiz/generate_quiz.py --help

HERE=$(cd "$(dirname "$0")/../.." && pwd)
VENV_DIR="../.venv"
REQS_FILE="$HERE/requirements.txt"
PYTHON=${PYTHON:-python3}

if [[ ! -d "$VENV_DIR" ]]; then
  echo "[setup] Creating venv at $VENV_DIR"
  $PYTHON -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
export PATH="$VENV_DIR/bin:$PATH"
python -m pip install --upgrade pip >/dev/null

if [[ -f "$REQS_FILE" ]]; then
  echo "[deps] Syncing requirements.txt"
  pip install -r "$REQS_FILE"
fi

if [[ $# -eq 0 ]]; then
  echo "[ok] Venv ready at $VENV_DIR"
  exit 0
fi

# Execute common tools directly to avoid 'python -m xxx' mismatch
case "$1" in
  pip|pip3|python|python3|pytest|jupyter)
    exec "$@" ;;
  *)
    exec python "$@" ;;
esac
