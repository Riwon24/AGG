#!/usr/bin/env bash
# run_wsl: WSL 가상환경(~/agg_exp_venv)의 python 으로 exp_bgaug 스크립트를 실행한다.  bash run_wsl.sh b03_extract_features.py [인자...]
export PYTHONDONTWRITEBYTECODE=1
export PYTHONIOENCODING=utf-8
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE" || exit 1
exec "$HOME/agg_exp_venv/bin/python" "$@"
