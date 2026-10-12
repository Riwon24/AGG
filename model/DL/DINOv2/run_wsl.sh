#!/usr/bin/env bash
# run_wsl: WSL 가상환경(~/agg_exp_venv)의 python 으로 exp 스크립트를 실행한다.  bash run_wsl.sh e00_make_splits.py [인자...]
# __pycache__ 를 만들지 않는다 (/mnt/c 의 프로젝트 폴더와 DINOv2 캐시 저장소를 깨끗하게 유지).
export PYTHONDONTWRITEBYTECODE=1
export PYTHONIOENCODING=utf-8
EXP="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$EXP" || exit 1
exec "$HOME/agg_exp_venv/bin/python" "$@"
