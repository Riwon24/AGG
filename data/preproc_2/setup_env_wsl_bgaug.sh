#!/usr/bin/env bash
# setup_env_wsl_bgaug: 기존 WSL 가상환경(~/agg_exp_venv)에 배경 증강 이미지 생성(b03)에 필요한 opencv 만 추가 설치한다.
# Windows 쪽 전처리와 같은 버전(opencv-python-headless 5.0.0.93)을 쓴다 (preproc/requirements_freeze.txt).
# 설치 후 전체 버전 목록을 requirements_freeze_wsl.txt 에 기록한다.  실행: wsl -d Ubuntu -- bash setup_env_wsl_bgaug.sh
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$HOME/agg_exp_venv"
OUT="$HERE/outputs/setup_env_wsl_bgaug"; mkdir -p "$OUT"
exec > >(tee "$OUT/log.txt") 2>&1
[ -x "$VENV/bin/python" ] || { echo "venv 없음: $VENV (exp/setup_env_wsl.sh 먼저)"; exit 2; }
export TMPDIR="$HOME/.tmp_pip"; mkdir -p "$TMPDIR"
echo "설치 전 cv2: $("$VENV/bin/python" -c 'import cv2; print(cv2.__version__)' 2>&1 | tail -1)"
"$VENV/bin/python" -m pip install --no-input "opencv-python-headless==5.0.0.93" || { echo "설치 실패"; exit 3; }
echo "설치 후 cv2: $("$VENV/bin/python" -c 'import cv2, numpy; print(cv2.__version__, "numpy", numpy.__version__)' 2>&1 | tail -1)"
"$VENV/bin/python" -m pip freeze > "$HERE/requirements_freeze_wsl.txt"
rm -rf "$TMPDIR"
echo "freeze 기록: $HERE/requirements_freeze_wsl.txt ($(wc -l < "$HERE/requirements_freeze_wsl.txt")개)"
