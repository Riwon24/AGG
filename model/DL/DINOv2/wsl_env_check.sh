#!/usr/bin/env bash
# wsl_env_check: WSL(Ubuntu) 안에서 실행 환경을 읽기 전용으로 점검한다. (설치/변경 없음)
EXP="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "배포판: $(. /etc/os-release && echo "$PRETTY_NAME")"
echo "사용자: $(whoami) | 커널: $(uname -r)"
echo "python3: $(python3 --version 2>&1) ($(command -v python3))"
echo "다른 python 버전: $(ls /usr/bin/python3.* 2>/dev/null | tr '\n' ' ')"
if python3 -m venv --help >/dev/null 2>&1; then echo "venv: OK"; else echo "venv: 없음"; fi
echo "pip: $(python3 -m pip --version 2>&1 | cut -c1-60)"
echo "--- GPU (WSL 안에서)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>&1 | head -3 || echo "nvidia-smi 실패"
echo "--- 프로젝트 폴더 접근 (exp)"
ls "$EXP" | head -30
echo "--- DINOv2 캐시 접근 (Windows 홈 경로)"
ls -la /mnt/c/Users/<USER>/.cache/torch/hub/checkpoints/ 2>&1 | tail -2
ls /mnt/c/Users/<USER>/.cache/torch/hub/ 2>&1 | head -5
echo "--- 인터넷: pytorch 휠 인덱스 / PyPI"
curl -sI -m 10 https://download.pytorch.org/whl/cu126/ 2>&1 | head -1 || echo "pytorch 인덱스 접속 실패"
curl -sI -m 10 https://pypi.org/simple/numpy/ 2>&1 | head -1 || echo "PyPI 접속 실패"
echo "--- 디스크/메모리"
df -h "$HOME" | tail -1
free -h | sed -n 2p
