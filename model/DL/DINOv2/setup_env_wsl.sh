#!/usr/bin/env bash
# setup_env_wsl: WSL(Ubuntu) 안에서 실행한다.  bash setup_env_wsl.sh
# - 가상환경은 WSL 홈(~/agg_exp_venv)에 만든다 (/mnt/c 위보다 빠르고 안정적).
# - 설치: torch, torchvision (CUDA 12.x 빌드, PyTorch 공식 인덱스: cu126 -> cu124 순), numpy, scipy, scikit-learn, pillow, pyyaml, matplotlib
# - 설치 버전은 exp/requirements_freeze_wsl.txt 에 기록, 설치 후 env_smoke_test.py 로 확인한다.
# - 실패하면 멈춘다 (python3-venv 가 없으면 안내 메시지를 내고 종료 코드 2).
# 로그: outputs/setup_env_wsl/log.txt
set -u
EXP="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$HOME/agg_exp_venv"
OUT="$EXP/outputs/setup_env_wsl"; mkdir -p "$OUT"; LOG="$OUT/log.txt"
exec > >(tee "$LOG") 2>&1

echo "== 환경 =="
echo "exp 폴더: $EXP"
echo "WSL 배포판: $(. /etc/os-release && echo "$PRETTY_NAME") | 커널: $(uname -r)"
echo "python3: $(python3 --version 2>&1)"
echo "== GPU (WSL 안에서 보이는지) =="
(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>&1 || echo "nvidia-smi 실패: WSL 에서 GPU 가 보이지 않음") | head -3

if [ ! -x "$VENV/bin/python" ]; then
  echo "== venv 생성: $VENV =="
  python3 -m venv "$VENV" || { echo "venv 생성 실패. Ubuntu 터미널에서 다음을 한 번 실행하세요: sudo apt update && sudo apt install -y python3-venv python3-pip"; exit 2; }
else
  echo "venv 이미 있음, 재사용: $VENV"
fi
# Ubuntu 26.04 의 /tmp 는 RAM 기반 tmpfs(약 3.9GB)라서 torch/nvidia 휠을 풀다가 'No space left on device' 가 난다.
# pip 의 임시 폴더를 디스크 위로 옮긴다 (설치 후 삭제).
export TMPDIR="$HOME/.tmp_pip"; mkdir -p "$TMPDIR"
echo "pip 임시 폴더(TMPDIR): $TMPDIR (디스크: $(df -h "$TMPDIR" | tail -1 | awk '{print $4}') 여유)"
PIP="$VENV/bin/python -m pip"

ok=0
for tag in cu126 cu124; do
  echo "== torch/torchvision 설치 시도: $tag =="
  if $PIP install --no-input torch torchvision --index-url "https://download.pytorch.org/whl/$tag" > "$OUT/pip_torch_$tag.txt" 2>&1; then
    tail -3 "$OUT/pip_torch_$tag.txt"; echo "torch CUDA 인덱스: $tag"; ok=1; break
  fi
  if grep -q -e "No matching distribution" -e "Could not find a version" "$OUT/pip_torch_$tag.txt"; then
    echo "$tag 인덱스에 맞는 배포판 없음 -> 다음 인덱스 시도"
  else
    tail -8 "$OUT/pip_torch_$tag.txt"; echo "설치 실패 (배포판 없음 이외의 원인). 멈춘다."; exit 2
  fi
done
[ "$ok" = 1 ] || { echo "torch 설치 실패. 멈춘다."; exit 2; }

echo "== 나머지 패키지 =="
$PIP install --no-input numpy scipy scikit-learn pillow pyyaml matplotlib > "$OUT/pip_rest.txt" 2>&1 || { tail -8 "$OUT/pip_rest.txt"; echo "설치 실패. 멈춘다."; exit 2; }
tail -2 "$OUT/pip_rest.txt"

$VENV/bin/python -m pip freeze > "$EXP/requirements_freeze_wsl.txt"
echo "requirements_freeze_wsl.txt 기록: $(wc -l < "$EXP/requirements_freeze_wsl.txt")개 패키지"

echo "== 스모크 테스트 =="
"$VENV/bin/python" "$EXP/env_smoke_test.py"
rc=$?
if [ $rc -ne 0 ]; then echo "스모크 테스트 실패 (종료 코드 $rc). 멈춘다."; exit 3; fi
rm -rf "$TMPDIR"
echo "스모크 테스트 통과. 환경 준비 완료."
