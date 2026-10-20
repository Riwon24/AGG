#!/usr/bin/env bash
# run_pipeline_wsl: WSL 에서 e00 ~ e11 을 순서대로 실행한다 (g00 은 먼저 실행해 outputs/g00_tile_list/tiles.csv 가 있어야 함).  bash run_pipeline_wsl.sh
# 한 단계라도 실패하면 멈추고 그 단계 이름과 종료 코드를 남긴다 (e05, e07 은 순열 기준을 어기면 종료 코드 4).
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE" || exit 1
[ -f outputs/g00_tile_list/tiles.csv ] || { echo "outputs/g00_tile_list/tiles.csv 없음: 먼저 g00_make_tile_list.py 실행"; exit 2; }
for step in e00_make_splits.py e01_extract_features.py e02_reference_baselines.py e03_run_cv.py e04_report.py e05_sanity_permutation.py e06_extra_checks.py e07_permutation_other_settings.py e11_train_final.py; do
  echo "[$(date +%H:%M:%S)] $step 시작"
  bash ./run_wsl.sh "$step" > "/tmp/exp_grid_${step%.py}.out" 2>&1
  rc=$?
  echo "[$(date +%H:%M:%S)] $step 종료 코드 $rc"
  [ $rc -eq 0 ] || { echo "중단: $step (종료 코드 $rc), 자세한 내용은 outputs/${step%.py}/log.txt"; exit $rc; }
done
echo "[$(date +%H:%M:%S)] e11 --verify 시작"
bash ./run_wsl.sh e11_train_final.py --verify > /tmp/exp_grid_e11_verify.out 2>&1; rc=$?
echo "[$(date +%H:%M:%S)] e11 --verify 종료 코드 $rc"
exit $rc
