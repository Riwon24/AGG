> **저장소 배치 안내**: 이 폴더는 로컬에서 `AGG_labeled_only/exp/` 로 실행한 코드의 복사본이다. 입력인 전처리 코드는 `data` 브랜치의 `data/preproc/` 에 있다. `config.yaml` 의 `paths`(`../preproc/outputs/...`)는 원래 폴더 배치 기준이라 이 저장소 배치에서는 경로를 맞춰야 한다. `outputs/`(특징, 예측, 로그, 최종 모델)와 이미지는 포함하지 않았고, 사용자 이름은 `<USER>` 로 바꿨다.

# 모델·실험 (고정 DINOv2 특징 → 이미지 단위 합치기 → 분류기)

전처리(`../preproc`)가 고른 10x 타일 1,514개에서 고정 DINOv2 ViT-B/14 특징을 뽑고, 이미지 단위로 합쳐 AGG0/1/2를 분류하는 실험(1단계).

- 범위: 특징 추출 → 합치기 → 분류기 → 이미지 단위 층화 반복 K-fold 평가. **attention 합치기, DINOv2 fine-tuning, pseudo-labeling, 7.5x/12.5x 배율은 하지 않았다** (결과를 보고 2단계에서 결정).
- 원본 이미지와 전처리 산출물(`../preproc/outputs/...`)은 읽기만 했다.
- 모든 실행은 단계 이름이 붙은 파이썬 스크립트이고, 파라미터는 `config.yaml` 한 곳에 있다.

## 1. 한눈에 보기

| 항목 | 값 |
|---|---|
| 입력 | 10x 이미지 60장 (AGG0 28 / AGG1 14 / AGG2 18), 선택 타일 1,514개 (`preproc` 의 `tiles.csv`, `selected==1`) |
| 평가 | 이미지 단위 층화 5-fold x 반복 10회 (seed 0), 한 이미지의 타일은 항상 같은 쪽 |
| primary (결과를 보기 전에 고정) | 특징 cls / 합치기 mean / 분류기 logreg → **accuracy 0.712±0.046, macro-F1 0.682±0.044** |
| 기준선 | 다수 클래스 0.467 / 0.212, **처리 조건 암기 0.900 / 0.893** |
| 16개 설정의 accuracy 범위 | 0.677 ~ 0.805 (최고 SL cls+patchmean 0.805 / 0.753) |
| 순열 검사 (20회) | 평균 accuracy 0.358, macro-F1 0.323 (최대 0.400): 기준 통과 |
| 실행 환경 | **WSL2 (Ubuntu 26.04) + RTX 3060**, torch 2.14.1+cu126 (Windows 에서는 못 돌림, 3절) |
| 재현성 | e00, e03 을 다시 돌려 `splits.csv`, `cv_predictions.csv` 의 sha256 이 같음 |

전체 보고서는 `outputs/e04_report/report.md`.

## 2. 용어

- 1단계 폴더(BASE, BASE-LC, BASE-LCSI, BASE-LSi, REF, REF-LN) = **금속 종류** (6종)
- 2단계 폴더(NPH1035, PH1180-1050 ...) = **처리 조건** = 처리 방법(NPH, PH1180, PH1205, PH1230, PH1255, 5종) x 온도(1035, 1050, 2종) = 10종
- 이미지 10x 는 (금속 종류 x 처리 조건)마다 1장, 모두 60장.
- `config.yaml` 의 `forbidden_terms` 에 있는 말은 스크립트, 로그, 보고서에서 쓰지 않는다 (e04, e05 가 검사).

## 3. 실행 환경 (중요)

이 PC 는 Windows **스마트 앱 컨트롤이 강제(켜짐)** 라서 pip 로 받은 서명 없는 DLL(`torch.dll`, `sklearn.neighbors`, pandas, skimage 등)이 차단된다 (`WinError 4551`, "애플리케이션 제어 정책"). 정책을 끄거나 우회하지 않고, 사용자가 **WSL2(Ubuntu)** 를 설치해서 쓰기로 했다. (이 PC 는 학교 소유이고 도메인/MDM 관리는 없다.)

- 실험 환경: WSL 홈의 `~/agg_exp_venv` (Python 3.14.4, torch 2.14.1+cu126, torchvision 0.29.1+cu126, scikit-learn 1.9.1, numpy 2.5.2, scipy 1.18.1, matplotlib 3.11.2). 전체 버전은 `requirements_freeze_wsl.txt`.
- GPU 는 WSL 안에서 보이고(`RTX 3060, 12GB`) CUDA 가 동작한다 (`env_smoke_test.py` 통과).
- DINOv2 저장소와 가중치는 Windows 쪽 캐시(`C:\Users\admin\.cache\torch\hub`)를 WSL 에서 `/mnt/c/...` 로 읽는다. `config.yaml` 의 경로 후보 목록이 Windows 와 WSL 양쪽을 처리한다.
- Ubuntu 26.04 의 `/tmp` 는 RAM 기반 tmpfs(약 3.9GB)라 torch 설치 시 pip 임시 폴더를 `~/.tmp_pip`(디스크)로 옮겼다 (`setup_env_wsl.sh`).
- Windows 쪽 `exp/.venv`(torch 설치됐지만 로드 차단, 4.71GB)는 **삭제했다.** 그 시도의 기록(설치 스크립트, 로그, 설치 버전 목록)을 모아 둔 `exp/_archive/`도 이후 삭제했다 (삭제 기록: `outputs/remove_exp_archive/log.txt`). 정책 차단 내용은 위 설명이 전부다.

### 환경을 다시 만들려면
Ubuntu 터미널에서 한 번 (`sudo` 비밀번호 필요):
```bash
sudo apt update && sudo apt install -y python3-venv python3-pip
```
그다음 PowerShell 에서:
```powershell
wsl -d Ubuntu -- bash "/mnt/c/Users/<USER>/Downloads/260929_images_for_AGG - manual labeling/AGG_labeled_only/exp/setup_env_wsl.sh"
```
설치와 스모크 테스트가 끝나면 `requirements_freeze_wsl.txt` 가 갱신된다. 실패하면 멈춘다.

## 4. 실행 방법

각 단계는 `run_wsl.sh` 로 실행한다 (WSL 가상환경의 python, `__pycache__` 를 만들지 않음).
```powershell
$EXP = "/mnt/c/Users/<USER>/Downloads/260929_images_for_AGG - manual labeling/AGG_labeled_only/exp"
wsl -d Ubuntu -- bash "$EXP/run_wsl.sh" e00_make_splits.py
```
순서: `e00 → e01 → e02 → e03 → e04 → e05` (이후 추가 확인 `e06`, `e07`). 소요(이번 실행 기준): e01 약 1분 이내(GPU), e03 약 12분(진행 로그상 SL 의 타일 단위 logreg 에서 대부분 소요), 나머지는 몇 초. 모든 스크립트가 `--config`, `--output` 인자를 받는다.

## 5. 폴더 구조

```
exp/
├── config.yaml                  # 모든 파라미터 (하드코딩 금지)
├── common.py                    # config, 로그, 입력 로딩, 조건 파싱, 지표, 부트스트랩, 금지어 검사
├── e00_make_splits.py ... e05_sanity_permutation.py   # 실험 6단계
├── e06_extra_checks.py, e07_permutation_other_settings.py, e11_train_final.py   # 추가 확인, 순열 검사, 최종 모델
├── setup_env_wsl.sh, run_wsl.sh, wsl_env_check.sh, env_smoke_test.py   # WSL 환경 설치/실행/점검
├── requirements_freeze_wsl.txt  # WSL 환경의 설치 버전
└── outputs/<스크립트명>/         # 단계별 출력과 log.txt (e00 ~ e07, e11_final_model, setup_env_wsl, remove_exp_archive)
```

## 6. 단계별 스크립트

| 단계 | 하는 일 | 주요 출력 |
|---|---|---|
| `e00_make_splits` | 처리 조건 파싱(10개 조건 전부와 일치, 조건마다 6장 assert) + 이미지 단위 층화 반복 K-fold 분할. 각 repeat 에서 모든 이미지가 정확히 한 번 test, 타일이 양쪽에 걸치지 않음 assert | `splits.csv`(3,000행), `conditions.csv` |
| `e01_extract_features` | 선택 타일 224x224 (resize 없음) → DINOv2 ViT-B/14 (고정, eval, no_grad, fp32, ImageNet 정규화). cls(768), patch 평균(768) 저장. 이미지별 마스크 bbox 크롭(224 resize)의 cls 도 저장 | `features.npz`, `tile_index.csv`, `whole_index.csv`, `meta.json`, `repo_files_sha256.csv` |
| `e02_reference_baselines` | 다수 클래스 예측, 처리 조건 암기(같은 처리 조건의 다른 이미지들의 다수 라벨, leave-one-image-out), 처리 조건과 라벨이 다른 이미지 목록 | `baseline_predictions.csv`, `baseline_metrics.csv`, `condition_discordant_images.csv` |
| `e03_run_cv` | 16개 설정을 e00 의 분할 그대로 실행: tiles(특징 2 x 합치기 3 x 분류기 2 = 12), SL 2, whole_image 2. 표준화 등 모든 통계는 fold 의 train 에서만 계산 | `cv_predictions.csv`(9,600행), `cv_metrics_per_repeat.csv`, `cv_summary.csv` |
| `e04_report` | 결과 표, 기준선 비교, 불일치 이미지 vs 나머지, 이미지별 오분류 빈도(4가지 집계), primary 혼동행렬, 부트스트랩 95% CI(이미지 단위 + 처리 조건 클러스터) | `report.md`, `results_table.csv` 등 |
| `e05_sanity_permutation` | 이미지 단위 라벨을 섞어(클래스별 개수 유지, fold 도 새로 층화) primary 를 20회. 기준을 어기면 누수 단정 없이 멈추고 보고 | `permutation_runs.csv` |
| `e06_extra_checks` (추가) | 새로 학습하지 않고 e00~e03 결과를 읽어 정리: primary 혼동행렬(풀링), 16개 설정 전체 표(6장 k/6, 54장 정확도), 비교 기준(6장 모두 AGG0, 금속 종류 단독), PH1255-1050 예측 분포, e01 결정성/NaN 검사 | `extra_report.md` 등 |
| `e07_permutation_other_settings` (추가) | e05 와 같은 순열/기준을 whole_image(logreg, knn)와 SL(cls+patchmean)에도 20회씩 (`config.yaml: permutation_extra`) | `permutation_other_runs.csv` |
| `e11_train_final` (마무리) | 60장 전체로 primary 를 config 값 그대로 학습해 저장 (새 실험 아님). `--verify` 로 저장본만 읽어 예측이 같은지 확인 | `outputs/e11_final_model/` (8.7절) |

## 7. 설정 (`config.yaml` 핵심)

- 평가: `n_splits=5`, `n_repeats=10`, `seed=0`.
- 모델: `dinov2_vitb14`, 입력 224, ImageNet mean/std (공식 repo 의 `IMAGENET_DEFAULT_MEAN/STD` 와 코드로 대조).
- 합치기: `mean`, `mean_std`(평균 + 표준편차 이어붙임, **ddof=0**), `max`. 타일 특징: `cls`, `cls+patchmean`.
- logreg: 다항 L2, `C=1.0`, `class_weight=balanced`, `StandardScaler` 는 train 에서만 fit, `max_iter=1000`(명세에 없는 수렴용 값; 수렴 경고 0건).
- kNN: `k=5`, cosine, 거리 가중(**SEM 논문의 Gaussian kernel 가중이 아니라 단순 거리 가중**), 표준화 없음.
- primary: `{features: cls, aggregator: mean, classifier: logreg}` — 결과를 보기 전에 고정했고 바꾸지 않았다. 나머지 15개는 모두 ablation 이고 표에 전부 싣는다 (성능순 정렬 없음).
- 부트스트랩 2,000회, 순열 20회(기준: 평균 macro-F1 0.33±0.10, 어느 회차도 0.55 이하 — **제안값이며 데이터에서 유도한 값이 아님**).

## 8. 결과

### 8.1 설정별 결과 (accuracy, macro-F1: 반복 10회의 평균±SD)

| 설정 | accuracy | macro-F1 | recall AGG0 / AGG1 / AGG2 |
|---|---|---|---|
| **tiles\|cls\|mean\|logreg (primary)** | 0.712±0.046 | 0.682±0.044 | 0.768 / 0.400 / 0.867 |
| tiles\|cls\|mean\|knn | 0.702±0.030 | 0.609±0.041 | 0.879 / 0.121 / 0.878 |
| tiles\|cls\|mean_std\|logreg | 0.728±0.032 | 0.687±0.033 | 0.807 / 0.329 / 0.917 |
| tiles\|cls\|mean_std\|knn | 0.697±0.039 | 0.627±0.045 | 0.832 / 0.179 / 0.889 |
| tiles\|cls\|max\|logreg | 0.708±0.029 | 0.668±0.022 | 0.804 / 0.321 / 0.861 |
| tiles\|cls\|max\|knn | 0.677±0.025 | 0.624±0.032 | 0.775 / 0.229 / 0.872 |
| tiles\|cls+patchmean\|mean\|logreg | 0.740±0.036 | 0.705±0.036 | 0.804 / 0.407 / 0.900 |
| tiles\|cls+patchmean\|mean\|knn | 0.718±0.027 | 0.629±0.028 | 0.889 / 0.143 / 0.900 |
| tiles\|cls+patchmean\|mean_std\|logreg | 0.745±0.035 | 0.700±0.042 | 0.846 / 0.329 / 0.911 |
| tiles\|cls+patchmean\|mean_std\|knn | 0.692±0.031 | 0.614±0.032 | 0.839 / 0.143 / 0.889 |
| tiles\|cls+patchmean\|max\|logreg | 0.738±0.037 | 0.690±0.031 | 0.850 / 0.314 / 0.894 |
| tiles\|cls+patchmean\|max\|knn | 0.680±0.028 | 0.633±0.030 | 0.768 / 0.250 / 0.878 |
| SL\|cls\|logreg | 0.790±0.026 | 0.736±0.038 | 0.918 / 0.357 / 0.928 |
| SL\|cls+patchmean\|logreg | 0.805±0.025 | 0.753±0.037 | 0.939 / 0.386 / 0.922 |
| whole_image\|cls\|logreg | 0.708±0.050 | 0.688±0.050 | 0.750 / 0.436 / 0.856 |
| whole_image\|cls\|knn | 0.692±0.020 | 0.621±0.022 | 0.839 / 0.207 / 0.839 |
| 다수 클래스 예측 (기준선) | 0.467 | 0.212 | 1.000 / 0.000 / 0.000 |
| **처리 조건 암기 (기준선)** | **0.900** | **0.893** | 0.857 / 0.929 / 0.944 |

SD 는 반복 간 값이라 반복이 독립이 아니므로 불확실성을 과소평가한다.

### 8.2 primary 부트스트랩 95% CI

| 대상 | 방식 | accuracy | macro-F1 |
|---|---|---|---|
| primary | 이미지 단위 | 0.712 [0.615, 0.802] | 0.682 [0.587, 0.768] |
| primary | 처리 조건 단위(클러스터) | 0.712 [0.583, 0.830] | 0.682 [0.555, 0.748] |
| 처리 조건 암기 | 이미지 단위 | 0.900 [0.817, 0.967] | 0.893 [0.798, 0.967] |
| 처리 조건 암기 | 처리 조건 단위(클러스터) | 0.900 [0.817, 0.983] | 0.893 [0.636, 0.952] |

이미지 단위 CI 는 이미지가 독립이라는 가정이라 같은 처리 조건 안의 상관 때문에 실제보다 좁다. 클러스터 표본 2,000개 중 18개는 클래스가 빠졌고(macro-F1 은 정답이 있는 클래스만 평균), 통계 검정은 하지 않았다.

### 8.3 처리 조건과 라벨이 다른 6장 vs 나머지 54장

- 6장은 BASE-LCSI 4장, BASE-LSi 1장, REF-LN 1장이다. 처리 조건 암기는 구조상 이 6장을 모두 틀린다 (0/6).
- 맞힌 수(반복 평균, 6장 중): primary 2.8 (1~4), 설정별 2.0 ~ 5.5 (whole_image 5.5, SL 4.9). 나머지 54장의 accuracy 는 primary 0.739.

### 8.4 오분류가 몰린 곳 (primary, 반복 10회 중 틀린 비율)

- 온도: 1050 → 0.400, 1035 → 0.177.
- 처리 조건: PH1205-1050 0.650, PH1230-1050 0.567, PH1255-1050 0.417 이 높고, NPH1035 0.033, NPH1050 0.100.
- 금속 종류: REF-LN 0.44, BASE-LSi 0.40, REF 0.37, BASE-LC/BASE-LCSI 0.19, BASE 0.14.
- primary 혼동행렬(반복 평균 개수와 행 비율): AGG1 의 51% 가 AGG0 으로, 40% 가 AGG1 로 예측된다.

### 8.5 해석할 때 주의 (추정)

- 처리 조건 암기 기준선은 **같은 처리 조건의 다른 금속 이미지 5장의 정답 라벨**을 쓴다. 모델은 이미지만 보므로, 기준선은 모델이 아니라 상한에 가까운 참고선으로 읽는 게 맞다. 어떤 설정도 이 기준선에 못 미쳤다.
- 6장에서 일부 설정이 높은 점수를 낸 것을 AGG 형태를 본다는 증거로 단정할 수 없다. 6장 중 4장이 BASE-LCSI 이고 이 금속은 라벨이 전반적으로 낮다(AGG0 70%). 모델이 **금속 종류를 알아봐서** 맞혔을 가능성을 이 데이터로는 구분할 수 없다.
- 라벨이 처리 조건에 90% 좌우되고(`preproc/outputs/s07_label_crosstab`), 이 평가는 이미지 단위 분할이라 test 이미지와 같은 처리 조건의 이미지가 항상 train 에 있다. 새 처리 조건에 대한 일반화는 이 평가로 알 수 없다.
- 60장이라 CI 가 넓고(±0.09 안팎), primary 와 ablation 의 차이(예: 0.712 vs 0.740)는 반복 간 SD 안이다.
- 순열 검사는 파이프라인 누수나 버그의 징후만 본다. 처리 조건을 알아보는 문제는 잡지 못한다.

### 8.6 추가 확인 (e06, e07; primary 와 설정은 바꾸지 않음)

- primary 혼동행렬(반복 10회 풀링, 600개 예측; 행=정답): AGG0 215 / 62 / 3, AGG1 72 / 56 / 12, AGG2 5 / 19 / 156 (행 합 280 / 140 / 180).
- 비교 기준 (accuracy, macro-F1, 불일치 6장 k/6, 나머지 54장 accuracy): 6장 모두 AGG0 예측 → 4/6. 금속 종류 단독(같은 금속 종류의 다른 9장의 다수 라벨, leave-one-image-out; 동률은 가장 작은 클래스 번호로 예측, 동률 이미지 14장) → 0.400, 0.190, 4/6, 0.370. 동률을 오답으로 세면 accuracy 0.200, 동률 클래스 중 균등 무작위의 기댓값이면 0.267.
- PH1255-1050 이미지 6장(모두 AGG0)에 대한 primary 의 예측(반복 10회 중 AGG0/AGG1/AGG2 횟수): BASE 10/0/0, BASE-LCSI 10/0/0, BASE-LSi 7/3/0, BASE-LC 4/4/2, REF-LN 4/6/0, REF 0/10/0 (합계 60번 중 AGG0 35, AGG1 23, AGG2 2).
- 순열 검사(20회, e05 와 같은 순열과 기준): whole_image|cls|logreg 평균 macro-F1 0.295 (최대 0.372), whole_image|cls|knn 0.300 (0.378), SL|cls+patchmean|logreg 0.318 (0.408) → 모두 제안한 기준 통과.
- e01 결정성: 같은 입력 두 번 실행한 최대 절대 차이 cls, patchmean, whole_cls 모두 0, `features.npz` 의 NaN/Inf 0개.

### 8.7 최종 모델 (e11)

60장 전체로 primary(특징 cls / 합치기 mean / 분류기 logreg, C=1.0, balanced, max_iter=1000 = config 값 그대로)를 한 번 학습해서 `outputs/e11_final_model/` 에 저장했다.

| 파일 | 내용 |
|---|---|
| `final_model.joblib` | StandardScaler + LogisticRegression 객체 (sklearn 1.9.1 에 묶임) |
| `final_model_params.npz` | 버전과 무관한 파라미터: `scaler_mean`, `scaler_scale`, `coef`, `intercept`, `classes` |
| `config_copy.yaml` | 학습에 쓴 config 사본 |
| `meta.json` | 클래스 순서 `["AGG0","AGG1","AGG2"]`(= 0,1,2), `features.npz` 의 sha256 (`e059374afaf6671c…`), 학습 이미지 60장 목록, 버전, 수렴 경고 0건 (n_iter 91) |
| `training_images.csv`, `train_predictions.csv/.npz` | 학습에 쓴 이미지 순서/라벨, 학습 시점의 예측과 확률 |
| `inputs_sha256.csv` | 보존 대상 파일의 실행 전/후 sha256 |
| `log.txt`, `verify_log.txt` | 학습 로그, 저장본 재검증 로그 |

- 추론 확인(`e11_train_final.py --verify`, 별도 프로세스): 저장본(joblib, npz)만 불러 60장에 예측을 다시 돌려서 학습 시점 예측과 비교했다. 예측 클래스가 60장 모두 같고(다른 이미지 0장), 확률도 완전히 같다(최대 절대 차이 0).
- **주의: 이 60장 정확도(1.000)는 학습 데이터에 대한 값이라 성능으로 쓰면 안 된다. 성능은 e03/e04 의 CV 결과(primary accuracy 0.712±0.046, macro-F1 0.682±0.044)만 쓴다.**
- 보존: `splits.csv`(`5899237284b70c31…`), `cv_predictions.csv`(`1e397d7fc8feb8a1…`), `features.npz`(`e059374afaf6671c…`)는 e11 이 읽기만 했고 실행 전후 sha256 이 같다.
- 새 이미지에 쓰려면 e01 과 같은 방식으로 타일 특징(cls)을 뽑아 이미지별로 평균(mean)한 768차원 벡터를 만들어 `final_model.joblib` 의 `scaler`, `clf` 에 넣는다 (`obj["clf"].predict_proba(obj["scaler"].transform(X))`). 타일 선택(전처리 `s04`)과 특징 추출 설정이 같아야 한다.

## 9. 검증

- 분할(e00): 각 repeat 에서 모든 이미지가 정확히 한 번 test, train/test 의 이미지와 타일이 서로소 (assert). fold 구성(AGG0/AGG1/AGG2)은 5/3/4, 6/2/4, 6/3/3 이고 AGG1 은 fold 당 2~3장.
- 모델(e01): 가중치 `strict=True` 로드에서 누락/초과 키 0, 파라미터 86,580,480 (기대 86.6M±0.5), 공식 코드에서 `x_norm_clstoken`/`x_norm_patchtokens` 키와 정규화 값 대조, NaN/Inf 없음, 같은 입력 두 번 실행 최대 절대 차이 0.
- 모델 출처: 캐시된 공식 저장소, **커밋 불명, 2026-10-09 다운로드**. 저장소 파일 203개와 가중치(`dinov2_vitb14_pretrain.pth`, sha256 `0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73`)의 sha256 목록(`repo_files_sha256.csv`)으로 검증하고, 다음 실행부터 달라지면 멈춘다.
- 기준선(e02): 손 계산 기대값(54/60, macro-F1 0.893, 불일치 6장, 동률 없음)과 모두 일치.
- 누수 방지(e03): fit_predict 는 train 행만 받아 표준화와 분류기를 학습하고, test 는 transform/predict 에만 쓴다. scaler 가 train 통계와 같은지 assert 450회 통과, 모든 fold 에서 train/test 이미지(SL 은 타일)가 서로소. kNN 은 표준화/정규화 통계를 쓰지 않는다.
- 재현성: e00, e03 을 다시 실행해 `splits.csv`(`5899237284b70c31…`), `cv_predictions.csv`(`1e397d7fc8feb8a1…`)의 sha256 이 같음. e01 은 같은 입력 두 번 실행 결과가 동일.
- 순열 검사(e05): 20회 평균 macro-F1 0.323, 최대 0.400 (accuracy 평균 0.358, 최대 0.433) → 제안한 기준 통과.

## 10. 명세와 달라진 점, 진행 중 고친 문제

- 실행 환경이 Windows 가 아니라 WSL2 이다 (3절). Windows 에서는 torch 가 정책에 막혀 e01 이 못 돈다.
- `max_iter=1000` 은 명세에 없는 수렴용 값이다 (수렴 경고 0건).
- `xFormers is not available` 경고는 xFormers 라이브러리가 없다는 뜻이며 기본 PyTorch 연산으로 정상 동작한다.
- 이미지 전체 크롭(whole_image)은 마스크 bbox 를 224x224 로 resize 한다. 종횡비 왜곡은 평균 1.056, 최대 1.233이고, 스케일바 제외 구역과 겹치는 이미지는 2/60장(수정하지 않고 기록만).
- 고친 버그: (1) e00 의 타일 겹침 검사가 `tile_id` 를 이미지 간에 구분하지 않아 오판 → `(image_id, tile_id)` 쌍으로 수정(분할 자체는 문제없었음). (2) e04, e05 의 금지어 검사가 실패 메시지에 금지어를 찍어 스스로 걸리는 순환 오류 → 파일명:줄 번호만 기록하도록 수정하고 오염된 로그를 재생성.

## 11. 하지 않은 것

- attention 합치기, DINOv2 fine-tuning, pseudo-labeling, 7.5x/12.5x 배율.
- 처리 조건(또는 처리 방법) 단위 분할, 금속 종류 단위 분할 평가. 이 평가는 이미지 단위이므로 새 처리 조건에 대한 일반화는 측정하지 않았다.
- 통계 검정, primary 와 하이퍼파라미터의 사후 변경. 2단계 방향은 사용자가 정한다.

## 12. 한계

이 결과를 쓸 때 아래를 함께 봐야 한다 (수치는 이 문서의 8절과 e04/e06 보고서에서 가져온 것).

1. **AGG1 은 이미지가 14장뿐이다 (recall 0.40).** 60장 중 23% 이고 fold 당 2~3장이다. primary 의 AGG1 recall 은 0.400 (풀링 56/140)이고, AGG1 의 51%(72/140)가 AGG0 으로 예측된다. 16개 설정의 AGG1 recall 은 0.121 ~ 0.436 이다. AGG1 이미지 1장이 AGG1 recall 에 주는 영향이 7%p 라서 AGG1 관련 수치는 불안정하다.
2. **처리 조건과 라벨이 강하게 얽혀 있다.** 처리 조건의 다수 라벨로 라벨을 맞히면 54/60 = 90% 이고(처리 조건 암기 accuracy 0.900, macro-F1 0.893), 16개 설정 중 어떤 것도 이에 못 미쳤다(최고 0.805). 이 평가는 이미지 단위 분할이라 test 이미지와 같은 처리 조건의 이미지가 항상 train 에 있으므로, **새 처리 조건에 대한 일반화는 측정하지 않았다.** 처리 조건과 라벨이 다른 6장은 4장이 BASE-LCSI 라서, 이 6장의 점수를 AGG 형태를 본 증거로 읽으면 안 된다 (금속 종류를 알아봤을 가능성을 구분할 수 없다). 오분류는 온도 1050 과 라벨이 섞인 처리 조건에 몰려 있다.
3. **타일 단위 AGG 검출은 아직 없다.** 라벨은 이미지 단위뿐이고, primary 는 타일 특징을 이미지별로 평균한 뒤 이미지 단위로 분류한다. 어느 타일에 AGG 가 있는지를 알려주는 정답도, 그것을 평가하는 단계도 없다.
4. **SL 의 타일 확률은 검증되지 않았다.** SL 은 이미지 라벨을 그 이미지의 모든 타일에 그대로 붙여 학습한다 (일부 타일에만 AGG 가 있어도 모든 타일을 같은 라벨로 취급). 평가는 타일 확률을 이미지별로 평균한 이미지 단위 예측만 했고, 타일 확률 자체의 정확도나 보정은 확인하지 않았다. SL 이 accuracy 0.805 로 가장 높았다는 것을 타일 확률이 믿을 만하다는 근거로 쓰면 안 된다.
5. **whole_image 는 해석에 주의한다.** 마스크 bbox 를 224x224 로 resize 하므로 종횡비가 왜곡되고(평균 1.056, 최대 1.233) 이미지 전체를 224 로 축소해 타일(원본 해상도)과 입력 해상도가 다르다. 마스크 bbox 가 스케일바 제외 구역과 겹치는 이미지가 2/60장이다. 처리 조건과 라벨이 다른 6장에서 k/6 가 가장 높지만(5.5/6) 나머지 54장 accuracy 는 낮은 편이다(logreg 0.685, knn 0.676). 6장의 점수가 높다는 이유로 AGG 형태를 더 잘 본다고 해석하면 안 된다.
6. 그 밖에: 이미지 60장이라 CI 가 넓고(이미지 단위 accuracy [0.615, 0.802]), primary 와 ablation 의 차이는 반복 간 SD 안이다. 최종 모델(e11)의 성능 추정은 CV 결과뿐이며, 이는 **같은 처리 조건의 새 금속 이미지**를 가정한 값이다.
