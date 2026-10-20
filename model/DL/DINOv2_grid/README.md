# 모델·실험, 겹치지 않는 격자 타일 버전 (고정 DINOv2 특징 → 이미지 단위 합치기 → 분류기)

`model/DL/DINOv2` 와 같은 형식(e00~e11 전체 파이프라인)을 **새 마스크(`ML_preproc` m01)와 새 타일 집합**으로 다시 돌린 실험이다. 10x 이미지 60장(AGG0 28 / AGG1 14 / AGG2 18)에서 고정 DINOv2 ViT-B/14 특징을 뽑고, 이미지 단위로 합쳐 AGG0/1/2 를 분류한다.
배경 증강은 없다 (원본 이미지만 쓴다). 배경 증강 실험은 `data/preproc_2` 에 있다.

- 범위: 특징 추출 → 합치기 → 분류기 → 이미지 단위 층화 반복 K-fold 평가. **모델은 학습하지 않았다** (DINOv2 고정, 분류기는 로지스틱 회귀/kNN). attention 합치기, fine-tuning, 배경 증강은 하지 않았다.
- 원본 이미지, `ML_preproc` 출력, 기존 `exp`(`model/DL/DINOv2`)의 분할 파일은 읽기만 했다.
- 모든 실행은 단계 이름이 붙은 파이썬 스크립트이고, 파라미터는 `config.yaml` 한 곳에 있다.

## 1. 한눈에 보기

| 항목 | 값 |
|---|---|
| 입력 | 10x 이미지 60장, 분석 타일 **2,591개**: 224px, 서로 겹치지 않음, 시편 픽셀 0인 타일 제외 (이미지당 36~48개, 클래스별 AGG0 1,216 / AGG1 590 / AGG2 785) |
| 시편 픽셀 커버리지 | 시편 픽셀의 **99.74%** 가 타일에 들어간다 (이전 실험의 타일 1,514개는 72.4%). 마스크 100% 타일 1,309개, 배경이 섞인 타일 1,282개 |
| 평가 | 이미지 단위 층화 5-fold x 반복 10회 (seed 0). 분할은 `model/DL/DINOv2` 와 sha256 이 같다 (같은 이미지, 같은 fold) |
| primary (결과를 보기 전에 고정, 이전과 같음) | 특징 cls / 합치기 mean / 분류기 logreg → **accuracy 0.813±0.024, macro-F1 0.790±0.028** (이미지 단위 95% CI [0.728, 0.892], 처리 조건 단위 [0.697, 0.913]) |
| 기준선 | 다수 클래스 0.467 / 0.212, **처리 조건 암기 0.900 / 0.893** |
| 16개 설정의 accuracy 범위 | 0.688 ~ 0.813 (최고는 primary) |
| 순열 검사 | primary 20회 평균 accuracy 0.365, macro-F1 0.326 (최대 0.469), 다른 설정 3개도 모두 기준 통과 |
| 실행 환경 | WSL2 (Ubuntu) + RTX 3060, `model/DL/DINOv2` 와 같은 가상환경 |

전체 보고서는 `outputs/e04_report/report.md`.

## 2. `model/DL/DINOv2` 와 다른 점

| 항목 | `model/DL/DINOv2` | 이 폴더 |
|---|---|---|
| 타일 | `preproc` s04: 마스크를 32px 침식한 안쪽에 완전히 들어가는 타일만. 겹치지 않는 부분 격자 중 그런 타일이 가장 많은 것을 골랐고(대부분 (1,1) 부분 격자, 즉 이미지 모서리에서 112px 안쪽으로 밀린 격자), 1,514개 | `ML_preproc` m02 격자(224px, stride 112)에서 `g00` 이 고른 2,591개: 침식 없음, 시편 픽셀이 하나라도 있는 타일, 시편 픽셀을 가장 많이 덮는 부분 격자 (60장 모두 (0,0), 즉 이미지 모서리에서 시작) |
| 이전 타일과의 관계 | | 이전 1,514개 중 같은 좌표가 이번 선택에도 있는 것은 **55개뿐**. 격자 위치(오프셋)가 달라서 안쪽 타일도 대부분 다른 자리를 자른 것이다 (이번 2,591개 중 마스크 100% 1,309개) |
| 시편 마스크 | `preproc` s02 (otsu_v) | `ML_preproc` m01 (밝기·색상·근접 감점 점수). 두 마스크는 거의 같다 (IoU 중앙값 0.998). 이 실험에서 마스크는 타일을 고르는 데와 whole_image 기준선(마스크 bbox 크롭)에만 쓰이고, 타일 픽셀 자체는 원본을 그대로 자른 것이다 |
| 타일 목록 만들기 | 없음 (`preproc` 출력을 바로 읽음) | **`g00_make_tile_list.py` 추가** → `outputs/g00_tile_list/tiles.csv` |
| 스크립트 | e00~e07, e11, common.py | **같은 파일**. 바뀐 줄은 e00 1줄, e01 3줄, e04 1줄뿐이다 (설명과 로그에 적힌 옛 입력 설명 `1,514개`, `s04`, `s02` 를 config 값으로 바꾼 것). 계산 코드는 같다 |
| `config.yaml` | 기존 | 입력 경로(`tiles_csv`, `masks_dir`), `expected_tiles` 2591, `g00` 항목 추가 |

## 3. 용어

- 1단계 폴더(BASE, BASE-LC, BASE-LCSI, BASE-LSi, REF, REF-LN) = **금속 종류** (6종)
- 2단계 폴더(NPH1035, PH1180-1050 ...) = **처리 조건** = 처리 방법(NPH, PH1180, PH1205, PH1230, PH1255, 5종) x 온도(1035, 1050, 2종) = 10종
- 이미지 10x 는 (금속 종류 x 처리 조건)마다 1장, 모두 60장.
- `config.yaml` 의 `forbidden_terms` 에 있는 말은 스크립트, 로그, 보고서에서 쓰지 않는다 (e04, e05 가 검사).

## 4. 실행 환경과 방법

환경은 `model/DL/DINOv2` 와 같다 (`setup_env_wsl.sh` 로 만든 WSL 가상환경 `~/agg_exp_venv`, torch + CUDA, scikit-learn; 이 PC 는 Windows 애플리케이션 제어 정책이 torch 를 막아 WSL 에서만 돌린다). 설치 버전은 `requirements_freeze_wsl.txt`.

선행 조건 (읽기 전용 입력, `config.yaml` 의 경로를 이 저장소 배치에 맞춰야 한다):
- `ML_preproc` 출력: m01 마스크 `masks/*.png`, m02 `tiles.csv` (sha256 앞 12자리 `eb9899d3d072`)
- `preproc` 출력: `s00b_meta_mag/manifest_v1.csv`, 10x 원본 이미지
- 기존 실험: `config.yaml` 의 `g00.reference_selected_ids_sha256` 은 배경 증강 실험(`data/preproc_2`)이 같은 규칙으로 고른 목록의 해시이다 (같은 선택인지 대조용)
- DINOv2 공식 저장소와 가중치 (`~/.cache/torch/hub`)

```
# Windows (전처리용 가상환경의 python, numpy + cv2 + PyYAML 만 필요)
python g00_make_tile_list.py

# WSL (이 폴더에서): e00 ~ e11 을 순서대로 실행하고 한 단계라도 실패하면 멈춘다
bash run_pipeline_wsl.sh
# 단계별로: bash run_wsl.sh e00_make_splits.py, e01_extract_features.py, ... , e11_train_final.py, e11_train_final.py --verify
```

소요 (이번 실행): e01 약 1분, e03 약 30분 (SL 의 타일 단위 logreg 가 대부분), e07 약 48분, 나머지는 각 몇 초.

## 5. 폴더 구조

```
exp_grid/
├── config.yaml                  # 모든 파라미터 (하드코딩 금지)
├── common.py                    # config, 로그, 입력 로딩, 조건 파싱, 지표, 부트스트랩, 금지어 검사
├── g00_make_tile_list.py        # 격자 9,900개 -> 분석 타일 2,591개 (이 폴더에서 추가)
├── e00_make_splits.py ... e05_sanity_permutation.py   # 실험 6단계
├── e06_extra_checks.py, e07_permutation_other_settings.py, e11_train_final.py   # 추가 확인, 순열 검사, 최종 모델
├── run_wsl.sh, run_pipeline_wsl.sh, requirements_freeze_wsl.txt
└── outputs/<스크립트명>/         # 단계별 출력과 log.txt (g00, e00 ~ e07, e11_final_model)
```

## 6. 단계별 스크립트

| 단계 | 하는 일 | 주요 출력 |
|---|---|---|
| `g00_make_tile_list` | m02 격자 9,900개에서 겹치지 않는 stride 224 부분 격자 4개 중 시편 픽셀을 가장 많이 덮는 것을 이미지마다 하나 고르고(동률이면 앞 번호) 시편 픽셀 0인 타일을 뺀다. 선택 타일끼리 겹치지 않음(모든 쌍), 선택 목록 해시가 기준과 같음을 검사 | `tiles.csv`, `tile_selection.csv`, `selected_tile_ids.csv` |
| `e00_make_splits` | 이미지 단위 층화 5-fold x 10 반복 분할 (seed 0), 한 이미지의 타일은 항상 같은 쪽 | `splits.csv`, `conditions.csv` |
| `e01_extract_features` | 타일 224x224 를 원본에서 잘라 고정 DINOv2 ViT-B/14 로 cls(768)와 patch 평균(768) 추출, 이미지 전체(마스크 bbox 를 224 로 resize) cls 도 추출. 모델 출처, 정규화, 결정성 검사 | `features.npz`, `tile_index.csv`, `meta.json` |
| `e02_reference_baselines` | 다수 클래스, 처리 조건 다수 라벨 암기 기준선 | `baseline_metrics.csv` |
| `e03_run_cv` | 16개 설정(타일 12 + SL 2 + whole_image 2) x 분할 전체 | `cv_predictions.csv`, `cv_summary.csv` |
| `e04_report` | 결과표, 부트스트랩 95% CI(이미지/처리 조건 단위), 그룹별 오분류, 보고서 | `report.md` |
| `e05_sanity_permutation` | primary 의 이미지 단위 라벨 순열 20회 (제안 기준: 평균 macro-F1 0.33±0.10, 어느 회차도 0.55 이하) | `permutation_runs.csv` |
| `e06_extra_checks` | 혼동행렬, 16개 설정 표, 금속 종류만으로 예측한 기준선 등 (새로 학습하지 않음) | `extra_report.md` |
| `e07_permutation_other_settings` | e05 와 같은 순열 검사를 whole_image(logreg, knn)와 SL(cls+patchmean)에도 | `permutation_other_runs.csv` |
| `e11_train_final` | primary 로 60장 전체를 학습한 최종 모델 저장, `--verify` 로 저장본 재예측이 같은지 확인 | `final_model.joblib`, `final_model_params.npz` |

설정(고정): logreg = 표준화(train 에서만 fit) + L2, C=1.0, balanced, max_iter=1000. kNN = k=5, cosine, 거리 가중, 표준화 없음. mean_std 의 표준편차는 ddof=0. SL = 타일 단위 logreg 후 이미지 안 타일 확률 평균.

## 7. 결과

accuracy (반복 10회 평균±SD). 이전 실험은 `model/DL/DINOv2` (타일 1,514개).

| 설정 | 이전 (1,514개) | 이번 (2,591개) |
|---|---|---|
| **cls · mean · logreg (primary)** | 0.712 | **0.813** |
| cls · mean_std · logreg | 0.728 | 0.808 |
| cls+patchmean · mean_std · logreg | 0.745 | 0.810 |
| cls · mean · kNN | 0.702 | 0.758 |
| SL · cls · logreg | 0.790 | 0.795 |
| SL · cls+patchmean · logreg | 0.805 | 0.775 |
| whole_image · cls · logreg | 0.708 | 0.710 |

- 타일 합치기 계열 12개 설정은 모두 올랐다(+0.03 ~ +0.10). SL 과 whole_image 는 거의 같거나(±0.005) 내렸다(SL cls+patchmean −0.030). 12개 중 어떤 설정이 가장 많이 올랐는지는 `outputs/e04_report/report.md` 와 `outputs/e06_extra_checks/extra_report.md` 의 16개 설정 표.
- primary 클래스별 재현율: AGG0 0.886 / AGG1 0.600 / AGG2 0.867. 혼동행렬(10회 합산): AGG1 의 32.9%가 AGG0, 7.1%가 AGG2 로 예측된다.
- 오분류가 많은 처리 조건: PH1180-1035 (0.567), PH1205-1050 (0.450). 금속 종류별로는 BASE-LSi 0.32, REF-LN 0.27, REF 0.24 가 높고 BASE-LCSI 0.03 이 낮다.
- 같은 타일, 같은 분할, 같은 설정으로 돌린 배경 증강 실험(`data/preproc_2`)의 증강 없는 팔(R0)과 primary 값이 같다 (0.813). 두 실험이 같은 이미지 벡터를 만들었다: 같은 타일 2,591개의 특징이 두 실행에서 완전히 같음 (최대 차이 0).
- 최종 모델은 학습 60장에서 accuracy 1.000 이지만 학습 데이터 값이라 성능이 아니다. 저장본을 다시 불러도 예측이 학습 시점과 완전히 같다 (`--verify`). 성능은 e03/e04 의 CV 결과만 쓴다.

## 8. 한계

1. **점수 상승의 원인을 모른다.** 이전 실험(0.712)과 이번(0.813)의 차이는 타일 집합이 바뀐 영향이다. 바뀐 것은 (a) 침식 마진을 없애고 시편 픽셀이 있는 타일까지 포함(배경이 섞인 타일 1,282개), (b) 격자 위치가 달라져 안쪽 타일도 대부분 다른 자리를 자름 (이전 타일과 같은 좌표는 55개뿐), (c) 이미지당 타일 수 증가 (20~30개 → 36~48개)이다. 마스크가 달라진 영향은 작을 것으로 보지만(두 마스크 IoU 중앙값 0.998) 확인하지 않았다. 이 요인들이 각각 얼마나 기여했는지는 이 실험으로 가르지 않았다.
2. **처리 조건 혼동**: 라벨의 90% 정도가 처리 조건으로 설명된다 (처리 조건 암기 0.900, primary 점수보다 높다). 같은 처리 조건의 이미지가 train 과 test 에 모두 있어서 점수가 형태를 읽은 결과인지 처리 조건을 알아본 결과인지 구분되지 않는다. 처리 조건 단위로 나누는 평가는 하지 않았다.
3. **배경 정보**: 타일의 49%(1,282개)에 배경이 섞여 있다. 시편 크기, 가장자리 모양이 처리 조건과 얽혀 있으면 점수에 기여했을 수 있다. 이 실험은 그것을 가르지 않는다 (배경 증강 실험은 `data/preproc_2`, 그 결과로는 배경 의존이 단정되지 않았다).
4. **모델은 학습하지 않았다**: 모델 자체를 개선했다고 말할 수 없다.
5. 표본이 이미지 60장이라 CI 가 넓다. 이미지 단위 CI 는 이미지가 독립이라는 가정이라 같은 처리 조건 안의 상관 때문에 실제보다 좁다 (처리 조건 단위 CI 가 더 보수적이다). 반복 10회는 독립이 아니다.
6. 격자가 이미지 오른쪽 32px, 아래 24px 를 덮지 않아 시편 픽셀의 0.26%는 어떤 타일에도 들어가지 않는다. `ML_preproc` 마스크가 시편 일부를 놓친 이미지(예: BASE-LCSI/NPH1035 오른쪽 위)는 그 부분의 시편 픽셀이 배경으로 취급된다.
