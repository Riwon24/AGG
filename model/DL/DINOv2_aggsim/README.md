# DINOv2_aggsim: 타일별 AGG 유사도 (aggsim)

`DINOv2_grid_augmentation` 이 만든 원본(k=0) 타일 cls 특징(고정 DINOv2 ViT-B/14, 768차원)으로 타일마다 "AGG 와 얼마나 닮았나"를 코사인 유사도(`aggsim`)로 계산하는 탐색 분석이다. 이미지 안에서 AGG 가 있는 위치(국소 부위)를 설명해 보려는 시도라서 `model/DL/MIL` (attention 으로 같은 질문을 본 것) 과 같은 성격이다. 모델은 학습하지 않았다 (DINOv2 고정, 유사도는 특징의 평균과의 코사인).

**타일 라벨은 그 타일이 속한 이미지의 라벨을 복사한 것이다.** 타일 안에 실제로 AGG 가 있는지는 라벨에 없고, 이 폴더의 두 가지 prototype 은 그 한계를 다른 방식으로 다룬다.

## 두 가지 prototype

| 버전 | prototype | 스크립트 | 비고 |
|---|---|---|---|
| 1 | 이미지 라벨이 AGG1 또는 AGG2 인 이미지의 train 타일 cls 평균 | `c03`, `c05` | (repeat, fold) 마다 train 타일만으로 만들고 test 타일의 값을 분석에 쓴다 |
| 2 | 사람이 직접 "AGG 가 있다"고 표시한 타일 17개의 cls 평균 | `c06` ~ `c09` | AGG1 이미지 14장에서 이미지당 1~2개 (3장은 2개). 목록은 `c08` 코드 안에 있다 |

## 결과

### 버전 1: AGG1+AGG2 이미지의 타일로 만든 prototype (`c03`, `c05`)

- 관찰: test 타일 aggsim 평균은 AGG0 0.778, AGG1 0.796, AGG2 0.812 (표준편차 약 0.07~0.08). 안쪽(마스크 100%, 시편만 있는 타일) 1,309개만 보면 AGG0 0.828, AGG1 0.844, AGG2 0.849 이고, 가장자리 타일은 클래스와 상관없이 안쪽보다 0.08~0.10 낮다 (AGG0 0.733, AGG1 0.751, AGG2 0.765).
- 안쪽 타일의 이미지 평균으로 본 클래스 분리 (이미지 단위 AUROC, 이미지 부트스트랩 95% CI, 0.5 = 구분 못 함):

  | 비교 | AUROC |
  |---|---|
  | AGG1+2 대 AGG0 | 0.686 [0.542, 0.814] |
  | AGG1 대 AGG0 | 0.640 [0.450, 0.811] |
  | AGG2 대 AGG0 | 0.722 [0.572, 0.865] |
  | AGG2 대 AGG1 | 0.571 [0.333, 0.781] |

- 해석 (가능한 설명이며 확정이 아님): 클래스 순서대로 평균이 오르지만 분포가 크게 겹친다. aggsim 의 공간 패턴은 AGG 정도보다 타일이 시편 안쪽인지 가장자리인지에 더 크게 좌우될 가능성이 있다. 이 점수는 "AGG1+2 이미지의 평균 타일과 닮은 정도"이지 AGG 영역을 잰 값이 아니다.

### 버전 2: 직접 표시한 AGG 타일 17개로 만든 prototype (`c06` ~ `c09`)

- 표시 방법: PPT 를 보고 AGG 가 있다고 판단한 타일을 `r01_c01` 형식의 번호로 적었다 (AGG1 이미지 14장, 17개). AGG2 이미지는 표시하지 않았고, **표시하지 않은 타일에 AGG 가 없는지는 확인되지 않았다.** 3건은 타일 경계 부근이라고 적었고, 이름을 적은 타일만 AGG 타일로 두었다. 이 표시는 도메인 전문가의 검증을 거치지 않았다.
- **순환성 주의**: 17개 타일은 prototype 자체를 만든 타일이라 자기 자신과 닮은 정도가 부풀려진다 (한 타일이 prototype 의 1/17). 그래서 부풀림을 제거한 두 변형을 함께 계산한다.
  - `loo`: 해당 AGG 타일 자신을 뺀 16개로 만든 prototype
  - `loio`: 해당 타일과 같은 이미지의 AGG 타일을 모두 뺀 prototype (같은 이미지 안의 닮은 정도가 섞이는 것을 막는다)
- 이미지 안 순위의 percentile (`1 - (rank - 1) / (n - 1)`, 1.0 = 이미지 안 최상위, 무작위 기대 평균 0.5, 상위 25% 이내 개수의 무작위 기대 약 4.3개):

  | prototype | percentile 평균 / 중앙값 | 이미지 안 상위 25% 이내 |
  |---|---|---|
  | 전체 17개 (부풀려짐) | 0.674 / 0.732 | 8 / 17 |
  | loo | 0.565 / 0.610 | 4 / 17 |
  | loio | 0.606 / 0.622 | 6 / 17 |

  안쪽 타일(마스크 100%)인 AGG 타일 9개를 안쪽 타일끼리만 비교하면 loo 평균 0.520 (상위 25% 이내 2/9), loio 평균 0.626 (3/9) 이다. 가장자리 타일인 AGG 타일 8개는 loo 평균 0.341 로 낮다.
- 해석 (가능한 설명이며 확정이 아님): 부풀림을 제거하면 AGG 타일이 같은 이미지의 다른 타일보다 prototype 과 닮았다는 증거가 보이지 않는다. 다만 17개라 표본이 작고 percentile 의 CI 는 계산하지 않았다. AGG 타일이 DINOv2 cls 특징으로 다른 타일과 구분되지 않을 정도로 비슷한 것일 수도 있고, 표시가 AGG 를 타일 단위로 정확히 가리키지 못했을 수도 있다 (경계에 걸친 경우). 이 실험으로 둘을 가를 수 없다.

## 스크립트 (실행 순서)

| 스크립트 | 하는 일 | 주요 출력 (`outputs/<스크립트명>/`) |
|---|---|---|
| `c03_tile_aggsim.py` | (repeat, fold) 마다 train 타일의 AGG1+2 prototype 을 만들어 모든 타일의 aggsim 계산 (누수 방지 assert 포함) | `tile_aggsim.csv` |
| `c04_aggsim_heatmap.py` `[라벨]` | 클래스별 이미지의 타일 격자에 aggsim 을 색으로 칠해 원본 위에 겹침 (repeat 0, fold 0, 색 범위 고정) | `fig_aggsim_overlay_rep0_fold0_AGG*.png` |
| `c05_aggsim_inner_distribution.py` | 안쪽 타일의 aggsim 클래스별 분포, 이미지 단위 AUROC | `stats_by_class.csv`, `auroc_pairs.csv` |
| `c06_aggsim_agg1_heatmaps.py` | 실제 라벨 AGG1 이미지 14장의 aggsim 히트맵과 수동 표시용 템플릿 | `NN_*.png`, `agg_tile_labels_template.csv` |
| `c07_agg1_tile_numbers.py` | AGG1 이미지 14장의 타일 번호 격자 (색 없음, 수동 표시용) | `NN_*.png` |
| `c08_agg1_manual_labels.py` | 직접 표시한 AGG 타일 17개(코드 안의 목록)를 기록하고 번호가 존재하는지 검증 | `agg_tile_labels_filled.csv`, `agg_tiles_list.csv` |
| `c09_manual_prototype_sim.py` | 17개 prototype 과 모든 타일의 유사도, 이미지 안 순위와 percentile (loo, loio 변형 포함) | `tile_sim_manual.csv` |

`c04`, `c06`, `c07` 의 그림은 올리지 않았다. 의존 순서는 `c03` → (`c04`, `c05`, `c06`), `c06` → `c08` → `c09` 이다.

## 입력과 실행

- 입력은 `DINOv2_grid_augmentation` 의 출력이다 (읽기 전용): `outputs/e01_extract_features/` (`features.npz`, `tile_index.csv`), `outputs/e00_make_splits/` (`splits.csv`, `conditions.csv`), `outputs/g00_tile_list/tiles.csv`. 스크립트 상단의 `GRID` 가 `../DINOv2_grid_augmentation/outputs` (저장소 구조) 또는 `../exp_grid_aug/outputs` (로컬 작업 폴더 구조) 중 있는 쪽을 쓴다. 이 출력들은 저장소에 올라와 있지 않으므로 `DINOv2_grid_augmentation` 의 파이프라인(`g00` → `e00` → `e01`) 을 먼저 돌려야 한다.
- `c03`, `c05`, `c08`, `c09` 는 numpy 만 필요하다. 그림을 그리는 `c04`, `c06`, `c07` 은 matplotlib, OpenCV 가 필요하고 **원본 이미지와 전처리 폴더의 `manifest_v1.csv` (이미지 이름과 파일 위치를 적은 표)** 를 읽는데 둘 다 저장소에 없다. 그래서 이 세 스크립트는 저장소만으로는 실행되지 않고, `c08` 은 `c06` 이 만든 템플릿을 읽으므로 `c06` 도 이미지가 있어야 한다.
- GPU 는 필요 없다. 출력(표, 그림)은 올리지 않았다.

## 한계

1. 타일 라벨이 이미지 라벨의 복사라서, 타일 단위 점수가 AGG 위치를 잰다고 말할 수 없다.
2. 수동 표시는 17개뿐이고 이미지당 1~2개다. 표시하지 않은 타일에 AGG 가 없는지, AGG2 이미지에 AGG 가 어디에 있는지는 알 수 없다. 표시는 한 사람의 판단이고 전문가 검증을 거치지 않았다.
3. 17개 prototype 은 순환성이 있어 loo/loio 로 보정했지만 percentile 의 신뢰구간은 계산하지 않았다.
4. 가장자리 타일은 클래스와 무관하게 aggsim 이 낮아, 가장자리에 걸친 AGG 타일의 낮은 순위가 AGG 때문인지 위치 때문인지 구분되지 않는다.
5. 이미지 60장, 처리 조건 10개의 작은 데이터이고 라벨이 처리 조건과 얽혀 있다. 위 결과는 모두 탐색적 관찰이다.
