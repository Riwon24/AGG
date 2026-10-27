# JEVision 타일 점수 시각화 (jv06, jv07, jv08)

`../jev_binary_choice`(jv06), `../jev_binary_choice_v2`(jv07), `../jev_binary_fewshot`(jv08)의 타일 점수를 이미지 단위로 보기 위한 스크립트와 그림이다.

| 파일 | 내용 |
|---|---|
| `jv_tile_scores_all_runs.py` | JEVision 6가지 실행(v1~v3, jv06~jv08)의 타일별 점수를 타일 하나당 한 줄로 합친 CSV(`tile_scores_jv06_jv08.csv`)를 만든다. 입력은 각 실행의 `tiles_grid_*.jsonl`이고, 이전 실행(v1~v3)은 타일 번호가 달라 `(image_id, x0, y0)`로 맞춘다. |
| `jv_tile_score_maps.py` | 타일별 점수를 원본 사진 위에 칠해서 이미지마다 한 장(60장)을 만든다. 색은 같은 이미지 안의 순위 백분위, 숫자는 실제 점수 p. **사진이 들어 있어서 그림은 올리지 않았다.** |
| `jv_score_boxplot.py`, `image_score_boxplot.png` | 이미지 점수(타일 p_flat의 이미지 평균, 60장)를 AGG0/AGG1/AGG2 별 박스플롯으로 그린다. 점 하나가 이미지 한 장이다. |

- **눈금 주의**: 실행마다 점수의 눈금이 달라서(jv06은 높은 쪽에 몰리고 jv08은 거의 0) y축을 실행별로 따로 뒀다. 실행 사이에 값을 직접 비교할 수 없다.
- 스크립트의 입출력 경로는 원본 작업 폴더(`AGG_labeled_only/exp_jev/outputs/`) 기준이다. 박스플롯은 `tile_scores_jv06_jv08.csv`가 있어야 다시 그릴 수 있고, 이 CSV는 올리지 않았다(`jv_tile_scores_all_runs.py`로 다시 만든다).
- AGG 신호 여부의 판단(AUROC, 정확도)은 각 실험 폴더의 README를 본다. 이 폴더는 점수 분포를 눈으로 보기 위한 것이다.
