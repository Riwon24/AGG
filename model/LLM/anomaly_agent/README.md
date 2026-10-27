# AnomalyAgent로 AGG 판별 시도 (Claude Sonnet 5.5 + 로컬 임베딩)

교수님이 추천한 [AnomalyAgent](https://github.com/mala-lab/AnomalyAgent)(저자 참고 구현, 커밋 `6db47f3`, MIT)를 우리 AGG 이미지에 돌려 본 실험이다. 에이전트의 판단 모델은 Claude API(`anthropic/claude-sonnet-5-5`)를 썼다. **결론: 이 설정에서는 AGG0와 AGG1+2(또는 AGG2)를 구분하지 못했다.** 이미지 단위 AUROC 0.47, 표시한 AGG 타일에서도 이상 판정은 17개 중 2~3개뿐이다. 다만 에이전트가 확대한 영역은 사용자가 표시한 AGG 위치와 우연보다 훨씬 많이 겹쳤다(아래 5절). 즉 AGG 자리를 보고도 "자연스러운 무늬"로 해석해 정상으로 판정했다.

> **저자 코드가 아닌 것**: 저자 저장소에는 `tools.py`와 `prompts.py`가 없다. `prompts.py`는 저자의 `prompts_v1_27.py`를 그대로 복사해서 썼고(내용 동일, 이 폴더에 올리지 않음), **`src_changes/tools.py`는 논문 설명(Eq. 1~6, 부록 A.1)을 보고 우리가 재구성한 구현이다.** 저자 코드가 아니므로 논문의 수치와 직접 비교할 수 없다.
> 이미지 파일, 사진에서 나온 타일 크롭, 사진 위에 그린 그림, 모델 가중치는 올리지 않았다. 문서와 스크립트 안의 데이터 경로는 원본 작업 폴더(`AGG_labeled_only`) 기준이다.

## 1. 한눈에 보기

| 실행 | 단위 / 샘플 | 후보 | planner | 판정 정확도 | AUROC 최종 점수 | AUROC margin |
|---|---|---|---|---|---|---|
| `agg_tile_smoke` | 타일 6 | 2+2 | **오류** | - | 0.500 (전부 정상, 점수 동점) | - |
| `agg_img_cand7` | 이미지 29 | 7+7 | **오류** | 0.379 | 0.394 | 0.414 |
| `agg_img_cand2` | 이미지 29 | 2+2 | **오류** | 0.448 | 0.467 | 0.424 |
| **`agg_img_cand7_v2`** | **이미지 29** | **7+7** | 정상 | **0.448** | **0.467** | **0.535** |
| `agg_tile34` | 타일 34 | 7+7 | 정상 | - | 0.529 | 0.664 (※) |
| `agg_tile34_v2` | 타일 34 | 중립 4+4 + 새 정의 | 정상 | - | 0.500 | 0.675 (※) |

- 이미지 29장 = AGG0 11장 + AGG2 18장. 후보 문장을 만든 AGG1 이미지 31장은 평가에서 뺐다(누수 방지).
- **※ 타일 34개(AGG1 표시 17 / AGG0 대조 17)는 후보 문장과 정의를 만든 바로 그 타일이라 margin 점수가 부풀려져 있다. 해석하지 말 것.**
- 최종 점수(`pred_score`)는 판정이 정상이면 0, 이상이면 1이라서 AUROC가 사실상 정확도 쪽이다. 연속 점수는 margin과 일반 템플릿 점수다(`template_log`).
- 같은 설정으로 다시 돌려도 판정의 약 14%가 바뀐다(세 실행 사이 일치율 0.86). 이미지 29장에서 AUROC 오차는 ±0.11 정도라 0.47과 0.55 같은 차이는 구분되지 않는다.

## 2. 설정

- 에이전트 그래프(planner, 도구, reasoner, reflector)와 템플릿 도구(일반 템플릿, 반사실 후보)는 저자 구조 그대로다. 판단용 LLM(planner, reflector, reasoner, 이미지 설명, 후보 생성)은 모두 Claude Sonnet 5.5 API. **임베딩은 항상 로컬 Qwen3-4B 텍스트 모델**(`models/qwen3_4b`)이다(API만으로는 돌릴 수 없음).
- 반사실 margin (식 5): 캡션 `c` 하나에 대해 `margin(c) = max_{a ∈ 이상 후보} cos(c, a) − max_{n ∈ 정상 후보} cos(c, n)`. 양수면 이상 후보 쪽에 가깝다. 이미지(타일)의 값은 캡션 3개 평균(`avg_margin`). 처음 만든 구현은 max 대신 logsumexp를 썼는데 논문 식이 max라서 고쳤다.
- 일반 템플릿 점수는 캡션과 정상/이상 프로토타입(WinCLIP 프롬프트 앙상블)의 `sigmoid(margin/τ)`, τ=0.3. 이 τ는 논문에서 확인하지 못한 값이고, WinCLIP 문장 목록도 재구성 쪽이 16개 템플릿이라 anomalib의 21개와 다를 수 있다(수정 안 함). 하드 라벨(0.5 기준)은 로그용이고 폴백일 때만 판정에 쓰인다.
- 캡션은 에이전트가 사진을 보고 만드는 3개의 짧은 관점 설명이고, 후보 문장 파일 형식은 `{"클래스": {"anomaly_candidates": [...], "normal_candidates": [...]}}`.
- 평가 스크립트는 저자의 `evaluate_headCT_v12_target.py`에서 `scripts/aa_make_eval_script.py`로 생성하고(`--unit image|tile`), `aa_patch_eval_log.py`로 사진마다 `template_log`(margin, 일반 템플릿 점수, 캡션)를 `results.json`에 남기게 했다. 생성된 파일 자체는 올리지 않았다.

## 3. 후보 문장과 이상 정의

| 파일 (`captions/tiles/`) | 만든 방법 |
|---|---|
| `atomic_candidates_agg_2x2.json` (2+2) | 타일 캡션에서 AGG1 표시 타일 쪽에 더 많이 나온 표현(차이 ≥ 3타일)만. 근거 개수는 `candidate_evidence.csv` |
| `atomic_candidates_agg.json` (7+7) | 위 기준을 차이 ≥ 2로 완화 + 근거 후보의 바꿔쓰기 8개(직접 작성, 새 근거 아님) |
| `atomic_candidates_agg_neutral.json` (4+4) | 밝기 방향에 중립인 문장을 직접 작성(5절 측정 결과 기준), "옅은 얼룩은 정상" 문장 제거. 이상 정의는 `target_anomaly_definition_neutral.txt` |

처음 후보가 "darker mottled patches" 쪽으로 쏠린 것은 캡션에서 가장 많이 나온 표현(AGG1 표시 14/17 대 AGG0 대조 3/17)을 따랐기 때문이다. 이 집계는 제 프롬프트로 만든 캡션 34개에 대한 정규식 개수이고, 표본이 작아 교란이 있을 수 있다.

## 4. planner 오류 (수정 전후)

처음 세 번의 실행(`before_planner_fix/`)에서 planner의 API 호출이 매번 `ValueError: Received multiple non-consecutive system messages`로 실패했다. planner 메시지가 `[system, 사용자 입력, system(템플릿 분석), user]` 순서인데 Anthropic API는 system 메시지가 한 덩어리로 맨 앞에 있어야 하기 때문이다. `aa_patch_system_merge.py`가 `safe_api_invoke`에서 system 메시지를 합치도록 고쳤고, 이후 실행(`valid/`)에서는 planner가 정상적으로 도구를 골랐다.
- 결과 파일로 확인한 것: 수정 전 세 실행은 planner의 도구 호출이 있는 샘플이 0개(6 / 29 / 29), 수정 뒤 네 실행은 전부 있다(2 / 29 / 34 / 34).
- margin과 일반 템플릿 점수는 planner 이전 단계에서 계산되므로 영향이 없다. 최종 판정은 영향을 받았다.
- 수정 뒤 이미지 29장 AUROC는 0.394에서 0.467로 올랐지만, 위 1절의 반복 변동 안이다. 점수가 나빴던 주된 이유가 planner는 아니었다.

## 5. 진단

**(a) 에이전트가 확대한 곳과 AGG의 위치.** 사용자가 AGG1 표시 타일 17개 중 15개에서 AGG 영역을 직접 표시했다(`labels/agg_regions.json`, 타일 크기 대비 비율 좌표, 7번·16번은 위치 미표시). 에이전트의 `image_zooming` 호출(중심, 배율)로 확대 창을 복원해 겹쳤다(`aa_zoom_overlay.py`, 그림은 사진이 들어 있어 올리지 않음).
- AGG 영역 중 확대 창 하나가 가장 많이 덮은 비율의 평균은 **49.9%** (같은 창을 다른 타일의 AGG에 대 본 기준값 16.2%). 확대 중심이 AGG 안에 있는 타일은 **6/15** (기준값 약 1.3/15). 타일별 값은 `labels/zoom_overlap.csv`.
- 한계: "확대하라고 한 곳"이지 모델의 실제 주의(attention)가 아니다.

**(b) 판정이 정상으로 가는 이유.** 확대가 AGG를 잘 잡은 타일(2, 4, 6, 9, 11, 14, 15번)의 이유 문장을 읽어 보면(`aa_read_reasons.py`), 에이전트는 AGG 자리의 "옅은 부드러운 경계의 어두운 얼룩"을 찾아 쓰고도 "낮은 대비, 자연스러운 음영이나 질감 변화"라며 정상으로 판정한다. 이상 정의를 "옅고 부드러운 밝기 차이도 AGG"로 바꿔도(`v2`) 판정은 거의 안 바뀌었다(표시 타일 이상 판정 2/17, 대조 타일 2/17, 바뀐 타일 3개). 원인으로 보이는 것(코드 확인, 실험으로 분리하지는 않음):
- `prompts.py`(저자 `prompts_v1_27.py`와 동일)의 규칙: 정상 샘플도 조명·질감 변화를 가질 수 있고, 구체적인 규칙 위반이나 눈에 보이는 구조적 결함이 아니면 이상으로 분류하지 말라는 내용.
- `agent_v1_16.py`의 종료 규칙: reasoner가 `normal`을 `normal_tolerance`(=2)번 내면 바로 종료.
- 최종 단계에서 `anomalous`/`normal`이 아니면 `normal`로 둔다.

**(c) 표시한 AGG 15곳의 밝기 방향과 질감** (`labels/agg_region_polarity.csv`, 상자와 그 주변 띠의 평균 밝기 비교, ±3 기준): 어두움 9 (3, 4, 5, 9, 10, 11, 14, 15, 17번), 밝음 3 (2, 6, 13번), 비슷 3 (1, 8, 12번). 타일 전체 평균과 비교하면 밝음 8 / 어두움 3으로 반대가 나왔는데, 조명 기울기 때문이어서 주변 띠 비교가 더 믿을 만하다. 상자 안 고주파 질감은 바깥보다 매끈하지 않았다(평균 1.07, 1 미만 3/15). 즉 "어두운 얼룩만 AGG"도 "매끈한 영역이 AGG"도 이 표시로는 지지되지 않는다. 상자는 손으로 그린 것이라 경계가 거칠고 표본은 15개다.

## 6. 한계

- 모든 실행이 1회 실행이다. 이미지 29장은 AGG0 11 / AGG2 18이라 AUROC 오차가 크다.
- 후보 문장, 이상 정의, 타일 평가는 같은 AGG1 타일에서 만들었다(타일 34개 지표는 부풀려짐). 이미지 29장은 이 이미지들과 겹치지 않지만 AGG2가 AGG1과 비슷하게 생겼다면 완전한 독립 검증은 아니다.
- 재구성 `tools.py`는 저자 코드가 아니다. τ=0.3, WinCLIP 템플릿 목록(16 대 21)은 확인하지 못했다.
- `agent_v1_16.py`에 쓰는 프롬프트는 `prompts_v1_27.py` 복사본이라, 저자가 v1.16에서 쓴 프롬프트와 같은지는 확인하지 못했다.
- API 비용이 드는 실험이라 정의, 후보, 프롬프트를 바꿔 가며 여러 번 돌려 보지는 않았다. 프롬프트의 "구조적 결함" 규칙과 종료 규칙을 AGG에 맞게 고쳐서 다시 돌리는 것은 하지 않았다.

## 7. 파일

| 경로 | 내용 |
|---|---|
| `scripts/aa_caption_phrases.py`, `aa_caption_tiles.py` | 전체 이미지·타일 캡션을 Claude로 만들고 반복 표현을 센다 (타일은 34개 × 관점 3개 = 102회 호출) |
| `scripts/aa_make_candidates.py`, `aa_make_neutral_setup.py` | 후보 문장 파일 생성 (근거 집계 / 중립 후보와 정의, 타일 재실행 스크립트) |
| `scripts/aa_make_eval_script.py`, `aa_patch_eval_log.py`, `aa_patch_system_merge.py`, `aa_patch_workspace_header.py` | 평가 스크립트 생성과 저자 코드 패치(점수 기록, system 메시지 병합, 워크스페이스 헤더) |
| `scripts/aa_api_check.py`, `run_*.sh`, `run_image_compare.ps1` | API 점검, 실행 명령 (키는 환경변수 `ANTHROPIC_API_KEY`, `ANTHROPIC_WORKSPACE_ID`) |
| `scripts/aa_compare_runs.py`, `aa_compare_tile_runs.py`, `aa_margin_rescore.py`, `aa_read_reasons.py` | 실행 비교, 같은 캡션으로 후보 세트를 바꿔 margin 재계산(API 없음), 이유 문장 읽기 |
| `scripts/aa_region_polarity.py`, `aa_zoom_overlay.py` | 표시한 AGG의 밝기·질감 측정, 확대 영역과 AGG 겹침 계산 |
| `src_changes/tools.py`, `tools_v1_27.py` | **우리가 재구성한** 도구 모듈과 재수출 래퍼 (저자 저장소의 `src/react_agent/`에 복사해서 사용) |
| `results/valid/`, `results/before_planner_fix/` | 실행별 `results.json`(`data_path`는 `<AGG_DATA_ROOT>`로 바꿈)과 사용한 후보 파일 |
| `captions/whole_image/`, `captions/tiles/` | 캡션 원문과 반복 표현표, 후보 문장 3종, 근거표, 이상 정의 |
| `labels/` | 사용자가 표시한 AGG 영역, 밝기·질감 측정, 확대 겹침 |

재현 순서(WSL): 저자 저장소를 `~/AnomalyAgent`에 받고 `src_changes/`를 복사한 뒤, 패치 스크립트 3개와 `aa_make_eval_script.py`를 실행하고, `aa_make_neutral_setup.py` 또는 `run_*.sh`로 평가한다. 올리지 않은 것: 이미지, 타일 크롭(`tiles/`), 확대 영역 그림, 라벨 표시용 HTML(타일이 들어 있음), 로그, 모델 가중치(`models/qwen3_4b`), `prompts.py`, 생성된 평가 스크립트.
