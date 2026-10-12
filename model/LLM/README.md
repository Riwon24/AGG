# 모델별 LLM 테스트 현황 (AGG0/1/2 분류)

> **폴더 구조**: `jev/`는 로컬에서 돌린 JEVision(스크립트 `jev/scripts/`, 결과 `jev/results/`), `api/`는 API로 호출한 모델(kimi-k3, Gemini 등)의 스크립트(`api/scripts/`)와 결과(`api/results/`)다. 이미지 파일은 올리지 않았다. 문서 안의 `../preproc/`, `../exp/` 경로와 스크립트 안의 데이터 경로는 원본 작업 폴더(`AGG_labeled_only`) 기준이다.

작성 시점: 2026-10-12. 같은 과제(`AGG_labeled_only` 이미지 179장을 AGG0/1/2로 zero-shot 분류)를 여러 모델로 시도한 결과를 한곳에 정리한다. 상세한 JEVision 결과와 한계는 [`jev/README.md`](jev/README.md).

## 1. 한눈에 보기

| 모델 | 제공처 | 호출 | 전체 사진 179장 | accuracy / macro-F1 | 순위 지표 (AUROC AGG2>AGG0) | 비고 |
|---|---|---|---|---|---|---|
| JEVision 0.8B (프롬프트 v1) | 로컬(WSL2 GPU) | 가능 | 완료 | 0.464 / 0.269 | 0.551 | 거의 전부 AGG0 예측 |
| JEVision 0.8B (프롬프트 v2) | 로컬 | 가능 | 완료 | 0.257 / 0.177 | 0.587 | 거의 전부 AGG1 예측 |
| **JEVision 0.8B 타일 방식** | 로컬 | 가능 | (10x 60장, 타일 1,514개) | 0.733 / 0.689 (※) | **0.944** | ※임계값을 같은 데이터에서 골라 낙관적 상한 |
| **kimi-k3** | NVIDIA API | 가능 | **완료** | 0.447 / 0.241 | 0.531 | 170/179를 AGG0 예측 |
| **gemini-3.5-flash-lite** | Gemini API | 가능 | **완료** | 0.464 / 0.251 | 0.473 | 173/179를 AGG0 예측, 분당 15건 제한 |
| **gemini-3.1-flash-lite** | Gemini API | 가능 | **완료** | 0.458 / 0.237 | 0.464 | 174/179를 AGG0 예측, 분당 15건 제한 |
| gemini-3.6-flash | Gemini API | 가능 | **미완료** (무료 한도) | 평가 안 함 | - | 하루 20건 제한 |
| gemini-3.8-flash | Gemini API | 가능 | **미완료** (무료 한도) | 평가 안 함 | - | 하루 20건 제한 |
| llama-3.2-90b-vision-instruct | NVIDIA API | 가능하나 매우 느림 | 미실행 | 평가 안 함 | - | 1건에 26분 |
| nemotron-3-nano-omni-30b-a3b-reasoning | NVIDIA API | 불안정 | 미실행 | 평가 안 함 | - | 504 오류 |

기준선(항상 다수 클래스 AGG0 예측): accuracy 0.464 (179장), 0.467 (10x 60장).

**요약**: 전체 사진 한 장을 넣는 방식은 모델 크기(0.8B, Gemini flash-lite, 대형 reasoning 모델 kimi-k3)와 프롬프트에 관계없이 거의 전부 한 클래스로 답했고, 심각도 점수의 AUROC도 0.5 근처였다. 신호가 나온 것은 JEVision에 224px 타일을 넣어 이미지 단위로 평균한 방식뿐이다(단, 조명 띠·스크래치 같은 아티팩트 오탐과 처리 조건 단서 가능성이 있음, README 5·6절).

## 2. 공통 조건

- 데이터: 이미지 179장(1824x1368 JPEG, 장당 약 2.4MB). 라벨 AGG0 83 / AGG1 43 / AGG2 53. API 모델에는 **원본 JPEG를 그대로** 보냈다(리사이즈 없음).
- 프롬프트(API 모델): 사용자 정의("이상 결정립 성장, 주변 질감과 다르게 편평한 부분이 있으면 AGG")를 반영한 v2와 같은 내용. JSON(`evidence`, `label`, `severity` 0~1)으로 답하게 했다. 프롬프트 전문은 `api/scripts/run_api.py`의 `PROMPT`.
- JEVision은 선택지 확률을 직접 출력하므로 AUROC를 낼 수 있고, API 모델은 확률이 없어 `severity` 점수로 AUROC를 계산했다(모델이 스스로 말한 값이라 JEVision 확률과 같은 성격은 아님).
- 평가는 `eval_agg.py`(이 폴더 바로 아래, `jev/`와 `api/` 결과 공용) (혼동행렬, 클래스별 precision/recall/F1, 배율·금속·처리 조건별 정확도).

## 3. 모델별 상세

### 3.1 JEVision 0.8B (로컬)
- 모델: [divyanshx11/JEVision](https://huggingface.co/divyanshx11/JEVision) (Qwen3.5-0.8B-Base + 어댑터, Apache-2.0, 교수님 추천). WSL2 Ubuntu, RTX 3060 12GB.
- 호출: 로컬, 장당 약 1초(전체 사진), 타일 1,514개도 수 분. 비용 없음.
- 결과 요약: 위 표와 `README.md`. 전체 사진은 프롬프트에 따라 기본 답(AGG0 또는 AGG1)만 바뀌었고, 타일 방식에서만 AGG2 구분 신호(AUROC 0.944)가 나왔다.
- 환경 차이: WSL에 C 컴파일러가 없어 `flash-linear-attention` 없이 `TORCH_DISABLE_NATIVE_JIT=1`로 실행(순수 PyTorch 대체 경로).

### 3.2 kimi-k3 (NVIDIA API, `moonshotai/kimi-k3`)
- 호출 설정(사용자가 준 예시 기준): temperature 1, seed 0, `reasoning_effort: max`, max_tokens 16384, 스트리밍 없음. 이미지는 base64 data URL(원본 2.4MB)로 전송되어 **그대로 받아들여졌다**.
- 실행: 179장 전체 성공, 오류 0건, 응답 형식(JSON) 파싱 실패 0건, `finish_reason` 전부 `stop`. 동시 호출 4개.
- 속도·토큰: 호출당 중앙값 71초(최대 388초). 입력 평균 3,560토큰, 출력 평균 593토큰(최대 3,490, 합계 106,070).
- 결과: accuracy 0.447, macro-F1 0.241.

  | 정답＼예측 | AGG0 | AGG1 | AGG2 | recall |
  |---|---|---|---|---|
  | AGG0 (83) | 77 | 6 | 0 | 0.928 |
  | AGG1 (43) | 40 | 3 | 0 | 0.070 |
  | AGG2 (53) | 53 | 0 | 0 | 0.000 |

  precision / recall / F1: AGG0 0.453 / 0.928 / 0.609, AGG1 0.333 / 0.070 / 0.115, AGG2 0 / 0 / 0.
- `severity` 점수: 정답별 평균 0.105 / 0.099 / 0.091(AGG0/1/2), AUROC AGG2>AGG0 0.531, AGG1>AGG0 0.462, AGG2>AGG1 0.559. 점수가 클래스를 구분하지 못한다.
- 배율별 accuracy: 7.5x 0.483, 10x 0.417, 12.5x 0.441. 처리 조건별 accuracy는 해당 조건의 AGG0 비율을 따라간다(NPH 0.000, PH1230-1035 1.000 등, `api/results/metrics_api_kimi-k3.txt`).
- 응답 설명(`evidence`)은 대체로 "균일한 미세 알갱이, 편평한 부분 없음"이었다.

### 3.3a gemini-3.5-flash-lite / gemini-3.1-flash-lite (Gemini API)
- 한도 확인: 두 모델 모두 무료 티어 **분당 15건**(429 후 약 50~60초 대기). 하루 한도는 이번 실행에서 닿지 않았다. 한도는 모델별이라 두 모델을 동시에 돌려도 서로 영향이 없다.
- 실행 설정: temperature 0, JSON 스키마 응답, 모델당 동시 호출 2개, 분당 13건으로 제한, **무작위 순서(seed 0)**. 두 모델 모두 179장 완료, 오류 0건, `finish_reason` 전부 `STOP`. 호출당 중앙값 3.4초, 입력 1,302토큰, 출력 평균 약 50토큰(생각 토큰 없음). 모델당 약 14분.
- 결과: 아래 표. 둘 다 거의 전부 AGG0으로 답했다.

  | 모델 | accuracy | macro-F1 | 예측 분포 (AGG0/AGG1/AGG2) | AGG1 recall | AGG2 recall | AUROC (AGG2>AGG0 / AGG1>AGG0 / AGG2>AGG1) |
  |---|---|---|---|---|---|---|
  | gemini-3.5-flash-lite | 0.464 | 0.251 | 173 / 4 / 2 | 0.070 | 0.000 | 0.473 / 0.499 / 0.474 |
  | gemini-3.1-flash-lite | 0.458 | 0.237 | 174 / 2 / 3 | 0.047 | 0.000 | 0.464 / 0.510 / 0.453 |

  혼동행렬: 3.5는 AGG0 80/1/2, AGG1 40/3/0, AGG2 53/0/0 (행=정답, 열=AGG0/1/2). 3.1은 AGG0 80/0/3, AGG1 41/2/0, AGG2 53/0/0.
- `severity` 점수가 클래스를 구분하지 못하고 오히려 거꾸로다: 정답별 평균이 AGG0 0.031/0.036, AGG1 0.023/0.019, AGG2 0.001/0.000(3.5/3.1). AUROC가 0.5 아래다. AGG2로 예측한 2~3건은 전부 실제로는 AGG0인 이미지였다.
- 조건별 accuracy는 해당 조건의 AGG0 비율을 따라간다(NPH 0.000, PH1205-1035 1.000 등).
- 이 모델들은 3.6/3.8 flash보다 작은 별개 모델이라, 3.6/3.8의 결과를 대신하지 못한다.

### 3.3 gemini-3.6-flash / gemini-3.8-flash (Gemini API)
- 호출: REST `generateContent`, temperature 0, JSON 스키마 응답. 두 모델 모두 모델 목록(`/v1beta/models`)에 있고 이미지 입력이 된다.
- 스모크 테스트(3장): 둘 다 형식 정상. 호출당 3.6은 약 6~7초, 3.8은 8~36초. 입력 1,302토큰, 생각 토큰 약 150~260.
- **전체 실행 실패 원인: 무료 티어 한도.** `generate_content_free_tier_requests, limit: 20`, 즉 **모델당 하루 20건**이다. 오늘 한도를 소진했고 응답에 복구까지 약 2시간 24분이라고 나왔다.
- 남은 데이터: 성공한 행이 3.6은 15건, 3.8은 12건뿐이고 전부 `BASE` 금속이라 라벨 분포가 치우쳐 있다. **정확도 평가는 하지 않았다**(오해 우려).
- 진행하려면: 결제 계정을 연결해 유료 티어로 전환(179장은 입력 약 30만 토큰). 전환은 사용자가 직접 해야 한다. 하루 20건씩 나누면 모델당 약 9일이 걸린다.
- 코드 수정 이력: 초기 실행에서 429 오류 행이 "처리됨"으로 기록되어 재개 시 건너뛰는 문제가 있어, 재개 시 성공한 행만 건너뛰도록 고쳤다.

### 3.4 llama-3.2-90b-vision-instruct (NVIDIA API)
- 호출 설정: temperature 0(사용자 예시는 1이었으나 재현성을 위해 변경), top_p 1, max_tokens 512.
- 원본 2.4MB 이미지는 받아들여졌다(입력 6,651토큰, 출력 88토큰). 1건 시험 결과는 AGG0(정답 AGG0)이었지만 **1건이라 성능 판단은 불가**.
- 문제: 그 1건에 **1,572초(26분)** 걸렸다(재시도 대기 포함). 별도 점검에서는 64x64 이미지나 텍스트만 보내도 90~150초 안에 응답이 오지 않았다. 서버 쪽 대기가 긴 것으로 보인다.
- 전체 179장은 현실적이지 않아 실행하지 않았다.
- 재측정(2026-10-12 09:46 KST, 텍스트 한 줄, 300초 대기): **응답 없음(ReadTimeout 300초)**. 시간대를 바꿔도 나아지지 않았다.

### 3.5 nemotron-3-nano-omni-30b-a3b-reasoning (NVIDIA API)
- 호출 설정(사용자 예시 기준): temperature 0.6, top_p 0.95, `reasoning_budget` 16384, max_tokens 65536.
- 1건 시험이 HTTP 504(게이트웨이 시간 초과)로 재시도 후 실패했다. 별도 점검에서도 150초 안에 응답이 없었다.
- 재측정(2026-10-12 09:46 KST, 텍스트 한 줄): **HTTP 503** `Worker local total request limit reached (16/16)`, 0.6초 만에 거절. 서버 워커의 동시 요청 한도가 가득 찬 상태로, 우리 쪽 요청 수와는 무관하다.
- 실행하지 않았다.

## 4. 알려진 한계와 주의

- API 모델 결과는 모두 **한 번 실행**이다. 같은 입력에서 출력이 재현되는지는 확인하지 않았다(kimi-k3는 temperature 1, seed 0 지정).
- 라벨이 처리 조건과 강하게 얽혀 있어(`../preproc/README.md` 7절), 조건별 정확도가 높게 나와도 AGG를 본 것이라고 말할 수 없다.
- API 모델에 보낸 이미지는 외부 서버(Google, NVIDIA)로 전송되었다(사용자 허가를 받고 진행).
- 모델명과 지원 기능은 2026-10-12 시점 확인 결과이며, 무료 티어 한도와 호스팅 상태는 바뀔 수 있다.
- Gemini, llama, nemotron은 평가 지표가 없으므로 위 표의 성능 비교에 넣을 수 없다.

## 5. 다음 단계 후보

1. kimi-k3에 타일 방식 적용(타일당 약 1분이라 일부 이미지·타일만 먼저).
2. Gemini 유료 전환 후 전체 사진과 타일 재개.
3. llama/nemotron은 서버 상태가 나아지면 소규모로 재시도.
4. 조명 띠·스크래치를 무시하도록 프롬프트를 보강한 타일 프롬프트(JEVision).

## 6. 파일

| 경로 | 내용 |
|---|---|
| `api/scripts/run_api.py` | Gemini/NVIDIA API 호출(재개 가능, 키는 `~/.agg_keys.env`에서 읽음) |
| `eval_agg.py` | 공용 평가 스크립트(JEVision과 API 결과 모두 처리, 오류 행 제외, `severity` 점수 AUROC 포함) |
| `api/results/api_kimi-k3.jsonl`, `api/results/metrics_api_kimi-k3.txt` | kimi-k3 이미지별 결과와 지표 |
| `api/results/api_gemini-3.6-flash.jsonl`, `api/results/api_gemini-3.8-flash.jsonl` | Gemini 부분 결과(BASE만, 평가 안 함) |
| `api/results/api_gemini-3.5-flash-lite.jsonl`, `api/results/api_gemini-3.1-flash-lite.jsonl` | flash-lite 이미지별 결과 (179줄) |
| `api/results/metrics_api_gemini-3.5-flash-lite.txt`, `api/results/metrics_api_gemini-3.1-flash-lite.txt` | flash-lite 지표 원본 |

`run_api.py` 옵션: `--rpm`(분당 요청 수 제한), `--shuffle`(무작위 순서), 하루 한도(429에 대기 5분 초과)가 나오면 재시도 없이 중단. 재개 시 성공한 행만 건너뛴다.
