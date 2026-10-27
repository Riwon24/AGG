"""aa_margin_rescore: 에이전트 실행에서 저장한 캡션(results.json 의 template_log.captions)으로, 후보 문장 세트를 바꿔 반사실 margin 을 오프라인(API 없이)에서 다시 계산한다.

margin(캡션) = max_{이상 후보} cos - max_{정상 후보} cos  (식 5), 이미지(타일)의 점수는 캡션 3개 평균. 임베딩은 에이전트와 같은 Qwen3-4B 텍스트 모델(TextModelEmbeddings).
같은 results.json 으로 후보 세트별 AUROC 를 비교하고, 실행 사이 판정 일치율도 낸다 (같은 설정 반복 시 판정이 얼마나 흔들리는지 보려는 것).
실행 (WSL, ~/anomalyagent_venv 에서): python aa_margin_rescore.py [results.json 이 있는 폴더]
  기본: ~/AnomalyAgent/results/agg_img_cand7_v2 (이미지 29장), 후보 7+7 / 2+2 두 벌.
"""
import json
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.expanduser("~/AnomalyAgent/src"))
from react_agent.utils import TextModelEmbeddings  # noqa: E402

RES = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/AnomalyAgent/results/agg_img_cand7_v2")
HERE = os.path.dirname(os.path.abspath(__file__))
CAND_DIR = os.path.join(HERE, "outputs/aa_caption_tiles")
CANDS = {"7+7": "atomic_candidates_agg.json", "2+2": "atomic_candidates_agg_2x2.json"}

rows = json.load(open(os.path.join(RES, "results.json"), encoding="utf-8"))["results"]
rows = [r for r in rows if (r.get("template_log") or {}).get("captions")]
y = [r["gt_anomaly"] for r in rows]
flat = [c for r in rows for c in r["template_log"]["captions"]]
idx = np.cumsum([0] + [len(r["template_log"]["captions"]) for r in rows])

emb = TextModelEmbeddings(os.path.expanduser("~/AnomalyAgent/models/qwen3_4b"), "sdpa")


def embed(texts):
    v = np.asarray(emb.embed_documents(texts), dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


C = embed(flat)
for name, fn in CANDS.items():
    d = json.load(open(os.path.join(CAND_DIR, fn)))["metal_sample_surface"]
    A, N = embed(d["anomaly_candidates"]), embed(d["normal_candidates"])
    m = (C @ A.T).max(1) - (C @ N.T).max(1)
    s = [float(m[idx[k]:idx[k + 1]].mean()) for k in range(len(rows))]
    print(f"{name}: 샘플 {len(rows)}개 | margin AUROC {roc_auc_score(y, s):.3f} | 정상 평균 {np.mean([v for v, t in zip(s, y) if t == 0]):+.4f} 이상 평균 {np.mean([v for v, t in zip(s, y) if t == 1]):+.4f}")
