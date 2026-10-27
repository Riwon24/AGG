"""aa_compare_runs: 이미지 단위 평가 결과(후보 7+7 vs 2+2)를 비교한다.
results.json 의 pred_score(에이전트 최종 점수), 반사실 margin 평균, 일반 템플릿 점수로 AGG0 vs AGG1+2 AUROC 를 내고, 최종 판정 정확도와 AGG별 분포를 보여준다.
실행 (WSL): python aa_compare_runs.py [결과폴더 ...]   기본: ~/AnomalyAgent/results/agg_img_cand7 agg_img_cand2
"""
import json
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

dirs = sys.argv[1:] or [os.path.expanduser(f"~/AnomalyAgent/results/{d}") for d in ("agg_img_cand7", "agg_img_cand2")]


def auc(y, s):
    ok = [(a, b) for a, b in zip(y, s) if b is not None and np.isfinite(b)]
    if len({a for a, _ in ok}) < 2:
        return float("nan")
    return roc_auc_score([a for a, _ in ok], [b for _, b in ok])


for d in dirs:
    r = json.load(open(os.path.join(d, "results.json"), encoding="utf-8"))["results"]
    y = [x["gt_anomaly"] for x in r]
    tl = [x.get("template_log") or {} for x in r]
    cf = [(t.get("counterfactual") or {}).get("avg_margin") for t in tl]
    gen = [(t.get("general") or {}).get("avg_score") for t in tl]
    pred = [x["pred_score"] for x in r]
    acc = np.mean([(x["pred_result"] == "anomalous") == bool(x["gt_anomaly"]) for x in r])
    print(f"\n== {os.path.basename(d)} (n={len(r)}, AGG0 {y.count(0)} / AGG1+2 {y.count(1)})")
    print(f"  판정 정확도 {acc:.3f} | 판정 분포 " + str({k: sum(x['pred_result'] == k for x in r) for k in ('anomalous', 'normal', 'uncertain')}))
    print(f"  AUROC  최종 점수 {auc(y, pred):.3f} | 반사실 margin {auc(y, cf):.3f} | 일반 템플릿 {auc(y, gen):.3f}   (로그 없는 항목 margin {sum(c is None for c in cf)}개)")
    for g in ("AGG0", "AGG2"):
        sel = [i for i, x in enumerate(r) if x["label_value"] == g]
        m = [cf[i] for i in sel if cf[i] is not None]
        print(f"  {g}: n={len(sel)} | 이상 판정 {sum(r[i]['pred_result'] == 'anomalous' for i in sel)} | margin 평균 {np.mean(m) if m else float('nan'):+.4f} | 최종 점수 평균 {np.mean([pred[i] for i in sel]):.3f}")
