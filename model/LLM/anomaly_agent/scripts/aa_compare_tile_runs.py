"""aa_compare_tile_runs: 같은 타일 34개(AGG1 표시 17 / AGG0 대조 17)를 다른 설정으로 두 번 돌린 결과를 비교한다.

판정 분포(표시/대조 타일 중 이상 판정 수), AUROC(에이전트 지표), 판정이 바뀐 타일, 반사실 margin·일반 템플릿 점수의 AUROC.
주의: 후보 문장과 이상 정의를 이 타일들에서 만들었기 때문에 margin AUROC 는 부풀려진 값이다 (해석하지 말 것).
실행 (WSL): python aa_compare_tile_runs.py [폴더A 폴더B]   기본: ~/AnomalyAgent/results/agg_tile34  agg_tile34_v2
"""
import json
import os
import sys
from collections import Counter

from sklearn.metrics import roc_auc_score

R = os.path.expanduser("~/AnomalyAgent/results")
names = sys.argv[1:3] or ["agg_tile34", "agg_tile34_v2"]


def load(d):
    j = json.load(open(os.path.join(R, d, "results.json"), encoding="utf-8"))
    return j["metrics"], {x["img_path"]: x for x in j["results"]}


runs = {n: load(n) for n in names}
for n, (m, d) in runs.items():
    c = Counter((x["label_value"], x["pred_result"]) for x in d.values())
    y = [1 if x["label_value"] == "AGG1표시" else 0 for x in d.values()]
    cf = [((x.get("template_log") or {}).get("counterfactual") or {}).get("avg_margin") for x in d.values()]
    gen = [((x.get("template_log") or {}).get("general") or {}).get("avg_score") for x in d.values()]
    ok = [i for i, v in enumerate(cf) if v is not None]
    print(f"{n}: AUROC {m['mean_auroc_sp']:.3f} | 표시 타일 이상 {c[('AGG1표시', 'anomalous')]}/17, 대조 타일 이상 {c[('AGG0대조', 'anomalous')]}/17"
          f" | margin AUROC {roc_auc_score([y[i] for i in ok], [cf[i] for i in ok]):.3f} | 일반 템플릿 AUROC {roc_auc_score([y[i] for i in ok], [gen[i] for i in ok]):.3f}")
da, db = runs[names[0]][1], runs[names[1]][1]
flip = [(k, da[k]["pred_result"], db[k]["pred_result"]) for k in sorted(da) if k in db and da[k]["pred_result"] != db[k]["pred_result"]]
print("판정이 바뀐 타일", len(flip), flip)
