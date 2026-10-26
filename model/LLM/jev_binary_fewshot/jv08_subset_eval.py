"""jv08 보조 분석: few-shot 예시가 나온 이미지 3장을 평가에서 뺀 57장으로 jv06, jv07, jv08 의 이진(AGG0 vs AGG1+2) 결과를 같은 지표로 다시 비교한다 (표준 라이브러리만 사용).

예시가 나온 이미지(outputs/jv08_tiles_fewshot/examples.json)는 평가 때 질의 타일 중 예시와 똑같은 타일(3개)이 있고, 같은 이미지의 다른 타일은 예시와 같은 사진에서 나온 것이라 jv08 점수가 예시의 영향을 받는다.
각 실행의 predictions.csv (임계값은 원래대로 60장의 학습 fold 에서 고른 것이고 평가에 쓰는 이미지만 줄인다)와 image_scores.csv 에서 primary 점수(mean_flat)만 쓴다.
출력: outputs/jv08_tiles_fewshot/subset_57_comparison.md
"""
import csv
import json
import random
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
RUNS = {"jv06 (선택형, run_tiles.py 프롬프트)": "jv06_tiles_choice", "jv07 (설명문 보강)": "jv07_tiles_choice_v2", "jv08 (예시 이미지 few-shot)": "jv08_tiles_fewshot"}
EX = json.loads((OUT / "jv08_tiles_fewshot/examples.json").read_text(encoding="utf-8"))
SRC = sorted({e["image_id"] for e in EX})
PRIM = "JEVision 타일 mean_flat"


def auc(pos, neg):
    return sum(1 if p > n else 0.5 if p == n else 0 for p in pos for n in neg) / (len(pos) * len(neg))


def load(d):
    pred = {}
    for r in csv.DictReader(open(OUT / d / "predictions.csv", encoding="utf-8-sig")):
        if r["method"] == PRIM:
            pred.setdefault(r["scheme"], {}).setdefault(int(r["repeat"]), {})[r["image_id"]] = (int(r["binary"]), int(r["pred"]))
    sc = {r["image_id"]: (int(r["binary"]), float(r["mean_flat"])) for r in csv.DictReader(open(OUT / d / "image_scores.csv", encoding="utf-8-sig"))}
    return pred, sc


def main():
    md = ["# few-shot 예시 이미지 3장을 뺀 비교 (이진 AGG0 vs AGG1+2, primary 점수 mean_flat)\n", "빼는 이미지: " + ", ".join(f"`{s}`" for s in SRC) + "\n"]
    data = {k: load(d) for k, d in RUNS.items()}
    ids_all = sorted(next(iter(data.values()))[1])
    keep = [i for i in ids_all if i not in SRC]
    yb = {i: data[next(iter(data))][1][i][0] for i in ids_all}
    md.append(f"60장 -> {len(keep)}장 (AGG0 {sum(1 for i in keep if yb[i] == 0)} / AGG1+2 {sum(1 for i in keep if yb[i] == 1)}, 다수 클래스(항상 AGG1+2) {max(sum(1 for i in keep if yb[i] == 1), sum(1 for i in keep if yb[i] == 0)) / len(keep):.3f}).\n")
    md += ["## 1. accuracy (IMG: 반복 10회 평균 ± SD)\n", "| 실행 | 전체 60장 IMG | 57장 IMG | 57장 LOCO | 57장 LOMO | 57장 recall AGG0 / AGG1+2 (IMG) | 57장 AUROC (AGG1+2 vs AGG0) |", "|---|---|---|---|---|---|---|"]
    per = {}
    for k, (pred, sc) in data.items():
        def acc(scheme, ims):
            a = []
            for rp, m in pred[scheme].items():
                a.append(sum(m[i][0] == m[i][1] for i in ims) / len(ims))
            return st.mean(a), (st.pstdev(a) if len(a) > 1 else 0.0)
        full = acc("IMG", ids_all); sub = {s: acc(s, keep) for s in ("IMG", "LOCO", "LOMO")}
        r0, r1 = [], []
        for rp, m in pred["IMG"].items():
            n0 = [i for i in keep if m[i][0] == 0]; n1 = [i for i in keep if m[i][0] == 1]
            r0.append(sum(m[i][1] == 0 for i in n0) / len(n0)); r1.append(sum(m[i][1] == 1 for i in n1) / len(n1))
        pos = [sc[i][1] for i in keep if sc[i][0] == 1]; neg = [sc[i][1] for i in keep if sc[i][0] == 0]
        a0 = auc(pos, neg); rng = random.Random(0); bs = []
        for _ in range(2000):
            ix = [rng.choice(keep) for _ in keep]
            p_ = [sc[i][1] for i in ix if sc[i][0] == 1]; n_ = [sc[i][1] for i in ix if sc[i][0] == 0]
            if p_ and n_:
                bs.append(auc(p_, n_))
        bs.sort()
        per[k] = {s: {i: st.mean(int(m[i][0] == m[i][1]) for m in pred[s].values()) for i in keep} for s in ("IMG", "LOCO", "LOMO")}
        md.append(f"| {k} | {full[0]:.3f} ± {full[1]:.3f} | {sub['IMG'][0]:.3f} ± {sub['IMG'][1]:.3f} | {sub['LOCO'][0]:.3f} | {sub['LOMO'][0]:.3f} | {st.mean(r0):.2f} / {st.mean(r1):.2f} | {a0:.3f} [{bs[49]:.3f}, {bs[1949]:.3f}] |")
    md += ["\n## 2. 57장에서 jv08 과 다른 실행의 차이 (같은 이미지끼리 짝지은 이미지 단위 부트스트랩 95% CI)\n", "| 비교 | 평가 | 차이 | 95% CI |", "|---|---|---|---|"]
    names = list(RUNS)
    for other in names[:2]:
        for s in ("IMG", "LOCO", "LOMO"):
            d = [per[names[2]][s][i] - per[other][s][i] for i in keep]
            rng = random.Random(0); n = len(d)
            bs = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(2000))
            md.append(f"| jv08 - {other.split(' ')[0]} | {s} | {st.mean(d):+.3f} | [{bs[49]:+.3f}, {bs[1949]:+.3f}] |")
    md.append("\n임계값은 60장의 학습 fold 에서 고른 것을 그대로 쓰고 평가 이미지만 57장으로 줄였다. 57장에서도 CI 는 넓다.\n")
    (OUT / "jv08_tiles_fewshot/subset_57_comparison.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
