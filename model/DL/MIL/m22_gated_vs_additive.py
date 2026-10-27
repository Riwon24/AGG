"""m22: gated MIL (outputs/) 과 additive MIL (outputs_additive/) 의 accuracy 를 같은 이미지끼리 짝지어 비교한다 (m10 의 mil_predictions.csv 만 읽음, numpy).

이미지 단위 부트스트랩 (2000회, 시드 0): 이미지 N장을 복원추출해 두 모델의 accuracy 차이 (additive - gated) 의 95% 구간을 구한다. IMG 는 반복 10회의 accuracy 를 평균한 뒤 차이를 낸다 (m12 와 같은 방식).
출력: outputs_additive/m22_gated_vs_additive/paired_gated_additive.csv, log.txt
"""
import csv
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
rd = lambda p: list(csv.DictReader(open(p, encoding="utf-8-sig")))


def load(run):
    P = {}
    for r in rd(HERE / run / "m10_train_eval_mil" / "mil_predictions.csv"):
        P.setdefault(r["scheme"], {}).setdefault(r["image_id"], {})[int(r["repeat"])] = (int(r["label"]), int(r["pred"]))
    return P


def main():
    G, A = load("outputs"), load("outputs_additive")
    out = HERE / "outputs_additive" / "m22_gated_vs_additive"; out.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(0); rows, lines = [], []
    for sc in ("IMG", "LOCO", "LOMO"):
        ids = sorted(G[sc]); assert ids == sorted(A[sc])
        cg = np.array([[G[sc][i][r][1] == G[sc][i][r][0] for r in sorted(G[sc][i])] for i in ids], float).mean(1)       # 이미지별 정답 비율 (IMG 는 반복 평균)
        ca = np.array([[A[sc][i][r][1] == A[sc][i][r][0] for r in sorted(A[sc][i])] for i in ids], float).mean(1)
        N = len(ids); d = []
        for _ in range(2000):
            idx = rng.randint(0, N, N); d.append(ca[idx].mean() - cg[idx].mean())
        lo, hi = np.percentile(d, [2.5, 97.5])
        rows.append(dict(scheme=sc, gated=round(float(cg.mean()), 4), additive=round(float(ca.mean()), 4), diff=round(float(ca.mean() - cg.mean()), 4), image_ci_lo=round(float(lo), 4), image_ci_hi=round(float(hi), 4),
                         verdict="구분되지 않는다 (CI 가 0 포함)" if lo <= 0 <= hi else "구분된다"))
        lines.append(f"{sc}: gated {cg.mean():.3f} additive {ca.mean():.3f} 차이 {ca.mean() - cg.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] -> {rows[-1]['verdict']}")
    with open(out / "paired_gated_additive.csv", "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (out / "log.txt").write_text("\n".join(lines) + "\n", encoding="utf-8"); print("\n".join(lines))


if __name__ == "__main__":
    main()
