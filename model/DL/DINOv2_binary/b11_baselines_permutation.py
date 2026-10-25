"""b11: 이진 라벨(AGG0 vs AGG1+2)의 기준선(라벨만 쓰는 단순 규칙)과 라벨 순열 검사를 3가지 평가(IMG, LOCO, LOMO)에서 계산한다 (WSL, sklearn 필요).

기준선 (학습 fold 이미지의 이진 라벨만 사용, 동률이면 작은 클래스 번호):
  majority               학습 fold 의 다수 클래스 (IMG 는 e00 의 모든 repeat/fold)
  condition_memorization (IMG 만) 같은 처리 조건의 다른 5장의 다수 이진 라벨 (leave-one-image-out). LOCO/LOMO 에서는 시험 조건이 학습에 없어 정의되지 않음
  metal                  같은 금속 종류의 학습 이미지 다수 라벨 (LOCO, LOMO)
  temp                   같은 온도의 학습 이미지 다수 라벨 (LOCO, LOMO)
  sibling_temp           (LOCO 만) 같은 처리 방법의 다른 온도 조건의 다수 라벨
순열 검사: primary 와 whole_image logreg 에 20회씩. 이진 라벨을 이미지 단위로 섞고(클래스별 개수 유지) 같은 학습 절차(증강 학습)로 평가.
  IMG 는 섞은 라벨로 5-fold 를 새로 층화(sklearn StratifiedKFold, DINOv2 의 e05 와 같은 방식), LOCO/LOMO 는 같은 그룹 분할. 임계 기준은 따로 두지 않고 실제 점수가 순열 20회 범위를 넘는지로 읽는다.
출력: baseline_bin_predictions.csv, baseline_bin.csv, bin_permutation_runs.csv, log.txt
"""
import argparse
import collections
import csv
from collections import defaultdict

import numpy as np
from sklearn.model_selection import StratifiedKFold

from bcommon import (Logger, build_settings, check_folds, fit_predict_proba_bin, image_vectors, load_config, load_features, load_images, make_folds, metrics_bin,
                     stack_versions, to_binary, write_csv)

SCRIPT = "b11_baselines_permutation"


def maj(ys, default=None):
    if len(ys) == 0:
        return default
    c = collections.Counter(int(v) for v in ys)
    top = max(c.values())
    return min(k for k, v in c.items() if v == top)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    log = Logger(cfg["paths"]["outputs_dir"], SCRIPT)
    log("config:", cfg["_config_path"])
    img = load_images(cfg)
    ids = img["ids"]; y = to_binary(img["y"], cfg); N = len(ids); pos = {i: k for k, i in enumerate(ids)}
    sp = list(csv.DictReader(open(cfg["paths"]["splits_csv"], encoding="utf-8-sig")))
    byk = defaultdict(lambda: {"train": [], "test": []})
    for r in sp:
        byk[(int(r["repeat"]), int(r["fold"]))][r["role"]].append(pos[r["image_id"]])
    reps = sorted({k[0] for k in byk}); nf = sorted({k[1] for k in byk})
    plan = {"IMG": [[(np.array(byk[(r, f)]["train"]), np.array(byk[(r, f)]["test"])) for f in nf] for r in reps]}
    for sch, unit in cfg["schemes"].items():
        folds = make_folds(img, img[unit]); check_folds(img, folds, sch)
        plan[sch] = [[(tr, te) for u, tr, te in folds]]

    # ---------- 기준선 ----------
    pred_rows, sum_rows = [], []
    log("[기준선] 이진 라벨 (AGG0 vs AGG1+2), 60장 pooled")
    for sch, reps_ in plan.items():
        rules = {"majority": lambda i, tr: maj(y[tr]),
                 "metal": lambda i, tr: maj(y[[j for j in tr if img["metal"][j] == img["metal"][i]]], maj(y[tr])),
                 "temp": lambda i, tr: maj(y[[j for j in tr if img["temp"][j] == img["temp"][i]]], maj(y[tr]))}
        if sch == "IMG":
            rules = {"majority": rules["majority"],
                     "condition_memorization": lambda i, tr: maj(y[[j for j in range(N) if j != i and img["condition"][j] == img["condition"][i]]])}
        if sch == "LOCO":
            rules["sibling_temp"] = lambda i, tr: maj(y[[j for j in tr if img["method"][j] == img["method"][i]]], maj(y[tr]))
        for rn, f in rules.items():
            R = len(reps_) if rn != "condition_memorization" else 1
            P = np.zeros((N, R), int)
            for r in range(R):
                for tr, te in reps_[r]:
                    for i in te:
                        P[i, r] = f(i, list(tr))
            ms = [metrics_bin(y, P[:, r]) for r in range(R)]
            sum_rows.append(dict(scheme=sch, rule=rn, accuracy=round(float(np.mean([m["accuracy"] for m in ms])), 4), balanced_acc=round(float(np.mean([m["balanced_acc"] for m in ms])), 4),
                                 macro_f1=round(float(np.mean([m["macro_f1"] for m in ms])), 4), n_repeats=R))
            for r in range(R):
                for i in range(N):
                    pred_rows.append(dict(scheme=sch, rule=rn, repeat=r, image_id=ids[i], label=int(y[i]), pred=int(P[i, r])))
            log(f"  [{sch}] {rn:23s} accuracy {sum_rows[-1]['accuracy']:.3f} | balanced {sum_rows[-1]['balanced_acc']:.3f} | macro-F1 {sum_rows[-1]['macro_f1']:.3f}")
    write_csv(f"{log.dir}/baseline_bin_predictions.csv", pred_rows)
    write_csv(f"{log.dir}/baseline_bin.csv", sum_rows)

    # ---------- 순열 검사 ----------
    settings, pid = build_settings(cfg)
    want = {s["id"]: s for s in settings if s["id"] in cfg["permutation"]["settings"]}
    assert set(want) == set(cfg["permutation"]["settings"])
    L = load_features(cfg["paths"]["aug_out"], True, ids)
    pm = cfg["permutation"]; guard = defaultdict(int); rows = []
    cnt = {c: int((y == c).sum()) for c in (0, 1)}
    for sch in plan:
        for sid, s in want.items():
            X = L["W"] if s["kind"] == "whole_image" else image_vectors(L, s["features"], s["aggregator"])
            accs, bals = [], []
            for k in range(pm["n_permutations"]):
                yp = np.random.RandomState(pm["seed"] + k).permutation(y)
                assert {c: int((yp == c).sum()) for c in (0, 1)} == cnt
                if sch == "IMG":
                    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=pm["seed"] + k).split(np.zeros(N), yp))
                else:
                    folds = [(tr, te) for tr, te in plan[sch][0]]
                P = np.full(N, -1)
                for tr, te in folds:
                    assert set(np.unique(yp[tr]).tolist()) == {0, 1}, "순열 라벨에서 학습에 두 클래스가 없음"
                    Xtr, ytr = stack_versions(X, yp, tr)
                    P[te] = np.argmax(fit_predict_proba_bin(s["classifier"], Xtr, ytr, X[te][:, 0], cfg, guard), 1)
                m = metrics_bin(yp, P)
                accs.append(m["accuracy"]); bals.append(m["balanced_acc"])
                rows.append(dict(scheme=sch, setting=sid, permutation=k, seed=pm["seed"] + k, accuracy=round(m["accuracy"], 4), balanced_acc=round(m["balanced_acc"], 4)))
            log(f"[순열 {sch} / {sid}] {len(accs)}회: accuracy 평균 {np.mean(accs):.3f} (범위 {np.min(accs):.3f}~{np.max(accs):.3f}), balanced accuracy 평균 {np.mean(bals):.3f} (최대 {np.max(bals):.3f})")
    write_csv(f"{log.dir}/bin_permutation_runs.csv", rows)
    log(f"\n누수 방지 assert: scaler 확인 {guard['scaler_checked']}회 통과. 저장: {log.dir}")


if __name__ == "__main__":
    main()
