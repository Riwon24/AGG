"""m15: MIL 과 같은 타일 집합(config tile_filter)에서 기존 baseline 과 같은 방식(타일 특징을 이미지 안에서 평균 -> 표준화 + 로지스틱 회귀, cls · mean · logreg)을 이진 라벨로 다시 학습/평가한다 (WSL, sklearn 필요).

tile_filter = all 이면 기존 이진 baseline(exp_binary) 과 같은 결과가 나와야 한다 (재현 확인), full 이면 마스크 100% 타일만으로 평균한 baseline 이다.
학습 행 = 학습 이미지의 모든 버전(원본 + 증강 5개), 평가 = 원본, 분할은 MIL 과 같다 (이미지 단위 e00, LOCO, LOMO).
출력: baseline_predictions.csv, baseline_summary.csv, log.txt
"""
import argparse
import csv
from collections import defaultdict

import numpy as np

from mcommon import Logger, check_folds, load_config, load_features, load_images, make_folds, metrics_bin, tile_keep, to_binary, write_csv
from bcommon import fit_predict_proba_bin

SCRIPT = "m15_baseline_pooled_logreg"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    log = Logger(cfg["paths"]["outputs_dir"], SCRIPT)
    log("config:", cfg["_config_path"], "| tile_filter:", cfg.get("tile_filter", "all"))
    img = load_images(cfg)
    ids = img["ids"]; y3 = img["y"]; y = to_binary(y3, cfg); N = len(ids); pos = {i: k for k, i in enumerate(ids)}
    L = load_features(cfg["paths"]["aug_out"], True, ids)
    F = L["F"][cfg["features"]]; K1 = L["K1"]
    keep = tile_keep(cfg, L["tiles_of"]); tiles_of = [L["tiles_of"][i][keep[i]] for i in range(N)]
    X = np.stack([np.stack([F[k][tiles_of[i]].mean(0) for i in range(N)]) for k in range(K1)], 1)      # (N, 버전, 768): 이미지 안 타일 평균
    log(f"이미지 {N}장, 타일 {sum(len(t) for t in tiles_of)}개 (이미지당 {min(len(t) for t in tiles_of)}~{max(len(t) for t in tiles_of)}개), 평균 특징 {X.shape}")
    sp = list(csv.DictReader(open(cfg["paths"]["splits_csv"], encoding="utf-8-sig")))
    byk = defaultdict(lambda: {"train": [], "test": []})
    for r in sp:
        byk[(int(r["repeat"]), int(r["fold"]))][r["role"]].append(pos[r["image_id"]])
    reps = sorted({k[0] for k in byk}); nf = sorted({k[1] for k in byk})
    plan = {"IMG": [[(np.array(byk[(r, f)]["train"]), np.array(byk[(r, f)]["test"])) for f in nf] for r in reps]}
    for sch, unit in cfg["schemes"].items():
        folds = make_folds(img, img[unit]); check_folds(img, folds, sch)
        plan[sch] = [[(tr, te) for u, tr, te in folds]]
    guard = defaultdict(int)
    pred_rows, sum_rows = [], []
    for sch, reps_ in plan.items():
        R = len(reps_); P = np.full((N, R), -1); S = np.zeros((N, R))
        for r, folds in enumerate(reps_):
            for tr, te in folds:
                assert set(tr.tolist()).isdisjoint(te.tolist())
                Xtr = X[tr].reshape(-1, X.shape[2]); ytr = np.repeat(y[tr], K1)
                proba = fit_predict_proba_bin("logreg", Xtr, ytr, X[te][:, 0], cfg, guard)
                P[te, r] = np.argmax(proba, 1); S[te, r] = proba[:, 1]
        assert (P >= 0).all()
        ms = [metrics_bin(y, P[:, r], S[:, r]) for r in range(R)]
        row = dict(scheme=sch, n_repeats=R)
        for k in ("accuracy", "balanced_acc", "macro_f1", "recall_neg", "recall_pos", "f1_pos", "auroc"):
            row[k] = round(float(np.mean([m[k] for m in ms])), 4); row[k + "_sd"] = round(float(np.std([m[k] for m in ms])), 4)
        sum_rows.append(row)
        for r in range(R):
            for i in range(N):
                pred_rows.append(dict(scheme=sch, repeat=r, image_id=ids[i], label=int(y[i]), pred=int(P[i, r]), prob=round(float(S[i, r]), 6)))
        log(f"[{sch}] baseline (타일 평균 + logreg) accuracy {row['accuracy']:.3f}" + (f" ± {row['accuracy_sd']:.3f}" if R > 1 else "") + f" | balanced {row['balanced_acc']:.3f} | macro-F1 {row['macro_f1']:.3f} | recall AGG0 {row['recall_neg']:.2f} / AGG1+2 {row['recall_pos']:.2f} | AUROC {row['auroc']:.3f}")
    write_csv(f"{log.dir}/baseline_predictions.csv", pred_rows)
    write_csv(f"{log.dir}/baseline_summary.csv", sum_rows)
    log(f"\n누수 방지: scaler 가 train 통계와 같은지 assert {guard['scaler_checked']}회 통과, 수렴 경고 {guard['conv_warnings']}건. 저장: {log.dir}")


if __name__ == "__main__":
    main()
