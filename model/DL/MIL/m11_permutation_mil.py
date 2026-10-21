"""m11: MIL 의 이진 라벨 순열 검사 (이미지 단위 분할, LOCO, LOMO 각각 20회). WSL, torch + sklearn 필요.

라벨을 이미지 단위로 섞고(클래스별 개수 유지) 같은 절차(학습 bag = 학습 이미지의 모든 버전, 표준화는 학습 타일만, 고정 epoch)로 학습해 섞은 라벨 기준 accuracy 를 구한다.
IMG 는 섞은 라벨로 5-fold 를 새로 층화(sklearn StratifiedKFold, 기존 이진 baseline 의 순열 검사 b11 과 같은 방식), LOCO/LOMO 는 같은 그룹 분할. 계산량 때문에 순열마다 시드 1개로 학습한다 (본 평가는 시드 5개 앙상블).
임계 기준은 따로 두지 않고 실제 점수가 순열 20회 범위(최댓값)를 넘는지로 읽는다 (이진 라벨의 다수 클래스 기준선이 0.533 이라 우연 수준은 0.5 부근).
출력: mil_permutation_runs.csv, log.txt
"""
import argparse

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

from mcommon import (Logger, check_folds, load_config, load_features, load_images, make_folds, metrics_bin, predict_mil, standardize_fit, tile_keep, to_binary, train_mil, write_csv)

SCRIPT = "m11_permutation_mil"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    log = Logger(cfg["paths"]["outputs_dir"], SCRIPT)
    log("config:", cfg["_config_path"])
    torch.use_deterministic_algorithms(True, warn_only=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    img = load_images(cfg)
    ids = img["ids"]; y = to_binary(img["y"], cfg); N = len(ids)
    L = load_features(cfg["paths"]["aug_out"], True, ids)
    F = L["F"][cfg["features"]]; K1 = L["K1"]
    keep = tile_keep(cfg, L["tiles_of"]); tiles_of = [L["tiles_of"][i][keep[i]] for i in range(N)]
    nmax = max(len(t) for t in tiles_of)
    bags = [[F[k][tiles_of[i]].astype(np.float32) for k in range(K1)] for i in range(N)]
    pm = cfg["permutation"]; cnt = {c: int((y == c).sum()) for c in (0, 1)}
    fixed = {}
    for sch, unit in cfg["schemes"].items():
        folds = make_folds(img, img[unit]); check_folds(img, folds, sch)
        fixed[sch] = [(tr, te) for u, tr, te in folds]
    rows = []
    for sch in ("IMG", "LOCO", "LOMO"):
        accs, bals = [], []
        for k in range(pm["n_permutations"]):
            yp = np.random.RandomState(pm["seed"] + k).permutation(y)
            assert {c: int((yp == c).sum()) for c in (0, 1)} == cnt
            folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=pm["seed"] + k).split(np.zeros(N), yp)) if sch == "IMG" else fixed[sch]
            P = np.full(N, np.nan)
            for fi, (tr, te) in enumerate(folds):
                assert set(tr.tolist()).isdisjoint(te.tolist()) and set(np.unique(yp[tr]).tolist()) == {0, 1}, "순열 라벨에서 학습에 두 클래스가 없음"
                tr_bags = [bags[i][kk] for i in tr for kk in range(K1)]; ytr = np.array([yp[i] for i in tr for kk in range(K1)])
                mu, sd = standardize_fit(tr_bags)
                tr_n = [(b - mu) / sd for b in tr_bags]
                te_b = [(bags[i][0] - mu) / sd for i in te]                                  # 평가는 원본 bag
                ps = np.zeros(len(te))
                for s in range(pm["seeds_per_permutation"]):
                    model, _, _ = train_mil(tr_n, ytr, cfg, 7000000 + 100000 * k + 1000 * fi + s, dev, nmax)
                    p, _ = predict_mil(model, te_b, nmax, dev)
                    ps += p / pm["seeds_per_permutation"]
                P[te] = ps
            assert not np.isnan(P).any()
            m = metrics_bin(yp, (P > 0.5).astype(int), P)
            accs.append(m["accuracy"]); bals.append(m["balanced_acc"])
            rows.append(dict(scheme=sch, permutation=k, seed=pm["seed"] + k, accuracy=round(m["accuracy"], 4), balanced_acc=round(m["balanced_acc"], 4), auroc=round(m["auroc"], 4)))
        log(f"[순열 {sch}] {len(accs)}회: accuracy 평균 {np.mean(accs):.3f} (범위 {np.min(accs):.3f}~{np.max(accs):.3f}), balanced accuracy 평균 {np.mean(bals):.3f} (최대 {np.max(bals):.3f})")
        write_csv(f"{log.dir}/mil_permutation_runs.csv", rows)
    log(f"\n저장: {log.dir}")


if __name__ == "__main__":
    main()
