"""b10: DINOv2_grid_augmentation 의 특징으로 라벨을 이진(AGG0 = 0, AGG1+AGG2 = 1)으로 바꿔 16개 설정을 학습하고 평가한다 (WSL, sklearn 필요, GPU 불필요).

- 특징: exp_grid_aug 의 e01 출력(원본 + 배경 증강 5개)을 읽기 전용으로 사용. 새 특징 추출 없음.
- 학습은 DINOv2_grid_augmentation 과 같다: 이미지 단위 설정(합치기, whole_image)은 학습 이미지의 모든 버전(원본 + 증강 5개)을 학습 행으로, SL 은 학습 타일마다 6개 버전 중 하나를 무작위로. 평가는 항상 원본.
- 분류기 정의는 3분류 실험과 같고(logreg: 표준화 + L2, C=1, balanced / kNN: k=5, cosine, 거리 가중) 라벨만 이진이다.
- 평가 3가지: IMG = e00 의 이미지 단위 층화 5-fold x 10 반복 (3분류 라벨로 층화된 기존 분할), LOCO = 처리 조건 1개 제외(10 fold), LOMO = 처리 방법 1개 제외(5 fold).
- 지표는 fold 를 합친(pooled) 60장 예측으로 계산한다. IMG 는 반복마다 구해 평균 +- SD, LOCO/LOMO 는 한 번. AUROC 는 양성 확률로 계산 (LOCO/LOMO 는 fold 마다 모델이 달라 참고용).
출력: cv_bin_predictions.csv, cv_bin_summary.csv, cv_bin_fold_accuracy.csv, log.txt
"""
import argparse
import csv
from collections import defaultdict

import numpy as np

from bcommon import (Logger, build_settings, check_folds, fit_predict_proba_bin, image_vectors, load_config, load_features, load_images, make_folds, metrics_bin,
                     pick_versions_tiles, stack_versions, to_binary, write_csv)

SCRIPT = "b10_run_binary_cv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    log = Logger(cfg["paths"]["outputs_dir"], SCRIPT)
    log("config:", cfg["_config_path"])
    img = load_images(cfg)
    ids = img["ids"]; y3 = img["y"]; y = to_binary(y3, cfg); N = len(ids); pos = {i: k for k, i in enumerate(ids)}
    log(f"이진 라벨: AGG0 -> 0 ({int((y == 0).sum())}장), AGG1+AGG2 -> 1 ({int((y == 1).sum())}장), 다수 클래스(양성) 기준선 {max(int((y == 0).sum()), int((y == 1).sum())) / N:.3f}")
    settings, pid = build_settings(cfg)
    L = load_features(cfg["paths"]["aug_out"], True, ids)
    tile_img, tiles_of, K1 = L["tile_img"], L["tiles_of"], L["K1"]
    log(f"설정 {len(settings)}개 (primary = {pid}), 버전 {K1}개 (원본 + 배경 증강 {K1 - 1}개), 타일 {len(tile_img)}개")
    vec = {}

    def image_matrix(f, agg):
        if (f, agg) not in vec:
            vec[(f, agg)] = image_vectors(L, f, agg)
        return vec[(f, agg)]

    # ---------- 분할 목록: scheme -> repeat 별 fold 목록 [(tr, te)] ----------
    sp = list(csv.DictReader(open(cfg["paths"]["splits_csv"], encoding="utf-8-sig")))
    byk = defaultdict(lambda: {"train": [], "test": []})
    for r in sp:
        byk[(int(r["repeat"]), int(r["fold"]))][r["role"]].append(pos[r["image_id"]])
    reps = sorted({k[0] for k in byk}); nf = sorted({k[1] for k in byk})
    plan = {"IMG": [[(np.array(byk[(r, f)]["train"]), np.array(byk[(r, f)]["test"]), f"{r}-{f}") for f in nf] for r in reps]}
    for sch, unit in cfg["schemes"].items():
        folds = make_folds(img, img[unit]); check_folds(img, folds, sch)
        plan[sch] = [[(tr, te, u) for u, tr, te in folds]]
    for sch, reps_ in plan.items():                               # 이진 라벨에서도 모든 fold 의 train 에 두 클래스가 있는지, 이미지가 서로소인지
        for folds in reps_:
            seen = np.zeros(N, int)
            for tr, te, _ in folds:
                assert set(tr.tolist()).isdisjoint(te.tolist()) and set(np.unique(y[tr]).tolist()) == {0, 1}, sch
                seen[te] += 1
            assert (seen == 1).all(), sch
    log("분할 assert 통과: 모든 repeat 에서 모든 이미지가 정확히 한 번 test, train/test 서로소, train 에 두 클래스")

    guard = defaultdict(int)
    pred_rows, sum_rows, fold_rows = [], [], []
    for si, (sch, reps_) in enumerate(plan.items()):
        R = len(reps_)
        for s in settings:
            P = np.full((N, R), -1); S = np.zeros((N, R))
            for r, folds in enumerate(reps_):
                for fi, (tr, te, u) in enumerate(folds):
                    if s["kind"] == "SL":
                        F = L["F"][s["features"]]
                        ttr = np.concatenate([tiles_of[i] for i in tr]); tte = np.concatenate([tiles_of[i] for i in te])
                        assert set(tile_img[ttr].tolist()).isdisjoint(tile_img[tte].tolist()), "타일이 train/test 양쪽에 있음"
                        Ftr = pick_versions_tiles(L, s["features"], ttr, np.random.RandomState(cfg["aug"]["seed"] + 1000 * fi + 10000 * r + 500 * si))
                        pt = fit_predict_proba_bin("logreg", Ftr, y[tile_img[ttr]], F[0][tte], cfg, guard)
                        proba = np.zeros((len(te), 2))
                        for k, i in enumerate(te):
                            proba[k] = pt[np.where(tile_img[tte] == i)[0]].mean(0)
                    else:
                        X = L["W"] if s["kind"] == "whole_image" else image_matrix(s["features"], s["aggregator"])
                        Xtr, ytr = stack_versions(X, y, tr)
                        proba = fit_predict_proba_bin(s["classifier"], Xtr, ytr, X[te][:, 0], cfg, guard)
                    pred = np.argmax(proba, 1)
                    P[te, r] = pred; S[te, r] = proba[:, 1]
                    if sch != "IMG":
                        fold_rows.append(dict(scheme=sch, setting=s["id"], fold=fi, unit=u, n_test=len(te), accuracy=round(float(np.mean(pred == y[te])), 4)))
            assert (P >= 0).all()
            ms = [metrics_bin(y, P[:, r], S[:, r]) for r in range(R)]
            row = dict(scheme=sch, setting=s["id"], primary=int(s["id"] == pid), n_repeats=R)
            for k in ("accuracy", "balanced_acc", "macro_f1", "recall_neg", "recall_pos", "f1_pos", "auroc"):
                row[k] = round(float(np.mean([m[k] for m in ms])), 4)
                row[k + "_sd"] = round(float(np.std([m[k] for m in ms])), 4)
            sum_rows.append(row)
            for r in range(R):
                for i in range(N):
                    pred_rows.append(dict(scheme=sch, setting=s["id"], repeat=r, image_id=ids[i], label=int(y[i]), label3=int(y3[i]), pred=int(P[i, r]), score_pos=round(float(S[i, r]), 6)))
        log(f"\n[{sch}] accuracy (이진, pooled; IMG 는 반복 {R}회 평균 ± SD)")
        for r in [r for r in sum_rows if r["scheme"] == sch]:
            log(f"  {r['setting']:<36} {'*' if r['primary'] else ' '} acc {r['accuracy']:.3f}" + (f" ± {r['accuracy_sd']:.3f}" if r["n_repeats"] > 1 else "") +
                f" | balanced {r['balanced_acc']:.3f} | macro-F1 {r['macro_f1']:.3f} | recall AGG0 {r['recall_neg']:.2f} / AGG1+2 {r['recall_pos']:.2f} | AUROC {r['auroc']:.3f}")
    log(f"\n누수 방지 확인: scaler 가 train 통계와 같은지 assert {guard['scaler_checked']}회 통과, 수렴 경고 {guard['conv_warnings']}건")
    write_csv(f"{log.dir}/cv_bin_predictions.csv", pred_rows)
    write_csv(f"{log.dir}/cv_bin_summary.csv", sum_rows)
    write_csv(f"{log.dir}/cv_bin_fold_accuracy.csv", fold_rows)
    log(f"저장: {log.dir}")


if __name__ == "__main__":
    main()
