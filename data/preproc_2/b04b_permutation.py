"""b04b: 이미지 단위 라벨 순열 검사 (primary 설정, primary 팔 R1 = 원본+증강본으로 학습, 원본으로 평가). exp/e05 와 같은 방식.

- 이미지 단위로 라벨을 섞는다 (클래스별 개수 유지). 순열마다 섞은 라벨로 fold 를 새로 나누고(층화) 5-fold 1회를 돈다.
  한 이미지의 증강본은 그 이미지의 (섞인) 라벨을 그대로 따른다.
- 통과 기준(제안값, 데이터에서 유도한 값이 아님): 20회 평균 macro-F1 이 0.33±0.10 이내이고 어느 회차도 0.55 를 넘지 않는다.
- 기준을 어기면 누수라고 단정하지 않고 멈춰서 보고한다 (종료 코드 4, 원인 미확인).
출력: permutation_runs.csv, log.txt
"""
import argparse
import csv
import sys
from collections import defaultdict

import numpy as np
from sklearn.model_selection import StratifiedKFold

from bg_common import Logger, aggregate, extraction_rows, forbidden_check, inputs, load_config, metrics, write_csv
from e03_run_cv import CLASSES, fit_predict_proba

SCRIPT = "b04b_permutation"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    out = cfg["paths"]["outputs_dir"]
    log = Logger(out, SCRIPT)
    log("config:", cfg["_config_path"])
    pm, cr, ex = cfg["permutation"], cfg["permutation"]["criteria"], cfg["experiment"]
    pr = ex["primary"]
    K = int(cfg["bgaug"]["n_aug"])
    log(f"primary = 특징 {pr['features']} / 합치기 {pr['aggregator']} / 분류기 {pr['classifier']}, 팔 {ex['primary_arm']} (원본+증강본 학습, 원본 평가) | 순열 {pm['n_permutations']}회, seed {pm['seed']}")
    log(f"통과 기준(제안값, 데이터에서 유도한 값이 아님): 평균 macro-F1 {cr['macro_f1_center']}±{cr['macro_f1_tolerance']} 이내, 어느 회차도 macro-F1 > {cr['macro_f1_max_single_run']} 금지")

    imgs, tiles = inputs(cfg)
    ids = [r["image_id"] for r in imgs]; y = np.array([r["label"] for r in imgs]); pos = {i: k for k, i in enumerate(ids)}
    z = np.load(f"{out}/b03_extract_features/features.npz")
    idx = extraction_rows(cfg, tiles)
    tile_img = np.array([pos[t["image_id"]] for t in tiles]); tiles_of = [np.where(tile_img == k)[0] for k in range(len(ids))]
    row_of = z["row_of"][idx]
    F = z["cls"].astype(np.float64) if pr["features"] == "cls" else np.concatenate([z["cls"], z["patchmean"]], 1).astype(np.float64)
    X = np.stack([aggregate(F[row_of[:, k]], tiles_of, pr["aggregator"]) for k in range(K + 1)], 1)     # (60, K+1, d)
    cnt = {int(c): int((y == c).sum()) for c in CLASSES}
    log(f"이미지 {len(ids)}장, 클래스별 개수 {cnt}, 벡터 {X.shape}")

    guard = defaultdict(int); rows = []
    for k in range(pm["n_permutations"]):
        rng = np.random.RandomState(pm["seed"] + k)
        yp = rng.permutation(y)
        assert {int(c): int((yp == c).sum()) for c in CLASSES} == cnt
        skf = StratifiedKFold(n_splits=cfg["cv"]["n_splits"], shuffle=True, random_state=pm["seed"] + k)
        pred = np.full(len(y), -1)
        for tr, te in skf.split(np.zeros(len(yp)), yp):
            assert set(tr.tolist()).isdisjoint(te.tolist())
            Xtr = X[tr].reshape(-1, X.shape[2]); ytr = np.repeat(yp[tr], K + 1)
            proba, _ = fit_predict_proba(pr["classifier"], Xtr, ytr, X[te][:, 0], cfg, guard)
            pred[te] = CLASSES[np.argmax(proba, 1)]
        assert (pred >= 0).all()
        m = metrics(yp, pred)
        rows.append(dict(permutation=k, seed=pm["seed"] + k, accuracy=m["accuracy"], macro_f1=m["macro_f1"], **{f"recall_AGG{c}": float(v) for c, v in zip(CLASSES, m["recall"])}))
    write_csv(f"{log.dir}/permutation_runs.csv", rows)
    acc = np.array([r["accuracy"] for r in rows]); f1 = np.array([r["macro_f1"] for r in rows])
    log(f"\n순열 {len(rows)}회 결과 (다수 클래스 정확도 참고값 {max(cnt.values()) / len(y):.3f}, macro-F1 우연 수준 약 {1 / 3:.2f}):")
    log(f"  accuracy : 평균 {acc.mean():.3f} | SD {acc.std():.3f} | 최소 {acc.min():.3f} | 최대 {acc.max():.3f}")
    log(f"  macro-F1 : 평균 {f1.mean():.3f} | SD {f1.std():.3f} | 최소 {f1.min():.3f} | 최대 {f1.max():.3f}")
    log("  회차별 accuracy : " + " ".join(f"{v:.2f}" for v in acc))
    lo, hi = cr["macro_f1_center"] - cr["macro_f1_tolerance"], cr["macro_f1_center"] + cr["macro_f1_tolerance"]
    ok_mean, ok_max = lo <= f1.mean() <= hi, f1.max() <= cr["macro_f1_max_single_run"]
    log(f"\n기준 점검 (제안값): 평균 macro-F1 {f1.mean():.3f} 이(가) [{lo:.2f}, {hi:.2f}] 안: {'예' if ok_mean else '아니오'} | 최대 macro-F1 {f1.max():.3f} <= {cr['macro_f1_max_single_run']}: {'예' if ok_max else '아니오'}")
    if not (ok_mean and ok_max):
        log("\n[멈춤] 제안한 기준을 벗어났다. 이것만으로 누수라고 단정할 수 없다 (기준은 제안값이고, 반복 20회의 분산, 작은 표본 때문일 수도 있다). 원인 미확인 상태로 보고한다.")
        sys.exit(4)
    log("\n통과: 라벨을 섞으면 우연 수준으로 떨어진다 (파이프라인 누수/버그의 징후는 이 검사에서는 보이지 않음).")
    log("한계: 이 검사는 이미지 단위 라벨 순열이라, 처리 조건을 알아보는 문제(라벨이 처리 조건에 90% 좌우됨)는 잡지 못한다.")
    bad = forbidden_check(cfg)
    log(f"금지어 검사: {'없음' if not bad else bad}")
    assert not bad


if __name__ == "__main__":
    main()
