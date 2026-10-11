"""e05: 이미지 단위 라벨 순열 검사 (primary 설정을 20번).

- 이미지 단위로 라벨을 섞는다 (클래스별 개수 유지). 순열마다 섞은 라벨로 fold 를 새로 나누고(층화) 5-fold 1회를 돈다.
- 통과 기준: 20회 평균 macro-F1 이 0.33±0.10 이내이고 어느 회차도 macro-F1 이 0.55 를 넘지 않는다.
  **이 기준은 제안값이지 데이터에서 유도한 값이 아니다 (config permutation.criteria).**
- 기준을 어기면 누수라고 단정하지 않고 멈춰서 보고한다 (종료 코드 4, 원인 미확인).
- 20회의 평균, 최댓값, 전체 분포를 보고한다. 순열 검사는 파이프라인 버그나 누수를 잡을 뿐, 처리 조건 인식 문제는 잡지 못한다.
출력: permutation_runs.csv, log.txt
"""
import argparse
import sys

import numpy as np
from sklearn.model_selection import StratifiedKFold

from common import Logger, aggregate, load_config, load_image_table, metrics, term_check, write_csv
from e03_run_cv import CLASSES, fit_predict_proba
from collections import defaultdict
import csv
from common import write_bytes_retry  # noqa: F401

SCRIPT = "e05_sanity_permutation"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--output", help="outputs 폴더 (기본: config paths.outputs_dir)")
    a = ap.parse_args()
    cfg = load_config(a.config)
    out = a.output or cfg["paths"]["outputs_dir"]
    log = Logger(out, SCRIPT)
    log("config:", cfg["_config_path"])
    pm = cfg["permutation"]; cr = pm["criteria"]; pr = cfg["experiment"]["primary"]
    log(f"primary = 특징 {pr['features']} / 합치기 {pr['aggregator']} / 분류기 {pr['classifier']} | 순열 {pm['n_permutations']}회, seed {pm['seed']}, 5-fold 1회씩 (fold 는 섞은 라벨로 층화)")
    log(f"통과 기준(제안값, 데이터에서 유도한 값이 아님): 평균 macro-F1 {cr['macro_f1_center']}±{cr['macro_f1_tolerance']} 이내, 어느 회차도 macro-F1 > {cr['macro_f1_max_single_run']} 금지")

    imgs = load_image_table(cfg); ids = [r["image_id"] for r in imgs]; y = np.array([r["label"] for r in imgs]); pos = {i: k for k, i in enumerate(ids)}
    z = np.load(f"{out}/e01_extract_features/features.npz")
    tile_idx = list(csv.DictReader(open(f"{out}/e01_extract_features/tile_index.csv", encoding="utf-8-sig")))
    tile_img = np.array([pos[t["image_id"]] for t in tile_idx]); tiles_of = [np.where(tile_img == k)[0] for k in range(len(ids))]
    F = z["tile_cls"].astype(np.float64) if pr["features"] == "cls" else np.concatenate([z["tile_cls"], z["tile_patchmean"]], 1).astype(np.float64)
    X = aggregate(F, tiles_of, pr["aggregator"])
    cnt = {int(c): int((y == c).sum()) for c in CLASSES}
    log(f"이미지 {len(ids)}장, 클래스별 개수 {cnt}, 벡터 차원 {X.shape[1]}")

    guard = defaultdict(int); rows = []
    for k in range(pm["n_permutations"]):
        rng = np.random.RandomState(pm["seed"] + k)
        yp = rng.permutation(y)
        assert {int(c): int((yp == c).sum()) for c in CLASSES} == cnt, "순열 후 클래스별 개수가 달라짐"
        skf = StratifiedKFold(n_splits=cfg["cv"]["n_splits"], shuffle=True, random_state=pm["seed"] + k)
        pred = np.full(len(y), -1)
        for tr, te in skf.split(np.zeros(len(yp)), yp):
            assert set(tr.tolist()).isdisjoint(te.tolist())
            proba, _ = fit_predict_proba(pr["classifier"], X[tr], yp[tr], X[te], cfg, guard)
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
    log("  회차별 macro-F1 : " + " ".join(f"{v:.2f}" for v in f1))

    lo, hi = cr["macro_f1_center"] - cr["macro_f1_tolerance"], cr["macro_f1_center"] + cr["macro_f1_tolerance"]
    ok_mean = lo <= f1.mean() <= hi; ok_max = f1.max() <= cr["macro_f1_max_single_run"]
    log(f"\n기준 점검 (제안값): 평균 macro-F1 {f1.mean():.3f} 이(가) [{lo:.2f}, {hi:.2f}] 안: {'예' if ok_mean else '아니오'} | 최대 macro-F1 {f1.max():.3f} <= {cr['macro_f1_max_single_run']}: {'예' if ok_max else '아니오'}")
    if not (ok_mean and ok_max):
        log("\n[멈춤] 제안한 기준을 벗어났다. 이것만으로 누수라고 단정할 수 없다 (기준은 제안값이고, 반복 20회의 분산, 작은 표본, 순열마다 다른 fold 구성 때문일 수도 있다). 원인 미확인 상태로 보고한다.")
        sys.exit(4)
    log("\n통과: 라벨을 섞으면 우연 수준으로 떨어진다 (파이프라인 누수/버그의 징후는 이 검사에서는 보이지 않음).")
    log("한계: 이 검사는 이미지 단위 라벨 순열이라, 처리 조건을 알아보는 문제(라벨이 처리 조건에 90% 좌우됨)는 잡지 못한다. 그건 e02 의 처리 조건 암기 기준선과 e04 의 불일치 이미지 결과로 본다.")

    bad = term_check(cfg)
    if bad:
        log("[실패] 금지어가 발견됨 (내용을 로그에 다시 쓰면 다음 검사에 걸리므로 파일명:줄 번호만 기록):")
        for b in bad:
            log(f"    {b[0]}:{b[1]}")
        sys.exit(5)
    log(f"금지어 검사 통과: 금지어 {len(cfg.get('forbidden_terms', []))}개 (config forbidden_terms)")


if __name__ == "__main__":
    main()
