"""e07: e05 와 같은 이미지 단위 라벨 순열 검사를 whole_image 와 SL(cls+patchmean)에도 20회씩 한다.

- 대상 설정은 config permutation_extra.settings (whole_image|cls|logreg, whole_image|cls|knn, SL|cls+patchmean|logreg).
- e05 와 같은 순열(seed+k 로 섞은 라벨, 클래스별 개수 유지), 순열마다 섞은 라벨로 새로 층화한 5-fold 1회, 같은 통과 기준을 쓴다.
  기준: 20회 평균 macro-F1 0.33±0.10 이내, 어느 회차도 macro-F1 0.55 초과 금지 (제안값이며 데이터에서 유도한 값이 아님).
- 기준을 어기면 누수라고 단정하지 않고 거기서 멈춰 보고한다 (종료 코드 4, 이후 설정은 실행하지 않음).
출력: permutation_other_runs.csv, log.txt
"""
import argparse
import csv
import sys
from collections import defaultdict

import numpy as np
from sklearn.model_selection import StratifiedKFold

from common import Logger, load_config, load_image_table, metrics, term_check, write_csv
from e03_run_cv import CLASSES, fit_predict_proba

SCRIPT = "e07_permutation_other_settings"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--output", help="outputs 폴더 (기본: config paths.outputs_dir)")
    a = ap.parse_args()
    cfg = load_config(a.config)
    out = a.output or cfg["paths"]["outputs_dir"]
    log = Logger(out, SCRIPT)
    log("config:", cfg["_config_path"])
    pm = cfg["permutation"]; cr = pm["criteria"]; sets = cfg["permutation_extra"]["settings"]
    lo, hi = cr["macro_f1_center"] - cr["macro_f1_tolerance"], cr["macro_f1_center"] + cr["macro_f1_tolerance"]
    log(f"대상 설정: {sets} | 순열 {pm['n_permutations']}회 (e05 와 같은 seed {pm['seed']}+k), 순열마다 섞은 라벨로 층화한 5-fold 1회")
    log(f"통과 기준(제안값, 데이터에서 유도한 값이 아님): 평균 macro-F1 [{lo:.2f}, {hi:.2f}] 이내, 어느 회차도 macro-F1 > {cr['macro_f1_max_single_run']} 금지. 어기면 누수 단정 없이 멈추고 보고")

    imgs = load_image_table(cfg); ids = [r["image_id"] for r in imgs]; y = np.array([r["label"] for r in imgs]); pos = {i: k for k, i in enumerate(ids)}
    z = np.load(f"{out}/e01_extract_features/features.npz")
    tile_idx = list(csv.DictReader(open(f"{out}/e01_extract_features/tile_index.csv", encoding="utf-8-sig")))
    tile_img = np.array([pos[t["image_id"]] for t in tile_idx]); tiles_of = [np.where(tile_img == k)[0] for k in range(len(ids))]
    feats = {"cls": z["tile_cls"].astype(np.float64), "cls+patchmean": np.concatenate([z["tile_cls"], z["tile_patchmean"]], 1).astype(np.float64)}
    whole = z["whole_cls"].astype(np.float64)
    cnt = {int(c): int((y == c).sum()) for c in CLASSES}
    log(f"이미지 {len(ids)}장, 클래스별 개수 {cnt}")

    guard = defaultdict(int); all_rows = []; status = 0
    for sid in sets:
        kind = sid.split("|")[0]
        rows = []
        for k in range(pm["n_permutations"]):
            rng = np.random.RandomState(pm["seed"] + k)
            yp = rng.permutation(y)                           # e05 와 같은 순열
            assert {int(c): int((yp == c).sum()) for c in CLASSES} == cnt
            skf = StratifiedKFold(n_splits=cfg["cv"]["n_splits"], shuffle=True, random_state=pm["seed"] + k)
            pred = np.full(len(y), -1)
            for tr, te in skf.split(np.zeros(len(yp)), yp):
                assert set(tr.tolist()).isdisjoint(te.tolist())
                if kind == "whole_image":
                    proba, _ = fit_predict_proba(sid.split("|")[2], whole[tr], yp[tr], whole[te], cfg, guard)
                elif kind == "SL":
                    F = feats[sid.split("|")[1]]
                    ttr = np.concatenate([tiles_of[i] for i in tr]); tte = np.concatenate([tiles_of[i] for i in te])
                    assert set(tile_img[ttr].tolist()).isdisjoint(tile_img[tte].tolist())
                    pt, _ = fit_predict_proba("logreg", F[ttr], yp[tile_img[ttr]], F[tte], cfg, guard)
                    proba = np.zeros((len(te), 3))
                    for j, i in enumerate(te):
                        proba[j] = pt[np.where(tile_img[tte] == i)[0]].mean(0)
                else:
                    raise ValueError(sid)
                pred[te] = CLASSES[np.argmax(proba, 1)]
            assert (pred >= 0).all()
            m = metrics(yp, pred)
            rows.append(dict(setting=sid, permutation=k, seed=pm["seed"] + k, accuracy=m["accuracy"], macro_f1=m["macro_f1"], **{f"recall_AGG{c}": float(v) for c, v in zip(CLASSES, m["recall"])}))
        all_rows += rows
        write_csv(f"{log.dir}/permutation_other_runs.csv", all_rows)      # 설정이 끝날 때마다 저장 (중간에 멈춰도 남음)
        acc = np.array([r["accuracy"] for r in rows]); f1 = np.array([r["macro_f1"] for r in rows])
        log(f"\n[{sid}] 순열 {len(rows)}회 (다수 클래스 정확도 참고값 {max(cnt.values()) / len(y):.3f}, macro-F1 우연 수준 약 0.33)")
        log(f"  accuracy : 평균 {acc.mean():.3f} | SD {acc.std():.3f} | 최소 {acc.min():.3f} | 최대 {acc.max():.3f}")
        log(f"  macro-F1 : 평균 {f1.mean():.3f} | SD {f1.std():.3f} | 최소 {f1.min():.3f} | 최대 {f1.max():.3f}")
        log("  회차별 accuracy : " + " ".join(f"{v:.2f}" for v in acc))
        log("  회차별 macro-F1 : " + " ".join(f"{v:.2f}" for v in f1))
        ok_mean = lo <= f1.mean() <= hi; ok_max = f1.max() <= cr["macro_f1_max_single_run"]
        log(f"  기준 점검 (제안값): 평균 macro-F1 {f1.mean():.3f} 이(가) [{lo:.2f}, {hi:.2f}] 안: {'예' if ok_mean else '아니오'} | 최대 {f1.max():.3f} <= {cr['macro_f1_max_single_run']}: {'예' if ok_max else '아니오'}")
        if not (ok_mean and ok_max):
            log(f"\n[멈춤] {sid} 가 제안한 기준을 벗어났다. 이것만으로 누수라고 단정할 수 없다 (기준은 제안값이고 반복 20회의 분산, 작은 표본, 순열마다 다른 fold 구성 때문일 수도 있다). 원인 미확인 상태로 보고하고 이후 설정은 실행하지 않는다.")
            status = 4
            break
        log("  -> 통과")
    log(f"\n누수 방지 assert: scaler 가 train 통계와 같은지 {guard['scaler_checked']}회 통과, 모든 fold 에서 train/test 이미지(SL 은 타일)가 서로소")
    if status:
        sys.exit(status)
    log("\n모든 대상 설정이 기준을 통과했다. 한계: 이 검사는 이미지 단위 라벨 순열이라 처리 조건을 알아보는 문제(라벨이 처리 조건에 90% 좌우됨)는 잡지 못한다.")
    bad = term_check(cfg)
    if bad:
        log("[실패] 금지어가 발견됨 (파일명:줄 번호만 기록):")
        for b in bad:
            log(f"    {b[0]}:{b[1]}")
        sys.exit(5)
    log(f"금지어 검사 통과: 금지어 {len(cfg.get('forbidden_terms', []))}개")


if __name__ == "__main__":
    main()
