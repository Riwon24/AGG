"""b04: e00 의 이미지 단위 층화 5-fold x 10 반복 분할 그대로, 설정 3개 x 팔 4개를 실행하고 예측을 저장한다.

이미지 벡터 = 버전 k (0 = 원본, 1..K = 증강본) 의 타일 특징을 이미지 안에서 평균 (aggregator mean).
팔 (config experiment.arms):
  R0 원본으로 학습 -> 원본으로 평가 (증강 없음)
  R1 원본+증강본으로 학습 -> 원본으로 평가  (primary 팔)
  R2 원본+증강본으로 학습 -> 증강본으로 평가 (증강본 K개의 확률 평균 -> 이미지 예측)
  R3 원본으로 학습 -> 증강본으로 평가 (K개 확률 평균)
누수 방지: train 이미지와 test 이미지는 서로소이고 한 이미지의 모든 버전(원본/증강본)은 같은 쪽에만 들어간다. 표준화는 train 행(증강 행 포함)만으로 fit.
분류기 정의는 exp/e03_run_cv.py 의 fit_predict_proba (logreg: 표준화 + L2 C=1 balanced / knn: k=5 cosine distance, 표준화 없음)를 그대로 쓴다.
출력: cv_predictions.csv, cv_metrics_per_repeat.csv, cv_summary.csv, settings.csv, log.txt
"""
import argparse
import csv
from collections import defaultdict

import numpy as np

from bg_common import Logger, aggregate, extraction_rows, inputs, load_config, metrics, sha256_file, write_csv
from e03_run_cv import CLASSES, fit_predict_proba  # exp/ 의 분류기 구현 (읽기 전용 import)

SCRIPT = "b04_run_cv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    out = cfg["paths"]["outputs_dir"]
    log = Logger(out, SCRIPT)
    log("config:", cfg["_config_path"])
    ex = cfg["experiment"]
    K = int(cfg["bgaug"]["n_aug"])
    log(f"logreg: {ex['logreg']} | knn: {ex['knn']} (표준화 없음) | 합치기 {ex['aggregator']} | 버전 k=0(원본), 1..{K}(증강)")

    imgs, tiles = inputs(cfg)
    ids = [r["image_id"] for r in imgs]; y = np.array([r["label"] for r in imgs]); pos = {i: k for k, i in enumerate(ids)}
    z = np.load(f"{out}/b03_extract_features/features.npz")
    idx = extraction_rows(cfg, tiles)                        # 분석 타일 -> b03 특징 배열의 행 (타일 목록은 config tiles 규칙으로 고른 부분집합)
    tile_img = np.array([pos[t["image_id"]] for t in tiles]); tiles_of = [np.where(tile_img == k)[0] for k in range(len(ids))]
    row_of = z["row_of"][idx]
    assert row_of.shape == (len(tiles), K + 1)
    log(f"분석 타일 {len(tiles)}개 (규칙 {cfg['tiles']['select']}, 시편 픽셀 0 제외 {cfg['tiles'].get('drop_zero_mask', False)}), 이미지당 {min(len(x) for x in tiles_of)}~{max(len(x) for x in tiles_of)}개")
    feats = {"cls": z["cls"].astype(np.float64), "cls+patchmean": np.concatenate([z["cls"], z["patchmean"]], 1).astype(np.float64)}
    X = {f: np.stack([aggregate(feats[f][row_of[:, k]], tiles_of, ex["aggregator"]) for k in range(K + 1)], 1) for f in feats}   # (60, K+1, d)
    log(f"이미지 {len(ids)}장, 타일 {len(tiles)}개, 이미지 벡터 {{ {', '.join(f'{f}: {v.shape}' for f, v in X.items())} }}")
    # 증강본 벡터가 원본과 얼마나 다른지 (기록용)
    for f in X:
        d_ = np.linalg.norm(X[f][:, 1:] - X[f][:, [0]], axis=2) / np.linalg.norm(X[f][:, [0]], axis=2)
        log(f"  {f}: 증강본-원본 이미지 벡터 상대 거리 중앙값 {np.median(d_):.4f}, 최대 {d_.max():.4f}")

    sp = list(csv.DictReader(open(cfg["paths"]["splits_csv"], encoding="utf-8-sig")))
    folds = defaultdict(lambda: {"train": [], "test": []})
    for r in sp:
        folds[(int(r["repeat"]), int(r["fold"]))][r["role"]].append(pos[r["image_id"]])
    reps = sorted({k[0] for k in folds}); nf = sorted({k[1] for k in folds})
    assert len(reps) == cfg["cv"]["n_repeats"] and len(nf) == cfg["cv"]["n_splits"]
    log(f"분할: 반복 {len(reps)}회 x {len(nf)} fold (기존 e00 splits.csv, sha256 {sha256_file(cfg['paths']['splits_csv'])[:16]}…)")

    settings = [dict(id=f"tiles|{s['features']}|{ex['aggregator']}|{s['classifier']}", **s) for s in ex["settings"]]
    pid = f"tiles|{ex['primary']['features']}|{ex['primary']['aggregator']}|{ex['primary']['classifier']}"
    assert pid == settings[0]["id"]
    write_csv(f"{log.dir}/settings.csv", [dict(**s, primary=int(s["id"] == pid)) for s in settings])
    arms = ex["arms"]
    log(f"설정 {len(settings)}개 (primary = {pid}) x 팔 {list(arms)} (primary 팔 {ex['primary_arm']})")

    def versions(spec, kind):
        return [0] if spec[kind] == "orig" else (list(range(K + 1)) if spec[kind] == "all" else list(range(1, K + 1)))

    guard = defaultdict(int)
    pred_rows, met_rows, nconv = [], [], defaultdict(int)
    for s in settings:
        for arm, spec in arms.items():
            vt, ve = versions(spec, "train"), versions(spec, "test")
            P = np.zeros((len(reps), len(ids)), int)
            for r in reps:
                for fo in nf:
                    tr, te = np.array(folds[(r, fo)]["train"]), np.array(folds[(r, fo)]["test"])
                    assert set(tr.tolist()).isdisjoint(te.tolist()) and len(tr) + len(te) == len(ids)
                    Xtr = X[s["features"]][tr][:, vt].reshape(-1, X[s["features"]].shape[2]); ytr = np.repeat(y[tr], len(vt))
                    assert len(Xtr) == len(tr) * len(vt) and set(np.repeat(tr, len(vt)).tolist()).isdisjoint(te.tolist())
                    # 한 번 학습해서 모든 평가 버전(k)을 예측하고 확률을 평균한다
                    Xte_all = X[s["features"]][te][:, ve].reshape(-1, X[s["features"]].shape[2])
                    proba_all, wc = fit_predict_proba(s["classifier"], Xtr, ytr, Xte_all, cfg, guard)
                    nconv[(s["id"], arm)] += wc
                    proba = proba_all.reshape(len(te), len(ve), 3).mean(1)
                    pred = CLASSES[np.argmax(proba, 1)]
                    P[r, te] = pred
                    for q, i in enumerate(te):
                        pred_rows.append(dict(setting=s["id"], arm=arm, repeat=r, fold=fo, image_id=ids[i], label=int(y[i]), pred=int(pred[q]),
                                              p0=round(float(proba[q, 0]), 6), p1=round(float(proba[q, 1]), 6), p2=round(float(proba[q, 2]), 6)))
            for r in reps:
                m = metrics(y, P[r])
                met_rows.append(dict(setting=s["id"], arm=arm, repeat=r, accuracy=m["accuracy"], macro_f1=m["macro_f1"], **{f"recall_AGG{c}": float(v) for c, v in zip(CLASSES, m["recall"])}))
            mm = [x for x in met_rows if x["setting"] == s["id"] and x["arm"] == arm]
            log(f"  {s['id']:<30} {arm}: acc {np.mean([x['accuracy'] for x in mm]):.3f}±{np.std([x['accuracy'] for x in mm]):.3f} | macro-F1 {np.mean([x['macro_f1'] for x in mm]):.3f}±{np.std([x['macro_f1'] for x in mm]):.3f}")
    log(f"\n누수 방지 확인: scaler 가 train 통계와 같은지 assert {guard['scaler_checked']}회 통과, 모든 fold 에서 train/test 이미지가 서로소 (한 이미지의 모든 버전은 같은 쪽)")
    log("근거: fit_predict_proba 는 train 행만으로 StandardScaler 와 분류기를 학습하고 test 행은 transform/predict 에만 쓴다. 증강본(학습 행)은 train 이미지에서만 만든 버전이다.")
    log(f"logreg 수렴 경고 합계: {sum(nconv.values())}건")

    write_csv(f"{log.dir}/cv_predictions.csv", pred_rows)
    write_csv(f"{log.dir}/cv_metrics_per_repeat.csv", met_rows)
    summ = []
    for s in settings:
        for arm in arms:
            mm = [x for x in met_rows if x["setting"] == s["id"] and x["arm"] == arm]
            row = dict(setting=s["id"], arm=arm, primary=int(s["id"] == pid and arm == ex["primary_arm"]))
            for k in ("accuracy", "macro_f1", "recall_AGG0", "recall_AGG1", "recall_AGG2"):
                row[f"{k}_mean"] = float(np.mean([x[k] for x in mm])); row[f"{k}_sd"] = float(np.std([x[k] for x in mm]))
            summ.append(row)
    write_csv(f"{log.dir}/cv_summary.csv", summ)
    log(f"\n예측 {len(pred_rows)}행 저장: {log.dir} | cv_predictions.csv sha256 {sha256_file(f'{log.dir}/cv_predictions.csv')}")


if __name__ == "__main__":
    main()
