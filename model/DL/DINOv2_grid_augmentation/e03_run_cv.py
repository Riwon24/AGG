"""e03 (배경 증강 판): e00 의 분할 그대로 모든 설정 조합을 반복/fold 별로 실행하고 예측을 저장한다. 학습에 배경 증강본을 함께 쓰고(아래), 평가는 원본으로만 한다.

설정 목록은 config.yaml 에서 만든다 (하드코딩 없음):
  tiles       : 타일 특징(cls | cls+patchmean) -> 이미지 벡터(mean | mean_std | max) -> 분류기(logreg | knn)
  SL          : 타일 단위 logreg(이미지 라벨을 타일마다 붙임) -> test 이미지의 타일 확률 평균 -> 이미지 예측 (특징 cls, cls+patchmean)
  whole_image : 마스크 bbox 크롭(224 resize) cls -> 분류기 (logreg | knn)
분류기: logreg = 다항 로지스틱 회귀 L2, C=1.0, class_weight=balanced, StandardScaler 는 train 에서만 fit
        knn    = k=5, cosine 거리, 거리 가중, 표준화 없음 (SEM 논문은 Gaussian kernel 가중이었으나 여기서는 단순 거리 가중)
        mean_std 의 표준편차는 ddof=0
누수 방지: 모든 통계(표준화 등)는 fit_predict 안에서 train 행만으로 계산한다. 함수는 train/test 행 번호만 받고, 입력 이미지 id 집합이
        서로소인지, scaler 가 train 통계와 같은지 assert 한다.
출력: cv_predictions.csv, cv_metrics_per_repeat.csv, cv_summary.csv, settings.csv, log.txt
"""
import argparse
import csv
import warnings
from collections import defaultdict

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from common import Logger, aggregate, image_vectors, load_aug_features, load_config, load_image_table, metrics, pick_versions_tiles, sha256_file, stack_versions, write_csv

SCRIPT = "e03_run_cv"
CLASSES = np.array([0, 1, 2])


def build_settings(cfg):
    ex = cfg["experiment"]; s = []
    for f in ex["features"]:
        for a in ex["aggregators"]:
            for c in ex["classifiers"]:
                s.append(dict(id=f"tiles|{f}|{a}|{c}", kind="tiles", features=f, aggregator=a, classifier=c))
    for f in ex["sl_features"]:
        s.append(dict(id=f"SL|{f}|logreg", kind="SL", features=f, aggregator="tile_proba_mean", classifier="logreg"))
    for c in ex["whole_image_classifiers"]:
        s.append(dict(id=f"whole_image|cls|{c}", kind="whole_image", features="cls", aggregator="-", classifier=c))
    p = ex["primary"]
    s_ids = [x["id"] for x in s]
    pid = f"tiles|{p['features']}|{p['aggregator']}|{p['classifier']}"
    assert pid in s_ids, f"primary 설정이 조합 목록에 없음: {pid}"
    return s, pid


def make_clf(name, cfg):
    ex = cfg["experiment"]
    if name == "logreg":
        lr = ex["logreg"]
        return LogisticRegression(C=lr["C"], penalty="l2", class_weight=lr["class_weight"], max_iter=lr["max_iter"])
    kn = ex["knn"]
    return KNeighborsClassifier(n_neighbors=kn["k"], metric=kn["metric"], weights=kn["weights"], algorithm="brute")


def fit_predict_proba(clf_name, Xtr, ytr, Xte, cfg, guard):
    """train 행만으로 (표준화 +) 분류기를 학습하고 test 행의 클래스 확률(3열)을 반환한다."""
    warn_cnt = 0
    if clf_name == "logreg":
        sc = StandardScaler().fit(Xtr)
        assert sc.n_samples_seen_ == len(Xtr) and np.allclose(sc.mean_, Xtr.mean(0)), "scaler 가 train 이외의 정보를 썼음"
        guard["scaler_checked"] += 1
        Ztr, Zte = sc.transform(Xtr), sc.transform(Xte)
        clf = make_clf("logreg", cfg)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            clf.fit(Ztr, ytr)
        warn_cnt = sum(issubclass(x.category, ConvergenceWarning) for x in w)
        proba = clf.predict_proba(Zte)
    else:
        clf = make_clf("knn", cfg).fit(Xtr, ytr)    # 표준화 없음
        proba = clf.predict_proba(Xte)
    assert np.array_equal(clf.classes_, CLASSES), f"train 에 3개 클래스가 모두 있어야 함: {clf.classes_}"
    return proba, warn_cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--output", help="outputs 폴더 (기본: config paths.outputs_dir)")
    a = ap.parse_args()
    cfg = load_config(a.config)
    out = a.output or cfg["paths"]["outputs_dir"]
    log = Logger(out, SCRIPT)
    log("config:", cfg["_config_path"])
    ex = cfg["experiment"]
    log(f"logreg: C={ex['logreg']['C']} class_weight={ex['logreg']['class_weight']} max_iter={ex['logreg']['max_iter']}(수렴용, 명세에 없는 값) | knn: k={ex['knn']['k']} metric={ex['knn']['metric']} weights={ex['knn']['weights']} (표준화 없음)")
    log("mean_std 의 표준편차: ddof=0")

    imgs = load_image_table(cfg); ids = [r["image_id"] for r in imgs]; y = np.array([r["label"] for r in imgs]); pos = {i: k for k, i in enumerate(ids)}
    L = load_aug_features(cfg, out)                       # 버전별(0 = 원본, 1..K = 배경 증강) 특징
    assert L["ids"] == ids, "특징의 이미지 순서가 이미지 표와 다름"
    tile_img, tiles_of, K1 = L["tile_img"], L["tiles_of"], L["K1"]
    log(f"이미지 {len(ids)}장, 타일 {len(tile_img)}개 (이미지당 {min(len(t) for t in tiles_of)}~{max(len(t) for t in tiles_of)}개), 버전 {K1}개 (원본 + 배경 증강 {K1 - 1}개)")
    log(f"학습: 이미지 단위 설정은 학습 이미지의 모든 버전({K1}개)을 학습 행으로, SL 은 학습 타일마다 {K1}개 버전 중 하나를 무작위로(시드 {cfg['aug']['seed']}+1000*반복+fold) 쓴다. 평가는 항상 원본(k=0).")
    feats = L["F"]; whole = L["W"]                         # feats[특징][k]: (타일, d), whole: (60, K1, 768)
    _vec = {}

    def image_matrix(f, agg):
        if (f, agg) not in _vec:
            _vec[(f, agg)] = image_vectors(L, f, agg)      # (60, K1, d)
        return _vec[(f, agg)]

    # 분할
    sp = list(csv.DictReader(open(f"{out}/e00_make_splits/splits.csv", encoding="utf-8-sig")))
    folds = defaultdict(lambda: {"train": [], "test": []})
    for r in sp:
        folds[(int(r["repeat"]), int(r["fold"]))][r["role"]].append(pos[r["image_id"]])
    reps = sorted({k[0] for k in folds}); nf = sorted({k[1] for k in folds})
    log(f"분할: 반복 {len(reps)}회 x {len(nf)} fold (e00 splits.csv)")

    settings, pid = build_settings(cfg)
    write_csv(f"{log.dir}/settings.csv", [dict(**s, primary=int(s["id"] == pid)) for s in settings])
    log(f"설정 {len(settings)}개 (primary = {pid}):")
    for s in settings:
        log(f"   {'*' if s['id'] == pid else ' '} {s['id']}")

    guard = defaultdict(int)
    pred_rows, met_rows = [], []
    for s in settings:
        conv = 0
        P = np.zeros((len(reps), len(ids)), int)
        for r in reps:
            for f in nf:
                tr, te = np.array(folds[(r, f)]["train"]), np.array(folds[(r, f)]["test"])
                assert set(tr.tolist()).isdisjoint(te.tolist()) and len(tr) + len(te) == len(ids)
                if s["kind"] == "SL":
                    F = feats[s["features"]]
                    ttr = np.concatenate([tiles_of[i] for i in tr]); tte = np.concatenate([tiles_of[i] for i in te])
                    assert set(tile_img[ttr].tolist()).isdisjoint(tile_img[tte].tolist()), "타일이 train/test 양쪽에 있음"
                    Ftr = pick_versions_tiles(L, s["features"], ttr, np.random.RandomState(cfg["aug"]["seed"] + 1000 * r + f))
                    proba_t, wc = fit_predict_proba("logreg", Ftr, y[tile_img[ttr]], F[0][tte], cfg, guard)
                    proba = np.zeros((len(te), 3))
                    for k, i in enumerate(te):
                        proba[k] = proba_t[np.where(tile_img[tte] == i)[0]].mean(0)
                else:
                    X = whole if s["kind"] == "whole_image" else image_matrix(s["features"], s["aggregator"])       # (60, K1, d)
                    Xtr, ytr = stack_versions(X, y, tr)                                                            # 학습 이미지의 모든 버전
                    proba, wc = fit_predict_proba(s["classifier"], Xtr, ytr, X[te][:, 0], cfg, guard)             # 평가는 원본
                conv += wc
                pred = CLASSES[np.argmax(proba, 1)]
                P[r, te] = pred
                for k, i in enumerate(te):
                    pred_rows.append(dict(setting=s["id"], repeat=r, fold=f, image_id=ids[i], label=int(y[i]), pred=int(pred[k]), p0=round(float(proba[k, 0]), 6), p1=round(float(proba[k, 1]), 6), p2=round(float(proba[k, 2]), 6)))
        for r in reps:       # 반복마다 fold 를 합친(pooled) 예측으로 지표
            m = metrics(y, P[r])
            met_rows.append(dict(setting=s["id"], repeat=r, accuracy=m["accuracy"], macro_f1=m["macro_f1"], **{f"recall_AGG{c}": float(v) for c, v in zip(CLASSES, m["recall"])}))
        s["conv_warnings"] = conv
        mm = [x for x in met_rows if x["setting"] == s["id"]]
        log(f"  {s['id']:<34} acc {np.mean([x['accuracy'] for x in mm]):.3f}±{np.std([x['accuracy'] for x in mm]):.3f} | macro-F1 {np.mean([x['macro_f1'] for x in mm]):.3f}±{np.std([x['macro_f1'] for x in mm]):.3f} | 수렴 경고 {conv}")

    log(f"\n누수 방지 확인: scaler 가 train 통계와 같은지 assert {guard['scaler_checked']}회 통과, 모든 fold 에서 train/test 이미지(및 SL 의 타일)가 서로소임을 assert")
    log("근거: fit_predict_proba 는 train 행(Xtr, ytr)만 받아 StandardScaler 와 분류기를 학습하고, test 행은 transform/predict 에만 쓴다. kNN 은 표준화/정규화 통계를 쓰지 않는다. 이미지 특징은 e01 에서 모델이 이미지 단위로 독립 계산한 값이다.")
    nconv = sum(s["conv_warnings"] for s in settings)
    log(f"logreg 수렴 경고 합계: {nconv}건 " + ("(없음)" if nconv == 0 else "(max_iter 부족 가능성; 설정별 건수는 위 줄 참조)"))

    # ---------- 저장 ----------
    write_csv(f"{log.dir}/cv_predictions.csv", pred_rows)
    write_csv(f"{log.dir}/cv_metrics_per_repeat.csv", met_rows)
    summ = []
    for s in settings:
        mm = [x for x in met_rows if x["setting"] == s["id"]]
        row = dict(setting=s["id"], kind=s["kind"], features=s["features"], aggregator=s["aggregator"], classifier=s["classifier"], primary=int(s["id"] == pid), conv_warnings=s["conv_warnings"])
        for k in ("accuracy", "macro_f1", "recall_AGG0", "recall_AGG1", "recall_AGG2"):
            row[f"{k}_mean"] = float(np.mean([x[k] for x in mm])); row[f"{k}_sd"] = float(np.std([x[k] for x in mm]))
        summ.append(row)
    write_csv(f"{log.dir}/cv_summary.csv", summ)
    log(f"\n예측 {len(pred_rows)}행 (설정 {len(settings)} x 반복 {len(reps)} x 이미지 {len(ids)}), 저장: {log.dir}")
    log(f"cv_predictions.csv sha256: {sha256_file(f'{log.dir}/cv_predictions.csv')}")


if __name__ == "__main__":
    main()
