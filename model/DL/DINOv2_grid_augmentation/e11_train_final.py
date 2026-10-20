"""e11 (배경 증강 판): 60장 전체(원본 + 배경 증강본 모두)로 primary(config 값 그대로)를 학습해 저장하고, 저장본을 다시 불러 예측이 같은지 확인한다. (새 실험이 아님)

primary = config experiment.primary (특징 cls / 합치기 mean / 분류기 logreg, C·class_weight·max_iter 는 config 값 그대로).
저장 (outputs/e11_final_model/):
  final_model.joblib        StandardScaler + LogisticRegression 객체 (sklearn 버전에 묶임)
  final_model_params.npz    버전 무관 파라미터: scaler_mean, scaler_scale, coef, intercept, classes (클래스 순서 AGG0/1/2)
  config_copy.yaml          학습에 쓴 config 사본
  meta.json                 클래스 순서, 입력 파일 sha256(features.npz 등), 버전, primary 설정, 수렴 경고 수
  training_images.csv       학습에 쓴 이미지 60장(순서, 라벨)
  train_predictions.csv / train_predictions.npz   학습 시점의 예측과 확률 (npz 는 정확한 비교용)
  inputs_sha256.csv         보존 대상 파일(splits.csv, cv_predictions.csv, features.npz 등)의 실행 전/후 sha256
--verify: 학습하지 않고 저장본만 읽어 60장에 예측을 다시 돌려 학습 시점 예측과 비교한다 (별도 프로세스로 실행).
주의: 이 60장 정확도는 학습 데이터에 대한 값이라 성능으로 쓰면 안 된다. 성능은 e03/e04 의 CV 결과만 쓴다.
읽기 전용 입력: splits.csv, cv_predictions.csv, features.npz (수정하지 않으며 해시로 확인한다).
"""
import argparse
import csv
import json
import shutil
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from common import Logger, image_vectors, load_aug_features, load_config, load_image_table, metrics, sha256_file, stack_versions, term_check, write_csv

SCRIPT = "e11_train_final"
CLASSES = np.array([0, 1, 2])
NOT_PERF = "주의: 이 60장 정확도는 학습 데이터에 대한 값이라 성능으로 쓰면 안 된다. 성능은 e03/e04 의 CV 결과만 쓴다."


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def load_inputs(cfg, out):
    """primary 설정에 따라 60장의 이미지 벡터를 버전별로 만든다: (ids, y, X), X 는 (60, 버전 수, d), 버전 0 = 원본 (e03/e05 와 같은 방식)."""
    L = load_aug_features(cfg, out)
    pr = cfg["experiment"]["primary"]
    return L["ids"], L["y"], image_vectors(L, pr["features"], pr["aggregator"])


def protected_files(out):
    return {"splits.csv": f"{out}/e00_make_splits/splits.csv", "cv_predictions.csv": f"{out}/e03_run_cv/cv_predictions.csv",
            "features.npz": f"{out}/e01_extract_features/features.npz", "tile_index.csv": f"{out}/e01_extract_features/tile_index.csv"}


def train(cfg, out, log, d):
    pr = cfg["experiment"]["primary"]; lr = cfg["experiment"]["logreg"]
    assert pr["classifier"] == "logreg", "e11 은 logreg primary 만 지원"
    log(f"primary: 특징 {pr['features']} / 합치기 {pr['aggregator']} / 분류기 {pr['classifier']} | logreg: C={lr['C']} class_weight={lr['class_weight']} max_iter={lr['max_iter']} (config 값 그대로)")
    prot = protected_files(out)
    before = {k: sha256_file(v) for k, v in prot.items()}
    log("보존 대상 파일 sha256 (실행 전): " + " | ".join(f"{k} {h[:16]}" for k, h in before.items()))

    ids, y, X = load_inputs(cfg, out)
    assert len(ids) == cfg["data"]["expected_images"] and X.shape[0] == 60
    X0 = X[:, 0]                                             # 원본 60장 벡터 (예측과 기록은 원본으로)
    Xfit, yfit = stack_versions(X, y, np.arange(len(y)))    # 학습: 60장의 모든 버전 (원본 + 배경 증강)
    log(f"학습 데이터: 이미지 {len(ids)}장 (AGG0 {int((y == 0).sum())} / AGG1 {int((y == 1).sum())} / AGG2 {int((y == 2).sum())}), 벡터 {X.shape} (이미지, 버전, 차원), 학습 행 {len(Xfit)}개 = 60장 x 버전 {X.shape[1]}개")

    sc = StandardScaler().fit(Xfit)
    assert sc.n_samples_seen_ == len(Xfit) and np.allclose(sc.mean_, Xfit.mean(0))
    clf = LogisticRegression(C=lr["C"], penalty="l2", class_weight=lr["class_weight"], max_iter=lr["max_iter"])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        clf.fit(sc.transform(Xfit), yfit)
    conv = sum(issubclass(x.category, ConvergenceWarning) for x in w)
    assert np.array_equal(clf.classes_, CLASSES), "클래스 순서가 AGG0/1/2 가 아님"
    proba = clf.predict_proba(sc.transform(X0)); pred = CLASSES[np.argmax(proba, 1)]
    log(f"학습 완료: 반복 횟수 n_iter={int(clf.n_iter_[0])}, 수렴 경고 {conv}건, 클래스 순서 {clf.classes_.tolist()} (= AGG0/AGG1/AGG2)")
    m = metrics(y, pred)
    log(f"학습 데이터에 대한 예측: accuracy {m['accuracy']:.3f} ({int((pred == y).sum())}/60), macro-F1 {m['macro_f1']:.3f}")
    log(NOT_PERF)

    joblib.dump(dict(scaler=sc, clf=clf, classes=CLASSES, primary=pr), d / "final_model.joblib")
    np.savez(d / "final_model_params.npz", scaler_mean=sc.mean_, scaler_scale=sc.scale_, coef=clf.coef_, intercept=clf.intercept_, classes=CLASSES)
    shutil.copyfile(cfg["_config_path"], d / "config_copy.yaml")      # copy 는 /mnt/c(권한 변경 불가)에서 chmod 에 실패하므로 내용만 복사
    np.savez(d / "train_predictions.npz", proba=proba, pred=pred, y=y)
    write_csv(d / "training_images.csv", [dict(order=i, image_id=ids[i], label=int(y[i])) for i in range(len(ids))])
    write_csv(d / "train_predictions.csv", [dict(image_id=ids[i], label=int(y[i]), pred=int(pred[i]), p0=float(proba[i, 0]), p1=float(proba[i, 1]), p2=float(proba[i, 2])) for i in range(len(ids))])

    after = {k: sha256_file(v) for k, v in prot.items()}
    assert before == after, "보존 대상 파일이 바뀌었음"
    log("보존 대상 파일 sha256 (실행 후): 모두 실행 전과 동일 -> splits.csv, cv_predictions.csv, features.npz 는 수정되지 않음")
    meta = dict(script=SCRIPT, class_order=["AGG0", "AGG1", "AGG2"], classes=CLASSES.tolist(), primary=pr,
                logreg=dict(C=lr["C"], class_weight=lr["class_weight"], max_iter=lr["max_iter"], n_iter=int(clf.n_iter_[0]), convergence_warnings=conv),
                n_images=len(ids), vector_dim=int(X.shape[2]), n_versions=int(X.shape[1]), n_train_rows=int(len(Xfit)), features_npz_sha256=after["features.npz"], tile_index_sha256=after["tile_index.csv"],
                config_sha256=sha256_file(cfg["_config_path"]), training_images=ids, sklearn=sklearn.__version__, numpy=np.__version__, python=sys.version.split()[0],
                note=NOT_PERF)
    (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(d / "inputs_sha256.csv", [dict(file=k, path=prot[k], sha256_before=before[k], sha256_after=after[k], unchanged=int(before[k] == after[k])) for k in prot]
              + [dict(file="config.yaml", path=cfg["_config_path"], sha256_before=meta["config_sha256"], sha256_after=sha256_file(cfg["_config_path"]), unchanged=1)])
    log(f"저장: final_model.joblib, final_model_params.npz, config_copy.yaml, meta.json, training_images.csv, train_predictions.csv/.npz, inputs_sha256.csv -> {d}")
    log(f"features.npz sha256 (meta.json 에 기록): {after['features.npz']}")


def verify(cfg, out, log, d):
    """저장본만 읽어서 60장에 예측을 다시 돌리고, 학습 시점의 예측과 같은지 확인한다."""
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    log("검증 모드: 학습하지 않고 저장된 파일만 불러서 예측한다 (별도 프로세스)")
    assert sha256_file(f"{out}/e01_extract_features/features.npz") == meta["features_npz_sha256"], "features.npz 가 학습 때와 다름"
    assert sha256_file(d / "config_copy.yaml") == meta["config_sha256"], "config 사본이 학습 때와 다름"
    log(f"입력 일치 확인: features.npz sha256 {meta['features_npz_sha256'][:16]}… 와 config 사본이 학습 때와 같음, 클래스 순서 {meta['class_order']}")
    ids, y, X = load_inputs(cfg, out)
    X0 = X[:, 0]                                             # 예측은 원본 60장 벡터로
    assert ids == meta["training_images"], "이미지 순서가 학습 때와 다름"
    saved = np.load(d / "train_predictions.npz")

    # (a) joblib 로 다시 불러서 예측
    obj = joblib.load(d / "final_model.joblib")
    proba_a = obj["clf"].predict_proba(obj["scaler"].transform(X0)); pred_a = obj["classes"][np.argmax(proba_a, 1)]
    same_pred_a = np.array_equal(pred_a, saved["pred"]); same_proba_a = np.array_equal(proba_a, saved["proba"])
    log(f"(a) joblib 로 불러서 60장 예측: 예측 클래스 완전히 같음 {same_pred_a} (다른 이미지 {int((pred_a != saved['pred']).sum())}장), 확률 값 완전히 같음 {same_proba_a} (최대 절대 차이 {np.abs(proba_a - saved['proba']).max():.3e})")
    # (b) 버전 무관 파라미터(npz)로 직접 계산
    p = np.load(d / "final_model_params.npz")
    assert np.array_equal(p["classes"], CLASSES)
    Z = (X0 - p["scaler_mean"]) / p["scaler_scale"]
    proba_b = softmax(Z @ p["coef"].T + p["intercept"]); pred_b = p["classes"][np.argmax(proba_b, 1)]
    same_pred_b = np.array_equal(pred_b, saved["pred"])
    log(f"(b) npz 파라미터로 직접 계산(표준화 -> 선형 -> softmax): 예측 클래스 완전히 같음 {same_pred_b} (다른 이미지 {int((pred_b != saved['pred']).sum())}장), 확률 최대 절대 차이 {np.abs(proba_b - saved['proba']).max():.3e}")
    csvp = list(csv.DictReader(open(d / "train_predictions.csv", encoding="utf-8-sig")))
    assert [r["image_id"] for r in csvp] == ids and [int(r["pred"]) for r in csvp] == pred_a.tolist()
    log("train_predictions.csv 의 예측도 위와 같음")
    ok = same_pred_a and same_proba_a and same_pred_b
    log("결과: " + ("저장한 모델을 다시 불러도 60장 예측이 학습 시점과 완전히 같음 (예측 클래스 100% 일치)" if ok else "불일치 발견!"))
    m = metrics(y, pred_a)
    log(f"(참고) 이 60장 accuracy {m['accuracy']:.3f}, macro-F1 {m['macro_f1']:.3f}")
    log(NOT_PERF)
    if not ok:
        sys.exit(6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--output", help="outputs 폴더 (기본: config paths.outputs_dir)")
    ap.add_argument("--verify", action="store_true", help="학습하지 않고 저장본만 불러 예측이 같은지 확인")
    a = ap.parse_args()
    cfg = load_config(a.config)
    out = a.output or cfg["paths"]["outputs_dir"]
    d = Path(out) / "e11_final_model"; d.mkdir(parents=True, exist_ok=True)
    log = Logger(out, "e11_final_model") if not a.verify else None
    if a.verify:                                    # 검증 로그는 학습 로그를 덮어쓰지 않고 따로 저장
        class L(Logger):
            def __init__(s):
                s.dir = d; s.fh = open(d / "verify_log.txt", "w", encoding="utf-8")
        log = L()
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    log("config:", cfg["_config_path"], "| 모드:", "검증(--verify)" if a.verify else "학습")
    (verify if a.verify else train)(cfg, out, log, d)
    bad = term_check(cfg)
    if bad:
        log("[실패] 금지어가 발견됨 (파일명:줄 번호만 기록):")
        for b in bad:
            log(f"    {b[0]}:{b[1]}")
        sys.exit(5)
    log(f"금지어 검사 통과: 금지어 {len(cfg.get('forbidden_terms', []))}개")


if __name__ == "__main__":
    main()
