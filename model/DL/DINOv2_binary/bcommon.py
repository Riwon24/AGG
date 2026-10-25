"""이진분류 확인 공통 함수: config, 이진 라벨 이미지 표, 이진 분류기, 이진 지표.

exp_group_cv/gcommon.py (이미지 표, 분할, 증강 특징 로딩), exp/common.py, exp/e03_run_cv.py 를 읽기 전용으로 불러 쓴다.
라벨 매핑: AGG0 -> 0 (음성), AGG1/AGG2 -> 1 (양성).
"""
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "exp_group_cv"))
from gcommon import (Logger, check_folds, image_vectors, load_config as _load_config, load_features, load_images, make_folds, pick_versions_tiles, sha256_file, stack_versions, write_csv)  # noqa: E402,F401
from e03_run_cv import build_settings  # noqa: E402,F401  (exp/ 의 설정 목록 생성)

BIN = np.array([0, 1])


def load_config(path=None):
    return _load_config(path or HERE / "config.yaml")


def to_binary(y3, cfg):
    return np.isin(y3, cfg["positive_labels"]).astype(int)


def fit_predict_proba_bin(clf_name, Xtr, ytr, Xte, cfg, guard):
    """exp/e03_run_cv.py 의 fit_predict_proba 와 같은 정의를 이진 라벨(0/1)에 맞춘 것: train 행만으로 (표준화 +) 분류기를 학습, test 행의 확률(2열) 반환."""
    wc = 0
    if clf_name == "logreg":
        sc = StandardScaler().fit(Xtr)
        assert sc.n_samples_seen_ == len(Xtr) and np.allclose(sc.mean_, Xtr.mean(0)), "scaler 가 train 이외의 정보를 썼음"
        guard["scaler_checked"] += 1
        lr = cfg["experiment"]["logreg"]
        clf = LogisticRegression(C=lr["C"], penalty="l2", class_weight=lr["class_weight"], max_iter=lr["max_iter"])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            clf.fit(sc.transform(Xtr), ytr)
        wc = sum(issubclass(x.category, ConvergenceWarning) for x in w)
        proba = clf.predict_proba(sc.transform(Xte))
    else:
        kn = cfg["experiment"]["knn"]
        clf = KNeighborsClassifier(n_neighbors=kn["k"], metric=kn["metric"], weights=kn["weights"], algorithm="brute").fit(Xtr, ytr)
        proba = clf.predict_proba(Xte)
    assert np.array_equal(clf.classes_, BIN), f"train 에 두 클래스가 모두 있어야 함: {clf.classes_}"
    guard["conv_warnings"] += wc
    return proba


def auroc(score, y):
    pos, neg = score[y == 1], score[y == 0]
    return float(np.mean([(1.0 if p > n else 0.5 if p == n else 0.0) for p in pos for n in neg]))


def metrics_bin(y, pred, score=None):
    """pooled 이진 지표: accuracy, balanced accuracy, macro-F1, recall(음성 AGG0, 양성 AGG1+2), 양성 F1, AUROC(score 가 있으면)"""
    y = np.asarray(y); pred = np.asarray(pred)
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum()); fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    rec1 = tp / (tp + fn) if tp + fn else 0.0; rec0 = tn / (tn + fp) if tn + fp else 0.0
    pre1 = tp / (tp + fp) if tp + fp else 0.0; pre0 = tn / (tn + fn) if tn + fn else 0.0
    f1_1 = 2 * pre1 * rec1 / (pre1 + rec1) if pre1 + rec1 else 0.0; f1_0 = 2 * pre0 * rec0 / (pre0 + rec0) if pre0 + rec0 else 0.0
    return dict(accuracy=(tp + tn) / len(y), balanced_acc=(rec0 + rec1) / 2, macro_f1=(f1_0 + f1_1) / 2, recall_neg=rec0, recall_pos=rec1, f1_pos=f1_1,
                auroc=auroc(score, y) if score is not None else float("nan"))
