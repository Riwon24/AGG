"""이진분류 공통 함수: config, 로그, 이미지 표, 분할, 특징 로딩, 합치기, 이진 분류기, 이진 지표 (이 폴더 안에서 완결, 다른 폴더의 코드를 import 하지 않는다).

DINOv2 실험(exp/common.py, exp/e03_run_cv.py)과 처리 조건/방법 단위 분할(exp_group_cv/gcommon.py)에서 필요한 함수를 그대로 옮겨 왔다 (동작 변경 없음).
라벨 매핑: AGG0 -> 0 (음성), AGG1/AGG2 -> 1 (양성).
"""
import csv
import hashlib
import sys
import warnings
from pathlib import Path

import numpy as np
import yaml
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
BIN = np.array([0, 1])


# ---------------------------------------------------------------- config / log / csv
def load_config(path=None):
    """config.yaml 을 읽는다. paths 의 상대경로는 config 파일이 있는 폴더 기준이고, 리스트는 후보 경로(존재하는 첫 번째를 쓴다)."""
    path = Path(path or HERE / "config.yaml").resolve()
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    base = path.parent

    def resolve(c):
        c = str(c)
        return str(Path(c).expanduser()) if c.startswith("~") else (c if c.startswith("/") else str((base / c).resolve()))

    for k, v in cfg["paths"].items():
        if isinstance(v, list):
            cands = [resolve(c) for c in v]
            cfg["paths"][k] = next((c for c in cands if Path(c).exists()), cands[0])
        else:
            cfg["paths"][k] = resolve(v)
    cfg["_config_path"] = str(path)
    return cfg


class Logger:
    """stdout + outputs/<script>/log.txt"""

    def __init__(self, outputs_dir, script_name):
        self.dir = Path(outputs_dir) / script_name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.fh = open(self.dir / "log.txt", "w", encoding="utf-8")
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    def __call__(self, *a):
        msg = " ".join(str(x) for x in a)
        print(msg); self.fh.write(msg + "\n"); self.fh.flush()

    def warn(self, *a):
        self("[WARN]", *a)


def write_csv(path, rows, fieldnames=None):
    rows = list(rows)
    fieldnames = fieldnames or list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- 이미지 표, 분할
def load_images(cfg):
    """이미지 60장 (image_id 순): ids, y(int, 3분류 라벨), metal, condition, method, temp"""
    rows = list(csv.DictReader(open(cfg["paths"]["conditions_csv"], encoding="utf-8-sig")))
    rows.sort(key=lambda r: r["image_id"])
    ids = [r["image_id"] for r in rows]
    return dict(ids=ids, y=np.array([int(r["label"]) for r in rows]), metal=[r["metal"] for r in rows], condition=[r["condition"] for r in rows],
                method=[r["method"] for r in rows], temp=[r["temp"] for r in rows])


def make_folds(img, unit_values):
    """단위(처리 조건/방법) 하나씩 통째로 test, 나머지가 train. 반환: [(단위 이름, train 인덱스, test 인덱스)] (단위 이름 정렬 순)"""
    folds = []
    for u in sorted(set(unit_values)):
        te = np.array([i for i, v in enumerate(unit_values) if v == u]); tr = np.array([i for i, v in enumerate(unit_values) if v != u])
        folds.append((u, tr, te))
    return folds


def check_folds(img, folds, name):
    """모든 이미지가 정확히 한 번 test, train/test 서로소, 시험 단위가 train 에 한 장도 없음, train 에 세 클래스가 모두 있음"""
    N = len(img["ids"]); cnt = np.zeros(N, int)
    for u, tr, te in folds:
        assert set(tr.tolist()).isdisjoint(te.tolist()) and len(tr) + len(te) == N, (name, u)
        assert set(np.unique(img["y"][tr]).tolist()) == {0, 1, 2}, f"{name} {u}: train 에 세 클래스가 없음"
        cnt[te] += 1
    assert (cnt == 1).all(), f"{name}: 모든 이미지가 정확히 한 번 test 여야 함"


# ---------------------------------------------------------------- 특징 로딩, 합치기
def aggregate(F, tiles_of, agg):
    """타일 특징 F (타일 수 x d) 를 이미지별로 합친다. tiles_of: 이미지마다 타일 행 번호 배열.
    mean | mean_std(평균 + 표준편차 ddof=0 이어붙임) | max"""
    rows = []
    for idx in tiles_of:
        T = F[idx]
        if agg == "mean":
            rows.append(T.mean(0))
        elif agg == "mean_std":
            rows.append(np.concatenate([T.mean(0), T.std(0, ddof=0)]))
        elif agg == "max":
            rows.append(T.max(0))
        else:
            raise ValueError(agg)
    return np.array(rows)


def load_features(out_dir, aug, ids):
    """한 실험의 특징을 버전별 구조로 통일: dict(tile_img, tiles_of, K1, F{특징: [버전별 (타일, d)]}, W (60, K1, 768)). 증강 없음은 K1 = 1."""
    pos = {i: k for k, i in enumerate(ids)}
    d = Path(out_dir) / "e01_extract_features"
    z = np.load(d / "features.npz")
    tindex = list(csv.DictReader(open(d / "tile_index.csv", encoding="utf-8-sig")))
    tile_img = np.array([pos[t["image_id"]] for t in tindex]); tiles_of = [np.where(tile_img == k)[0] for k in range(len(ids))]
    cls = z["tile_cls"].astype(np.float64); both = np.concatenate([z["tile_cls"], z["tile_patchmean"]], 1).astype(np.float64)
    if aug:
        row_of = z["tile_row_of"]; K1 = row_of.shape[1]
        F = {"cls": [cls[row_of[:, k]] for k in range(K1)], "cls+patchmean": [both[row_of[:, k]] for k in range(K1)]}
        W = z["whole_cls"].astype(np.float64)
    else:
        K1 = 1
        F = {"cls": [cls], "cls+patchmean": [both]}
        W = z["whole_cls"].astype(np.float64)[:, None, :]
    assert cls.shape[0] == len(tindex) if not aug else row_of.shape[0] == len(tindex)
    assert W.shape[0] == len(ids)
    return dict(tile_img=tile_img, tiles_of=tiles_of, K1=K1, F=F, W=W)


def image_vectors(L, f, agg):
    return np.stack([aggregate(L["F"][f][k], L["tiles_of"], agg) for k in range(L["K1"])], 1)


def stack_versions(X, y, idx):
    return X[idx].reshape(-1, X.shape[2]), np.repeat(np.asarray(y)[idx], X.shape[1])


def pick_versions_tiles(L, f, tile_rows, rng):
    v = rng.randint(0, L["K1"], len(tile_rows))
    out = np.empty((len(tile_rows), L["F"][f][0].shape[1]))
    for k in range(L["K1"]):
        m = v == k
        out[m] = L["F"][f][k][np.asarray(tile_rows)[m]]
    return out


def build_settings(cfg):
    """설정 목록: 이미지 단위(특징 x 합치기 x 분류기), SL(타일 단위), whole_image. 반환: (설정 리스트, primary 설정 id)"""
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


# ---------------------------------------------------------------- 이진 라벨, 분류기, 지표
def to_binary(y3, cfg):
    return np.isin(y3, cfg["positive_labels"]).astype(int)


def fit_predict_proba_bin(clf_name, Xtr, ytr, Xte, cfg, guard):
    """3분류 실험의 분류기와 같은 정의를 이진 라벨(0/1)에 맞춘 것: train 행만으로 (표준화 +) 분류기를 학습, test 행의 확률(2열) 반환."""
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
