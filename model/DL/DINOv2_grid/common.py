"""모델·실험 공통 유틸: config, 로그, 입력 로딩, 조건 파싱, 지표, 부트스트랩, 금지어 검사.

용어 (사용자 확인): 1단계 폴더 = 금속 종류(6), 2단계 폴더 = 처리 조건(10) = 처리 방법(5) x 온도(2).
"""
import csv
import hashlib
import re
import sys
from pathlib import Path

import numpy as np
import yaml

EXP_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------- config / log
def load_config(path=None):
    path = Path(path or EXP_DIR / "config.yaml").resolve()
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    base = path.parent

    def resolve(c):
        c = str(c)
        return str(Path(c).expanduser()) if c.startswith("~") else (c if c.startswith("/") else str((base / c).resolve()))

    for k, v in cfg["paths"].items():
        if isinstance(v, list):     # 후보 목록(Windows/WSL 경로): 존재하는 첫 번째를 쓴다
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


def write_bytes_retry(path, data, retries=8):
    """Windows 에서 열려 있는 파일을 덮어쓸 때 간헐적으로 OSError 가 나서 재시도한다."""
    import time
    for i in range(retries):
        try:
            Path(path).write_bytes(data); return
        except OSError:
            if i == retries - 1:
                raise RuntimeError(f"파일을 쓸 수 없음 (다른 프로그램에서 열려 있을 수 있음): {path}")
            time.sleep(0.5 * (i + 1))


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


# ---------------------------------------------------------------- 입력 / 조건 파싱 (R1)
def parse_condition(name):
    """2단계 폴더명 -> (처리 방법, 온도). '-' 가 있으면 첫 '-' 로 나누고, 없으면 ^(NPH)(\\d{4})$."""
    if "-" in name:
        a, b = name.split("-", 1)
        return a, b
    m = re.match(r"^(NPH)(\d{4})$", name)
    if not m:
        raise ValueError(f"처리 조건 이름을 해석할 수 없음: {name}")
    return m.group(1), m.group(2)


def image_parts(image_id):
    """'BASE/NPH1035/x10#1 - AGG2' -> (금속 종류, 처리 조건)"""
    parts = image_id.split("/")
    assert len(parts) == 3, f"image_id 형식 이상: {image_id}"
    return parts[0], parts[1]


def load_selected_tiles(cfg):
    """tiles.csv 에서 selected==1 타일을 읽는다 (행 순서 고정: image_id, tile_id 순)."""
    rows = [r for r in csv.DictReader(open(cfg["paths"]["tiles_csv"], encoding="utf-8-sig")) if r["selected"] == "1"]
    rows.sort(key=lambda r: (r["image_id"], r["tile_id"]))
    return rows


def load_image_table(cfg):
    """이미지 60장의 표: image_id, label, 금속 종류, 처리 조건, 처리 방법, 온도, 원본 경로."""
    man = {r["image_id"]: r for r in csv.DictReader(open(cfg["paths"]["manifest_csv"], encoding="utf-8-sig"))}
    ids = sorted({r["image_id"] for r in load_selected_tiles(cfg)})
    out = []
    for iid in ids:
        metal, cond = image_parts(iid)
        method, temp = parse_condition(cond)
        out.append(dict(image_id=iid, label=int(man[iid]["label"]), metal=metal, condition=cond, method=method, temp=temp,
                        path=str(Path(cfg["paths"]["image_root"]) / man[iid]["path"]), mask=str(Path(cfg["paths"]["masks_dir"]) / (iid.replace("/", "__") + ".png"))))
    return out


def assert_conditions(imgs, cfg, log=None):
    """파싱 결과가 10개 조건(5 방법 x 2 온도) 전부와 정확히 일치하는지, 조건마다 6장(금속 6종 x 1)인지 확인."""
    c = cfg["conditions"]
    expected = {(m, t) for m in c["methods"] for t in c["temps"]}
    got = {(r["method"], r["temp"]) for r in imgs}
    assert got == expected, f"파싱된 조건이 기대와 다름: 누락 {sorted(expected - got)}, 초과 {sorted(got - expected)}"
    cnt = {}
    for r in imgs:
        cnt.setdefault((r["method"], r["temp"]), []).append(r["metal"])
    for k, v in cnt.items():
        assert len(v) == c["images_per_condition"] and sorted(v) == sorted(c["metals"]), f"조건 {k} 의 금속 구성이 이상함: {sorted(v)}"
    assert sorted({r["metal"] for r in imgs}) == sorted(c["metals"])
    if log:
        log(f"조건 파싱 assert 통과: 처리 방법 {c['methods']} x 온도 {c['temps']} = {len(expected)}개 조건, 조건마다 {c['images_per_condition']}장 (금속 6종 x 1장)")


# ---------------------------------------------------------------- 이미지 벡터 (합치기)
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


# ---------------------------------------------------------------- 지표
def confusion(y, p, labels=(0, 1, 2)):
    k = len(labels)
    return np.bincount(np.asarray(y) * k + np.asarray(p), minlength=k * k).reshape(k, k)


def metrics_from_confusion(cm):
    """macro-F1 은 정답이 하나라도 있는 클래스만 평균 (부트스트랩 표본에서 클래스가 빠질 때를 위한 규칙; 전체 데이터에서는 3클래스 모두 존재)."""
    tp = np.diag(cm).astype(float); true = cm.sum(1).astype(float); pred = cm.sum(0).astype(float)
    recall = np.divide(tp, true, out=np.zeros_like(tp), where=true > 0)
    prec = np.divide(tp, pred, out=np.zeros_like(tp), where=pred > 0)
    f1 = np.divide(2 * prec * recall, prec + recall, out=np.zeros_like(tp), where=(prec + recall) > 0)
    present = true > 0
    return dict(accuracy=float(tp.sum() / cm.sum()), macro_f1=float(f1[present].mean()), recall=recall, f1=f1, n_present=int(present.sum()))


def metrics(y, p, labels=(0, 1, 2)):
    return metrics_from_confusion(confusion(y, p, labels))


# ---------------------------------------------------------------- 부트스트랩
def bootstrap_ci(y, P, groups, n, seed, labels=(0, 1, 2)):
    """y: (N,) 정답, P: (N, R) 반복별 예측 (기준선은 R=1), groups: (N,) 처리 조건 id.
    이미지 단위: N장 복원추출. 클러스터: 처리 조건을 복원추출(조건 하나 = 그 조건의 이미지 전부).
    각 표본에서 반복(R)마다 지표를 구해 평균 -> 백분위 95% 구간. 반환: {'image': {...}, 'cluster': {...}}"""
    y = np.asarray(y); P = np.asarray(P); groups = np.asarray(groups)
    N, R = P.shape
    uniq = sorted(set(groups.tolist()))
    members = {g: np.where(groups == g)[0] for g in uniq}
    rng = np.random.RandomState(seed)
    out = {}
    for mode in ("image", "cluster"):
        acc, f1, missing = [], [], 0
        for _ in range(n):
            if mode == "image":
                idx = rng.randint(0, N, N)
            else:
                pick = rng.randint(0, len(uniq), len(uniq))
                idx = np.concatenate([members[uniq[i]] for i in pick])
            if len(set(y[idx].tolist())) < len(labels):
                missing += 1
            ms = [metrics(y[idx], P[idx, r], labels) for r in range(R)]
            acc.append(np.mean([m["accuracy"] for m in ms])); f1.append(np.mean([m["macro_f1"] for m in ms]))
        out[mode] = dict(acc=tuple(np.percentile(acc, [2.5, 97.5])), f1=tuple(np.percentile(f1, [2.5, 97.5])), missing_class_resamples=missing, n=n)
    return out


# ---------------------------------------------------------------- 금지어 검사
def term_check(cfg, extra_files=()):
    """exp/*.py, config.yaml, outputs/*/log.txt, report 에 금지어가 있는지 검사. 있으면 (파일, 줄, 내용) 목록을 반환."""
    bad = []
    files = list(EXP_DIR.glob("*.py")) + [EXP_DIR / "config.yaml"] + list(Path(cfg["paths"]["outputs_dir"]).glob("*/log.txt")) + \
        list(Path(cfg["paths"]["outputs_dir"]).glob("*/report*.md")) + [Path(f) for f in extra_files]
    for f in files:
        if not f.exists():
            continue
        for i, line in enumerate(f.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if f.name in ("config.yaml",) and line.lstrip().startswith("forbidden_terms"):
                continue
            if f.name == "common.py" and "forbidden" in line:
                continue
            for t in cfg.get("forbidden_terms", []):
                if t in line:
                    bad.append((f.name, i, line.strip()[:100]))
    return bad
