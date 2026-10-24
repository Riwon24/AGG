"""배경 증강 실험 공통 유틸: config, 입력 로딩, 마스크, 배경 증강 함수 (b02/b03 이 같은 함수를 쓴다).

기존 exp/common.py 의 로그, CSV, 지표, 조건 파싱 함수를 재사용한다 (읽기 전용 import).
증강은 이미지 전체에 한 번 적용한 뒤 타일을 자른다. 마스크 안쪽 픽셀은 절대 바꾸지 않는다.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
EXP_DIR = HERE.parent / "exp"
sys.path.insert(0, str(EXP_DIR))
from common import (Logger, aggregate, assert_conditions, bootstrap_ci, image_parts, load_image_table, load_selected_tiles,  # noqa: E402,F401
                    metrics, parse_condition, sha256_file, write_csv)


def _is_abs(c):
    return c.startswith("/") or (len(c) > 1 and c[1] == ":")


def load_config(path=None):
    path = Path(path or HERE / "config.yaml").resolve()
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))

    def resolve(c):
        c = str(c)
        return str(Path(c).expanduser()) if c.startswith("~") else (c if _is_abs(c) else str((path.parent / c).resolve()))

    for k, v in cfg["paths"].items():
        if isinstance(v, list):
            cands = [resolve(c) for c in v]
            cfg["paths"][k] = next((c for c in cands if Path(c).exists()), cands[0])
        else:
            cfg["paths"][k] = resolve(v)
    cfg["_config_path"] = str(path)
    return cfg


def read_rgb(path):
    """RGB uint8 (H, W, 3). 한글/공백 경로 때문에 np.fromfile + imdecode 를 쓴다."""
    arr = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    assert arr is not None, f"이미지를 읽을 수 없음: {path}"
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)


def read_mask(path):
    m = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    assert m is not None, f"마스크를 읽을 수 없음: {path}"
    return m > 127


def write_png(path, rgb):
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    assert ok
    Path(path).write_bytes(buf.tobytes())


def _rc(tile_id):
    """tile_id 'r05_c12' -> (행 번호, 열 번호) (stride 112 격자 위치)."""
    a, b = tile_id.split("_c")
    return int(a[1:]), int(b)


PHASES = [(0, 0), (0, 1), (1, 0), (1, 1)]     # (행 홀짝, 열 홀짝): stride 112 격자에서 서로 겹치지 않는 stride 224 부분 격자 4개


def select_tiles(rows, tcfg):
    """타일 선택 (config tiles). 모든 이미지에 같은 규칙, 라벨과 무관.
      all_grid                : 격자 전체
      selected                : tiles.csv 의 selected==1 (마스크 비율 >= 0.5)
      nonoverlap_best_phase   : stride 112 격자에서 서로 겹치지 않는 부분 격자 4개 중 시편 픽셀을 가장 많이 덮는 것(동률이면 앞 번호) 하나
    drop_zero_mask 가 true 이면 시편 픽셀이 0인 타일(mask_px == 0)은 뺀다. 반환: (선택 타일 목록, 이미지별 정보 목록)"""
    mode, drop0 = tcfg["select"], bool(tcfg.get("drop_zero_mask", False))
    assert mode in ("all_grid", "selected", "nonoverlap_best_phase"), mode
    by = {}
    for r in rows:
        by.setdefault(r["image_id"], []).append(r)
    out, info = [], []
    for iid in sorted(by):
        g = by[iid]
        keep = lambda r: (int(r["mask_px"]) > 0) if drop0 else True
        if mode == "all_grid":
            chosen, ph = [r for r in g if keep(r)], None
        elif mode == "selected":
            chosen, ph = [r for r in g if r["selected"] == "1"], None
        else:
            scores = []
            for pr, pc in PHASES:
                cand = [r for r in g if _rc(r["tile_id"])[0] % 2 == pr and _rc(r["tile_id"])[1] % 2 == pc and keep(r)]
                scores.append((sum(int(r["mask_px"]) for r in cand), cand))
            best = max(range(4), key=lambda i: (scores[i][0], -i))
            chosen, ph = scores[best][1], PHASES[best]
            info.append(dict(image_id=iid, phase=f"{ph[0]}{ph[1]}", n_tiles=len(chosen), specimen_px=scores[best][0], phase_px=[s[0] for s in scores]))
        out += chosen
    out.sort(key=lambda r: (r["image_id"], r["tile_id"]))
    return out, info


def inputs(cfg, extract=False):
    """이미지 표(60장, image_id 순)와 타일 목록.
    extract=True : 특징 추출 대상(tiles.csv 의 격자 전체, (image_id, tile_id) 순). b03 이 이 전체에 대해 특징을 만든다.
    extract=False: 분석에 쓸 타일 (config tiles 의 select / drop_zero_mask 규칙으로 격자 전체의 부분집합을 고름)."""
    import csv
    cfg_exp = {"paths": {"tiles_csv": cfg["paths"]["tiles_csv"], "manifest_csv": cfg["paths"]["manifest_csv"],
                         "image_root": cfg["paths"]["image_root"], "masks_dir": cfg["paths"]["masks_dir"]}}
    imgs = load_image_table(cfg_exp)
    rows = list(csv.DictReader(open(cfg["paths"]["tiles_csv"], encoding="utf-8-sig")))
    if extract:
        rows.sort(key=lambda r: (r["image_id"], r["tile_id"]))
        return imgs, rows
    tiles, _ = select_tiles(rows, cfg["tiles"])
    return imgs, tiles


def extraction_rows(cfg, tiles, out_dir=None):
    """분석 타일 목록 -> b03 특징 배열(row_of 의 행)에서의 위치. b03 의 tile_index.csv 와 (image_id, tile_id) 로 대조한다."""
    import csv
    out_dir = out_dir or cfg["paths"]["outputs_dir"]
    tindex = list(csv.DictReader(open(f"{out_dir}/b03_extract_features/tile_index.csv", encoding="utf-8-sig")))
    pos = {(t["image_id"], t["tile_id"]): i for i, t in enumerate(tindex)}
    idx = np.array([pos[(t["image_id"], t["tile_id"])] for t in tiles])
    for t, i in zip(tiles, idx):
        assert tindex[i]["x0"] == t["x0"] and tindex[i]["y0"] == t["y0"], "b03 타일 좌표가 현재 타일 목록과 다름"
    return idx


def dist_to_mask(mask):
    """마스크 밖 픽셀에서 가장 가까운 마스크 픽셀까지의 거리 (마스크 안쪽은 0)."""
    return cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5)


def bg_region(mask, d, buffer_px, margin_px, box):
    """배경 측정 영역: 마스크 밖이고 경계에서 buffer+margin 보다 멀며 스케일바 구역 밖."""
    w = d > (buffer_px + margin_px)
    x0, y0, x1, y1 = box
    w[y0:y1, x0:x1] = False
    return w


def measure_background(img, w, bg):
    """배경 픽셀 w 에서 채널별 평균(mean), 저주파 변동(lf), 중간 대역 질감(mid), 미세 노이즈(hf)와 그 채널 간 공분산을 측정한다 (RGB, 0~255 단위).

    hf  = 이미지 - σ(hf_blur_sigma_px) 흐림 (채널 3x3 공분산 hf_cov 도 저장: 원본 노이즈는 채널끼리 상관이 있다)
    mid = σ(hf) 흐림 - σ(mid_hi_sigma_px) 흐림 (배경의 얼룩 같은 중간 크기 질감; 흐림은 배경 픽셀만으로 정규화)
    lf  = σ(lowfreq_sigma_px) 로 흐린 값(배경만으로 정규화)의 표준편차
    """
    f = img.astype(np.float32)
    mean = f[w].mean(0)
    l2 = cv2.GaussianBlur(f, (0, 0), float(bg["hf_blur_sigma_px"]))
    res = f - l2
    hf_cov = np.cov(res[w].T)
    hf = np.sqrt(np.diag(hf_cov))
    wf = w.astype(np.float32)

    def nblur(s):
        num = cv2.GaussianBlur(f * wf[..., None], (0, 0), s)
        den = cv2.GaussianBlur(wf, (0, 0), s)
        return num, den

    num, den = nblur(float(bg["lowfreq_sigma_px"]))
    ok = w & (den > 0.05)
    lf = (num[ok] / den[ok][:, None]).std(0)
    num8, den8 = nblur(float(bg["mid_hi_sigma_px"]))
    ok8 = w & (den8 > 0.3)
    band = l2[ok8] - num8[ok8] / den8[ok8][:, None]
    mid = band.std(0)
    mid_corr = np.corrcoef(band.T)
    return dict(mean=mean, lf=lf, hf=hf, hf_cov=hf_cov, mid=mid, mid_corr=mid_corr, n=int(w.sum()))


def lowfreq_field(h, w, sigma_px, down, rng):
    """단위 표준편차의 매끄러운 무작위 장 (h, w)."""
    hs, ws = h // down + 2, w // down + 2
    z = rng.standard_normal((hs, ws)).astype(np.float32)
    z = cv2.GaussianBlur(z, (0, 0), sigma_px / down)
    z = cv2.resize(z, (w, h), interpolation=cv2.INTER_CUBIC)
    return z / (z.std() + 1e-8)


def make_rng(seed, image_idx, k):
    return np.random.default_rng(np.random.SeedSequence([int(seed), int(image_idx), int(k)]))


_HF_GAIN = {}


def _hf_gain(sigma):
    """흰 노이즈를 (노이즈 - σ 흐림) 으로 측정하면 표준편차가 줄어드는 비율의 역수 (원본 노이즈를 같은 방식으로 측정했으므로 맞춰 준다)."""
    if sigma not in _HF_GAIN:
        z = np.random.default_rng(12345).standard_normal((512, 512)).astype(np.float32)
        _HF_GAIN[sigma] = 1.0 / float((z - cv2.GaussianBlur(z, (0, 0), sigma)).std())
    return _HF_GAIN[sigma]


def augment_image(img, d, pool, bg, image_idx, k, feather_px):
    """증강 이미지 k (>=1) 를 만든다. 반환: (aug uint8, alpha float32, src_index).

    원본을 저주파 L_o (σ(hf) 흐림)와 미세 노이즈 H_o (나머지)로 나누고, 합성 배경도 L_s + H_s 로 만든다.
      L_s = 평균색 + 저주파 장 x lf + 중간 대역 질감 x mid,   H_s = 채널 상관(hf_cov)을 맞춘 흰 노이즈
    세 값 세트는 60장 배경 측정 풀에서 무작위로 고른 한 이미지(src)의 것이다 (라벨 미사용, 같은 이미지일 수도 있음).
    alpha = 0 (마스크 안쪽과 buffer 이내) -> 1 (buffer + feather 바깥).
      out = (1-a) L_o + a L_s + c(a) [(1-a) H_o + a H_s],  c(a) = 1/sqrt((1-a)^2 + a^2)   (분산 보존 블렌딩: 노이즈 크기가 경계에서 줄지 않게)
    alpha == 0 인 픽셀은 원본 uint8 을 그대로 쓴다. 같은 (image_idx, k) 이면 feather 와 상관없이 같은 합성 배경(alpha 만 다름).
    """
    h, w = img.shape[:2]
    rng = make_rng(bg["seed"], image_idx, k)
    src = int(rng.integers(0, len(pool)))
    P = pool[src]
    mean, lf, mid = (np.asarray(P[key], dtype=np.float32) for key in ("mean", "lf", "mid"))
    cov = np.asarray(P["hf_cov"], dtype=np.float64)
    field = lowfreq_field(h, w, bg["lowfreq_sigma_px"], int(bg["lowfreq_downscale"]), rng)
    white = rng.standard_normal((h, w)).astype(np.float32)
    s_lo, s_hi = float(bg["hf_blur_sigma_px"]), float(bg["mid_hi_sigma_px"])
    band = cv2.GaussianBlur(white, (0, 0), s_lo) - cv2.GaussianBlur(white, (0, 0), s_hi)
    band /= band.std() + 1e-8
    z = rng.standard_normal((h, w, 3)).astype(np.float32)
    chol = np.linalg.cholesky(cov + 1e-6 * np.eye(3)).astype(np.float32)
    h_s = (z @ chol.T) * _hf_gain(s_lo)
    l_s = mean[None, None, :] + field[..., None] * lf[None, None, :] + band[..., None] * mid[None, None, :]
    f = img.astype(np.float32)
    l_o = cv2.GaussianBlur(f, (0, 0), s_lo)
    h_o = f - l_o
    b = float(bg["buffer_px"])
    alpha = (d > b).astype(np.float32) if feather_px <= 0 else np.clip((d - b) / float(feather_px), 0, 1).astype(np.float32)
    a = alpha[..., None]
    c = 1.0 / np.sqrt((1 - a) ** 2 + a ** 2)
    out = (1 - a) * l_o + a * l_s + c * ((1 - a) * h_o + a * h_s)
    out = np.where(a > 0, out, f)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8), alpha, src


def grad_mag(img):
    g = img.astype(np.float32).mean(2)
    return np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))


def forbidden_check(cfg):
    """exp_bgaug 의 스크립트, config, 로그, 보고서에 금지어가 있는지 검사한다 (파일, 줄 번호만 반환)."""
    bad = []
    files = list(HERE.glob("*.py")) + [HERE / "config.yaml"] + list(Path(cfg["paths"]["outputs_dir"]).glob("*/log.txt")) + \
        list(Path(cfg["paths"]["outputs_dir"]).glob("*/report*.md")) + [HERE / "README.md"]
    for f in files:
        if not f.exists():
            continue
        for i, line in enumerate(f.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if line.lstrip().startswith("forbidden_terms"):
                continue
            if any(t in line for t in cfg.get("forbidden_terms", [])):
                bad.append((f.name, i))
    return bad
