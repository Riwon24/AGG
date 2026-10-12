"""s01: 스케일바(검은 박스 + 흰 막대 + 눈금 글자) 위치와 px/mm 를 이미지에서 직접 측정한다.

- 179장 전부 측정하고 배율별로 요약한다. 제외 구역은 use_mags(10x) 이미지의 박스 모델 합집합 + 여유 마진.
- 방법: 우하단 탐색 영역에서 (1) 가장 긴 흰 가로 줄(막대)을 찾고, (2) 어두운 마스크를 열기/닫기 한 뒤
  막대 아래 점을 포함하는 연결 영역의 bbox 를 박스로 본다.
- 막대의 눈금 값은 이미지마다 다르다(1 mm / 750 um / 2.5 mm ...). 글자를 읽지 않고, 막대 길이(px)를
  메타데이터 기준 px/mm 로 나눠 config 의 nominal_bar_mm 후보 중 가장 가까운 값으로 추정하고,
  (배율, 막대 길이) 군집마다 대표 이미지를 근거 범례로 저장해 사람이 눈으로 확인한다.
- 검출된 박스는 바로 옆의 어두운 배경이 붙어 바깥으로 늘어날 수 있어서, 군집별 중앙값 박스 모델을 같이 만들고
  검출값이 모델과 얼마나 다른지 로그에 남긴다.
- 측정 실패/이상치는 건너뛰지 않고 status 에 남긴다.
산출: scalebar_measurements.csv, scalebar_clusters.csv, exclusion_zone.json, evidence_*.png, log.txt
"""
import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from common import Logger, PREPROC_DIR, imwrite_png, load_config

SCRIPT = "s01_scalebar_locate"
SEARCH = dict(x0=1500, y0=1220)          # 탐색 영역 좌상단 (우하단 끝까지)
WHITE_MIN_GRAY = 200                      # 막대 흰색 판정
BAR_LEN_RANGE = (40, 260)                 # 막대 길이 허용 범위(px). 눈금 값이 1mm/750um/2.5mm 등으로 달라짐
BLACK_MAX_GRAY = 30                       # 박스 검정 판정
BOX_W_RANGE, BOX_H_RANGE = (80, 320), (40, 90)   # 7.5x 1mm 박스는 폭이 약 100px
EDGE_GRAY = 128                           # 막대 끝/눈금 두께는 반값 임계로 재측정 (200은 끝이 잘림)
CLUSTER_PX = 3                            # 막대 길이 군집 폭(px)


def longest_run(row):
    """bool 1D 배열에서 가장 긴 True 구간 (start, length)."""
    best = (0, 0)
    d = np.diff(np.concatenate(([0], row.astype(np.int8), [0])))
    s, e = np.where(d == 1)[0], np.where(d == -1)[0]
    for a, b in zip(s, e):
        if b - a > best[1]:
            best = (a, b - a)
    return best


def measure(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    sx, sy = SEARCH["x0"], SEARCH["y0"]
    reg = gray[sy:, sx:]
    res = dict(status="ok")

    # ---- 막대: 가장 긴 흰 가로 줄 (+ 인접 행도 비슷하게 길어야 함) ----
    white = reg >= WHITE_MIN_GRAY
    runs = [longest_run(white[r]) for r in range(white.shape[0])]
    cand = [(r, s, L) for r, (s, L) in enumerate(runs) if BAR_LEN_RANGE[0] <= L <= BAR_LEN_RANGE[1]]
    if not cand:
        res["status"] = "no_bar"; return res
    r0, s0, L0 = max(cand, key=lambda t: t[2])
    rows = [r for r, s, L in cand if abs(s - s0) <= 3 and abs(L - L0) <= 6 and abs(r - r0) <= 8]
    if len(rows) < 2:
        res["status"] = "bar_too_thin"; return res
    bar_row = int(round(np.mean(rows)))
    xs = [runs[r][0] for r in rows]; ls = [runs[r][1] for r in rows]
    bx0, bx1 = int(np.median(xs)), int(np.median([x + l for x, l in zip(xs, ls)]))
    # 끝 재측정: 막대 행들의 평균 밝기가 반값(EDGE_GRAY) 이상인, 막대 중심을 포함한 구간
    prof = reg[min(rows):max(rows) + 1].astype(np.float32).mean(axis=0)
    cx = (bx0 + bx1) // 2
    lo, hi = cx, cx
    while lo > 0 and prof[lo - 1] >= EDGE_GRAY:
        lo -= 1
    while hi < len(prof) - 1 and prof[hi + 1] >= EDGE_GRAY:
        hi += 1
    bx0, bx1 = lo, hi + 1
    outer = bx1 - bx0
    # 눈금 두께: 막대 위쪽(눈금만 있는 행)에서 왼쪽 끝 구간 폭 (반값 임계)
    tick_w = None
    for dr in range(3, 9):
        rr = bar_row - dr
        if rr < 0:
            break
        rowv = reg[rr].astype(np.float32)
        seg = rowv[max(bx0 - 3, 0):bx0 + 12] >= EDGE_GRAY
        if seg.any() and not (rowv[bx0 + 12:bx1 - 12] >= EDGE_GRAY).any():
            tw = longest_run(seg)[1]
            if 1 <= tw <= 8:
                tick_w = tw; break
    if tick_w is None:
        tick_w = 3
        res["status"] = "ok(tick_w_default)"
    c2c = outer - tick_w
    res.update(bar_x0=int(bx0 + sx), bar_x1=int(bx1 + sx), bar_y=int(bar_row + sy), bar_outer_px=int(outer),
               tick_w=int(tick_w), bar_c2c_px=int(c2c))

    # ---- 박스: 어두운 마스크 열기/닫기 후 막대 아래 점을 포함하는 연결 영역 ----
    dark = (reg <= BLACK_MAX_GRAY).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=4)
    seed = (min(bar_row + 12, dark.shape[0] - 1), (bx0 + bx1) // 2)
    k = lab[seed]
    if k == 0:
        res["status"] = "no_box"; return res
    x, y, bw, bh, _ = stats[k]
    x, y, bw, bh = int(x), int(y), int(bw), int(bh)
    res.update(box_x0=x + sx, box_y0=y + sy, box_x1=x + bw + sx, box_y1=y + bh + sy, box_w=bw, box_h=bh,
               black_med=float(np.median(reg[lab == k])))
    if not (BOX_W_RANGE[0] <= bw <= BOX_W_RANGE[1] and BOX_H_RANGE[0] <= bh <= BOX_H_RANGE[1]):
        res["status"] = "box_size_out_of_range"
    return res


def crop_annotated(img_bgr, m, scale=1.0, crop=(1480, 1180, 1824, 1368), text=None):
    x0, y0, x1, y1 = crop
    c = img_bgr[y0:y1, x0:x1].copy()
    if "box_x0" in m:   # 초록: 검출 박스
        cv2.rectangle(c, (m["box_x0"] - x0, m["box_y0"] - y0), (m["box_x1"] - x0, m["box_y1"] - y0), (0, 255, 0), 1)
    if "model_box" in m:  # 노랑: 군집 모델 박스
        a, b, cc, d = m["model_box"]
        cv2.rectangle(c, (a - x0, b - y0), (cc - x0, d - y0), (0, 255, 255), 1)
    if "bar_x0" in m:   # 빨강: 막대 길이
        cv2.line(c, (m["bar_x0"] - x0, m["bar_y"] - y0 + 8), (m["bar_x1"] - x0, m["bar_y"] - y0 + 8), (0, 0, 255), 1)
    if scale != 1.0:
        c = cv2.resize(c, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_NEAREST)
    if text:
        cv2.putText(c, text, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA)
    return c


def sheet_of(tiles, cols):
    th, tw = tiles[0].shape[:2]
    rws = (len(tiles) + cols - 1) // cols
    sheet = np.zeros((rws * th, cols * tw, 3), np.uint8)
    for i, t in enumerate(tiles):
        sheet[(i // cols) * th:(i // cols + 1) * th, (i % cols) * tw:(i % cols + 1) * tw] = t
    return sheet


def imwrite(path, img):
    imwrite_png(path, img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", help="이미지 루트 (기본: config paths.image_root)")
    ap.add_argument("--manifest", help="manifest_v1.csv (기본: outputs/s00b_meta_mag/manifest_v1.csv)")
    ap.add_argument("--output", help="outputs 폴더 (기본: config paths.outputs_dir)")
    ap.add_argument("--config", default=str(PREPROC_DIR / "config.yaml"))
    a = ap.parse_args()

    cfg = load_config(a.config)
    root = Path(a.input or cfg["paths"]["image_root"])
    out = a.output or cfg["paths"]["outputs_dir"]
    manifest = Path(a.manifest or Path(out) / "s00b_meta_mag" / "manifest_v1.csv")
    margin = int(cfg["scalebar"]["extra_margin_px"])
    nominal = [float(x) for x in cfg["scalebar"]["nominal_bar_mm"]]
    use = cfg["experiment"]["use_mags"]
    known = cfg["naming"]["known_mags"]
    W, H = cfg["naming"]["expected_resolution"]
    log = Logger(out, SCRIPT)
    o = Path(out) / SCRIPT
    log("config:", cfg["_config_path"], "| manifest:", manifest, "| margin:", margin, "| use_mags:", use,
        "| nominal_bar_mm:", nominal)
    log(f"탐색 영역 x>={SEARCH['x0']}, y>={SEARCH['y0']} | 흰색 gray>={WHITE_MIN_GRAY} | 검정 gray<={BLACK_MAX_GRAY} | 막대 끝 반값 {EDGE_GRAY}")

    rows = list(csv.DictReader(open(manifest, encoding="utf-8-sig")))
    meas = []
    for r in rows:
        img = cv2.imdecode(np.fromfile(str(root / r["path"]), dtype=np.uint8), cv2.IMREAD_COLOR)
        assert img is not None and img.shape[1] == W and img.shape[0] == H, f"해상도 이상: {r['path']}"
        m = measure(img)
        m.update(image_id=r["image_id"], path=r["path"], mag_final=r["mag_final"], mag_source=r["mag_source"],
                 meta_check=r["meta_check"], meta_px_per_mm=r["px_per_mm"])
        meas.append((r, m, img))
    ms = [m for _, m, _ in meas]

    log(f"\n[1] 측정 상태: {dict(Counter(m['status'] for m in ms))}")
    for m in ms:
        if not m["status"].startswith("ok"):
            log(f"    [이상] {m['path']}: {m['status']}")
    good = [m for m in ms if "box_x0" in m and "bar_c2c_px" in m]

    # ---------- 배율별 기준 px/mm (메타데이터, 이름·메타 일치 파일의 중앙값) ----------
    refmap = {}
    for mag in known:
        v = [float(m["meta_px_per_mm"]) for m in good if m["mag_final"] == mag and m["meta_px_per_mm"]
             and m["mag_source"] in ("name", "meta+name_agree") and m["meta_check"] == "agree"]
        refmap[mag] = statistics.median(v)
    log("\n[2] 배율별 기준 px/mm (xlif 메타데이터, 이름·메타 일치 파일 중앙값):", {k: round(v, 2) for k, v in refmap.items()})

    # ---------- 눈금 값 추정 (글자를 읽지 않음) ----------
    for m in good:
        ratio = m["bar_c2c_px"] / refmap[m["mag_final"]]
        m["bar_mm_est"] = min(nominal, key=lambda x: abs(np.log(ratio / x)))
        m["snap_resid_pct"] = round(100 * (ratio / m["bar_mm_est"] - 1), 1)
        m["px_per_mm_bar"] = round(m["bar_c2c_px"] / m["bar_mm_est"], 2)
        if m["meta_px_per_mm"]:
            m["rel_diff_vs_meta_pct"] = round(100 * (m["px_per_mm_bar"] / float(m["meta_px_per_mm"]) - 1), 2)

    # ---------- (배율, 막대 길이) 군집 + 박스 모델 ----------
    clusters = defaultdict(list)
    for m in good:
        m["cluster"] = f"{m['mag_final']}x_bar{int(round(m['bar_c2c_px'] / CLUSTER_PX) * CLUSTER_PX)}"
        clusters[m["cluster"]].append(m)
    log("\n[3] (확정배율, 막대 길이) 군집  — 눈금 값은 추정이며 evidence_clusters.png 에서 눈으로 확인")
    crow = []
    for key in sorted(clusters, key=lambda k: (float(k.split("x")[0]), int(k.split("bar")[1]))):
        g = clusters[key]
        mm = Counter(x["bar_mm_est"] for x in g).most_common(1)[0][0]
        mdl = [int(statistics.median([y[f] for y in g])) for f in ("box_x0", "box_y0", "box_x1", "box_y1")]
        for x in g:
            x["model_box"] = tuple(mdl) if len(g) >= 3 else (x["box_x0"], x["box_y0"], x["box_x1"], x["box_y1"])
            x["model_src"] = "cluster_median" if len(g) >= 3 else "own_detected(n<3)"
        within = sum(1 for x in g if all(abs(x[f] - mdl[i]) <= 2 for i, f in enumerate(("box_x0", "box_y0", "box_x1", "box_y1"))))
        outward = sum(1 for x in g if x["box_x0"] <= mdl[0] + 2 and x["box_y0"] <= mdl[1] + 2
                      and x["box_x1"] >= mdl[2] - 2 and x["box_y1"] >= mdl[3] - 2)
        px = statistics.median([x["px_per_mm_bar"] for x in g])
        log(f"    {key:<14} n={len(g):>2} 눈금 추정={mm:g}mm px/mm(막대)={px:.1f} | 모델 박스={mdl} "
            f"| 검출이 모델과 ±2px 이내 {within}/{len(g)}, 바깥쪽으로만 벗어남(모델 포함) {outward}/{len(g)}")
        crow.append(dict(cluster=key, n=len(g), bar_mm_est=mm, px_per_mm_bar=round(px, 2),
                         model_box=mdl, example=g[0]["path"]))
    with open(o / "scalebar_clusters.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(crow[0].keys()))
        w.writeheader(); w.writerows(crow)

    # ---------- 막대 vs 메타 / 다른 배율 군집 ----------
    big = [m for m in good if m.get("rel_diff_vs_meta_pct") is not None and abs(m["rel_diff_vs_meta_pct"]) > 10]
    log(f"\n[4] 막대 기반 px/mm 가 메타데이터와 10% 넘게 다른 파일: {len(big)}장 "
        f"(중앙값 차이 {statistics.median([m['rel_diff_vs_meta_pct'] for m in good if 'rel_diff_vs_meta_pct' in m]):.1f}%)")
    for m in big:
        log(f"    {m['path']} 확정={m['mag_final']}({m['mag_source']}) 막대 c2c={m['bar_c2c_px']}px 눈금추정={m['bar_mm_est']:g}mm "
            f"막대px/mm={m['px_per_mm_bar']} 메타={float(m['meta_px_per_mm']):.1f} 차이={m['rel_diff_vs_meta_pct']}%")
    nometa = [m for m in good if not m["meta_px_per_mm"]]
    for m in nometa:
        log(f"    (메타 없음) {m['path']} 확정={m['mag_final']} 막대 c2c={m['bar_c2c_px']}px 눈금추정={m['bar_mm_est']:g}mm")

    log("\n[5] 확정 배율 군집에 속하지 않고 다른 배율 군집에 속하는 막대 (n>=3 군집 기준)")
    big_cl = {k: g for k, g in clusters.items() if len(g) >= 3}
    n5 = 0
    for m in good:
        if m["cluster"] in big_cl:
            continue
        for k, g in big_cl.items():
            if abs(int(k.split("bar")[1]) - m["bar_c2c_px"]) <= CLUSTER_PX and k.split("x")[0] != m["mag_final"]:
                n5 += 1
                log(f"    {m['path']} 확정={m['mag_final']}({m['mag_source']}) 막대 {m['bar_c2c_px']}px -> 군집 {k} 와 같은 길이")
    if not n5:
        log("    없음")

    # ---------- 제외 구역 (use_mags) ----------
    sel = [m for m in good if m["mag_final"] in use]
    boxes = np.array([m["model_box"] for m in sel])
    bu = [int(boxes[:, 0].min()), int(boxes[:, 1].min()), int(boxes[:, 2].max()), int(boxes[:, 3].max())]
    zone = dict(basis=f"use_mags={use}, n={len(sel)}, 군집 모델 박스의 합집합 + margin {margin}px",
                margin_px=margin, model_box_union=bu,
                exclude_box_xyxy=[max(bu[0] - margin, 0), max(bu[1] - margin, 0), min(bu[2] + margin, W), min(bu[3] + margin, H)],
                clusters_used=sorted({m["cluster"] for m in sel}),
                detected_box_union_raw=[min(m["box_x0"] for m in sel), min(m["box_y0"] for m in sel),
                                        max(m["box_x1"] for m in sel), max(m["box_y1"] for m in sel)])
    (o / "exclusion_zone.json").write_text(json.dumps(zone, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"\n[6] 제외 구역 제안 ({'/'.join(use)}x {len(sel)}장): 모델 박스 합집합={bu} -> margin {margin}px 포함 = {zone['exclude_box_xyxy']}")
    log(f"    (참고: 검출 박스 원시 합집합 {zone['detected_box_union_raw']} — 박스 옆 어두운 배경이 붙어 바깥으로 늘어난 값 포함)")
    log("    사용 군집:", zone["clusters_used"])

    # ---------- CSV ----------
    keys = ["image_id", "path", "mag_final", "mag_source", "meta_check", "status", "cluster", "bar_x0", "bar_x1", "bar_y",
            "bar_outer_px", "tick_w", "bar_c2c_px", "bar_mm_est", "snap_resid_pct", "px_per_mm_bar", "meta_px_per_mm",
            "rel_diff_vs_meta_pct", "box_x0", "box_y0", "box_x1", "box_y1", "box_w", "box_h", "model_box", "model_src", "black_med"]
    with open(o / "scalebar_measurements.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader(); w.writerows(ms)

    # ---------- 근거 PNG ----------
    for mag in known:
        g = [(m, img) for _, m, img in meas if m["mag_final"] == mag and "box_x0" in m]
        if g:
            imwrite(o / f"evidence_{mag}x.png",
                    sheet_of([crop_annotated(img, m, 0.5, text=f"{m['bar_mm_est']:g}mm") for m, img in g], 8))
    leg = []
    for key in sorted(clusters, key=lambda k: (float(k.split("x")[0]), int(k.split("bar")[1]))):
        m = clusters[key][0]
        img = next(i for _, mm, i in meas if mm is m)
        leg.append(crop_annotated(img, m, 1.5, text=f"{key} n={len(clusters[key])} est={m['bar_mm_est']:g}mm"))
    imwrite(o / "evidence_clusters.png", sheet_of(leg, 3))
    fl = [crop_annotated(img, m, 1.5, text=f"{m['mag_final']}x({m['mag_source']}) c2c={m.get('bar_c2c_px')}")
          for _, m, img in meas if m["mag_source"] == "user_override" and "box_x0" in m]
    if fl:
        imwrite(o / "evidence_user_override.png", sheet_of(fl, 3))
    log("\n근거 PNG: evidence_<mag>x.png (초록=검출 박스, 노랑=군집 모델 박스, 빨강=막대 길이), "
        "evidence_clusters.png (군집별 대표), evidence_user_override.png")
    log(f"저장: {o}")


if __name__ == "__main__":
    main()
