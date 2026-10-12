"""s02: 시편(파란색, 채도 높음)과 어두운 배경을 분리해 마스크를 만든다. (use_mags = 10x 만)

- 후보 방법: otsu_s(채도 Otsu), otsu_v(명도 Otsu), manual_*(S>=s & V>=v). 모두 계산해 비교하고
  config mask.method 로 기본 방법을 고른다(확정은 사람이 s03 QC 후).
- 후처리: 가우시안 블러(입력) -> 임계 -> 모폴로지 닫기/열기 -> 가장 큰 연결 영역 -> 구멍 메우기.
- 원본 이미지는 읽기만 한다. 배경을 칠하거나 지우지 않는다(마스크는 별도 PNG).
- overrides: overrides/mask_<safe_id>.json  ({"method": "...", "s": 50, "v": 60})  개별 임계값
             overrides/mask_<safe_id>.png   수동 마스크(흰색=시편, 원본과 같은 크기)  <- 있으면 이게 우선
- 시편이 프레임 밖으로 잘렸는지(마스크가 이미지 경계에 닿는지) 플래그를 남긴다.
산출: masks/*.png, overlays/*.png(0.5배), mask_stats.csv, log.txt
"""
import argparse
import csv
import json
import statistics
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi

from common import Logger, PREPROC_DIR, imwrite_png, load_config, safe_id

SCRIPT = "s02_specimen_mask"


def imread(path):
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def imwrite(path, img):
    imwrite_png(path, img)


def ellipse(d):
    d = int(d) | 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))


def raw_mask(hsv, method, params):
    s, v = hsv[..., 1], hsv[..., 2]
    if method == "otsu_s":
        t, m = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return (m > 0), {"thr_s": float(t)}
    if method == "otsu_v":
        t, m = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return (m > 0), {"thr_v": float(t)}
    if method == "otsu_sv":   # 채도와 명도 둘 다 Otsu 이상
        ts, ms = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        tv, mv = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return (ms > 0) & (mv > 0), {"thr_s": float(ts), "thr_v": float(tv)}
    if method == "hue_sv":    # 시편은 파란색(H 약 100~115): 색상 범위 & 채도 & 명도 하한. 명도만으로 안 갈리는 그림자/밝은 배경용
        h = hsv[..., 0]
        return ((h >= params["h_lo"]) & (h <= params["h_hi"]) & (s >= params["s"]) & (v >= params["v"])), \
            {"thr_h": (params["h_lo"], params["h_hi"]), "thr_s": params["s"], "thr_v": params["v"]}
    # manual: S>=s & V>=v
    return (s >= params["s"]) & (v >= params["v"]), {"thr_s": params["s"], "thr_v": params["v"]}


def postprocess(raw, close_px, open_px):
    m = raw.astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, ellipse(close_px))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, ellipse(open_px))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return np.zeros_like(m, bool), 0, 0.0
    areas = st[1:, cv2.CC_STAT_AREA]
    order = np.argsort(areas)[::-1]
    big = int(order[0]) + 1
    second = float(areas[order[1]] / areas[order[0]]) if len(order) > 1 else 0.0
    keep = ndi.binary_fill_holes(lab == big)
    return keep, int(n - 1), second


def iou(a, b):
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 1.0


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
    ovd = Path(cfg["paths"]["overrides_dir"])
    mc = cfg["mask"]
    use = cfg["experiment"]["use_mags"]
    W, H = cfg["naming"]["expected_resolution"]
    ex = cfg["scalebar"]["exclude_box"]
    assert ex and len(ex) == 4, "config scalebar.exclude_box 가 비어 있음 (s01 필요)"
    np.random.seed(int(cfg.get("seed", 0)))
    log = Logger(out, SCRIPT)
    o = Path(out) / SCRIPT
    (o / "masks").mkdir(exist_ok=True)
    (o / "overlays").mkdir(exist_ok=True)
    log("config:", cfg["_config_path"], "| manifest:", manifest, "| use_mags:", use, "| 기본 방법:", mc["method"])
    log(f"blur={mc['blur_ksize']} close={mc['close_px']} open={mc['open_px']} | 수동 후보: {mc['manual_candidates']}")

    methods = [("otsu_s", {}), ("otsu_v", {}), ("otsu_sv", {}), ("hue_sv", mc["hue"])] + [(c["name"], c) for c in mc["manual_candidates"]]
    AUTO = ("otsu_s", "otsu_v", "otsu_sv", "hue_sv")
    names = [n for n, _ in methods]
    assert mc["method"] in names, f"method {mc['method']} 가 후보에 없음: {names}"

    rows = [r for r in csv.DictReader(open(manifest, encoding="utf-8-sig")) if r["mag_final"] in use]
    log(f"대상 이미지: {len(rows)}장 (전체 manifest 중 mag_final in {use})")

    stats, skipped = [], []
    for r in rows:
        sid = safe_id(r["image_id"])
        img = imread(root / r["path"])
        if img is None or img.shape[1] != W or img.shape[0] != H:
            skipped.append((r["path"], "읽기 실패/해상도 이상")); continue
        blur = cv2.GaussianBlur(img, (int(mc["blur_ksize"]) | 1,) * 2, 0)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

        cands, thr = {}, {}
        for name, p in methods:
            raw, th = raw_mask(hsv, name if name in AUTO else "manual", p)
            cands[name], thr[name] = postprocess(raw, mc["close_px"], mc["open_px"]), th
        # --- override ---
        src, ov = "auto", ""
        png, js = ovd / f"mask_{sid}.png", ovd / f"mask_{sid}.json"
        n_cc, second = cands[mc["method"]][1], cands[mc["method"]][2]
        final = cands[mc["method"]][0]
        if png.exists():
            mk = cv2.imdecode(np.fromfile(str(png), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            assert mk.shape == (H, W), f"수동 마스크 크기 불일치: {png}"
            final, src, ov, n_cc, second = mk > 127, "manual_png", png.name, 1, 0.0
        elif js.exists():
            p = json.loads(js.read_text(encoding="utf-8"))
            meth = p.get("method", "manual")
            pp = {**mc["hue"], **p} if meth == "hue_sv" else p
            raw, th = raw_mask(hsv, meth if meth in AUTO else "manual", pp)
            final, n_cc, second = postprocess(raw, p.get("close_px", mc["close_px"]), p.get("open_px", mc["open_px"]))
            src, ov = "override_threshold", js.name
        imwrite(o / "masks" / f"{sid}.png", (final * 255).astype(np.uint8))

        # --- overlay (0.5배): 윤곽선 초록, 제외 구역 노랑 ---
        ov_img = img.copy()
        cnts, _ = cv2.findContours(final.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(ov_img, cnts, -1, (0, 255, 0), 3)
        cv2.rectangle(ov_img, (ex[0], ex[1]), (ex[2] - 1, ex[3] - 1), (0, 255, 255), 2)
        imwrite(o / "overlays" / f"{sid}.png", cv2.resize(ov_img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))

        d = cands[mc["method"]][0]
        row = dict(image_id=r["image_id"], path=r["path"], label=r["label"], mask_source=src, override_file=ov,
                   method=mc["method"], area_ratio=round(float(final.mean()), 4),
                   n_components=n_cc, second_ratio=round(second, 3),
                   touch_left=bool(final[:, 0].any()), touch_right=bool(final[:, -1].any()),
                   touch_top=bool(final[0].any()), touch_bottom=bool(final[-1].any()),
                   mask_px_in_exclude=int(final[ex[1]:ex[3], ex[0]:ex[2]].sum()),
                   thr_otsu_s=thr["otsu_s"]["thr_s"], thr_otsu_v=thr["otsu_v"]["thr_v"])
        row["touches_border"] = any(row[k] for k in ("touch_left", "touch_right", "touch_top", "touch_bottom"))
        for name in names:
            row[f"area_{name}"] = round(float(cands[name][0].mean()), 4)
            row[f"iou_{name}_vs_default"] = round(iou(cands[name][0], d), 3)
        stats.append(row)

    keys = list(stats[0].keys())
    with open(o / "mask_stats.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader(); w.writerows(stats)

    # ---------- 요약 ----------
    log(f"\n[1] 처리 {len(stats)}장, 건너뜀 {len(skipped)}장")
    for p, why in skipped:
        log("    건너뜀:", p, why)
    log("    마스크 출처:", dict(Counter(s["mask_source"] for s in stats)))
    ar = [s["area_ratio"] for s in stats]
    log(f"\n[2] 기본 방법 {mc['method']} 마스크 면적 비율: median={statistics.median(ar):.3f} min={min(ar):.3f} max={max(ar):.3f}")
    q = np.percentile(ar, [25, 75]); lo, hi = q[0] - mc["area_ratio_outlier_iqr"] * (q[1] - q[0]), q[1] + mc["area_ratio_outlier_iqr"] * (q[1] - q[0])
    log(f"    IQR 기준 이상치 범위 [{lo:.3f}, {hi:.3f}] 밖: {[s['path'] for s in stats if not lo <= s['area_ratio'] <= hi] or '없음'}")
    for lab in sorted({s['label'] for s in stats}):
        v = [s["area_ratio"] for s in stats if s["label"] == lab]
        log(f"    라벨 AGG{lab}: n={len(v)} 면적 median={statistics.median(v):.3f} min={min(v):.3f} max={max(v):.3f}")
    log("\n[3] 방법 비교 (기본 방법 대비, 후처리 후)")
    for name in names:
        a_ = [s[f"area_{name}"] for s in stats]; i_ = [s[f"iou_{name}_vs_default"] for s in stats]
        low = sum(1 for x in i_ if x < mc["min_iou_agree"])
        log(f"    {name:<16} 면적 median={statistics.median(a_):.3f} [{min(a_):.3f}~{max(a_):.3f}] | IoU(vs {mc['method']}) median={statistics.median(i_):.3f} min={min(i_):.3f} | IoU<{mc['min_iou_agree']}: {low}장")
    log("    Otsu 채도 임계값 분포:", f"median={statistics.median(s['thr_otsu_s'] for s in stats):.0f} min={min(s['thr_otsu_s'] for s in stats):.0f} max={max(s['thr_otsu_s'] for s in stats):.0f}",
        "| Otsu 명도:", f"median={statistics.median(s['thr_otsu_v'] for s in stats):.0f} min={min(s['thr_otsu_v'] for s in stats):.0f} max={max(s['thr_otsu_v'] for s in stats):.0f}")
    log("\n[4] 플래그")
    tb = [s for s in stats if s["touches_border"]]
    log(f"    마스크가 이미지 경계에 닿음(프레임 잘림 가능): {len(tb)}장 — 방향별 L/R/T/B:",
        sum(s['touch_left'] for s in stats), sum(s['touch_right'] for s in stats), sum(s['touch_top'] for s in stats), sum(s['touch_bottom'] for s in stats))
    mb = [s for s in stats if s["n_components"] > 1 and s["second_ratio"] >= 0.05]
    log(f"    큰 덩어리가 2개 이상(둘째/첫째 면적 >= 5%): {len(mb)}장")
    for s in mb:
        log(f"      {s['path']} 덩어리={s['n_components']} 둘째/첫째={s['second_ratio']}")
    inex = [s for s in stats if s["mask_px_in_exclude"] > 0]
    log(f"    마스크가 스케일바 제외 구역과 겹침: {len(inex)}장 (겹침 픽셀 합계 {sum(s['mask_px_in_exclude'] for s in inex)})")
    dis = [s for s in stats if any(s[f"iou_{n}_vs_default"] < mc["min_iou_agree"] for n in names)]
    log(f"    방법 간 IoU<{mc['min_iou_agree']} 인 이미지(임계값 민감): {len(dis)}장")
    log(f"\n저장: {o} (masks/, overlays/ 0.5배, mask_stats.csv)")


if __name__ == "__main__":
    main()
