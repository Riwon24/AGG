"""s04: 유효 타일 좌표만 계산한다 (픽셀을 자르거나 저장하지 않는다. QC 용 격자 overlay 는 클래스별 샘플에 한해 저장).

- 타일은 원본 해상도 그대로(resize 없음). 격자는 config tiling.grid_origin(top_left)에서 시작한다.
- 파라미터 조합마다 param_id = t<타일>_s<stride>_m<침식마진> 를 붙인다 (tile_size x stride x erosion_margin_px).
- valid 규칙: 타일 전체가 (안쪽으로 침식한 마스크) 안에 100% 들어오고, 스케일바 제외 구역과 겹치지 않을 때만 valid.
  침식은 config tiling.erode_frame_as_boundary=false 이면 이미지 프레임을 시편 경계로 보지 않는다.
- 배경을 칠하거나 지우지 않는다. 불량 타일은 제외만 한다 (제외 사유를 기록).
- 마스크는 s02 결과(outputs/s02_specimen_mask/masks, override 반영)를 그대로 쓴다.
산출: tiles.csv, tile_counts.csv(이미지 x 조합별 valid 수), qc_grid_*.png, log.txt
"""
import argparse
import csv
import statistics
import time
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from common import Logger, PREPROC_DIR, farthest_point, load_config, safe_id

SCRIPT = "s04_tile_grid"


def imwrite(path, img, retries=6):
    buf = cv2.imencode(".png", img)[1]
    for i in range(retries):
        try:
            buf.tofile(str(path)); return
        except OSError:
            if i == retries - 1:
                raise
            time.sleep(0.5 * (i + 1))


def erode(mask, margin, frame_as_boundary):
    if margin == 0:
        return mask.copy()
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1,) * 2)
    bv = 0 if frame_as_boundary else 1   # 1: 프레임 밖을 시편으로 간주(프레임에서는 침식 안 됨)
    return cv2.erode(mask.astype(np.uint8), k, borderType=cv2.BORDER_CONSTANT, borderValue=bv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", help="manifest_v1.csv (기본: outputs/s00b_meta_mag/manifest_v1.csv)")
    ap.add_argument("--masks", help="마스크 폴더 (기본: outputs/s02_specimen_mask/masks)")
    ap.add_argument("--input", help="이미지 루트: QC overlay 배경용 (기본: config paths.image_root)")
    ap.add_argument("--output", help="outputs 폴더 (기본: config paths.outputs_dir)")
    ap.add_argument("--config", default=str(PREPROC_DIR / "config.yaml"))
    a = ap.parse_args()

    cfg = load_config(a.config)
    root = Path(a.input or cfg["paths"]["image_root"])
    out = a.output or cfg["paths"]["outputs_dir"]
    manifest = Path(a.manifest or Path(out) / "s00b_meta_mag" / "manifest_v1.csv")
    masks_dir = Path(a.masks or Path(out) / "s02_specimen_mask" / "masks")
    tc = cfg["tiling"]
    use = cfg["experiment"]["use_mags"]
    W, H = cfg["naming"]["expected_resolution"]
    ex = cfg["scalebar"]["exclude_box"]
    assert ex and len(ex) == 4, "config scalebar.exclude_box 가 비어 있음 (s01 필요)"
    assert tc["grid_origin"] == "top_left", "grid_origin 은 top_left 만 지원"
    frame_b = bool(tc["erode_frame_as_boundary"])
    log = Logger(out, SCRIPT)
    o = Path(out) / SCRIPT
    log("config:", cfg["_config_path"], "| manifest:", manifest, "| masks:", masks_dir)
    log(f"tile_size={tc['tile_size']} stride={tc['stride']} margin={tc['erosion_margin_px']} origin={tc['grid_origin']} "
        f"erode_frame_as_boundary={frame_b} | 스케일바 제외 구역={ex} | 마스크 방법={cfg['mask']['method']}")

    # 마스크 출처 추적성
    s2 = Path(out) / "s02_specimen_mask" / "mask_stats.csv"
    if s2.exists():
        st = list(csv.DictReader(open(s2, encoding="utf-8-sig")))
        log("s02 마스크 출처:", dict(Counter(r["mask_source"] for r in st)), "| 기본 방법:", {r["method"] for r in st})
        log("  override 적용:", [r["image_id"] for r in st if r["mask_source"] != "auto"])

    rows = [r for r in csv.DictReader(open(manifest, encoding="utf-8-sig")) if r["mag_final"] in use]
    params = [(t, s, m) for t in tc["tile_size"] for s in tc["stride"] for m in tc["erosion_margin_px"]]
    pid = lambda t, s, m: f"t{t}_s{s}_m{m}"
    log(f"대상 이미지 {len(rows)}장 ({'/'.join(use)}x) | 파라미터 조합 {len(params)}개: {[pid(*p) for p in params]}")
    log("격자 끝 남는 픽셀(오른쪽/아래): " + ", ".join(f"t{t}s{s}: x {W - ((W - t) // s * s + t)}px, y {H - ((H - t) // s * s + t)}px" for t in tc["tile_size"] for s in tc["stride"]))

    cap = tc.get("cap_per_image")
    sel_method = tc.get("selection_method", "random_seeded")
    assert sel_method in ("random_seeded", "nonoverlap_best_phase"), f"지원하지 않는 selection_method: {sel_method}"
    seed = int(cfg.get("seed", 0))
    log(f"타일 선택 방식 selection_method={sel_method} | 이미지당 상한 cap_per_image={cap} | seed={seed} (random_seeded 에서만 사용)")
    tiles, counts = [], []
    for r in rows:
        sid = safe_id(r["image_id"])
        m = cv2.imdecode(np.fromfile(str(masks_dir / f"{sid}.png"), dtype=np.uint8), cv2.IMREAD_GRAYSCALE) > 127
        assert m.shape == (H, W), f"마스크 크기 이상: {sid}"
        for margin in tc["erosion_margin_px"]:
            e = erode(m, margin, frame_b)
            integ = cv2.integral(e.astype(np.uint8))   # (H+1, W+1) int32
            for t in tc["tile_size"]:
                for s in tc["stride"]:
                    p = pid(t, s, margin)
                    nv = 0
                    cur, rcs = [], []
                    for ri, y0 in enumerate(range(0, H - t + 1, s)):
                        for ci, x0 in enumerate(range(0, W - t + 1, s)):
                            rcs.append((ri, ci))
                            x1, y1 = x0 + t, y0 + t
                            assert x1 <= W and y1 <= H, "타일이 이미지 밖"
                            inside = int(integ[y1, x1] - integ[y0, x1] - integ[y1, x0] + integ[y0, x0])
                            frac = inside / (t * t)
                            reasons = []
                            if inside != t * t:
                                reasons.append("outside_mask")
                            if x0 < ex[2] and x1 > ex[0] and y0 < ex[3] and y1 > ex[1]:
                                reasons.append("scalebar_overlap")
                            valid = not reasons
                            nv += valid
                            cur.append((r["image_id"], r["label"], f"{p}_r{ri:02d}_c{ci:02d}", x0, y0, t, t,
                                        int(valid), "+".join(reasons), p, round(frac, 4)))
                    # ---- 이미지당 타일 선택 ----
                    vidx = [i for i, tt in enumerate(cur) if tt[7]]
                    pool = vidx
                    if sel_method == "nonoverlap_best_phase":
                        # 겹치지 않는 부분 격자(격자 인덱스가 가로세로 한 칸씩 건너뜀) 4가지 오프셋 중 valid 가 가장 많은 것.
                        # 동점이면 (0,0),(0,1),(1,0),(1,1) 순서로 먼저인 것 -> 결정적
                        assert t == 2 * s, "nonoverlap_best_phase 는 tile_size == 2*stride 일 때만 겹치지 않음"
                        pool = []
                        for pr in (0, 1):
                            for pc in (0, 1):
                                sub = [i for i in vidx if rcs[i][0] % 2 == pr and rcs[i][1] % 2 == pc]
                                if len(sub) > len(pool):
                                    pool = sub
                        if cap is not None and len(pool) > cap:   # 상한: 서로 가장 멀리 떨어지게 (결정적)
                            keep = farthest_point([(cur[i][3], cur[i][4]) for i in pool], cap, t)
                            chosen = set(pool[j] for j in keep)
                        else:
                            chosen = set(pool)
                    elif cap is not None and len(vidx) > cap:   # random_seeded: seed 고정 무작위 비복원 추출 (이미지/조합별 seed 파생)
                        rs = np.random.RandomState(zlib.crc32(f"{seed}|{r['image_id']}|{p}".encode()) % (2 ** 32))
                        chosen = set(int(i) for i in rs.choice(vidx, size=cap, replace=False))
                    else:
                        chosen = set(vidx)
                    for i, tt in enumerate(cur):
                        tiles.append(tt + (int(i in chosen),))
                    ntot = len(range(0, H - t + 1, s)) * len(range(0, W - t + 1, s))
                    counts.append(dict(image_id=r["image_id"], label=r["label"], param_id=p, n_tiles=ntot, n_valid=nv,
                                       n_pool=len(pool), n_selected=len(chosen)))

    # ---------- 저장 ----------
    with open(o / "tiles.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["image_id", "label", "tile_id", "x0", "y0", "w", "h", "valid", "exclude_reason", "param_id", "mask_frac", "selected"])
        w.writerows(tiles)
    with open(o / "tile_counts.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=["image_id", "label", "param_id", "n_tiles", "n_valid", "n_pool", "n_selected"])
        w.writeheader(); w.writerows(counts)
    log(f"\n[1] tiles.csv: {len(tiles)}행 (이미지 {len(rows)} x 조합별 격자) -> {o / 'tiles.csv'}")

    # ---------- 요약 / 가정 확인 ----------
    log("\n[2] 파라미터 조합별 valid 타일 수 (이미지당)")
    by_p = defaultdict(list)
    for c in counts:
        by_p[c["param_id"]].append(c)
    order = lambda x: (int(x.split("_")[0][1:]), int(x.split("_")[1][1:]), int(x.split("_")[2][1:]))   # (타일, stride, 마진)
    for p in sorted(by_p, key=order):
        v =[c["n_valid"] for c in by_p[p]]; nt = by_p[p][0]["n_tiles"]
        z = [c["image_id"] for c in by_p[p] if c["n_valid"] == 0]
        log(f"    {p:<14} 격자 {nt:>3}개/장 | valid 합계 {sum(v):>5} | 평균 {statistics.mean(v):6.1f} (비율 {100*statistics.mean(v)/nt:4.1f}%) "
            f"min {min(v):>3} median {statistics.median(v):>5.1f} max {max(v):>3} | 0개인 이미지 {len(z)}장")
        if z:
            log("      !! valid 타일이 0인 이미지:", z[:8])
    log("\n[3] 제외 사유 (조합별, 전체 타일 중)")
    rc = defaultdict(Counter)
    for t in tiles:
        rc[t[9]]["valid" if t[7] else t[8]] += 1
    for p in sorted(rc, key=order):
        n = sum(rc[p].values())
        log(f"    {p:<14} " + ", ".join(f"{k} {v} ({100*v/n:.1f}%)" for k, v in sorted(rc[p].items(), key=lambda kv: -kv[1])))
    log("\n[4] 클래스별 valid 타일 평균 (이미지당)")
    for p in sorted(by_p, key=order):
        parts = []
        for lab in sorted({c["label"] for c in by_p[p]}):
            v = [c["n_valid"] for c in by_p[p] if c["label"] == lab]
            parts.append(f"AGG{lab}: {statistics.mean(v):.1f} (n={len(v)})")
        log(f"    {p:<14} " + " | ".join(parts))
    # 가정 확인
    assert all(0 <= t[3] and t[3] + t[5] <= W and 0 <= t[4] and t[4] + t[6] <= H for t in tiles), "타일이 이미지 범위 밖"
    assert all(t[5] == t[6] for t in tiles)
    assert all((t[7] == 1) == (t[8] == "") for t in tiles)
    vfrac = sum(t[7] for t in tiles) / len(tiles)
    log(f"\n[5] 가정 확인: 모든 타일이 이미지 범위 안(assert 통과), valid 비율(전체) {100*vfrac:.1f}%")
    if vfrac < 0.05 or vfrac > 0.95:
        log.warn("전체 valid 비율이 비정상적임:", vfrac)
    # valid 타일의 마스크 비율은 정의상 1.0
    assert all(abs(t[10] - 1.0) < 1e-9 for t in tiles if t[7]), "valid 타일인데 마스크 비율이 1이 아님"
    assert all(t[11] <= t[7] for t in tiles), "선택된 타일이 valid 가 아님"
    for c in counts:
        # random_seeded: 선택 수 = min(valid, 상한). nonoverlap_best_phase: 선택 수 = min(겹치지 않는 부분 격자 크기, 상한)
        base = c["n_valid"] if sel_method == "random_seeded" else c["n_pool"]
        assert c["n_selected"] == (base if cap is None else min(base, cap)), f"상한 적용 이상: {c}"
        assert c["n_selected"] > 0, f"선택된 타일이 없는 이미지: {c}"
    log(f"    상한 적용 assert 통과: 모든 이미지/조합에서 선택 수 = min({'valid' if sel_method == 'random_seeded' else '겹치지 않는 부분 격자 크기'}, 상한)")
    if sel_method == "nonoverlap_best_phase":
        by_img = defaultdict(list)
        for t in tiles:
            if t[11]:
                by_img[(t[0], t[9])].append((t[3], t[4], t[5]))
        for key, lst in by_img.items():
            for i in range(len(lst)):
                for j in range(i + 1, len(lst)):
                    assert abs(lst[i][0] - lst[j][0]) >= lst[i][2] or abs(lst[i][1] - lst[j][1]) >= lst[i][2], f"선택 타일이 겹침: {key}"
        npool = [c["n_pool"] for c in counts]
        log(f"    겹침 assert 통과: 같은 이미지의 선택 타일끼리 겹치는 쌍 0개 | 겹치지 않는 부분 격자 크기(상한 전) 이미지당 평균 {statistics.mean(npool):.1f}, 최소 {min(npool)}, 최대 {max(npool)}")
        if cap is not None and min(npool) < cap:
            log.warn(f"일부 이미지는 부분 격자 크기가 상한({cap})보다 작아 선택 수가 상한에 못 미침: {sum(1 for x in npool if x < cap)}장")
    log("\n[5b] 선택(selected) 타일 수 (이미지당 상한 적용 후)")
    for p in sorted(by_p, key=order):
        parts = []
        for lab in sorted({c["label"] for c in by_p[p]}):
            v = [c["n_selected"] for c in by_p[p] if c["label"] == lab]
            parts.append(f"AGG{lab}: 합계 {sum(v)} (이미지 {len(v)}장, 평균 {statistics.mean(v):.1f})")
        tot = sum(c["n_selected"] for c in by_p[p])
        log(f"    {p:<14} 전체 {tot} | " + " | ".join(parts))

    # ---------- QC overlay (클래스별 샘플) ----------
    k_img = int(tc["qc_overlay_images_per_class"])
    ar = {r["image_id"]: None for r in rows}
    if s2.exists():
        ar = {r["image_id"]: float(r["area_ratio"]) for r in st}
    tl_by = defaultdict(list)
    for t in tiles:
        tl_by[(t[0], t[9])].append(t)
    qc_params = [pid(t, s, m) for t in tc["tile_size"][:1] for s in tc["stride"] for m in tc["erosion_margin_px"]]
    for lab in sorted({r["label"] for r in rows}):
        cand = [r for r in rows if r["label"] == lab]
        med = statistics.median(ar[r["image_id"]] for r in cand)
        for r in sorted(cand, key=lambda r: abs(ar[r["image_id"]] - med))[:k_img]:
            img = cv2.imdecode(np.fromfile(str(root / r["path"]), dtype=np.uint8), cv2.IMREAD_COLOR)
            m = cv2.imdecode(np.fromfile(str(masks_dir / f"{safe_id(r['image_id'])}.png"), dtype=np.uint8), cv2.IMREAD_GRAYSCALE) > 127
            panels = []
            for p in qc_params:
                mg = int(p.split("_m")[1]); e = erode(m, mg, frame_b)
                vis = img.copy()
                cn, _ = cv2.findContours(e.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for t in tl_by[(r["image_id"], p)]:
                    col = (0, 200, 0) if t[11] else ((160, 160, 0) if t[7] else (0, 0, 220))   # 선택=초록 굵게, valid지만 상한에서 빠짐=청록 가늘게, 제외=빨강 가늘게
                    cv2.rectangle(vis, (t[3], t[4]), (t[3] + t[5] - 1, t[4] + t[6] - 1), col, 2 if t[11] else 1)
                cv2.drawContours(vis, cn, -1, (0, 255, 255), 3)
                cv2.rectangle(vis, (ex[0], ex[1]), (ex[2] - 1, ex[3] - 1), (255, 0, 255), 3)
                nv = f"{sum(t[7] for t in tl_by[(r['image_id'], p)])} 선택={sum(t[11] for t in tl_by[(r['image_id'], p)])}"
                vis = cv2.resize(vis, (608, 456), interpolation=cv2.INTER_AREA)
                cv2.rectangle(vis, (0, 0), (608, 18), (0, 0, 0), -1)
                cv2.putText(vis, f"{p} valid={nv}", (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                panels.append(vis)
            cols = 3
            rws = (len(panels) + cols - 1) // cols
            sheet = np.zeros((rws * 456, cols * 608, 3), np.uint8)
            for i, pn in enumerate(panels):
                sheet[(i // cols) * 456:(i // cols + 1) * 456, (i % cols) * 608:(i % cols + 1) * 608] = pn
            fn = f"qc_grid_AGG{lab}_{safe_id(r['image_id'])[:60]}.png"
            imwrite(o / fn, sheet)
            log(f"    QC overlay: {fn} (초록=valid, 빨강=제외, 노랑=침식 마스크, 자홍=스케일바 제외 구역)")
    log(f"\n저장: {o}")


if __name__ == "__main__":
    main()
