"""m02_tiles: m01 마스크로 타일 좌표를 계산한다 (10x 60장). 픽셀을 자르거나 저장하지 않고 좌표와 마스크 비율만 낸다.

- 격자: tile_size x tile_size 타일을 stride 간격으로 (0,0) 에서 시작해 깐다 (224/112 -> 이미지당 15 x 11 = 165개). 겹침 50%.
- 마스크 비율 = 타일 안에서 마스크(m01 결과)에 속하는 픽셀의 비율. 스케일바 제외 구역 픽셀은 마스크에서 뺀다
  (m01 마스크는 60장 모두 이 구역과 겹치지 않으므로 보통 변화가 없고, 바뀌면 로그에 남긴다).
- 선택 = 마스크 비율 >= min_mask_frac. 침식 마진은 쓰지 않는다.
- 패치 라벨은 없다. label 열은 이미지 라벨(0/1/2)을 그대로 복사한 것이다. 분할은 이미지 단위로 해야 한다.
- 특징 추출 단계에서는 마스크 안쪽 픽셀만 쓴다 (이 단계는 좌표와 비율만 만든다).
산출: tiles.csv, tile_counts.csv, qc_grid_AGG<c>_*.png, log.txt
"""
import argparse
import csv
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

ML_DIR = Path(__file__).resolve().parent
SCRIPT = "m02_tiles"


def load_ml_config(path):
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg["paths"] = {k: str((path.parent / v).resolve()) for k, v in cfg["paths"].items()}
    cfg["_config_path"] = str(path)
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ML_DIR / "config.yaml"))
    ap.add_argument("--output", help="outputs 폴더 (기본: config paths.outputs_dir)")
    a = ap.parse_args()

    cfg = load_ml_config(a.config)
    pre_cfg_path = Path(cfg["preproc"]["config"])
    sys.path.insert(0, str(pre_cfg_path.parent))
    from common import Logger, imwrite_png, load_config, safe_id
    from s02_specimen_mask import imread

    pre = load_config(pre_cfg_path)
    tc = cfg["tiling"]
    root = Path(pre["paths"]["image_root"])
    out = a.output or cfg["paths"]["outputs_dir"]
    W, H = pre["naming"]["expected_resolution"]
    ex = pre["scalebar"]["exclude_box"]
    t, s, thr = int(tc["tile_size"]), int(tc["stride"]), float(tc["min_mask_frac"])
    assert tc["grid_origin"] == "top_left", "grid_origin 은 top_left 만 지원"
    log = Logger(out, SCRIPT)
    o = Path(out) / SCRIPT
    m01 = Path(out) / "m01_masks"
    log("config:", cfg["_config_path"], "| preproc config:", pre["_config_path"])
    log(f"tile={t} stride={s} (겹침 {100 * (1 - s / t):.0f}%) min_mask_frac={thr} | 스케일바 제외 구역={ex} | 마스크: {m01 / 'masks'}")

    sel = list(csv.DictReader(open(m01 / "selection.csv", encoding="utf-8-sig")))
    path_of = {r["image_id"]: r["path"] for r in csv.DictReader(open(Path(pre["paths"]["outputs_dir"]) / "s02_specimen_mask" / "mask_stats.csv", encoding="utf-8-sig"))}
    log(f"대상 이미지 {len(sel)}장 (m01 selection.csv)")
    xs, ys = list(range(0, W - t + 1, s)), list(range(0, H - t + 1, s))
    log(f"격자: x {len(xs)}개 x y {len(ys)}개 = {len(xs) * len(ys)}개/장 | 격자 끝 남는 픽셀: 오른쪽 {W - (xs[-1] + t)}px, 아래 {H - (ys[-1] + t)}px")

    tiles, counts, masks_cache = [], [], {}
    rng = np.random.RandomState(0)
    n_checked = 0
    for r in sel:
        iid, sid = r["image_id"], safe_id(r["image_id"])
        m = cv2.imdecode(np.fromfile(str(m01 / "masks" / f"{sid}.png"), dtype=np.uint8), cv2.IMREAD_GRAYSCALE) > 127
        assert m.shape == (H, W), f"마스크 크기 이상: {sid}"
        n_sb = int(m[ex[1]:ex[3], ex[0]:ex[2]].sum())
        if n_sb:
            log.warn(f"마스크가 스케일바 제외 구역과 겹침 ({n_sb}px, 제외함): {iid}")
            m = m.copy(); m[ex[1]:ex[3], ex[0]:ex[2]] = False
        integ = cv2.integral(m.astype(np.uint8))
        cur = []
        for ri, y0 in enumerate(ys):
            for ci, x0 in enumerate(xs):
                x1, y1 = x0 + t, y0 + t
                assert x1 <= W and y1 <= H, "타일이 이미지 밖"
                inside = int(integ[y1, x1] - integ[y0, x1] - integ[y1, x0] + integ[y0, x0])
                frac = inside / (t * t)
                sb = int(x0 < ex[2] and x1 > ex[0] and y0 < ex[3] and y1 > ex[1])
                cur.append((iid, r["label"], f"r{ri:02d}_c{ci:02d}", x0, y0, t, t, round(frac, 6), int(frac >= thr), sb, inside))
        # 적분 영상 검증: 무작위 타일 일부를 직접 합산해 비교
        for k in rng.choice(len(cur), size=min(4, len(cur)), replace=False):
            _, _, _, x0, y0, _, _, fr, _, _, px = cur[int(k)]
            assert px == int(m[y0:y0 + t, x0:x0 + t].sum()), f"적분 영상 합이 직접 합산과 다름: {iid}"
            n_checked += 1
        tiles += cur
        nsel = sum(c[8] for c in cur)
        counts.append(dict(image_id=iid, label=r["label"], n_tiles=len(cur), n_selected=nsel, n_full=sum(1 for c in cur if c[10] == t * t),
                           n_partial_selected=sum(1 for c in cur if c[8] and c[10] < t * t), n_scalebar_overlap_selected=sum(1 for c in cur if c[8] and c[9])))
        masks_cache[iid] = (m, cur)

    with open(o / "tiles.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["image_id", "label", "tile_id", "x0", "y0", "w", "h", "mask_frac", "selected", "overlaps_scalebar", "mask_px"])
        w.writerows(tiles)
    with open(o / "tile_counts.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(counts[0].keys()))
        w.writeheader(); w.writerows(counts)

    # ---------- 가정 확인 ----------
    assert all(0 <= tl[3] and tl[3] + tl[5] <= W and 0 <= tl[4] and tl[4] + tl[6] <= H for tl in tiles), "타일이 이미지 범위 밖"
    assert all(c["n_selected"] > 0 for c in counts), "선택된 타일이 없는 이미지"
    assert all(tl[8] == (1 if tl[10] / (t * t) >= thr else 0) for tl in tiles), "선택 규칙 불일치"
    assert len({(tl[0], tl[2]) for tl in tiles}) == len(tiles), "타일 ID 중복"
    log(f"\n[0] assert 통과: 타일이 이미지 범위 안, 이미지마다 선택 >= 1, 선택 규칙 일치, 타일 ID 중복 없음, 적분 영상 직접 합산 검증 {n_checked}개")

    # ---------- 요약 ----------
    log(f"\n[1] tiles.csv: {len(tiles)}행 (이미지 {len(sel)} x 격자 {len(xs) * len(ys)})")
    nsel = [c["n_selected"] for c in counts]
    tot_sel = sum(nsel)
    log(f"[2] 선택 타일 {tot_sel}개 ({100 * tot_sel / len(tiles):.1f}%) | 이미지당 min {min(nsel)} median {statistics.median(nsel):.0f} mean {statistics.mean(nsel):.1f} max {max(nsel)}")
    for lab in sorted({c["label"] for c in counts}):
        cc = [c for c in counts if c["label"] == lab]
        log(f"    AGG{lab}: 이미지 {len(cc)}장, 선택 타일 {sum(c['n_selected'] for c in cc)}개 (이미지당 평균 {statistics.mean(c['n_selected'] for c in cc):.1f})")
    full = sum(c["n_full"] for c in counts)
    part = sum(c["n_partial_selected"] for c in counts)
    log(f"[3] 선택 타일 중 마스크 100%인 타일 {tot_sel - part}개, 일부만 마스크({100 * thr:.0f}%~100% 미만, 임계값이 100% 이면 0개)인 타일 {part}개 ({100 * part / tot_sel:.1f}%)")
    fr_sel = [tl[7] for tl in tiles if tl[8]]
    log(f"    선택 타일의 마스크 비율: median {statistics.median(fr_sel):.3f} min {min(fr_sel):.3f} | 비율 0.75 미만 {sum(1 for x in fr_sel if x < 0.75)}개, 0.75~1.0 미만 {sum(1 for x in fr_sel if 0.75 <= x < 1.0)}개")
    sb_sel = sum(c["n_scalebar_overlap_selected"] for c in counts)
    log(f"[4] 스케일바 제외 구역과 겹치지만 선택된 타일: {sb_sel}개 (마스크가 스케일바를 포함하지 않으므로 해당 픽셀은 특징에서 제외됨)")
    by_class_unsel = Counter(tl[1] for tl in tiles if not tl[8])
    log(f"    선택되지 않은 타일: {sum(by_class_unsel.values())}개 {dict(by_class_unsel)}")

    # ---------- QC overlay (클래스별: 마스크 면적 중앙값에 가까운 이미지) ----------
    area = {r["image_id"]: float(r["area"]) for r in sel}
    k_img = int(tc["qc_images_per_class"])
    for lab in sorted({r["label"] for r in sel}):
        cand = [r for r in sel if r["label"] == lab]
        med = statistics.median(area[r["image_id"]] for r in cand)
        for r in sorted(cand, key=lambda r: abs(area[r["image_id"]] - med))[:k_img]:
            img = imread(root / path_of[r["image_id"]])
            m, cur = masks_cache[r["image_id"]]
            vis = img.copy()
            for c in cur:
                col = (0, 200, 0) if c[8] else (0, 0, 220)
                cv2.rectangle(vis, (c[3], c[4]), (c[3] + c[5] - 1, c[4] + c[6] - 1), col, 2 if c[8] else 1)
            cn, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cn, -1, (0, 255, 255), 3)
            cv2.rectangle(vis, (ex[0], ex[1]), (ex[2] - 1, ex[3] - 1), (255, 0, 255), 3)
            vis = cv2.resize(vis, (912, 684), interpolation=cv2.INTER_AREA)
            cv2.rectangle(vis, (0, 0), (912, 20), (0, 0, 0), -1)
            ns = sum(c[8] for c in cur)
            cv2.putText(vis, f"AGG{lab} {r['image_id'][:50]} selected={ns}/{len(cur)}", (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            fn = f"qc_grid_AGG{lab}_{safe_id(r['image_id'])[:50]}.png"
            imwrite_png(o / fn, vis)
            log(f"    QC overlay: {fn} (초록=선택, 빨강=비선택, 노랑=마스크 윤곽, 자홍=스케일바 구역)")
    import hashlib
    h = hashlib.sha256((o / "tiles.csv").read_bytes()).hexdigest()
    log(f"\ntiles.csv sha256: {h[:12]}… (같은 입력이면 같아야 함)")
    log(f"저장: {o}")


if __name__ == "__main__":
    main()
