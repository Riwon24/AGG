"""jv07 준비: jv06 의 타일별 점수에서 오류 패턴을 눈으로 보기 위한 확인용 시트를 만든다.

- FP 후보: 라벨이 AGG0 인 이미지의 타일 중 p_flat(두 순서 평균)이 가장 높은 타일. 이미지당 최대 2개, 12개.
- FN 후보: 라벨이 AGG1 또는 AGG2 인 이미지의 타일 중 p_flat 이 가장 낮은 타일. 이미지당 최대 2개, 12개.
- 이미지는 원본 사진에서 타일(224px)을 잘라 2배 확대(448px)해 4열 x 3행 시트 하나에 담는다. 칸 아래에 이미지 이름, 타일 번호, p_flat, p_A, p_B, mask_frac 를 쓴다.
- 이 목록은 평가 이미지(60장)에서 고른 것이다. 여기서 본 오류로 설명문을 보강하면 같은 이미지로 평가할 때 점수가 낙관적이 된다 (jv07 보고서에 명시).
출력: <out-dir>/FP_AGG0_high.png, FN_AGG12_low.png, review_tiles.csv
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def pick(rows, label_ok, key, reverse, n=12, per_image=2):
    cand = sorted((r for r in rows if label_ok(int(r["label"][3:]))), key=key, reverse=reverse)
    cnt, out = {}, []
    for r in cand:
        if cnt.get(r["image_id"], 0) >= per_image:
            continue
        cnt[r["image_id"]] = cnt.get(r["image_id"], 0) + 1; out.append(r)
        if len(out) == n:
            break
    return out


def sheet(sel, path, title):
    S, cols, pad, cap = 448, 4, 6, 64
    rowsn = (len(sel) + cols - 1) // cols
    W, H = cols * (S + pad) + pad, rowsn * (S + cap + pad) + pad + 34
    im = np.full((H, W, 3), 255, np.uint8)
    cv2.putText(im, title, (pad, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    cache = {}
    for k, r in enumerate(sel):
        if r["image_id"] not in cache:
            a = cv2.imdecode(np.fromfile(str(ROOT / (r["image_id"] + ".jpg")), np.uint8), cv2.IMREAD_COLOR); cache = {r["image_id"]: a}
        x0, y0 = int(r["x0"]), int(r["y0"])
        t = cv2.resize(cache[r["image_id"]][y0:y0 + 224, x0:x0 + 224], (S, S), interpolation=cv2.INTER_CUBIC)
        cx, cy = pad + (k % cols) * (S + pad), 34 + pad + (k // cols) * (S + cap + pad)
        im[cy:cy + S, cx:cx + S] = t
        iid = r["image_id"].replace(" - ", " ")
        lines = [f"{k + 1}. {r['label']} {iid[:44]}", f"{r['tile_id']}  p={float(r['p_flat']):.2f} A={float(r['p_A']):.2f} B={float(r['p_B']):.2f}  mask={float(r['mask_frac']):.2f}"]
        for j, s in enumerate(lines):
            cv2.putText(im, s, (cx, cy + S + 20 + 22 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imencode(".png", im)[1].tofile(str(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tile-scores", required=True); ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(a.tile_scores, encoding="utf-8-sig")))
    fp = pick(rows, lambda l: l == 0, lambda r: (float(r["p_flat"]), float(r["p_A"])), True)
    fn = pick(rows, lambda l: l >= 1, lambda r: (float(r["p_flat"]), float(r["p_A"])), False)
    inner = [r for r in rows if float(r["mask_frac"]) >= 0.999999]
    fpi = pick(inner, lambda l: l == 0, lambda r: (float(r["p_flat"]), float(r["p_A"])), True)
    sheet(fpi, out / "FP_AGG0_high_inner.png", "AGG0 images, specimen-only tiles (mask 1.0): highest p_flat (max 2 per image)")
    print("FP(안쪽 타일만) p_flat 범위", min(float(r["p_flat"]) for r in fpi), "~", max(float(r["p_flat"]) for r in fpi), "| 이미지", len({r["image_id"] for r in fpi}), "장")
    sheet(fp, out / "FP_AGG0_high.png", "AGG0 images: highest p_flat tiles (max 2 per image)")
    sheet(fn, out / "FN_AGG12_low.png", "AGG1+2 images: lowest p_flat tiles (max 2 per image)")
    with open(out / "review_tiles.csv", "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh); w.writerow(["kind", "no", "image_id", "label", "tile_id", "x0", "y0", "mask_frac", "p_flat", "p_A", "p_B"])
        for kind, sel in (("FP", fp), ("FN", fn)):
            for k, r in enumerate(sel, 1):
                w.writerow([kind, k, r["image_id"], r["label"], r["tile_id"], r["x0"], r["y0"], r["mask_frac"], r["p_flat"], r["p_A"], r["p_B"]])
    for kind, sel in (("FP", fp), ("FN", fn)):
        print(kind, "p_flat 범위", min(float(r["p_flat"]) for r in sel), "~", max(float(r["p_flat"]) for r in sel), "| 안쪽 타일", sum(float(r["mask_frac"]) >= 0.999999 for r in sel), "/", len(sel), "| 이미지", len({r["image_id"] for r in sel}), "장")


if __name__ == "__main__":
    main()
