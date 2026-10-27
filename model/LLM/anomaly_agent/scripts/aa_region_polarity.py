"""aa_region_polarity: 사용자가 표시한 AGG 영역(agg_regions.json)이 주변 타일 표면보다 어두운지 밝은지, 질감이 매끈한지 숫자로 센다.

타일 안에서 AGG 상자 바깥(타일 가장자리 어두운 배경은 제외)의 밝기와 비교한다.
  밝기차 = 상자 안 평균 밝기 - 상자 바깥 평균 밝기 (표면 픽셀만),  z = 밝기차 / 상자 바깥 '같은 크기 창' 평균 밝기의 표준편차
  매끈함 = 상자 안 고주파 표준편차 / 상자 바깥 고주파 표준편차 (1보다 작으면 더 매끈함)
출력: outputs/aa_caption_tiles/agg_region_polarity.csv 와 요약
주의: 상자는 사람이 손으로 그린 것이라 경계가 거칠다.
"""
import csv
import json
import os

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, uniform_filter

D = "/mnt/c/Users/admin/Downloads/260929_images_for_AGG - manual labeling/AGG_labeled_only/exp_aa/outputs/aa_caption_tiles"
reg = json.load(open(os.path.join(D, "agg_regions.json"), encoding="utf-8"))["tiles"]
rows = []
for name, v in sorted(reg.items()):
    if not v["boxes"]:
        continue
    im = np.asarray(Image.open(os.path.join(D, "tiles", name + ".png")).convert("L"), dtype=np.float32)
    H, W = im.shape
    surface = gaussian_filter(im, 6) > 40                      # 어두운 배경(가장자리) 제외
    hp = im - gaussian_filter(im, 3)                           # 고주파(결 질감)
    mask = np.zeros_like(surface)
    for b in v["boxes"]:
        mask[int(b[1] * H):int(b[3] * H), int(b[0] * W):int(b[2] * W)] = True
    inside = mask & surface
    outside = ~mask & surface
    if inside.sum() < 50:
        continue
    diff = float(im[inside].mean() - im[outside].mean())
    ws = max(8, int(np.sqrt(mask.sum())))
    loc = uniform_filter(im, ws)
    ok = outside & (np.arange(H)[:, None] > ws // 2) & (np.arange(H)[:, None] < H - ws // 2) & (np.arange(W)[None, :] > ws // 2) & (np.arange(W)[None, :] < W - ws // 2)
    sd = float(loc[ok].std()) if ok.sum() > 100 else float("nan")
    # 상자 바로 주변 띠(상자를 상자 변 길이의 0.6배만큼 넓힌 고리)와만 비교 -> 조명 기울기 영향을 줄인다
    from scipy.ndimage import binary_dilation
    ring = binary_dilation(mask, iterations=max(4, int(0.6 * np.sqrt(mask.sum())))) & ~mask & surface
    rdiff = float(im[inside].mean() - im[ring].mean()) if ring.sum() > 50 else float("nan")
    rows.append({"tile": name, "AGG면적비": round(mask.mean(), 3), "밝기차(타일 전체 대비)": round(diff, 1), "z": round(diff / sd, 2) if sd == sd else None,
                 "밝기차(주변 띠 대비)": round(rdiff, 1), "방향(주변 띠, ±3)": "어두움" if rdiff < -3 else "밝음" if rdiff > 3 else "비슷",
                 "매끈함(안/밖 고주파)": round(float(hp[inside].std() / hp[outside].std()), 2)})
with open(os.path.join(D, "agg_region_polarity.csv"), "w", encoding="utf-8-sig", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
for r in rows:
    print(r)
dirs = [r["방향(주변 띠, ±3)"] for r in rows]
print("\n어두움", dirs.count("어두움"), "| 밝음", dirs.count("밝음"), "| 비슷", dirs.count("비슷"), "| 전체", len(rows))
print("매끈함 평균", round(float(np.mean([r["매끈함(안/밖 고주파)"] for r in rows])), 2), "| 1 미만(더 매끈) 타일", sum(r["매끈함(안/밖 고주파)"] < 1 for r in rows))
