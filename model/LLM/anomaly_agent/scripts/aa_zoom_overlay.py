"""aa_zoom_overlay: AnomalyAgent 가 타일마다 image_zooming 으로 지정한 확대 영역(= 에이전트가 보려고 한 곳)과, 사용자가 직접 표시한 AGG 영역을 한 타일 위에 겹쳐 그린다.

입력: ~/AnomalyAgent/results/agg_tile34/results.json (tool_calls 의 image_zooming 중심·배율)
      outputs/aa_caption_tiles/agg_regions.json (label_agg_regions.html 로 표시한 AGG 영역, 타일 크기 대비 비율)
출력: outputs/aa_caption_tiles/zoom_overlay/zoom_overlay_AGG1.png (표시 타일 17장), zoom_overlay_AGG0.png (대조 타일 17장),
      zoom_overlap.csv (타일별 겹침), 터미널에 요약 + 무작위 대조(순열) 기준

확대 영역의 의미는 tools.py 의 image_zooming 과 같다: 현재 이미지에서 폭 1/z, 높이 1/z 짜리 창을 (center_x, center_y) 에 두고(가장자리에서는 밀어 넣음) 크롭한다.
image_target == "raw" 이면 원본 타일 기준, 아니면 직전까지 처리된 이미지(앞의 확대가 있으면 그 확대 창 안) 기준이므로 창을 합성한다.
주의: 이것은 "확대하라고 한 곳"이지 모델의 실제 주의(attention)가 아니다.
"""
import csv
import json
import os
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mp
from matplotlib import font_manager
from PIL import Image

R = os.path.expanduser("~/AnomalyAgent/results/agg_tile34/results.json")
D = "/mnt/c/Users/admin/Downloads/260929_images_for_AGG - manual labeling/AGG_labeled_only/exp_aa/outputs/aa_caption_tiles"
OUT = os.path.join(D, "zoom_overlay")
os.makedirs(OUT, exist_ok=True)

for fp in ("/mnt/c/Windows/Fonts/malgun.ttf", "C:/Windows/Fonts/malgun.ttf"):
    if os.path.exists(fp):
        font_manager.fontManager.addfont(fp)
        break
for name in ("Malgun Gothic", "NanumGothic", "Noto Sans CJK KR", "AppleGothic"):
    if any(name == f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = name
        break
plt.rcParams["axes.unicode_minus"] = False

COLORS = ["#1f77b4", "#ff7f0e", "#9467bd", "#17becf", "#8c564b", "#e377c2"]


def zoom_windows(tool_calls):
    """tool_calls 순서대로 확대 창을 합성해서 [(x0,y0,x1,y1, 배율), ...] (원본 타일 기준 0~1) 를 돌려준다."""
    frame = (0.0, 0.0, 1.0, 1.0)
    out = []
    for t in tool_calls:
        if t["tool_name"] != "image_zooming":
            continue
        a = t["args"]
        z = max(1.0, min(8.0, float(a.get("zoom_factor", 2.0))))
        if z <= 1.0:
            continue
        base = (0.0, 0.0, 1.0, 1.0) if a.get("image_target") == "raw" else frame
        cx = min(1.0, max(0.0, float(a.get("center_x", 0.5))))
        cy = min(1.0, max(0.0, float(a.get("center_y", 0.5))))
        cw = 1.0 / z
        fx0 = min(max(cx - cw / 2, 0.0), 1.0 - cw)
        fy0 = min(max(cy - cw / 2, 0.0), 1.0 - cw)
        bw, bh = base[2] - base[0], base[3] - base[1]
        win = (base[0] + fx0 * bw, base[1] + fy0 * bh, base[0] + (fx0 + cw) * bw, base[1] + (fy0 + cw) * bh)
        out.append((*win, z))
        frame = win
    return out


def inter(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return max(0.0, w) * max(0.0, h)


def area(b):
    return max(1e-9, (b[2] - b[0]) * (b[3] - b[1]))


def best_cover(wins, box):
    """AGG 영역 중 확대 창 하나가 가장 많이 덮은 비율"""
    return max((inter(w, box) / area(box) for w in wins), default=0.0)


def union_cover(wins, box, n=60):
    """AGG 영역 중 확대 창들의 합집합이 덮은 비율 (격자 표본)"""
    hit = tot = 0
    for i in range(n):
        for j in range(n):
            x = box[0] + (i + 0.5) / n * (box[2] - box[0])
            y = box[1] + (j + 0.5) / n * (box[3] - box[1])
            tot += 1
            hit += any(w[0] <= x <= w[2] and w[1] <= y <= w[3] for w in wins)
    return hit / tot


res = json.load(open(R))["results"]
regions = json.load(open(os.path.join(D, "agg_regions.json"), encoding="utf-8"))["tiles"]
meta = {c["n"]: c for c in json.load(open(os.path.join(D, "captions.json"), encoding="utf-8"))["captions"]}
byname = {x["img_path"]: x for x in res}


def draw(label, fname):
    names = sorted(n for n in byname if n.startswith(label))
    cols = 6
    rows = (len(names) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.1, rows * 4.0), dpi=110)
    axes = axes.ravel()
    for ax in axes:
        ax.axis("off")
    for ax, n in zip(axes, names):
        x = byname[n]
        ax.imshow(Image.open(os.path.join(D, "tiles", n)))
        W = 448
        for k, w in enumerate(zoom_windows(x["tool_calls"])):
            c = COLORS[k % len(COLORS)]
            ax.add_patch(mp.Rectangle((w[0] * W, w[1] * W), (w[2] - w[0]) * W, (w[3] - w[1]) * W, fill=False, ec=c, lw=2, ls="--"))
            ax.text(w[0] * W + 4, w[1] * W + 14, f"{k + 1}", color="white", fontsize=9, bbox=dict(fc=c, ec="none", pad=1.2))
        extra = ""
        if label.startswith("AGG1"):
            for b in regions.get(n.replace(".png", ""), {}).get("boxes", []):
                ax.add_patch(mp.Rectangle((b[0] * W, b[1] * W), (b[2] - b[0]) * W, (b[3] - b[1]) * W, fill=False, ec="#00e676", lw=2.5))
            if regions.get(n.replace(".png", ""), {}).get("skipped"):
                extra = " (AGG 위치 미표시)"
        c = meta[int(n.split("_")[1].split(".")[0])]
        mm = byname[n]["id"].split("/")
        ax.set_title(f"{n.replace('.png', '')}{extra}\n{mm[0]} {mm[1]} · {c['tile_id']}\n판정 {'이상' if x['pred_result'] == 'anomalous' else '정상'}", fontsize=8)
    fig.suptitle(("AGG1 표시 타일: 초록 실선 = 사용자가 표시한 AGG, 점선(번호) = 에이전트가 확대한 영역(순서)" if label.startswith("AGG1")
                  else "AGG0 대조 타일: 점선(번호) = 에이전트가 확대한 영역(순서)"), fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(os.path.join(OUT, fname))
    plt.close(fig)


draw("AGG1", "zoom_overlay_AGG1.png")
draw("AGG0", "zoom_overlay_AGG0.png")

# 겹침 통계 (사용자가 표시한 타일만)
rows = []
labeled = {n: regions[n.replace(".png", "")]["boxes"][0] for n in byname if n.startswith("AGG1") and regions.get(n.replace(".png", ""), {}).get("boxes")}
wins = {n: zoom_windows(byname[n]["tool_calls"]) for n in byname}
for n, b in sorted(labeled.items()):
    w = wins[n]
    ctr_in = any(b[0] <= (q[0] + q[2]) / 2 <= b[2] and b[1] <= (q[1] + q[3]) / 2 <= b[3] for q in w)
    rows.append({"tile": n.replace(".png", ""), "AGG면적비": round(area(b), 3), "확대 횟수": len(w), "확대창 평균 면적비": round(sum(area(q) for q in w) / max(1, len(w)), 3),
                 "AGG 가장 많이 덮은 창(%)": round(100 * best_cover(w, b)), "모든 창 합쳐 덮은(%)": round(100 * union_cover(w, b)), "확대 중심이 AGG 안": "예" if ctr_in else "아니오"})
with open(os.path.join(OUT, "zoom_overlap.csv"), "w", encoding="utf-8-sig", newline="") as fh:
    wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
    wr.writeheader()
    wr.writerows(rows)
for r in rows:
    print(r)

real = sum(r["AGG 가장 많이 덮은 창(%)"] for r in rows) / len(rows)
real_u = sum(r["모든 창 합쳐 덮은(%)"] for r in rows) / len(rows)
real_c = sum(r["확대 중심이 AGG 안"] == "예" for r in rows)
# 순열 기준: 같은 확대 창들을 다른 타일의 AGG 영역에 대 봄
random.seed(0)
names = list(labeled)
perm_b, perm_u, perm_c = [], [], []
for n in names:
    for m in names:
        if m == n:
            continue
        b = labeled[m]
        w = wins[n]
        perm_b.append(100 * best_cover(w, b))
        perm_u.append(100 * union_cover(w, b))
        perm_c.append(any(b[0] <= (q[0] + q[2]) / 2 <= b[2] and b[1] <= (q[1] + q[3]) / 2 <= b[3] for q in w))
print(f"\n표시한 타일 {len(rows)}장: 가장 많이 덮은 창 평균 {real:.1f}% | 합집합 평균 {real_u:.1f}% | 확대 중심이 AGG 안 {real_c}/{len(rows)}")
print(f"무작위 기준(같은 창을 다른 타일의 AGG 에 대 봄): {sum(perm_b) / len(perm_b):.1f}% | {sum(perm_u) / len(perm_u):.1f}% | 중심이 안 {sum(perm_c) / len(perm_c) * len(rows):.1f}/{len(rows)}")
