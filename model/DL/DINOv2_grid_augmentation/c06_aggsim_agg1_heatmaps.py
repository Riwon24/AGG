"""c06: 실제 라벨이 AGG1 인 이미지 14장 전부의 타일별 aggsim 히트맵.

이미지마다 한 장씩: 원본 위에 타일(224px 격자)을 aggsim 색으로 칠하고, 타일 안에 타일 번호(tile_id, 예 r00_c01 = 위에서 0번째 줄, 왼쪽에서 1번째 칸)와 aggsim 값을 쓴다.
제목에 금속, 처리 조건, 처리 방법, 온도, 라벨, 이미지 이름을 쓴다 (conditions.csv).
aggsim = 해당 타일이 test 였을 때의 값(prototype 은 그 fold 의 train 타일로만 계산, c03)을 10 반복에 걸쳐 평균한 값. 이미지의 모든 타일은 반복마다 정확히 한 번 test 이다.
색 범위는 c04 와 같은 고정값(0.537 ~ 0.923, repeat 0 fold 0 의 모든 test 타일 범위)이다.
수동 표시용: agg_tile_labels_template.csv (image_id, tile_id, aggsim, has_agg 빈 칸). "이 타일에 AGG 있음" 은 has_agg 에 1 을 적는 식으로 채우면 된다 (이 스크립트는 값을 채우지 않는다).
출력: outputs/c06_aggsim_agg1_heatmaps/NN_<금속>_<처리조건>.png (14장), agg_tile_labels_template.csv, log.txt
"""
import csv
import os
import re
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402
from matplotlib import font_manager as _fm  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

for _f in ("C:/Windows/Fonts/malgun.ttf", "/mnt/c/Windows/Fonts/malgun.ttf"):
    if os.path.exists(_f):
        _fm.fontManager.addfont(_f)
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
D = OUT / "c06_aggsim_agg1_heatmaps"
D.mkdir(parents=True, exist_ok=True)
T, LABEL, VMIN, VMAX = 224, 1, 0.537, 0.923
lines = []


def log(m):
    print(m); lines.append(m)


rd = lambda p: list(csv.DictReader(open(p, encoding="utf-8-sig")))
cond = [r for r in rd(OUT / "e00_make_splits/conditions.csv") if int(r["label"]) == LABEL]
man = {r["image_id"]: r for r in rd(HERE.parent / "preproc/outputs/s00b_meta_mag/manifest_v1.csv")}
xy = {(r["image_id"], r["tile_id"]): (int(r["x0"]), int(r["y0"])) for r in rd(OUT / "e01_extract_features/tile_index.csv")}
acc = {}
for r in rd(OUT / "c03_tile_aggsim/tile_aggsim.csv"):
    if r["split"] == "test" and int(r["label"]) == LABEL:
        acc.setdefault((r["image_id"], r["tile_id"]), []).append(float(r["aggsim"]))
assert all(len(v) == 10 for v in acc.values()), "각 타일은 반복마다 한 번씩 test 여야 함"
log(f"실제 라벨 AGG{LABEL} 이미지 {len(cond)}장 (14장이어야 함), 타일 {len(acc)}개, 색 범위 {VMIN} ~ {VMAX} (c04 와 같음)")
assert len(cond) == 14
cmap = plt.get_cmap("viridis")
tmpl = []
for n, c in enumerate(sorted(cond, key=lambda r: (r["metal"], r["condition"], r["image_id"])), 1):
    iid = c["image_id"]
    tiles = sorted((t for (i, t) in acc if i == iid), key=lambda t: (int(t[1:3]), int(t[5:7])))
    vals = {t: float(np.mean(acc[(iid, t)])) for t in tiles}
    im = cv2.cvtColor(cv2.imdecode(np.fromfile(str(HERE.parent / man[iid]["path"]), dtype=np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(13.5, 10.2))
    ax.imshow(im); ax.axis("off")
    for t in tiles:
        x0, y0 = xy[(iid, t)]; v = vals[t]
        ax.add_patch(Rectangle((x0, y0), T, T, facecolor=cmap((v - VMIN) / (VMAX - VMIN)), alpha=0.6, edgecolor="white", linewidth=1.2))
        ax.text(x0 + T / 2, y0 + T * 0.34, t, ha="center", va="center", fontsize=8, color="white", path_effects=[pe.withStroke(linewidth=2, foreground="black")])
        ax.text(x0 + T / 2, y0 + T * 0.68, f"{v:.2f}", ha="center", va="center", fontsize=11, color="white", fontweight="bold", path_effects=[pe.withStroke(linewidth=2.2, foreground="black")])
        tmpl.append(dict(image_id=iid, metal=c["metal"], condition=c["condition"], tile_id=t, x0=x0, y0=y0, aggsim=round(v, 4), has_agg=""))
    ax.set_title(f"[{n:02d}/14] 금속 {c['metal']} | 처리 조건 {c['condition']} (방법 {c['method']}, 온도 {c['temp']}) | 라벨 AGG{c['label']}\n{iid} | 타일 {len(tiles)}개, aggsim 평균 {np.mean(list(vals.values())):.3f}", fontsize=11)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(VMIN, VMAX))
    cb = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.01); cb.set_label("aggsim (test 타일, 10 반복 평균)", fontsize=9)
    fn = re.sub(r"[^\w\-]+", "_", f"{n:02d}_{c['metal']}_{c['condition']}") + ".png"
    fig.savefig(D / fn, dpi=100, bbox_inches="tight"); plt.close(fig)
    log(f"  {n:02d} {c['metal']:<10} {c['condition']:<12} 타일 {len(tiles):2d}개 aggsim 평균 {np.mean(list(vals.values())):.3f} (최소 {min(vals.values()):.3f}, 최대 {max(vals.values()):.3f}) -> {fn}")
with open(D / "agg_tile_labels_template.csv", "w", encoding="utf-8-sig", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(tmpl[0])); w.writeheader(); w.writerows(tmpl)
(D / "log.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
log(f"저장: {D} (그림 14장 + 템플릿 {len(tmpl)}행)")
