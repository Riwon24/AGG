"""c04: AGG 유사도(aggsim) 히트맵 오버레이.

c03 의 tile_aggsim.csv 에서 repeat = 0, fold = 0 의 test 이미지 중 label = 2 인 이미지만 골라, 각 이미지의 타일 위치(x0, y0, 224px)에 aggsim 을 색으로 칠해 원본 위에 겹친다.
타일 좌표는 tile_index.csv 에서 (image_id, tile_id) 로 가져온다. 색 범위는 같은 (repeat, fold) 의 test 타일 전체(모든 라벨)의 aggsim 최소~최대로 고정해 클래스 사이도 같은 색으로 비교할 수 있다.
사용: python c04_aggsim_heatmap.py [라벨 0|1|2, 기본 2]
출력: outputs/c04_aggsim_heatmap/fig_aggsim_overlay_rep0_fold0_AGG{라벨}.png, tiles_plotted_AGG{라벨}.csv, log_AGG{라벨}.txt
"""
import csv
import os
import sys
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
OUT = HERE / "outputs"                                  # 이 폴더의 결과 (c03 -> c04, c05, c06 -> c08 -> c09 가 서로의 출력을 읽는다)
GRID = next((p for p in (HERE.parent / "exp_grid_aug" / "outputs", HERE.parent / "DINOv2_grid_augmentation" / "outputs") if p.exists()), HERE.parent / "DINOv2_grid_augmentation" / "outputs")
#                                                       # 입력: DINOv2_grid_augmentation 의 출력 (읽기 전용; 로컬 작업 폴더 구조 또는 저장소 구조 중 있는 쪽)
D = OUT / "c04_aggsim_heatmap"
D.mkdir(parents=True, exist_ok=True)
REPEAT, FOLD, T = 0, 0, 224
LABEL = int(sys.argv[1]) if len(sys.argv) > 1 else 2
assert LABEL in (0, 1, 2)
lines = []


def log(m):
    print(m); lines.append(m)


rd = lambda p: list(csv.DictReader(open(p, encoding="utf-8-sig")))
test_all = [r for r in rd(OUT / "c03_tile_aggsim/tile_aggsim.csv") if int(r["repeat"]) == REPEAT and int(r["fold"]) == FOLD and r["split"] == "test"]
sim = [r for r in test_all if int(r["label"]) == LABEL]
xy = {(r["image_id"], r["tile_id"]): (int(r["x0"]), int(r["y0"])) for r in rd(GRID / "e01_extract_features/tile_index.csv")}
man = {r["image_id"]: r for r in rd(HERE.parent / "preproc/outputs/s00b_meta_mag/manifest_v1.csv")}
by = {}
for r in sim:
    by.setdefault(r["image_id"], []).append((xy[(r["image_id"], r["tile_id"])], float(r["aggsim"])))
ids = sorted(by)
log(f"repeat {REPEAT}, fold {FOLD}, test 이미지 중 label {LABEL}: {len(ids)}장 -> {ids}")
assert 3 <= len(ids) <= 5, len(ids)
allv = np.array([float(r["aggsim"]) for r in test_all])
vmin, vmax = float(allv.min()), float(allv.max())
log(f"색 범위(공통, 같은 fold 의 모든 test 타일 {len(allv)}개): aggsim {vmin:.3f} ~ {vmax:.3f}")
cmap = plt.get_cmap("viridis")
n = len(ids); nc = 2; nr = (n + 1) // 2
fig, axes = plt.subplots(nr, nc, figsize=(6.2 * nc, 5.0 * nr), squeeze=False)
rows_out = []
for k, iid in enumerate(ids):
    ax = axes[k // nc, k % nc]
    im = cv2.cvtColor(cv2.imdecode(np.fromfile(str(HERE.parent / man[iid]["path"]), dtype=np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    ax.imshow(im); ax.axis("off")
    vals = []
    for (x0, y0), v in by[iid]:
        ax.add_patch(Rectangle((x0, y0), T, T, facecolor=cmap((v - vmin) / (vmax - vmin)), alpha=0.6, edgecolor="white", linewidth=1.0))
        ax.text(x0 + T / 2, y0 + T / 2, f"{v:.2f}", ha="center", va="center", fontsize=8, color="white", fontweight="bold", path_effects=[pe.withStroke(linewidth=2, foreground="black")])
        vals.append(v); rows_out.append(dict(image_id=iid, x0=x0, y0=y0, aggsim=v))
    ax.set_title(f"{iid.rsplit('/', 1)[0]}  (실제 AGG{LABEL}, 타일 {len(vals)}개, aggsim 평균 {np.mean(vals):.3f})", fontsize=9)
    log(f"  {iid}: 타일 {len(vals)}개, aggsim 평균 {np.mean(vals):.3f}, 최소 {min(vals):.3f}, 최대 {max(vals):.3f}")
for k in range(n, nr * nc):
    axes[k // nc, k % nc].axis("off")
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin, vmax))
cb = fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02, aspect=35); cb.set_label("aggsim = cos(타일 cls, AGG1+2 prototype)", fontsize=9)
fig.suptitle(f"타일별 AGG 유사도 (repeat {REPEAT}, fold {FOLD}, test 이미지, label {LABEL}) | prototype 은 해당 fold 의 train 타일만으로 계산", fontsize=11)
fig.savefig(D / f"fig_aggsim_overlay_rep0_fold0_AGG{LABEL}.png", dpi=110, bbox_inches="tight"); plt.close(fig)
with open(D / f"tiles_plotted_AGG{LABEL}.csv", "w", encoding="utf-8-sig", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["image_id", "x0", "y0", "aggsim"]); w.writeheader(); w.writerows(rows_out)
(D / f"log_AGG{LABEL}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
log(f"저장: {D}")
