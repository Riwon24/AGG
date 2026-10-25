"""c07: 실제 라벨이 AGG1 인 이미지 14장의 타일 번호 격자 (색 없음, 수동 표시용).

원본 이미지 위에 타일(224px 격자) 경계선과 타일 번호(tile_id, 예 r00_c01 = 위에서 0번째 줄, 왼쪽에서 1번째 칸)만 그린다. 타일 안은 색을 칠하지 않아 원본이 그대로 보인다.
번호는 분석 타일(시편 픽셀이 있는 타일)에만 붙인다. 제목에 금속, 처리 조건, 처리 방법, 온도, 라벨, 이미지 이름을 쓴다 (conditions.csv).
출력: outputs/c07_agg1_tile_numbers/NN_<금속>_<처리조건>.png (14장), log.txt
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
D = OUT / "c07_agg1_tile_numbers"
D.mkdir(parents=True, exist_ok=True)
T, LABEL = 224, 1
lines = []


def log(m):
    print(m); lines.append(m)


rd = lambda p: list(csv.DictReader(open(p, encoding="utf-8-sig")))
cond = [r for r in rd(OUT / "e00_make_splits/conditions.csv") if int(r["label"]) == LABEL]
man = {r["image_id"]: r for r in rd(HERE.parent / "preproc/outputs/s00b_meta_mag/manifest_v1.csv")}
tiles = {}
for r in rd(OUT / "e01_extract_features/tile_index.csv"):
    if int(r["label"]) == LABEL:
        tiles.setdefault(r["image_id"], []).append((r["tile_id"], int(r["x0"]), int(r["y0"])))
assert len(cond) == 14
for n, c in enumerate(sorted(cond, key=lambda r: (r["metal"], r["condition"], r["image_id"])), 1):
    iid = c["image_id"]
    im = cv2.cvtColor(cv2.imdecode(np.fromfile(str(HERE.parent / man[iid]["path"]), dtype=np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(13.5, 10.2))
    ax.imshow(im); ax.axis("off")
    for t, x0, y0 in sorted(tiles[iid]):
        ax.add_patch(Rectangle((x0, y0), T, T, facecolor="none", edgecolor="yellow", linewidth=1.2))
        ax.text(x0 + T / 2, y0 + T / 2, t, ha="center", va="center", fontsize=13, color="white", fontweight="bold", path_effects=[pe.withStroke(linewidth=2.5, foreground="black")])
    ax.set_title(f"[{n:02d}/14] 금속 {c['metal']} | 처리 조건 {c['condition']} (방법 {c['method']}, 온도 {c['temp']}) | 라벨 AGG{c['label']}\n{iid} | 타일 {len(tiles[iid])}개", fontsize=11)
    fn = re.sub(r"[^\w\-]+", "_", f"{n:02d}_{c['metal']}_{c['condition']}") + ".png"
    fig.savefig(D / fn, dpi=100, bbox_inches="tight"); plt.close(fig)
    log(f"  {n:02d} {c['metal']:<10} {c['condition']:<12} 타일 {len(tiles[iid]):2d}개 -> {fn}")
(D / "log.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
log(f"저장: {D}")
