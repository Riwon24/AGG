"""jv_tile_score_maps: JEVision jv06, jv07, jv08 의 타일별 점수를 원본 사진 위에 그린다 (이미지마다 한 장, 60장 전부; matplotlib, PIL).

입력: outputs/tile_scores_jv06_jv08/tile_scores_jv06_jv08.csv (jv_tile_scores_all_runs.py 가 만든 표), 원본 사진.
한 장의 구성: [원본(직접 표시한 AGG 타일은 하늘색 테두리) | jv06 | jv07 | jv08].
  - 타일(224px 격자)은 같은 이미지 안에서의 순위 백분위(pct)로 칠한다 (1.0 = 이 이미지에서 점수가 가장 높은 타일, 0.0 = 가장 낮은 타일). 실행마다 점수의 크기와 분포가 달라서(jv06 은 거의 1.0, jv08 은 거의 0)
    같은 색 눈금으로 실제 점수를 칠하면 구조가 안 보이기 때문에 이미지 안 순위로 칠했다.
  - 타일 안의 숫자는 그 실행의 실제 점수 p (두 순서 평균 P(FLAT_REGION), 0~1)이다.
  - 이미지가 사진 전체를 덮는 격자 밖(오른쪽 32px, 아래 24px)이나 시편이 없는 타일은 그리지 않는다 (분석 타일 2,591개만).
출력: outputs/tile_score_maps_jv06_jv08/<AGG0|AGG1|AGG2>/NN_<금속>_<조건>.png (60장)
"""
import csv
import os
import re
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image

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
ROOT = HERE.parent
TABLE = HERE / "outputs/tile_scores_jv06_jv08/tile_scores_jv06_jv08.csv"
OUT = HERE / "outputs/tile_score_maps_jv06_jv08"
T = 224
RUNS = [("jv06", "jv06 선택형\n(run_tiles.py 프롬프트)"), ("jv07", "jv07 설명문 보강"), ("jv08", "jv08 예시 이미지 3장 포함")]


def main():
    rows = list(csv.DictReader(open(TABLE, encoding="utf-8-sig")))
    by = {}
    for r in rows:
        by.setdefault(r["image_id"], []).append(r)
    ids = sorted(by)
    assert len(ids) == 60
    cmap = plt.get_cmap("inferno")
    for n, iid in enumerate(ids, 1):
        ts = by[iid]; lab = ts[0]["label"]
        im = Image.open(ROOT / (iid + ".jpg")).convert("RGB")
        fig, axes = plt.subplots(1, 4, figsize=(22, 4.9))
        for k, ax in enumerate(axes):
            ax.imshow(im); ax.axis("off")
            for r in ts:
                x0, y0 = int(r["x0"]), int(r["y0"])
                if k == 0:
                    if r["has_agg"] == "1":
                        ax.add_patch(Rectangle((x0 + 6, y0 + 6), T - 12, T - 12, facecolor="none", edgecolor="#00e5ff", linewidth=3.2))
                    ax.add_patch(Rectangle((x0, y0), T, T, facecolor="none", edgecolor="white", linewidth=0.6, alpha=0.5))
                else:
                    run = RUNS[k - 1][0]; pc = float(r[f"pct_{run}"])
                    ax.add_patch(Rectangle((x0, y0), T, T, facecolor=cmap(pc), alpha=0.62, edgecolor="white", linewidth=0.7))
                    ax.text(x0 + T / 2, y0 + T / 2, f"{float(r[f'p_{run}']):.2f}", ha="center", va="center", fontsize=6.5, color="white", fontweight="bold",
                            path_effects=[pe.withStroke(linewidth=1.6, foreground="black")])
                    if r["has_agg"] == "1":
                        ax.add_patch(Rectangle((x0 + 6, y0 + 6), T - 12, T - 12, facecolor="none", edgecolor="#00e5ff", linewidth=3.2))
            if k == 0:
                nm = int(sum(1 for r in ts if r["has_agg"] == "1"))
                ax.set_title("원본 (하늘색 테두리 = 직접 표시한 AGG 타일 " + (f"{nm}개)" if nm else "없음)"), fontsize=9)
            else:
                run, title = RUNS[k - 1]
                ps = [float(r[f"p_{run}"]) for r in ts]
                ax.set_title(f"{title}\n이 이미지 점수 평균 {np.mean(ps):.2f} (최소 {min(ps):.2f}, 최대 {max(ps):.2f})", fontsize=9)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
        cb = fig.colorbar(sm, ax=axes.tolist(), fraction=0.012, pad=0.01, aspect=30); cb.set_label("이미지 안 순위 백분위 (1.0 = 이 이미지에서 점수 최상위)", fontsize=8)
        fig.suptitle(f"[{n:02d}/60] {lab} | 금속 {ts[0]['metal']} | 처리 조건 {ts[0]['condition']} | {iid} | 타일 {len(ts)}개", fontsize=11)
        d = OUT / lab; d.mkdir(parents=True, exist_ok=True)
        fn = re.sub(r"[^\w\-]+", "_", f"{n:02d}_{ts[0]['metal']}_{ts[0]['condition']}") + ".png"
        fig.savefig(d / fn, dpi=85, bbox_inches="tight"); plt.close(fig)
        if n % 10 == 0:
            print(f"  {n}/60", flush=True)
    print("저장:", OUT)


if __name__ == "__main__":
    main()
