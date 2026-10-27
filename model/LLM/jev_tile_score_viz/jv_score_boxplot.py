"""jv_score_boxplot: jv06, jv07, jv08 의 이미지 점수(타일 p_flat 의 이미지 평균, 60장)를 AGG0/AGG1/AGG2 별 박스플롯으로 그린다.

입력: outputs/tile_scores_jv06_jv08/tile_scores_jv06_jv08.csv
출력: outputs/tile_score_maps_jv06_jv08/image_score_boxplot.png
점 하나 = 이미지 한 장. 실행마다 점수의 눈금이 달라서 y 축을 실행별로 따로 둔다 (실행 사이 값 비교 불가).
"""
import csv
import os
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager as _fm  # noqa: E402

for _f in ("C:/Windows/Fonts/malgun.ttf", "/mnt/c/Windows/Fonts/malgun.ttf"):
    if os.path.exists(_f):
        _fm.fontManager.addfont(_f)
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).resolve().parent
TABLE = HERE / "outputs/tile_scores_jv06_jv08/tile_scores_jv06_jv08.csv"
OUT = HERE / "outputs/tile_score_maps_jv06_jv08/image_score_boxplot.png"
RUNS = [("jv06", "jv06 선택형"), ("jv07", "jv07 설명문 보강"), ("jv08", "jv08 예시 이미지 포함")]
LABS = ["AGG0", "AGG1", "AGG2"]
COL = {"AGG0": "#4c78a8", "AGG1": "#f58518", "AGG2": "#e45756"}


def main():
    rows = list(csv.DictReader(open(TABLE, encoding="utf-8-sig")))
    img = {}
    for r in rows:
        img.setdefault(r["image_id"], dict(label=r["label"], tiles=[]))["tiles"].append(r)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
    rng = np.random.RandomState(0)
    for ax, (run, title) in zip(axes, RUNS):
        data = {l: [np.mean([float(t[f"p_{run}"]) for t in v["tiles"]]) for v in img.values() if v["label"] == l] for l in LABS}
        bp = ax.boxplot([data[l] for l in LABS], positions=range(3), widths=0.5, showfliers=False, patch_artist=True, medianprops=dict(color="black"))
        for p, l in zip(bp["boxes"], LABS):
            p.set_facecolor(COL[l]); p.set_alpha(0.35)
        for i, l in enumerate(LABS):
            ax.scatter(i + rng.uniform(-0.14, 0.14, len(data[l])), data[l], s=18, color=COL[l], edgecolor="white", linewidth=0.4, zorder=3)
        ax.set_xticks(range(3)); ax.set_xticklabels([f"{l}\n(n={len(data[l])})" for l in LABS])
        ax.set_title(f"{title}\n중앙값 " + " / ".join(f"{np.median(data[l]):.2f}" for l in LABS), fontsize=10)
        ax.set_ylabel("이미지 점수 (타일 p_flat 평균)" if run == "jv06" else "")
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("JEVision 이미지 점수 분포 (점 = 이미지 1장, 실행마다 y 축 눈금이 다름)", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT, dpi=110); plt.close(fig)
    print("저장:", OUT)


if __name__ == "__main__":
    main()
