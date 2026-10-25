"""c02: DINOv2_grid_augmentation primary 설정(cls, mean, logreg)의 3분류 혼동행렬 그림.

e03 의 cv_predictions.csv (이미지 단위 층화 5-fold x 10 반복) 에서 primary 설정의 예측을 읽어
  한 장에 실제 클래스별 비율(%)과 그 아래 괄호로 반복 10회 평균 이미지 수(반복마다 60장이 한 번씩 test).
출력: outputs/c02_confusion_figure/fig_confusion_primary.png, confusion_by_repeat.csv, log.txt
"""
import csv
import os

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

from common import Logger, load_config, write_csv  # noqa: E402

PRIMARY = "tiles|cls|mean|logreg"


def main():
    cfg = load_config()
    out = cfg["paths"]["outputs_dir"]
    log = Logger(out, "c02_confusion_figure")
    rows = [r for r in csv.DictReader(open(f"{out}/e03_run_cv/cv_predictions.csv", encoding="utf-8-sig")) if r["setting"] == PRIMARY]
    reps = sorted({int(r["repeat"]) for r in rows})
    cms = np.zeros((len(reps), 3, 3))
    for r in rows:
        cms[reps.index(int(r["repeat"])), int(r["label"]), int(r["pred"])] += 1
    assert (cms.sum((1, 2)) == 60).all(), "반복마다 60장이 한 번씩 예측돼야 함"
    mean = cms.mean(0); rate = mean / mean.sum(1, keepdims=True)
    acc = np.array([np.trace(c) / 60 for c in cms])
    log(f"설정 {PRIMARY}, 반복 {len(reps)}회, 반복별 accuracy 평균 {acc.mean():.3f} (sd {acc.std():.3f})")
    log("반복 평균 혼동행렬 (행 = 실제, 열 = 예측):\n" + "\n".join(f"  AGG{i}: " + "  ".join(f"{mean[i, j]:5.1f}" for j in range(3)) for i in range(3)))
    write_csv(f"{log.dir}/confusion_by_repeat.csv", [dict(repeat=r, true=f"AGG{i}", **{f"pred_AGG{j}": int(cms[k, i, j]) for j in range(3)}) for k, r in enumerate(reps) for i in range(3)])
    fig, a = plt.subplots(figsize=(6.4, 5.4))
    lab = ["AGG0", "AGG1", "AGG2"]
    a.imshow(rate, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            a.text(j, i, f"{rate[i, j]:.0%}\n({mean[i, j]:.1f}장)", ha="center", va="center", fontsize=14, color="white" if rate[i, j] > 0.5 else "black")
    a.set_xticks(range(3)); a.set_yticks(range(3)); a.set_xticklabels(lab); a.set_yticklabels([f"{l}\n(n={int(mean[i].sum().round())})" for i, l in enumerate(lab)])
    a.set_xlabel("예측"); a.set_ylabel("실제")
    fig.suptitle(f"DINOv2_grid_augmentation primary (cls, mean, logreg), 3분류 | accuracy {acc.mean():.3f} ± {acc.std():.3f} (반복 {len(reps)}회)", fontsize=10, y=0.98)
    fig.tight_layout(); fig.savefig(f"{log.dir}/fig_confusion_primary.png", dpi=130); plt.close(fig)
    log(f"저장: {log.dir}")


if __name__ == "__main__":
    main()
