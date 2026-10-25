"""m18: MIL(gated attention, 이진 AGG0 vs AGG1+AGG2) 혼동행렬 그림.

m10 의 mil_predictions.csv 에서 이미지 단위 층화 5-fold x 10 반복(scheme IMG) 예측을 읽어, 한 장에 실제 클래스별 비율(%)과 그 아래 괄호로 반복 10회 평균 이미지 수를 그린다
(반복마다 60장이 한 번씩 test). --config 로 마스크 100% 타일 판(config_full.yaml)도 쓸 수 있다.
출력: outputs/m18_confusion_figure/fig_confusion_mil.png, confusion_by_repeat.csv, log.txt
"""
import argparse
import csv
import os
from pathlib import Path

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager as _fm  # noqa: E402

for _f in ("C:/Windows/Fonts/malgun.ttf", "/mnt/c/Windows/Fonts/malgun.ttf"):
    if os.path.exists(_f):
        _fm.fontManager.addfont(_f)
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    a = ap.parse_args()
    cfgp = Path(a.config).resolve()
    out = (cfgp.parent / yaml.safe_load(cfgp.read_text(encoding="utf-8"))["paths"]["outputs_dir"]).resolve()
    d = out / "m18_confusion_figure"; d.mkdir(parents=True, exist_ok=True)
    lines = []

    def log(m):
        print(m); lines.append(m)
    rows = [r for r in csv.DictReader(open(out / "m10_train_eval_mil/mil_predictions.csv", encoding="utf-8-sig")) if r["scheme"] == "IMG"]
    reps = sorted({int(r["repeat"]) for r in rows})
    cms = np.zeros((len(reps), 2, 2))
    for r in rows:
        cms[reps.index(int(r["repeat"])), int(r["label"]), int(r["pred"])] += 1
    assert (cms.sum((1, 2)) == 60).all(), "반복마다 60장이 한 번씩 예측돼야 함"
    mean = cms.mean(0); rate = mean / mean.sum(1, keepdims=True)
    acc = np.array([np.trace(c) / 60 for c in cms])
    lab = ["AGG0", "AGG1+2"]
    log(f"결과 폴더 {out}, 반복 {len(reps)}회, 반복별 accuracy 평균 {acc.mean():.3f} (sd {acc.std():.3f})")
    log("반복 평균 혼동행렬 (행 = 실제, 열 = 예측):\n" + "\n".join(f"  {n}: " + "  ".join(f"{mean[i, j]:5.1f}" for j in range(2)) for i, n in enumerate(lab)))
    with open(d / "confusion_by_repeat.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f); w.writerow(["repeat", "true", "pred_AGG0", "pred_AGG1+2"])
        for k, r in enumerate(reps):
            for i in range(2):
                w.writerow([r, lab[i], int(cms[k, i, 0]), int(cms[k, i, 1])])
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    ax.imshow(rate, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{rate[i, j]:.0%}\n({mean[i, j]:.1f}장)", ha="center", va="center", fontsize=14, color="white" if rate[i, j] > 0.5 else "black")
    ax.set_xticks(range(2)); ax.set_yticks(range(2)); ax.set_xticklabels(lab); ax.set_yticklabels([f"{l}\n(n={int(mean[i].sum().round())})" for i, l in enumerate(lab)])
    ax.set_xlabel("예측"); ax.set_ylabel("실제")
    fig.suptitle(f"MIL (gated attention), 이진\naccuracy {acc.mean():.3f} ± {acc.std():.3f} (반복 {len(reps)}회)", fontsize=10, y=0.98)
    fig.tight_layout(); fig.savefig(d / "fig_confusion_mil.png", dpi=130); plt.close(fig)
    (d / "log.txt").write_text("\n".join(lines) + f"\n저장: {d}\n", encoding="utf-8")
    print("저장:", d)


if __name__ == "__main__":
    main()
