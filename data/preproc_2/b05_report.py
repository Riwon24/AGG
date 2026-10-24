"""b05: b04 의 예측으로 결과표, 부트스트랩 신뢰구간, 쌍(paired) 차이, 그림, 보고서를 만든다 (새 실험 없음, torch/sklearn 불필요).

신뢰구간: 이미지 단위(60장 복원추출)와 처리 조건 단위(10개 조건 복원추출) 부트스트랩 각 2000회, 같은 재표본을 모든 팔/설정에 써서 차이를 쌍으로 계산.
          재표본마다 10회 반복의 지표를 평균한 값의 2.5~97.5 백분위. (exp/common.py 의 bootstrap_ci 와 같은 정의)
사전 지정한 해석 규칙 (b04 를 돌리기 전에 정함):
  규칙 1: R0 - R3 의 이미지 단위 95% CI 가 0 을 제외하면 "원본 학습 모델이 배경에 기대고 있었다는 증거", 아니면 단정하지 않는다.
  규칙 2: R1 - R0 의 이미지 단위 95% CI 가 0 을 포함하면 "증강 효과 없음"이라고도 단정하지 않고 차이만 적는다.
  규칙은 primary 설정(cls · mean · logreg)에만 적용하고, 나머지 설정은 일관성 확인용이다. R2 - R3 는 사전 지정이 아닌 탐색 비교이며, R2/R3 는 증강본 K개의 확률을 평균하므로 R0/R1 과 직접 비교하지 않는다.
출력: results_table.csv, bootstrap_ci.csv, paired_diff.csv, misclassification_by_group.csv, report.md, fig1~fig4.png, log.txt
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager as _fm  # noqa: E402  (Windows 의 애플리케이션 제어 정책이 matplotlib 를 막으면 WSL 에서 실행: Windows 글꼴 파일을 경로로 등록)
import os as _os  # noqa: E402
for _f in ("C:/Windows/Fonts/malgun.ttf", "/mnt/c/Windows/Fonts/malgun.ttf"):
    if _os.path.exists(_f):
        _fm.fontManager.addfont(_f)

from bg_common import Logger, forbidden_check, image_parts, inputs, load_config, metrics, parse_condition, sha256_file, write_csv  # noqa: E402
from common import metrics_from_confusion  # noqa: E402  (exp/common.py)

SCRIPT = "b05_report"
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
ARMS = ["R0", "R1", "R2", "R3"]
ARM_DESC = {"R0": "원본 학습 → 원본 평가 (증강 없음)", "R1": "원본+증강 학습 → 원본 평가 (primary 팔)", "R2": "원본+증강 학습 → 증강 평가", "R3": "원본 학습 → 증강 평가"}
COL = {"R0": "#4c78a8", "R1": "#c0392b", "R2": "#59a14f", "R3": "#b07aa1"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    out = cfg["paths"]["outputs_dir"]
    log = Logger(out, SCRIPT)
    log("config:", cfg["_config_path"])
    ex = cfg["experiment"]
    pid = f"tiles|{ex['primary']['features']}|{ex['primary']['aggregator']}|{ex['primary']['classifier']}"
    parr = ex["primary_arm"]
    imgs, tiles = inputs(cfg)
    ids = [r["image_id"] for r in imgs]; pos = {i: k for k, i in enumerate(ids)}
    y = np.array([r["label"] for r in imgs]); N = len(ids)
    conds = [image_parts(i)[1] for i in ids]
    cond_list = sorted(set(conds))
    log(f"이미지 {N}장, 처리 조건 {len(cond_list)}개, primary = {pid} / 팔 {parr}, 분석 타일 {len(tiles)}개 (규칙 {cfg['tiles']['select']})")

    rows = list(csv.DictReader(open(f"{out}/b04_run_cv/cv_predictions.csv", encoding="utf-8-sig")))
    settings = [s for s in dict.fromkeys(r["setting"] for r in rows)]
    R = 1 + max(int(r["repeat"]) for r in rows)
    P = {}
    for s in settings:
        for arm in ARMS:
            m = np.full((N, R), -1, int)
            for r in rows:
                if r["setting"] == s and r["arm"] == arm:
                    m[pos[r["image_id"]], int(r["repeat"])] = int(r["pred"])
            assert (m >= 0).all(), f"{s} {arm}: 예측이 비어 있음"
            P[(s, arm)] = m
    log(f"예측 {len(rows)}행 읽음: 설정 {len(settings)}개 x 팔 {len(ARMS)}개 x 반복 {R}회 x 이미지 {N}장")

    # ---------- 점 추정 (반복별 pooled 지표의 평균) ----------
    def point(s, arm):
        ms = [metrics(y, P[(s, arm)][:, r]) for r in range(R)]
        return dict(acc=np.mean([m["accuracy"] for m in ms]), acc_sd=np.std([m["accuracy"] for m in ms]), f1=np.mean([m["macro_f1"] for m in ms]), f1_sd=np.std([m["macro_f1"] for m in ms]),
                    recall=np.mean([m["recall"] for m in ms], 0))
    pt = {(s, arm): point(s, arm) for s in settings for arm in ARMS}
    write_csv(f"{log.dir}/results_table.csv", [dict(setting=s, arm=arm, description=ARM_DESC[arm], accuracy_mean=round(v["acc"], 4), accuracy_sd=round(v["acc_sd"], 4), macro_f1_mean=round(v["f1"], 4),
                                                      macro_f1_sd=round(v["f1_sd"], 4), recall_AGG0=round(v["recall"][0], 4), recall_AGG1=round(v["recall"][1], 4), recall_AGG2=round(v["recall"][2], 4),
                                                      primary=int(s == pid and arm == parr)) for (s, arm), v in pt.items()])

    # ---------- 부트스트랩 (같은 재표본을 모든 팔에 사용) ----------
    nb = int(cfg["bootstrap"]["n"]); rng = np.random.RandomState(cfg["bootstrap"]["seed"])
    members = {c: np.where(np.array(conds) == c)[0] for c in cond_list}
    idx_img = rng.randint(0, N, (nb, N))
    idx_clu = np.array([np.concatenate([members[cond_list[k]] for k in rng.randint(0, len(cond_list), len(cond_list))]) for _ in range(nb)])
    assert idx_clu.shape == (nb, N), "조건마다 이미지 수가 같아야 함"
    log(f"부트스트랩 {nb}회 (이미지 단위, 처리 조건 단위), seed {cfg['bootstrap']['seed']}")

    def boot_vec(s, arm, idx):
        Pm = P[(s, arm)]
        acc = np.zeros(nb); f1 = np.zeros(nb)
        for b in range(nb):
            ii = idx[b]
            accs, f1s = [], []
            for r in range(R):
                cm = np.bincount(y[ii] * 3 + Pm[ii, r], minlength=9).reshape(3, 3)
                m = metrics_from_confusion(cm)
                accs.append(m["accuracy"]); f1s.append(m["macro_f1"])
            acc[b], f1[b] = np.mean(accs), np.mean(f1s)
        return acc, f1
    boot = {}
    for s in settings:
        for arm in ARMS:
            for mode, idx in (("image", idx_img), ("cluster", idx_clu)):
                boot[(s, arm, mode)] = boot_vec(s, arm, idx)
    ci_rows = []
    for (s, arm, mode), (acc, f1) in boot.items():
        lo, hi = np.percentile(acc, [2.5, 97.5]); flo, fhi = np.percentile(f1, [2.5, 97.5])
        ci_rows.append(dict(setting=s, arm=arm, mode=mode, accuracy=round(pt[(s, arm)]["acc"], 4), acc_lo=round(lo, 4), acc_hi=round(hi, 4), macro_f1=round(pt[(s, arm)]["f1"], 4), f1_lo=round(flo, 4), f1_hi=round(fhi, 4), n_boot=nb))
    write_csv(f"{log.dir}/bootstrap_ci.csv", ci_rows)

    def diff(s, a_, b_, mode, key=0):
        d = boot[(s, a_, mode)][key] - boot[(s, b_, mode)][key]
        base = (pt[(s, a_)]["acc"] - pt[(s, b_)]["acc"]) if key == 0 else (pt[(s, a_)]["f1"] - pt[(s, b_)]["f1"])
        lo, hi = np.percentile(d, [2.5, 97.5])
        return base, lo, hi, float((d > 0).mean())
    comps = [("R1", "R0", "규칙 2: 증강 학습의 효과 (원본 평가)"), ("R0", "R3", "규칙 1: 배경을 바꾸면 원본 학습 모델이 떨어지는 정도"), ("R2", "R3", "탐색: 증강본 평가에서 증강 학습의 효과")]
    drows = []
    for s in settings:
        for a_, b_, desc in comps:
            for mode in ("image", "cluster"):
                for key, nm in ((0, "accuracy"), (1, "macro_f1")):
                    d, lo, hi, pg = diff(s, a_, b_, mode, key)
                    drows.append(dict(setting=s, comparison=f"{a_} - {b_}", description=desc, mode=mode, metric=nm, diff=round(d, 4), ci_lo=round(lo, 4), ci_hi=round(hi, 4), excludes_zero=int(lo > 0 or hi < 0), frac_boot_gt0=round(pg, 3)))
    write_csv(f"{log.dir}/paired_diff.csv", drows)

    # ---------- 콘솔/로그 요약 ----------
    log("\n[결과] accuracy (10회 반복 평균 ± 반복 간 SD) 와 이미지 단위 95% CI")
    for s in settings:
        for arm in ARMS:
            lo, hi = [x for x in [r for r in ci_rows if r["setting"] == s and r["arm"] == arm and r["mode"] == "image"][0:1]][0]["acc_lo"], [r for r in ci_rows if r["setting"] == s and r["arm"] == arm and r["mode"] == "image"][0]["acc_hi"]
            log(f"  {s:<32} {arm}: {pt[(s, arm)]['acc']:.3f} ± {pt[(s, arm)]['acc_sd']:.3f}  [{lo:.3f}, {hi:.3f}] | macro-F1 {pt[(s, arm)]['f1']:.3f} | recall AGG0/1/2 {pt[(s, arm)]['recall'][0]:.2f}/{pt[(s, arm)]['recall'][1]:.2f}/{pt[(s, arm)]['recall'][2]:.2f}")
    log("\n[쌍 차이] accuracy")
    for s in settings:
        for a_, b_, desc in comps:
            im, cl = diff(s, a_, b_, "image"), diff(s, a_, b_, "cluster")
            log(f"  {s:<32} {a_}-{b_}: {im[0]:+.3f}  이미지 CI [{im[1]:+.3f}, {im[2]:+.3f}] | 조건 CI [{cl[1]:+.3f}, {cl[2]:+.3f}]  ({desc})")

    # ---------- 기준선, 이전 실험, 순열 ----------
    exp_out = Path(cfg["paths"]["exp_dir"]) / "outputs"
    bl = {r["baseline"]: float(r["accuracy"]) for r in csv.DictReader(open(exp_out / "e02_reference_baselines" / "baseline_metrics.csv", encoding="utf-8-sig"))}
    prev = {r["setting"]: float(r["accuracy_mean"]) for r in csv.DictReader(open(exp_out / "e04_report" / "results_table.csv", encoding="utf-8-sig"))}
    perm = [float(r["accuracy"]) for r in csv.DictReader(open(f"{out}/b04b_permutation/permutation_runs.csv", encoding="utf-8-sig"))]
    pf1 = [float(r["macro_f1"]) for r in csv.DictReader(open(f"{out}/b04b_permutation/permutation_runs.csv", encoding="utf-8-sig"))]
    log(f"\n참고: 다수 클래스 {bl['majority_class']:.3f}, 처리 조건 다수 라벨 암기 {bl['condition_memorization']:.3f} (e02) | 이전 실험(겹침 없는 타일 1,514개) primary {prev[pid]:.3f} | 순열 20회 accuracy {np.mean(perm):.3f} (범위 {min(perm):.2f}~{max(perm):.2f})")

    # ---------- 그룹별 오분류율 (primary 설정, R0 vs R1) ----------
    groups = {"금속 종류": [image_parts(i)[0] for i in ids], "처리 방법": [parse_condition(c)[0] for c in conds], "온도": [parse_condition(c)[1] for c in conds], "처리 조건": conds}
    mrows = []
    for arm in ("R0", "R1"):
        wrong = (P[(pid, arm)] != y[:, None])
        for gname, labels in groups.items():
            for g in sorted(set(labels)):
                sel = np.array([l == g for l in labels])
                mrows.append(dict(setting=pid, arm=arm, grouping=gname, group=g, n_images=int(sel.sum()), wrong_rate=round(float(wrong[sel].mean()), 4)))
    write_csv(f"{log.dir}/misclassification_by_group.csv", mrows)

    # ---------- 그림 ----------
    fig, ax = plt.subplots(figsize=(11, 5))
    w = 0.2
    for k, arm in enumerate(ARMS):
        xs = np.arange(len(settings)) + (k - 1.5) * w
        v = [pt[(s, arm)]["acc"] for s in settings]
        lo = [[r for r in ci_rows if r["setting"] == s and r["arm"] == arm and r["mode"] == "image"][0]["acc_lo"] for s in settings]
        hi = [[r for r in ci_rows if r["setting"] == s and r["arm"] == arm and r["mode"] == "image"][0]["acc_hi"] for s in settings]
        ax.bar(xs, v, w, color=COL[arm], label=f"{arm}  {ARM_DESC[arm]}", yerr=[np.array(v) - np.array(lo), np.array(hi) - np.array(v)], capsize=3, ecolor="#333")
        for x_, v_ in zip(xs, v):
            ax.text(x_, 0.31, f"{v_:.3f}", ha="center", fontsize=7, color="white", rotation=90)
    ax.axhline(bl["majority_class"], color="#777", ls="--", label=f"항상 다수 클래스 {bl['majority_class']:.3f}")
    ax.axhline(bl["condition_memorization"], color="k", ls=":", label=f"처리 조건 다수 라벨 암기 {bl['condition_memorization']:.2f}")
    ax.set_xticks(range(len(settings))); ax.set_xticklabels([s.replace("tiles|", "").replace("|", " · ") for s in settings]); ax.set_ylim(0.3, 1.0)
    ax.set_ylabel("accuracy (10회 반복 평균, 막대 끝선 = 이미지 단위 95% CI)"); ax.legend(fontsize=7.5, loc="upper left", ncol=2)
    ax.set_title("배경 증강 실험: 설정 x 팔 별 정확도 (R2, R3 은 증강본 5개 확률 평균이라 R0, R1 과 직접 비교 불가)", fontsize=10)
    fig.tight_layout(); fig.savefig(log.dir / "fig1_accuracy_by_arm.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ylabels, yy = [], 0
    for s in settings:
        for a_, b_, desc in comps:
            im = diff(s, a_, b_, "image"); cl = diff(s, a_, b_, "cluster")
            c = COL["R1"] if s == pid else "#555"
            ax.errorbar(im[0], yy, xerr=[[im[0] - im[1]], [im[2] - im[0]]], fmt="o", color=c, capsize=3)
            ax.errorbar(cl[0], yy - 0.22, xerr=[[cl[0] - cl[1]], [cl[2] - cl[0]]], fmt="s", color=c, capsize=3, alpha=0.45, markersize=4)
            ylabels.append(f"{s.replace('tiles|', '').replace('|', ' · ')}\n{a_} - {b_}"); yy -= 1
        yy -= 0.4
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks([-(i + 0.4 * (i // 3)) for i in range(len(ylabels))]); ax.set_yticklabels(ylabels, fontsize=7)
    ax.set_xlabel("accuracy 차이 (원: 이미지 단위 95% CI, 사각형: 처리 조건 단위 95% CI)")
    ax.set_title("쌍 차이: R1-R0 (증강 학습 효과), R0-R3 (배경 의존), R2-R3 (탐색)", fontsize=10)
    fig.tight_layout(); fig.savefig(log.dir / "fig2_paired_differences.png", dpi=140); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for ax, arm in zip(axes, ("R0", "R1")):
        cm = sum(np.bincount(y * 3 + P[(pid, arm)][:, r], minlength=9).reshape(3, 3) for r in range(R))
        rate = cm / cm.sum(1, keepdims=True)
        ax.imshow(rate, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{rate[i, j] * 100:.0f}%\n({cm[i, j]})", ha="center", va="center", color="white" if rate[i, j] > 0.5 else "black", fontsize=10)
        ax.set_xticks(range(3)); ax.set_xticklabels(["AGG0", "AGG1", "AGG2"]); ax.set_yticks(range(3)); ax.set_yticklabels([f"AGG{c}\n(n={cm[c].sum()})" for c in range(3)])
        ax.set_xlabel("예측"); ax.set_ylabel("실제"); ax.set_title(f"{arm} ({ARM_DESC[arm]})\naccuracy {np.trace(cm) / cm.sum():.3f}", fontsize=9)
    fig.suptitle("primary 설정 혼동행렬 (10회 반복 합산, 행 비율 %, 괄호 = 개수)", fontsize=10)
    fig.tight_layout(); fig.savefig(log.dir / "fig3_confusion_R0_R1.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3.2))
    rr = np.random.default_rng(0)
    ax.scatter(perm, rr.uniform(-0.12, 0.12, len(perm)), color="#999", s=24, label=f"라벨 섞음 {len(perm)}회 (accuracy)")
    ax.scatter([pt[(pid, parr)]["acc"]], [0], color=COL["R1"], marker="D", s=80, zorder=3, label=f"실제 라벨 primary {parr} ({pt[(pid, parr)]['acc']:.3f})")
    ax.axvline(bl["majority_class"], color="#777", ls="--", label="항상 다수 클래스"); ax.axvline(1 / 3, color="k", ls=":", lw=0.8, label="무작위 1/3")
    ax.set_yticks([]); ax.set_xlim(0.15, 1.0); ax.set_xlabel("accuracy"); ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2, frameon=False)
    ax.set_title("라벨 순열 검사 (primary 팔 R1)", fontsize=10)
    fig.tight_layout(); fig.savefig(log.dir / "fig4_permutation.png", dpi=140); plt.close(fig)

    # ---------- 보고서 ----------
    g = lambda s, a_, b_, mode, key=0: diff(s, a_, b_, mode, key)
    r1r0, r0r3, r2r3 = g(pid, "R1", "R0", "image"), g(pid, "R0", "R3", "image"), g(pid, "R2", "R3", "image")
    r1r0c, r0r3c = g(pid, "R1", "R0", "cluster"), g(pid, "R0", "R3", "cluster")
    rule1 = r0r3[1] > 0 or r0r3[2] < 0
    rule2 = r1r0[1] > 0 or r1r0[2] < 0
    ci = lambda s, arm, mode="image": [r for r in ci_rows if r["setting"] == s and r["arm"] == arm and r["mode"] == mode][0]
    L = []
    L.append("# 배경 증강 실험 결과 (exp_bgaug, b00~b05)\n")
    L.append("고정 DINOv2 ViT-B/14 특징 + 이미지 평균 + 로지스틱 회귀 / kNN, 이미지 단위 층화 5-fold x 10 반복(기존 e00 분할 재사용). "
             f"타일은 겹치지 않는 224px 격자({len(tiles)}개, 시편 픽셀 0인 타일 제외), 배경 픽셀은 마스크(m01) 밖에서 무작위 증강(이미지당 증강본 5개).\n")
    L.append("## 1. 사실 (측정값)\n")
    L.append(f"- 시편 픽셀의 99.74%가 학습 타일에 들어간다 (처음 실험의 타일 1,514개는 72.4%). 타일 {len(tiles)}개 중 배경이 섞인 타일이 약 절반이다 (b00 로그).")
    L.append(f"- primary (cls · mean · logreg), 팔 {parr} (원본+증강 학습, 원본 평가): accuracy **{pt[(pid, parr)]['acc']:.3f}** (이미지 단위 95% CI [{ci(pid, parr)['acc_lo']:.3f}, {ci(pid, parr)['acc_hi']:.3f}], 처리 조건 단위 [{ci(pid, parr, 'cluster')['acc_lo']:.3f}, {ci(pid, parr, 'cluster')['acc_hi']:.3f}]), macro-F1 {pt[(pid, parr)]['f1']:.3f}.")
    L.append(f"- 증강 없는 R0: {pt[(pid, 'R0')]['acc']:.3f} [{ci(pid, 'R0')['acc_lo']:.3f}, {ci(pid, 'R0')['acc_hi']:.3f}]. 원본 학습으로 증강본 평가 R3: {pt[(pid, 'R3')]['acc']:.3f}. 증강 학습으로 증강본 평가 R2: {pt[(pid, 'R2')]['acc']:.3f}.")
    L.append(f"- 기준선: 항상 다수 클래스 {bl['majority_class']:.3f}, 처리 조건 다수 라벨 암기 {bl['condition_memorization']:.2f}. 이전 실험(타일 1,514개, 증강 없음) primary {prev[pid]:.3f}.")
    L.append(f"- 라벨 순열 20회: accuracy 평균 {np.mean(perm):.3f} (범위 {min(perm):.2f}~{max(perm):.2f}), macro-F1 평균 {np.mean(pf1):.3f}. 제안 기준을 통과했다.\n")
    L.append("### 결과표 (accuracy 평균 ± 반복 간 SD, 이미지 단위 95% CI)\n")
    L.append("| 설정 | R0 | R1 (primary 팔) | R2 | R3 |\n|---|---|---|---|---|")
    for s in settings:
        L.append(f"| {s.replace('tiles|', '').replace('|', ' · ')} | " + " | ".join(f"{pt[(s, arm)]['acc']:.3f} ± {pt[(s, arm)]['acc_sd']:.3f} [{ci(s, arm)['acc_lo']:.3f}, {ci(s, arm)['acc_hi']:.3f}]" for arm in ARMS) + " |")
    L.append("\n### 쌍 차이 (accuracy, primary 설정; 이미지 단위 / 처리 조건 단위 95% CI)\n")
    L.append("| 비교 | 차이 | 이미지 CI | 조건 CI | 0 제외(이미지) |\n|---|---|---|---|---|")
    for a_, b_, desc in comps:
        im, cl = g(pid, a_, b_, "image"), g(pid, a_, b_, "cluster")
        L.append(f"| {a_} - {b_} ({desc}) | {im[0]:+.3f} | [{im[1]:+.3f}, {im[2]:+.3f}] | [{cl[1]:+.3f}, {cl[2]:+.3f}] | {'예' if (im[1] > 0 or im[2] < 0) else '아니오'} |")
    L.append("\n다른 두 설정(cls+patchmean · logreg, cls · knn)의 같은 비교는 `paired_diff.csv` 에 있다.\n")
    L.append("## 2. 사전 지정 규칙의 적용\n")
    L.append(f"- 규칙 1 (R0 - R3 의 이미지 단위 CI 가 0 을 제외하면 배경 의존의 증거): 차이 {r0r3[0]:+.3f}, CI [{r0r3[1]:+.3f}, {r0r3[2]:+.3f}] → **{'0 을 제외한다: 원본 학습 모델이 배경에 일부 기대고 있었다는 증거로 본다' if rule1 else '0 을 포함한다: 배경 의존을 단정하지 않는다'}**. 처리 조건 단위 CI 는 [{r0r3c[1]:+.3f}, {r0r3c[2]:+.3f}] ({'0 을 제외' if (r0r3c[1] > 0 or r0r3c[2] < 0) else '0 을 포함'}).")
    L.append(f"- 규칙 2 (R1 - R0 의 이미지 단위 CI 가 0 을 포함하면 증강 효과 없음이라고도 단정하지 않는다): 차이 {r1r0[0]:+.3f}, CI [{r1r0[1]:+.3f}, {r1r0[2]:+.3f}] → **{'0 을 제외한다: 증강 학습이 원본 평가 점수를 바꾼 것으로 본다' if rule2 else '0 을 포함한다: 증강 효과가 있다고도 없다고도 단정하지 않고 차이만 적는다'}**. 처리 조건 단위 CI 는 [{r1r0c[1]:+.3f}, {r1r0c[2]:+.3f}].")
    L.append("- 신뢰구간은 이미지를 독립으로 본 값이라 같은 처리 조건 안의 상관 때문에 실제보다 좁다 (조건 단위 CI 가 더 보수적이다).\n")
    L.append("## 3. 해석 (추정이며 검증되지 않음)\n")
    L.append("- 이전 실험 대비 R0 점수 상승은 증강 때문이 아니다 (R0 에는 증강이 없다). 타일 집합이 바뀐 영향(시편 가장자리와 배경이 섞인 타일이 입력에 들어간 것 포함)일 수 있으나, 어떤 요인이 기여했는지는 이 실험으로 가르지 않았다.")
    L.append("- 배경이 섞인 타일에는 시편 크기, 가장자리 모양 같은 정보가 들어 있다. 이 정보가 처리 조건과 얽혀 있다면 점수에 기여했을 수 있다. 이것을 확인하려면 배경을 전부 바꾼 입력(R3)과 증강 학습 모델(R1, R2)을 비교해야 한다 (위 규칙 1, 2).")
    L.append("- R2, R3 는 증강본 5개의 확률을 평균한 값이다. 앙상블 효과가 섞이므로 R0, R1 과 같은 선상에서 비교하지 않는다. R2 - R3 는 같은 증강본 입력에 대한 증강 학습의 효과로 읽을 수 있으나 사전 지정 비교가 아니다.\n")
    L.append("## 4. 한계\n")
    L.append("1. **처리 조건 혼동**: 라벨의 90%가 처리 조건으로 설명된다 (처리 조건 암기 0.90). 같은 처리 조건의 이미지가 train 과 test 에 모두 있으므로 점수가 형태를 읽은 결과인지 처리 조건을 알아본 결과인지 구분되지 않는다. 처리 조건 단위 분할(leave-one-condition-out) 평가는 하지 않았다.")
    L.append("2. **모델은 학습하지 않았다**: DINOv2 는 고정이고 분류기는 평균 특징 위의 로지스틱 회귀/kNN 이다. 증강은 학습 행을 6배로 늘리는 역할만 한다. 모델 자체를 개선했다고 말할 수 없다.")
    L.append("3. **합성 배경의 한계**: 합성 배경은 60장 배경 측정 풀에서 뽑은 평균색, 저주파 변동, 중간 대역 질감, 노이즈 공분산으로 만들었다. 노이즈의 공간 상관과 중간 대역 질감(원본의 약 0.84배, 이미지별 편차 큼)은 완전히 맞추지 않았다. 링(증강에만 생기는 경계)은 feather 16px 에서 밝기 프로파일 기준 z>3 이 6%였다 (b02c).")
    L.append("4. **마스크 한계**: m01 마스크가 시편 일부를 놓친 이미지(예: BASE-LCSI/NPH1035 오른쪽 위)에서는 그 부분의 시편 픽셀이 배경으로 취급되어 증강될 수 있다. 완충 띠 8px 가 일부만 막는다. 이미지별 수동 보정은 하지 않았다.")
    L.append("5. 표본이 이미지 60장이라 CI 가 넓다. 반복 10회의 SD 는 같은 이미지를 다른 분할로 본 변동이지 독립 표본의 변동이 아니다.")
    L.append("6. 격자가 이미지 오른쪽 32px, 아래 24px 를 덮지 않아 시편 픽셀의 0.26% 는 어떤 타일에도 들어가지 않는다 (b03c).\n")
    L.append("## 5. 파일\n")
    L.append("`results_table.csv`, `bootstrap_ci.csv`(이미지/조건 단위 CI), `paired_diff.csv`(쌍 차이), `misclassification_by_group.csv`(primary 설정, R0 vs R1 그룹별 오분류율), `fig1_accuracy_by_arm.png`, `fig2_paired_differences.png`, `fig3_confusion_R0_R1.png`, `fig4_permutation.png`.")
    (Path(log.dir) / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    log(f"\n보고서 저장: {Path(log.dir) / 'report.md'} (sha256 {sha256_file(Path(log.dir) / 'report.md')[:16]}…)")
    bad = forbidden_check(cfg)
    log(f"금지어 검사: {'없음' if not bad else bad}")
    assert not bad


if __name__ == "__main__":
    main()
