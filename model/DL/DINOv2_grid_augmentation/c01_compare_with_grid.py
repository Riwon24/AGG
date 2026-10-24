"""c01: 배경 증강 판(이 폴더)과 증강 없는 같은 실험(exp_grid = DINOv2_grid)의 예측을 설정별로 쌍(paired)으로 비교하고, 두 primary 를 기준선(다수 클래스, 처리 조건 암기)과도 쌍으로 비교한다 (새 학습 없음, numpy 와 matplotlib 만 필요).

두 실험은 같은 이미지 60장, 같은 타일 2,591개, 같은 분할(e00 splits.csv sha256 같음), 같은 설정 16개, 같은 평가(원본 test 이미지)를 쓰고, 학습에 배경 증강본을 쓰느냐만 다르다.
방법: 같은 재표본(이미지 단위 60장 복원추출, 처리 조건 단위 10개 조건 복원추출, 각 2000회)으로 두 실험의 정확도(반복 10회 평균)를 구해 차이(증강 - 원본)의 95% 구간을 낸다.
해석 규칙 (결과를 보기 전에 정함): primary 설정의 차이에 대해 이미지 단위 95% CI 가 0 을 포함하면 "증강 효과가 있다고도 없다고도 단정하지 않고 차이만 적는다".
  나머지 15개 설정은 탐색용이며 16개를 한꺼번에 보고 있으므로(다중 비교) CI 가 0 을 제외하는 설정이 있어도 우연일 수 있다고 함께 적는다.
출력: paired_diff_vs_grid.csv, paired_diff_vs_baselines.csv, compare_report.md, fig_diff_vs_grid.png, log.txt
"""
import argparse
import csv
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

from common import Logger, image_parts, load_config, load_image_table, metrics_from_confusion, sha256_file, write_csv  # noqa: E402

SCRIPT = "c01_compare_with_grid"
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    out = cfg["paths"]["outputs_dir"]
    log = Logger(out, SCRIPT)
    log("config:", cfg["_config_path"])
    grid = Path(cfg["paths"]["exp_grid_dir"]) / "outputs"
    imgs = load_image_table(cfg)
    ids = [r["image_id"] for r in imgs]; pos = {i: k for k, i in enumerate(ids)}
    y = np.array([r["label"] for r in imgs]); N = len(ids)
    conds = [image_parts(i)[1] for i in ids]; cond_list = sorted(set(conds))
    sha_a, sha_b = sha256_file(f"{out}/e00_make_splits/splits.csv"), sha256_file(grid / "e00_make_splits" / "splits.csv")
    assert sha_a == sha_b, "두 실험의 분할이 다름"
    log(f"같은 분할 확인: splits.csv sha256 {sha_a[:16]}… (두 실험 동일)")

    def load(path):
        rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
        R = 1 + max(int(r["repeat"]) for r in rows)
        P = {}
        for r in rows:
            P.setdefault(r["setting"], np.full((N, R), -1, int))[pos[r["image_id"]], int(r["repeat"])] = int(r["pred"])
        return P, R
    Pa, R = load(f"{out}/e03_run_cv/cv_predictions.csv")
    Pg, R2 = load(grid / "e03_run_cv" / "cv_predictions.csv")
    assert R == R2 and list(Pa) == list(Pg), "설정 목록 또는 반복 횟수가 다름"
    settings = list(Pa)
    for s in settings:
        assert (Pa[s] >= 0).all() and (Pg[s] >= 0).all(), s
    pid = f"tiles|{cfg['experiment']['primary']['features']}|{cfg['experiment']['primary']['aggregator']}|{cfg['experiment']['primary']['classifier']}"
    log(f"설정 {len(settings)}개, 반복 {R}회, 이미지 {N}장, primary = {pid}")

    nb = int(cfg["bootstrap"]["n"]); rng = np.random.RandomState(cfg["bootstrap"]["seed"])
    members = {c: np.where(np.array(conds) == c)[0] for c in cond_list}
    idx_img = rng.randint(0, N, (nb, N))
    idx_clu = np.array([np.concatenate([members[cond_list[k]] for k in rng.randint(0, len(cond_list), len(cond_list))]) for _ in range(nb)])
    assert idx_clu.shape == (nb, N)

    def boot(P, idx):
        acc = np.zeros(nb); f1 = np.zeros(nb)
        for b in range(nb):
            ii = idx[b]; aa, ff = [], []
            for r in range(P.shape[1]):
                m = metrics_from_confusion(np.bincount(y[ii] * 3 + P[ii, r], minlength=9).reshape(3, 3))
                aa.append(m["accuracy"]); ff.append(m["macro_f1"])
            acc[b], f1[b] = np.mean(aa), np.mean(ff)
        return acc, f1

    def point(P):
        ms = [metrics_from_confusion(np.bincount(y * 3 + P[:, r], minlength=9).reshape(3, 3)) for r in range(R)]
        return np.mean([m["accuracy"] for m in ms]), np.mean([m["macro_f1"] for m in ms]), np.mean([m["recall"] for m in ms], 0)

    rows, res = [], {}
    for s in settings:
        pa, pg = point(Pa[s]), point(Pg[s])
        res[s] = dict(acc_aug=pa[0], acc_grid=pg[0], f1_aug=pa[1], f1_grid=pg[1], rec_aug=pa[2], rec_grid=pg[2])
        for mode, idx in (("image", idx_img), ("cluster", idx_clu)):
            aa, fa = boot(Pa[s], idx); ag, fg = boot(Pg[s], idx)
            for key, nm, va, vg, da, dg in ((0, "accuracy", pa[0], pg[0], aa, ag), (1, "macro_f1", pa[1], pg[1], fa, fg)):
                d = da - dg
                lo, hi = np.percentile(d, [2.5, 97.5])
                rows.append(dict(setting=s, mode=mode, metric=nm, aug=round(va, 4), grid=round(vg, 4), diff=round(va - vg, 4), ci_lo=round(lo, 4), ci_hi=round(hi, 4), excludes_zero=int(lo > 0 or hi < 0)))
                if nm == "accuracy":
                    res[s][f"ci_{mode}"] = (lo, hi)
    write_csv(f"{log.dir}/paired_diff_vs_grid.csv", rows)

    log("\n[증강 판 - 증강 없음(DINOv2_grid)] accuracy 차이, 95% CI (이미지 단위 / 처리 조건 단위)")
    n_ex = 0
    for s in settings:
        r_ = res[s]; d = r_["acc_aug"] - r_["acc_grid"]; li, lc = r_["ci_image"], r_["ci_cluster"]
        ex = (li[0] > 0 or li[1] < 0); n_ex += int(ex)
        log(f"  {s:<34} 증강 {r_['acc_aug']:.3f} | 증강 없음 {r_['acc_grid']:.3f} | 차이 {d:+.3f}  이미지 [{li[0]:+.3f}, {li[1]:+.3f}]  조건 [{lc[0]:+.3f}, {lc[1]:+.3f}]" + ("  *이미지 CI 가 0 제외*" if ex else ""))
    log(f"\n이미지 단위 CI 가 0 을 제외하는 설정: {n_ex}/{len(settings)}개 (16개를 함께 보는 다중 비교이므로 우연일 수 있음)")
    ps = res[pid]; li, lc = ps["ci_image"], ps["ci_cluster"]
    rule = "0 을 제외한다" if (li[0] > 0 or li[1] < 0) else "0 을 포함한다"
    log(f"\n사전 지정 규칙 (primary {pid}): 차이 {ps['acc_aug'] - ps['acc_grid']:+.3f}, 이미지 단위 CI [{li[0]:+.3f}, {li[1]:+.3f}] -> {rule}")


    # ---- 기준선(다수 클래스, 처리 조건 암기)과 두 primary 의 쌍 차이 ----
    bl = list(csv.DictReader(open(Path(cfg["paths"]["exp_dir"]) / "outputs" / "e02_reference_baselines" / "baseline_predictions.csv", encoding="utf-8-sig")))
    Pb = {"다수 클래스 예측": np.zeros((N, 1), int), "처리 조건 암기": np.zeros((N, 1), int)}
    for r in bl:
        Pb["다수 클래스 예측"][pos[r["image_id"]], 0] = int(r["pred_majority"]); Pb["처리 조건 암기"][pos[r["image_id"]], 0] = int(r["pred_condition_memorization"])
    brows, bres = [], {}
    prim_sets = {"배경 증강 학습": Pa[pid], "증강 없음 (DINOv2_grid)": Pg[pid]}
    log(f"\n[primary {pid} 와 기준선의 쌍 차이] accuracy (primary - 기준선), 95% CI")
    for bname, Pbase in Pb.items():
        bacc = float(np.mean(Pbase[:, 0] == y))
        for pname, Pp in prim_sets.items():
            pacc_ = point(Pp)[0]
            for mode, idx in (("image", idx_img), ("cluster", idx_clu)):
                dp = boot(Pp, idx)[0] - boot(Pbase, idx)[0]
                lo, hi = np.percentile(dp, [2.5, 97.5])
                brows.append(dict(primary=pname, baseline=bname, mode=mode, primary_acc=round(pacc_, 4), baseline_acc=round(bacc, 4), diff=round(pacc_ - bacc, 4), ci_lo=round(lo, 4), ci_hi=round(hi, 4), excludes_zero=int(lo > 0 or hi < 0)))
                bres[(bname, pname, mode)] = (pacc_ - bacc, lo, hi)
            li, lc = bres[(bname, pname, "image")], bres[(bname, pname, "cluster")]
            log(f"  {pname:<24} vs {bname:<12} : {pacc_:.3f} vs {bacc:.3f}, 차이 {li[0]:+.3f}  이미지 [{li[1]:+.3f}, {li[2]:+.3f}] | 조건 [{lc[1]:+.3f}, {lc[2]:+.3f}]")
    write_csv(f"{log.dir}/paired_diff_vs_baselines.csv", brows)

    # ---- 그림 ----
    fig, ax = plt.subplots(figsize=(9.5, 7.5))
    ys = np.arange(len(settings))[::-1]
    for y_, s in zip(ys, settings):
        r_ = res[s]; d = r_["acc_aug"] - r_["acc_grid"]; li, lc = r_["ci_image"], r_["ci_cluster"]
        c = "#c0392b" if s == pid else "#444"
        ax.errorbar(d, y_, xerr=[[d - li[0]], [li[1] - d]], fmt="o", color=c, capsize=3)
        ax.errorbar(d, y_ - 0.25, xerr=[[d - lc[0]], [lc[1] - d]], fmt="s", color=c, capsize=3, alpha=0.4, markersize=4)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(ys); ax.set_yticklabels([s.replace("tiles|", "").replace("|", " · ") for s in settings], fontsize=8)
    ax.set_xlabel("accuracy 차이 = 배경 증강 학습 - 증강 없음 (원: 이미지 단위 95% CI, 사각형: 처리 조건 단위 95% CI)")
    ax.set_title("배경 증강 학습의 효과, 16개 설정 (같은 타일, 같은 분할, 원본 평가). 빨강 = primary", fontsize=10)
    fig.tight_layout(); fig.savefig(log.dir / "fig_diff_vs_grid.png", dpi=140); plt.close(fig)

    # ---- 보고서 ----
    L = ["# 배경 증강 학습 vs 증강 없음 (DINOv2_grid), 같은 타일, 같은 분할, 원본 평가\n",
         "두 실험은 학습에 배경 증강본을 쓰느냐만 다르다 (이미지 60장, 타일 2,591개, 분할 sha256, 설정 16개, 평가 방식이 같다). 아래 차이는 같은 재표본 2000회로 계산한 95% 구간이다.\n",
         "| 설정 | 증강 없음 | 배경 증강 학습 | 차이 | 이미지 단위 95% CI | 처리 조건 단위 95% CI |", "|---|---|---|---|---|---|"]
    for s in settings:
        r_ = res[s]; d = r_["acc_aug"] - r_["acc_grid"]; li, lc = r_["ci_image"], r_["ci_cluster"]
        nm = ("**" + s + "**") if s == pid else s
        L.append(f"| {nm} | {r_['acc_grid']:.3f} | {r_['acc_aug']:.3f} | {d:+.3f} | [{li[0]:+.3f}, {li[1]:+.3f}] | [{lc[0]:+.3f}, {lc[1]:+.3f}] |")
    L.append(f"\n- **사전 지정 규칙** (primary): 차이 {ps['acc_aug'] - ps['acc_grid']:+.3f}, 이미지 단위 CI [{li[0] if False else ps['ci_image'][0]:+.3f}, {ps['ci_image'][1]:+.3f}] 가 {rule}"
             + (" -> 증강 효과가 있다고도 없다고도 단정하지 않고 차이만 적는다." if rule.endswith("포함한다") else " -> 증강 학습이 점수를 바꾼 것으로 본다."))
    L.append(f"- 16개 설정 중 이미지 단위 CI 가 0 을 제외하는 것은 {n_ex}개이다. 16개를 한꺼번에 보는 탐색이라 우연일 수 있다 (다중 비교 보정은 하지 않았다).")
    L.append("- 이미지 단위 CI 는 이미지를 독립으로 보기 때문에 같은 처리 조건 안의 상관을 반영하지 못해 처리 조건 단위 CI 보다 좁다.")
    L.append("\n## primary 와 기준선의 쌍 차이 (accuracy, primary - 기준선)\n")
    L.append("| primary | 기준선 | primary | 기준선 | 차이 | 이미지 단위 95% CI | 처리 조건 단위 95% CI |")
    L.append("|---|---|---|---|---|---|---|")
    for bname in Pb:
        for pname in prim_sets:
            d_, lo_i, hi_i = bres[(bname, pname, "image")]; _, lo_c, hi_c = bres[(bname, pname, "cluster")]
            pa_ = [r for r in brows if r["primary"] == pname and r["baseline"] == bname][0]
            L.append(f"| {pname} | {bname} | {pa_['primary_acc']:.3f} | {pa_['baseline_acc']:.3f} | {d_:+.3f} | [{lo_i:+.3f}, {hi_i:+.3f}] | [{lo_c:+.3f}, {hi_c:+.3f}] |")
    L.append("")
    L.append("- 이 비교는 증강 학습이 점수를 바꾸는지만 본다. 점수가 형태를 읽은 결과인지 처리 조건을 알아본 결과인지는 구분하지 못한다 (처리 조건 암기 기준선 0.90).")
    (Path(log.dir) / "compare_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    log(f"\n저장: {log.dir}")


if __name__ == "__main__":
    main()
