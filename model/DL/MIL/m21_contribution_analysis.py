"""m21: gated MIL 의 attention 과 additive MIL 의 attention, 기여도(a_k z_k)에 같은 정량 지표(m13 의 A1~A4)를 나란히 계산하고 오버레이를 그린다 (새 학습 없음; numpy, cv2, matplotlib).

입력: gated 결과 (outputs/, config.yaml) 와 additive 결과 (outputs_additive/, config_additive.yaml) 의 m10 출력 (attention_arrays.npz, attention_tiles.csv, mil_predictions.csv). 두 run 은 같은 분할, 같은 타일 순서, 같은 학습 절차이고 모델 종류만 다르다.
metric_basis (지표를 계산하는 가중치):
  gated_attention        gated 모델의 attention a_k (m13 과 같은 값)
  additive_attention     additive 모델의 attention a_k
  contrib_pos            additive 모델의 기여도 a_k z_k 의 양의 부분 max(c, 0) 을 이미지마다 합 1 로 정규화 (AGG1+2 로 미는 타일)
  contrib_neg            같은 기여도의 음의 부분 max(-c, 0) 을 합 1 로 정규화 (AGG0 으로 미는 타일). 기여도는 시드 5개 평균(부호 있음)을 먼저 구한 뒤 부분으로 나눈다.
  AGG0 이미지는 음의 기여가, AGG1/AGG2 이미지는 양의 기여가 bag logit 을 결정한다 (정답 쪽으로 미는 기여). 두 부분을 모든 그룹에 대해 보고한다.
지표 (m13 과 같은 정의, 결과를 보기 전에 정함):
  A1 가중치 질량: 배경이 섞인 타일(mask_frac < 1)이 가중치를 차지하는 비율 대 타일 비율 (배율 = 질량 / 비율, 1 = 균일)
  A2 집중도: effective number / 타일 수, 최대 가중치
  A3 안정성: 시드끼리, 원본 대 배경 증강 bag 의 순위 상관(Spearman). 기여도는 부호 있는 a_k z_k 자체로 계산 (contrib_signed 행).
  A4 상위 10% 타일 중 배경이 섞인 타일 비율과 기대 비율
추가 (기여도만): 부호 있는 합. 배경이 섞인 타일들의 기여도 합과 안쪽 타일들의 합, 둘의 합 (= bag logit - bias), logit 단위, 이미지 평균.
그림: m13 의 대표 이미지(클래스마다 맞힌 이미지 중 양성 확률이 중앙값에 가까운 1장, 틀린 이미지 중 오차가 가장 큰 1장, LOCO) 를 세 가지 기준으로 나란히 칠한다.
출력 (outputs_additive/m21_contribution_analysis/): attention_vs_contribution_stats.csv, signed_contribution_sums.csv, stability.csv, fig_contribution_typical.png, fig_contribution_errors.png, log.txt
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "exp_group_cv"))
from gcommon import Logger, load_images, write_csv  # noqa: E402
from gcommon import load_config as _lc  # noqa: E402
import os as _os  # noqa: E402
from matplotlib import font_manager as _fm  # noqa: E402

for _f in ("C:/Windows/Fonts/malgun.ttf", "/mnt/c/Windows/Fonts/malgun.ttf"):
    if _os.path.exists(_f):
        _fm.fontManager.addfont(_f)
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
SCRIPT = "m21_contribution_analysis"
BASES = ["gated_attention", "additive_attention", "contrib_pos", "contrib_neg"]


def rank(x):
    return np.argsort(np.argsort(x)).astype(float)


def spearman(a, b):
    ra, rb = rank(a), rank(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gated", default="outputs"); ap.add_argument("--additive", default="outputs_additive"); ap.add_argument("--config", default=str(HERE / "config_additive.yaml"))
    a = ap.parse_args()
    cfg = _lc(a.config)
    gdir, adir = HERE / a.gated, HERE / a.additive
    log = Logger(adir, SCRIPT)
    img = load_images(cfg); ids = img["ids"]; y3 = img["y"]; y = np.isin(y3, cfg["positive_labels"]).astype(int); N = len(ids)
    Zg = np.load(gdir / "m10_train_eval_mil" / "attention_arrays.npz"); Za = np.load(adir / "m10_train_eval_mil" / "attention_arrays.npz")
    assert "contrib_LOCO" in Za.files and "contrib_LOCO" not in Zg.files, "gated 결과에는 기여도 배열이 없어야 하고 additive 결과에는 있어야 함"
    counts = Zg["tile_counts"]; assert np.array_equal(counts, Za["tile_counts"])
    tg = list(csv.DictReader(open(gdir / "m10_train_eval_mil" / "attention_tiles.csv", encoding="utf-8-sig")))
    ta = list(csv.DictReader(open(adir / "m10_train_eval_mil" / "attention_tiles.csv", encoding="utf-8-sig")))
    assert [(r["image_id"], r["x0"], r["y0"]) for r in tg] == [(r["image_id"], r["x0"], r["y0"]) for r in ta], "두 run 의 타일 순서가 다름"
    tby = defaultdict(list)
    for r in ta:
        tby[int(r["image_idx"])].append(r)
    full = {i: np.array([float(r["mask_frac"]) >= 0.999999 for r in tby[i]]) for i in range(N)}
    pred = {}
    for nm, d in (("gated", gdir), ("additive", adir)):
        for r in csv.DictReader(open(d / "m10_train_eval_mil" / "mil_predictions.csv", encoding="utf-8-sig")):
            if r["repeat"] == "0":
                pred.setdefault((nm, r["scheme"]), {})[ids.index(r["image_id"])] = (float(r["prob"]), int(r["pred"]))
    topf = cfg["attention"]["top_fraction"]; main_s = cfg["attention"]["main_scheme"]
    log("config:", cfg["_config_path"], "| gated:", a.gated, "| additive:", a.additive)
    schemes = [s for s in ("IMG", "LOCO", "LOMO") if f"contrib_{s}" in Za.files]

    def mean_seed(arr, i, k=0):
        return np.nanmean(arr[i, k, :, :counts[i]], axis=0)

    def weights(i, sc, basis):
        """이미지 i 의 타일 n 개에 대한 합 1 가중치 (contrib_pos/neg 는 합이 0 이면 None)"""
        if basis == "gated_attention":
            return mean_seed(Zg[f"att_{sc}"], i)
        if basis == "additive_attention":
            return mean_seed(Za[f"att_{sc}"], i)
        c = mean_seed(Za[f"contrib_{sc}"], i)
        v = np.maximum(c, 0) if basis == "contrib_pos" else np.maximum(-c, 0)
        return v / v.sum() if v.sum() > 1e-12 else None

    stats, sums, stab = [], [], []
    for sc in schemes:
        for lab in ("전체", "AGG0", "AGG1", "AGG2", "양성(AGG1+2)"):
            sel = [i for i in range(N) if lab == "전체" or lab == f"AGG{y3[i]}" or (lab.startswith("양성") and y[i] == 1)]
            for basis in BASES:
                mp, fp, eff, mx, tp, ep = [], [], [], [], [], []
                for i in sel:
                    w = weights(i, sc, basis)
                    if w is None:
                        continue
                    n = len(w); pm = ~full[i]
                    mp.append(float(w[pm].sum())); fp.append(float(pm.mean())); eff.append(float(np.exp(-(w * np.log(w + 1e-12)).sum()) / n)); mx.append(float(w.max()))
                    top = np.argsort(-w)[:max(1, int(round(topf * n)))]; tp.append(float(pm[top].mean())); ep.append(float(pm.mean()))
                if not mp:
                    continue
                stats.append(dict(scheme=sc, group=lab, metric_basis=basis, n_images=len(mp), weight_mass_on_partial_tiles=round(float(np.mean(mp)), 4), tile_fraction_partial=round(float(np.mean(fp)), 4),
                                  ratio_partial=round(float(np.mean(mp)) / float(np.mean(fp)), 3), effective_n_over_n=round(float(np.mean(eff)), 4), max_weight=round(float(np.mean(mx)), 4),
                                  top10_partial_share=round(float(np.mean(tp)), 4), top10_expected=round(float(np.mean(ep)), 4)))
            # 부호 있는 합 (logit 단위)
            c_part, c_in, c_tot = [], [], []
            for i in sel:
                c = mean_seed(Za[f"contrib_{sc}"], i)
                c_part.append(float(c[~full[i]].sum())); c_in.append(float(c[full[i]].sum())); c_tot.append(float(c.sum()))
            sums.append(dict(scheme=sc, group=lab, n_images=len(sel), net_contrib_partial_tiles=round(float(np.mean(c_part)), 3), net_contrib_inner_tiles=round(float(np.mean(c_in)), 3), net_contrib_total=round(float(np.mean(c_tot)), 3)))
        # A3 안정성
        K1, S = Za[f"att_{sc}"].shape[1], Za[f"att_{sc}"].shape[2]
        for basis, arr_g in (("gated_attention", Zg[f"att_{sc}"]), ("additive_attention", Za[f"att_{sc}"]), ("contrib_signed", Za[f"contrib_{sc}"])):
            sd, vr = [], []
            for i in range(N):
                n = counts[i]
                sd.append(float(np.nanmean([spearman(arr_g[i, 0, s1, :n], arr_g[i, 0, s2, :n]) for s1 in range(S) for s2 in range(s1 + 1, S)])))
                a0 = mean_seed(arr_g, i, 0); vr.append(float(np.nanmean([spearman(a0, mean_seed(arr_g, i, k)) for k in range(1, K1)])))
            stab.append(dict(scheme=sc, metric_basis=basis, seed_to_seed_spearman_mean=round(float(np.mean(sd)), 3), seed_to_seed_min=round(float(np.min(sd)), 3),
                             orig_vs_augmented_spearman_mean=round(float(np.mean(vr)), 3), orig_vs_augmented_min=round(float(np.min(vr)), 3)))
    write_csv(f"{log.dir}/attention_vs_contribution_stats.csv", stats); write_csv(f"{log.dir}/signed_contribution_sums.csv", sums); write_csv(f"{log.dir}/stability.csv", stab)
    log("\n[A1/A2/A4] 가중치 기준별 지표 (시험 단위를 뺀 모델, 시드 5개 평균, 이미지 평균)")
    for r in stats:
        if r["scheme"] == main_s:
            log(f"  {r['scheme']:<4} {r['group']:<12} {r['metric_basis']:<19} n={r['n_images']:>2} | 배경 섞인 타일 질량 {r['weight_mass_on_partial_tiles']:.3f} vs 타일 비율 {r['tile_fraction_partial']:.3f} (배율 {r['ratio_partial']:.2f}) | "
                f"effective n / n {r['effective_n_over_n']:.3f}, 최대 {r['max_weight']:.3f} | 상위 10% 중 배경 섞인 {r['top10_partial_share']:.3f} (기대 {r['top10_expected']:.3f})")
    log("\n[기여도의 부호 있는 합 (logit 단위, 이미지 평균)]")
    for r in sums:
        if r["scheme"] == main_s:
            log(f"  {r['scheme']:<4} {r['group']:<12} 배경 섞인 타일 {r['net_contrib_partial_tiles']:+.2f} | 안쪽 타일 {r['net_contrib_inner_tiles']:+.2f} | 합 {r['net_contrib_total']:+.2f}")
    log("\n[A3] 안정성 (Spearman)")
    for r in stab:
        log(f"  {r['scheme']:<4} {r['metric_basis']:<19} 시드끼리 {r['seed_to_seed_spearman_mean']:.3f} (최소 {r['seed_to_seed_min']:.3f}) | 원본 대 배경 증강 {r['orig_vs_augmented_spearman_mean']:.3f} (최소 {r['orig_vs_augmented_min']:.3f})")

    # ---------- 오버레이 ----------
    reps_csv = HERE / a.gated / "m13_attention_analysis" / "representative_images.csv"
    reps = [(r["kind"], int(r["label3"]), ids.index(r["image_id"])) for r in csv.DictReader(open(reps_csv, encoding="utf-8-sig"))]
    man = {r["image_id"]: r for r in csv.DictReader(open(cfg["paths"]["manifest_csv"], encoding="utf-8-sig"))}
    root = Path(cfg["paths"]["images_root"])

    def read_rgb(path):
        arr = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)

    def panel(ax, base, coords, vals, cmap, vmin, vmax, fmt, title):
        ov = base.copy().astype(np.float32); s = 56
        norm = lambda v: (v - vmin) / (vmax - vmin)
        for (cx, cy), v in zip(coords, vals):
            col = np.array(cmap(min(max(norm(v), 0), 1))[:3]) * 255
            ov[cy:cy + s, cx:cx + s] = base[cy:cy + s, cx:cx + s] * 0.35 + col * 0.65
        ov = ov.astype(np.uint8)
        for (cx, cy) in coords:
            cv2.rectangle(ov, (cx, cy), (cx + s - 1, cy + s - 1), (255, 255, 255), 1)
        ax.imshow(ov); ax.axis("off"); ax.set_title(title, fontsize=8)
        for (cx, cy), v in zip(coords, vals):
            ax.text(cx + 28, cy + 28, fmt.format(v), ha="center", va="center", fontsize=6.5, color="white", fontweight="bold", path_effects=[pe.withStroke(linewidth=1.6, foreground="black")])

    def figure(items, fname, title):
        if not items:
            return
        fig, axes = plt.subplots(len(items), 3, figsize=(17, 3.7 * len(items)), squeeze=False)
        for row, (kind, lab, i) in enumerate(items):
            base = cv2.resize(read_rgb(root / man[ids[i]]["path"]), None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
            coords = [(int(int(r["x0"]) * 0.25), int(int(r["y0"]) * 0.25)) for r in tby[i]]; n = len(coords)
            ag, aa = mean_seed(Zg[f"att_{main_s}"], i) * n, mean_seed(Za[f"att_{main_s}"], i) * n
            cc = mean_seed(Za[f"contrib_{main_s}"], i)
            tag = f"{ids[i].rsplit('/', 1)[0]} (실제 AGG{lab}, gated 양성 확률 {pred[('gated', main_s)][i][0]:.2f}, additive {pred[('additive', main_s)][i][0]:.2f})"
            panel(axes[row, 0], base, coords, ag, plt.cm.inferno, 0, 4.0, "{:.1f}", f"{tag}\ngated attention x 타일 수 (1 = 균일)")
            panel(axes[row, 1], base, coords, aa, plt.cm.inferno, 0, 4.0, "{:.1f}", "additive attention x 타일 수 (1 = 균일)")
            vm = float(np.abs(cc).max()) + 1e-9
            panel(axes[row, 2], base, coords, cc, plt.cm.bwr, -vm, vm, "{:+.1f}", f"additive 기여도 a_k z_k (빨강 = AGG1+2 쪽, 파랑 = AGG0 쪽, 합 {cc.sum():+.2f} + bias = logit)")
        fig.suptitle(title, fontsize=10); fig.tight_layout(); fig.savefig(log.dir / fname, dpi=110, bbox_inches="tight"); plt.close(fig)
    figure([r for r in reps if r[0] == "typical"], "fig_contribution_typical.png", f"gated attention 대 additive attention 대 additive 기여도 ({main_s} fold 모델, 시드 5개 평균): 클래스별 대표 이미지 (맞힌 것)")
    figure([r for r in reps if r[0] == "error"], "fig_contribution_errors.png", f"gated attention 대 additive attention 대 additive 기여도 ({main_s} fold 모델): 클래스별 가장 크게 틀린 이미지")
    log(f"\n저장: {log.dir}")


if __name__ == "__main__":
    main()
