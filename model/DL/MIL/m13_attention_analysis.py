"""m13: MIL attention 의 정량 분석과 원본 이미지 위 패치 단위 오버레이 시각화 (새 학습 없음, numpy, cv2, matplotlib 필요).

attention 은 시험 단위를 학습에서 뺀 fold 모델의 것이다 (main_scheme = LOCO: 처리 조건 하나를 통째로 뺀 모델이 그 조건의 이미지에 대해 낸 attention). 시드 5개 평균.
이 파일은 측정값만 만든다. 관찰과 해석을 구분한 서술은 m14 가 이 결과를 읽어 쓴다.

정량 지표 (결과를 보기 전에 정의):
  A1 attention 질량: 마스크 100% 타일(시편만)과 배경이 섞인 타일(mask_frac < 1)이 attention 을 차지하는 비율을 타일 수 비율(균일한 attention 이면 얻을 비율)과 비교한 배율.
  A2 attention 집중도: effective number = exp(엔트로피) / 타일 수 (1 = 균일, 작을수록 소수 타일에 집중). 이미지의 최대 attention 가중치도 함께.
  A3 안정성: (a) 같은 버전에서 시드끼리 attention 의 Spearman 순위 상관, (b) 원본 bag 과 배경 증강 bag(같은 이미지, 시편 픽셀은 같고 배경만 다름)의 attention 순위 상관. (b) 가 (a) 보다 크게 낮으면 attention 이 배경에 의존한다는 신호.
  A4 상위 타일 구성: attention 상위 10% 타일 중 배경이 섞인 타일의 비율과 그 기대 비율(전체 타일 중 비율).
시각화: 원본 이미지 위에 타일(224px, 겹침 없음)마다 균일 attention 대비 배율(attention x 타일 수)로 색을 칠한다. 대표 이미지 선택 규칙은 config attention.representative_rule.
출력: attention_stats.csv, attention_stability.csv, attention_by_image.csv, fig_attention_typical.png, fig_attention_errors.png, fig_attention_dist.png, representative_images.csv, log.txt
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402

HERE = Path(__file__).resolve().parent
import sys  # noqa: E402
sys.path.insert(0, str(HERE.parent / "exp_group_cv"))
from gcommon import Logger, load_images, write_csv  # noqa: E402
from gcommon import load_config as _lc  # noqa: E402

SCRIPT = "m13_attention_analysis"
import os as _os  # noqa: E402  (WSL 에서는 Windows 글꼴 파일을 경로로 등록해야 한글이 나온다)
from matplotlib import font_manager as _fm  # noqa: E402
for _f in ("C:/Windows/Fonts/malgun.ttf", "/mnt/c/Windows/Fonts/malgun.ttf"):
    if _os.path.exists(_f):
        _fm.fontManager.addfont(_f)
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


def rank(x):
    return np.argsort(np.argsort(x)).astype(float)


def spearman(a, b):
    ra, rb = rank(a), rank(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = _lc(a.config or HERE / "config.yaml")
    out = Path(cfg["paths"]["outputs_dir"])
    log = Logger(out, SCRIPT)
    log("config:", cfg["_config_path"])
    img = load_images(cfg)
    ids = img["ids"]; y3 = img["y"]; y = np.isin(y3, cfg["positive_labels"]).astype(int); N = len(ids)
    Z = np.load(out / "m10_train_eval_mil" / "attention_arrays.npz")
    counts = Z["tile_counts"]
    tiles = list(csv.DictReader(open(out / "m10_train_eval_mil" / "attention_tiles.csv", encoding="utf-8-sig")))
    tby = defaultdict(list)
    for r in tiles:
        tby[int(r["image_idx"])].append(r)
    for i in range(N):
        assert len(tby[i]) == counts[i], "타일 수가 attention 배열과 다름"
    full = {i: np.array([float(r["mask_frac"]) >= 0.999999 for r in tby[i]]) for i in range(N)}          # 시편만 있는 타일
    man0 = {r["image_id"]: r for r in csv.DictReader(open(cfg["paths"]["manifest_csv"], encoding="utf-8-sig"))}
    md_ = cfg["paths"]["masks_dir"]
    mdir0 = Path(md_) if isinstance(md_, str) else Path(next(p_ for p_ in md_ if Path(p_).exists()))
    bdist = {}                                                                                            # 타일 중심에서 시편 경계(마스크 밖)까지의 거리(px)
    for i in range(N):
        mk0 = cv2.imdecode(np.fromfile(str(mdir0 / (ids[i].replace("/", "__") + ".png")), dtype=np.uint8), cv2.IMREAD_GRAYSCALE) > 127
        dt = cv2.distanceTransform(mk0.astype(np.uint8), cv2.DIST_L2, 5)
        bdist[i] = np.array([dt[int(r["y0"]) + 112, int(r["x0"]) + 112] for r in tby[i]], dtype=float)
    mfrac = {i: np.array([float(r["mask_frac"]) for r in tby[i]]) for i in range(N)}
    pred = {}
    for r in csv.DictReader(open(out / "m10_train_eval_mil" / "mil_predictions.csv", encoding="utf-8-sig")):
        if r["repeat"] == "0":
            pred.setdefault(r["scheme"], {})[ids.index(r["image_id"])] = (float(r["prob"]), int(r["pred"]))
    schemes = [s for s in ("IMG", "LOCO", "LOMO") if f"att_{s}" in Z.files]
    main_s = cfg["attention"]["main_scheme"]
    topf = cfg["attention"]["top_fraction"]
    stats_rows, stab_rows, img_rows = [], [], []

    def mean_att(att, i, k=0):
        n = counts[i]
        return np.nanmean(att[i, k, :, :n], axis=0)

    for sc in schemes:
        att = Z[f"att_{sc}"]                                                      # (N, K1, S, nmax)
        K1, S = att.shape[1], att.shape[2]
        for lab in ("전체", "AGG0", "AGG1", "AGG2", "양성(AGG1+2)"):
            sel = [i for i in range(N) if lab == "전체" or lab == f"AGG{y3[i]}" or (lab.startswith("양성") and y[i] == 1)]
            m_full, m_part, f_part, eff, mx, top_part, exp_part, corr_mf, corr_bd, wdist = [], [], [], [], [], [], [], [], [], []
            for i in sel:
                a_ = mean_att(att, i); n = len(a_)
                pm_ = ~full[i]
                m_part.append(float(a_[pm_].sum())); f_part.append(float(pm_.mean()))
                eff.append(float(np.exp(-(a_ * np.log(a_ + 1e-12)).sum()) / n)); mx.append(float(a_.max()))
                k_top = max(1, int(round(topf * n))); top = np.argsort(-a_)[:k_top]
                top_part.append(float(pm_[top].mean())); exp_part.append(float(pm_.mean()))
                if mfrac[i].std() > 0:
                    corr_mf.append(spearman(a_, mfrac[i]))
                corr_bd.append(spearman(a_, bdist[i])); wdist.append(float((a_ * bdist[i]).sum() / max(bdist[i].mean(), 1e-9)))
            stats_rows.append(dict(scheme=sc, group=lab, n_images=len(sel), attn_mass_on_partial_tiles=round(float(np.mean(m_part)), 4), tile_fraction_partial=round(float(np.mean(f_part)), 4),
                                   ratio_partial=(round(float(np.mean(m_part)) / float(np.mean(f_part)), 3) if np.mean(f_part) > 0 else float("nan")), effective_n_over_n=round(float(np.mean(eff)), 4), max_attention=round(float(np.mean(mx)), 4),
                                   top10_partial_share=round(float(np.mean(top_part)), 4), top10_expected=round(float(np.mean(exp_part)), 4), spearman_attn_vs_maskfrac=(round(float(np.nanmean(corr_mf)), 3) if corr_mf else float("nan")),
                                   spearman_attn_vs_boundary_dist=round(float(np.nanmean(corr_bd)), 3), attn_weighted_dist_over_uniform=round(float(np.mean(wdist)), 3)))
        # A3 안정성
        seed_c, ver_c = [], []
        for i in range(N):
            n = counts[i]
            sp_ = [spearman(att[i, 0, s1, :n], att[i, 0, s2, :n]) for s1 in range(S) for s2 in range(s1 + 1, S)]
            seed_c.append(float(np.nanmean(sp_)))
            a0 = mean_att(att, i, 0)
            vs = [spearman(a0, mean_att(att, i, k)) for k in range(1, K1)]
            ver_c.append(float(np.nanmean(vs)))
            img_rows.append(dict(scheme=sc, image_id=ids[i], label3=int(y3[i]), prob_pos=round(pred[sc][i][0], 4), n_tiles=int(n), seed_spearman=round(seed_c[-1], 3), version_spearman=round(ver_c[-1], 3),
                                 max_attention=round(float(a0.max()), 4), effective_n_over_n=round(float(np.exp(-(a0 * np.log(a0 + 1e-12)).sum()) / n), 4)))
        stab_rows.append(dict(scheme=sc, seed_to_seed_spearman_mean=round(float(np.mean(seed_c)), 3), seed_to_seed_min=round(float(np.min(seed_c)), 3),
                              orig_vs_augmented_spearman_mean=round(float(np.mean(ver_c)), 3), orig_vs_augmented_min=round(float(np.min(ver_c)), 3), n_images=N))
    write_csv(f"{log.dir}/attention_stats.csv", stats_rows)
    write_csv(f"{log.dir}/attention_stability.csv", stab_rows)
    write_csv(f"{log.dir}/attention_by_image.csv", img_rows)
    log("[A1/A2/A4] attention 통계 (시험 단위를 뺀 모델, 시드 5개 평균, 이미지 평균)")
    for r in stats_rows:
        log(f"  {r['scheme']:<4} {r['group']:<12} n={r['n_images']:>2} | 배경 섞인 타일 attention 질량 {r['attn_mass_on_partial_tiles']:.3f} vs 타일 비율 {r['tile_fraction_partial']:.3f} (배율 {r['ratio_partial']:.2f}) | "
            f"effective n / n {r['effective_n_over_n']:.3f}, 최대 가중치 {r['max_attention']:.3f} | 상위 10% 중 배경 섞인 타일 {r['top10_partial_share']:.3f} (기대 {r['top10_expected']:.3f}) | attention vs mask_frac Spearman {r['spearman_attn_vs_maskfrac']:+.3f} | attention vs 경계까지 거리 Spearman {r['spearman_attn_vs_boundary_dist']:+.3f}, attention 가중 평균 거리 / 균일 평균 거리 {r['attn_weighted_dist_over_uniform']:.2f}")
    log("\n[A3] 안정성 (Spearman 순위 상관, 타일 간)")
    for r in stab_rows:
        log(f"  {r['scheme']:<4} 시드끼리 {r['seed_to_seed_spearman_mean']:.3f} (최소 {r['seed_to_seed_min']:.3f}) | 원본 vs 배경 증강 {r['orig_vs_augmented_spearman_mean']:.3f} (최소 {r['orig_vs_augmented_min']:.3f})")

    # ---------- 대표 이미지 선택 (규칙은 config 에 고정) ----------
    P = {i: pred[main_s][i] for i in range(N)}
    reps, rows = [], []
    for lab in (0, 1, 2):
        idx = [i for i in range(N) if y3[i] == lab]
        ok = [i for i in idx if P[i][1] == y[i]]
        if ok:
            med = float(np.median([P[i][0] for i in idx]))
            i_a = min(ok, key=lambda i: abs(P[i][0] - med))
            reps.append(("typical", lab, i_a)); rows.append(dict(kind="typical", label3=lab, image_id=ids[i_a], prob_pos=round(P[i_a][0], 4), pred=P[i_a][1]))
        wrong = [i for i in idx if P[i][1] != y[i]]
        if wrong:
            i_b = max(wrong, key=lambda i: abs(P[i][0] - y[i]))
            reps.append(("error", lab, i_b)); rows.append(dict(kind="error", label3=lab, image_id=ids[i_b], prob_pos=round(P[i_b][0], 4), pred=P[i_b][1]))
    write_csv(f"{log.dir}/representative_images.csv", rows)
    log("\n대표 이미지 (규칙: 클래스마다 맞힌 이미지 중 양성 확률이 클래스 중앙값에 가장 가까운 1장 / 틀린 이미지 중 오차가 가장 큰 1장):")
    for r in rows:
        log(f"  {r['kind']:<8} AGG{r['label3']} {r['image_id']}  양성 확률 {r['prob_pos']:.3f} 예측 {'AGG1+2' if r['pred'] else 'AGG0'}")

    man = {r["image_id"]: r for r in csv.DictReader(open(cfg["paths"]["manifest_csv"], encoding="utf-8-sig"))}
    mdir = Path(next((p for p in cfg["paths"]["masks_dir"] if Path(p).exists())) if isinstance(cfg["paths"]["masks_dir"], list) else cfg["paths"]["masks_dir"])

    def read_rgb(path):
        arr = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)

    def overlay(i, vmax=4.0):
        im = read_rgb(Path(cfg["paths"]["images_root"]) / man[ids[i]]["path"])
        mk = cv2.imdecode(np.fromfile(str(mdir / (ids[i].replace("/", "__") + ".png")), dtype=np.uint8), cv2.IMREAD_GRAYSCALE) > 127
        a_ = mean_att(Z[f"att_{main_s}"], i); n = len(a_); rel = a_ * n
        vis = cv2.resize(im, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA).astype(np.float32)
        base = vis.copy(); cmap = plt.get_cmap("inferno"); ov = vis.copy()
        for j, r in enumerate(tby[i]):
            x0, y0 = int(int(r["x0"]) * 0.25), int(int(r["y0"]) * 0.25); s = int(224 * 0.25)
            col = np.array(cmap(min(rel[j] / vmax, 1.0))[:3]) * 255
            ov[y0:y0 + s, x0:x0 + s] = vis[y0:y0 + s, x0:x0 + s] * 0.35 + col * 0.65
        cn, _ = cv2.findContours(mk.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ovu = ov.astype(np.uint8); cv2.drawContours(ovu, [(c * 0.25).astype(np.int32) for c in cn], -1, (90, 150, 255), 1)
        for j, r in enumerate(tby[i]):
            x0, y0 = int(int(r["x0"]) * 0.25), int(int(r["y0"]) * 0.25); s = int(224 * 0.25)
            cv2.rectangle(ovu, (x0, y0), (x0 + s - 1, y0 + s - 1), (255, 255, 255), 1)
        coords = [(int(int(r["x0"]) * 0.25), int(int(r["y0"]) * 0.25)) for r in tby[i]]
        return base.astype(np.uint8), ovu, rel, coords

    def figure(items, fname, title):
        if not items:
            return
        fig, axes = plt.subplots(len(items), 2, figsize=(10.5, 3.7 * len(items)), squeeze=False)
        for row, (kind, lab, i) in enumerate(items):
            base, ov, rel, coords = overlay(i)
            axes[row, 0].imshow(base); axes[row, 0].axis("off")
            axes[row, 0].set_title(f"{ids[i].rsplit('/', 1)[0]}  (실제 AGG{lab}, 모델 양성 확률 {P[i][0]:.2f})", fontsize=8)
            im_ = axes[row, 1].imshow(ov); axes[row, 1].axis("off")
            for (cx, cy), v in zip(coords, rel):      # 격자 안에 균일 대비 배율(attention x 타일 수) 숫자 표기
                axes[row, 1].text(cx + 28, cy + 28, f"{v:.1f}", ha="center", va="center", fontsize=7, color="white", fontweight="bold",
                                  path_effects=[pe.withStroke(linewidth=1.8, foreground="black")])
            axes[row, 1].set_title(f"attention 오버레이 (균일 대비 배율; 최대 {rel.max():.1f}배, 흰 격자 = 타일, 숫자 = 타일의 배율, 파랑 = 마스크 윤곽)", fontsize=8)
        sm = plt.cm.ScalarMappable(cmap="inferno", norm=plt.Normalize(0, 4.0)); cb = fig.colorbar(sm, ax=axes[:, 1].tolist(), fraction=0.02, pad=0.03, aspect=35); cb.set_label("attention x 타일 수 (1 = 균일, 4 이상 같은 색)", fontsize=7)
        fig.suptitle(title, fontsize=10); fig.savefig(log.dir / fname, dpi=120, bbox_inches="tight"); plt.close(fig)
    figure([r for r in reps if r[0] == "typical"], "fig_attention_typical.png", f"MIL attention ({main_s} fold 모델, 시드 5개 평균): 클래스별 대표 이미지 (맞힌 것)")
    figure([r for r in reps if r[0] == "error"], "fig_attention_errors.png", f"MIL attention ({main_s} fold 모델): 클래스별 가장 크게 틀린 이미지")

    # 분포 그림: 클래스별 이미지의 attention 집중도
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, key, lab in ((axes[0], "effective_n_over_n", "effective number / 타일 수 (1 = 균일)"), (axes[1], "version_spearman", "원본 vs 배경 증강 attention 순위 상관")):
        data = [[r[key] for r in img_rows if r["scheme"] == main_s and r["label3"] == c] for c in (0, 1, 2)]
        ax.boxplot(data, tick_labels=["AGG0", "AGG1", "AGG2"]); ax.set_ylabel(lab, fontsize=8); ax.set_title(f"{main_s} fold 모델, 이미지별", fontsize=9)
    fig.tight_layout(); fig.savefig(log.dir / "fig_attention_dist.png", dpi=130); plt.close(fig)
    log(f"\n저장: {log.dir}")


if __name__ == "__main__":
    main()
