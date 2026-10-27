"""m19: MIL attention 이 직접 표시한 AGG 타일(AGG1 이미지 14장의 17개)과 얼마나 겹치는지 본다 (m10 이 저장한 attention 만 읽음, 새 학습 없음; numpy, matplotlib).

attention: m10 의 attention_arrays.npz (scheme 별 [이미지, 버전, 시드, 타일]) 에서 원본 버전(k=0)의 시드 5개 평균을 쓴다. 기본 scheme 은 LOCO (그 이미지의 처리 조건을 통째로 뺀 fold 모델), IMG, LOMO 도 같이 본다.
표시한 타일: ../exp_aggsim/outputs/c08_agg1_manual_labels/agg_tile_labels_filled.csv 의 has_agg == 1 (17개, AGG1 이미지 14장). 경계 부근이라고 적은 3개는 이웃 타일(neighbor_tile)도 함께 본다.
지표 (이미지 안에서 비교, 이미지마다 타일 수가 다름):
  - 배율 = attention x 타일 수 (1 = 균일)
  - percentile = 1 - (rank - 1) / (n - 1), rank 1 = 그 이미지에서 attention 이 가장 큰 타일 (1.0 = 최상위, 무작위 기대 0.5)
  - 상위 25% 이내 개수 (rank / n <= 0.25)
  - 이미지의 attention 1위, 3위 안에 표시한 타일이 든 이미지 수
무작위 대조 (순열): 같은 이미지에서 표시한 타일 수만큼 타일을 무작위로 골라 같은 지표를 20,000번 계산한다. 두 가지:
  (a) 이미지 안 아무 타일, (b) 위치 통제: 표시한 타일과 같은 종류(안쪽 = 마스크 비율 100%, 가장자리 = 그 외)의 타일에서만 고른다. attention 이 가장자리 여부와 강하게 얽혀 있어서(MIL README) 위치를 통제한 대조가 더 공정하다.
설정(run): outputs (모든 분석 타일), outputs_mf02 (마스크 비율 >= 0.2), outputs_full (마스크 비율 100% 타일만). bag 에 없는 표시 타일은 그 run 에서 제외한다.
타일은 번호가 아니라 좌표 (x0, y0) 로 맞춘다 (outputs, outputs_full 은 stride 224 전환 전에 돌려 옛 타일 번호를 쓰고, outputs_mf02 는 새 번호를 쓴다. 타일의 위치와 특징은 같다).
출력: <outputs>/m19_attention_vs_manual/overlap_tiles.csv (표시한 타일별), overlap_summary.md, fig_attention_vs_manual_LOCO.png (AGG1 이미지 14장의 타일 격자, 표시한 타일은 빨간 테두리)
"""
import argparse
import csv
import os
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager as _fm  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

for _f in ("C:/Windows/Fonts/malgun.ttf", "/mnt/c/Windows/Fonts/malgun.ttf"):
    if os.path.exists(_f):
        _fm.fontManager.addfont(_f)
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).resolve().parent
LAB = HERE.parent / "exp_aggsim/outputs/c08_agg1_manual_labels/agg_tile_labels_filled.csv"
N_PERM = 20000


def pct(rank, n):
    return 1 - (rank - 1) / (n - 1) if n > 1 else 1.0


def analyse(run, lines):
    out_dir = HERE / run
    Z = np.load(out_dir / "m10_train_eval_mil/attention_arrays.npz")
    counts = Z["tile_counts"]
    tiles = list(csv.DictReader(open(out_dir / "m10_train_eval_mil/attention_tiles.csv", encoding="utf-8-sig")))
    by = {}
    for r in tiles:                                                                    # 키 = (x0, y0): 옛 번호(stride 112)와 새 번호(stride 224)가 run 마다 달라서 좌표로 맞춘다
        by.setdefault(r["image_id"], {})[(int(r["x0"]), int(r["y0"]))] = r
    idx = {r["image_id"]: int(r["image_idx"]) for r in tiles}
    for iid, d in by.items():
        assert len(d) == counts[idx[iid]], "타일 수가 attention 배열과 다름"
    def nb_key(s):
        return (int(s[5:7]) * 224, int(s[1:3]) * 224) if s else None                   # 새 번호 rXX_cYY -> (x0, y0)
    marked = [(r["image_id"], (int(r["x0"]), int(r["y0"])), nb_key(r["neighbor_tile"]), r["tile_id"]) for r in csv.DictReader(open(LAB, encoding="utf-8-sig")) if r["has_agg"] == "1"]
    pred = {}
    for r in csv.DictReader(open(out_dir / "m10_train_eval_mil/mil_predictions.csv", encoding="utf-8-sig")):
        pred.setdefault(r["scheme"], {}).setdefault(r["image_id"], []).append(float(r["prob"]))
    lines.append(f"\n## run: {run}\n")
    lines.append(f"bag 의 타일 {len(tiles):,}개, 이미지 {len(by)}장. 표시한 타일 {len(marked)}개 중 이 run 의 bag 에 있는 것 {sum(1 for m in marked if m[1] in by.get(m[0], {}))}개.\n")
    rows, main_rows = [], {}
    for sc in [s for s in ("LOCO", "IMG", "LOMO") if f"att_{s}" in Z.files]:
        att = Z[f"att_{sc}"]
        rel, rnk, ntile = {}, {}, {}
        for iid, d in by.items():
            i = idx[iid]; a = np.nanmean(att[i, 0, :, :counts[i]], axis=0)
            order = {k_: r_ for r_, (k_, _) in enumerate(sorted(((k2, float(a[int(rr["j"])])) for k2, rr in d.items()), key=lambda x: -x[1]), 1)}
            for tid, rr in d.items():
                rel[(iid, tid)] = float(a[int(rr["j"])]) * len(d); rnk[(iid, tid)] = order[tid]
            ntile[iid] = len(d)
        present = [(iid, tid, nb, tname) for iid, tid, nb, tname in marked if (iid, tid) in rel]
        pc = np.array([pct(rnk[(iid, tid)], ntile[iid]) for iid, tid, _, _ in present])
        top25 = np.array([rnk[(iid, tid)] / ntile[iid] <= 0.25 for iid, tid, _, _ in present])
        inner = np.array([float(by[iid][tid]["mask_frac"]) >= 0.999999 for iid, tid, _, _ in present])
        # 순열 대조
        rng = np.random.RandomState(0)
        img_groups = {}
        for k, (iid, tid, _, _) in enumerate(present):
            img_groups.setdefault(iid, []).append(k)
        sim_a, sim_b, top_a, top_b = np.zeros(N_PERM), np.zeros(N_PERM), np.zeros(N_PERM), np.zeros(N_PERM)
        pool_all, pool_cat = {}, {}
        for iid, ks in img_groups.items():
            allt = list(by[iid]); pool_all[iid] = np.array([pct(rnk[(iid, t)], ntile[iid]) for t in allt]); pool_cat[iid] = {}
            for cat in (True, False):
                sel = [t for t in allt if (float(by[iid][t]["mask_frac"]) >= 0.999999) == cat]
                pool_cat[iid][cat] = np.array([pct(rnk[(iid, t)], ntile[iid]) for t in sel])
        for s in range(N_PERM):
            va, vb = [], []
            for iid, ks in img_groups.items():
                va += list(rng.choice(pool_all[iid], len(ks), replace=False))
                for k in ks:
                    vb.append(rng.choice(pool_cat[iid][bool(inner[k])]))
            va, vb = np.array(va), np.array(vb)
            sim_a[s], sim_b[s] = va.mean(), vb.mean()
            top_a[s], top_b[s] = (va >= 0.75).sum(), (vb >= 0.75).sum()
        obs_mean = pc.mean(); obs_top = int(top25.sum())
        p_a = float((sim_a >= obs_mean).mean()); p_b = float((sim_b >= obs_mean).mean())
        # 이미지 attention 1위, 3위 안에 표시한 타일
        hit1 = sum(1 for iid, ks in img_groups.items() if any(rnk[(present[k][0], present[k][1])] == 1 for k in ks))
        hit3 = sum(1 for iid, ks in img_groups.items() if any(rnk[(present[k][0], present[k][1])] <= 3 for k in ks))
        # 경계 표기 타일은 이웃 타일까지
        near3 = sum(1 for iid, ks in img_groups.items() if any(rnk[(iid, t)] <= 3 for k in ks for t in [present[k][1]] + ([present[k][2]] if present[k][2] and present[k][2] in by[iid] else [])))
        # attention 질량
        mass = sum(rel[(iid, tid)] / ntile[iid] for iid, tid, _, _ in present) / len(present)
        uni = 1 / np.mean([ntile[iid] for iid, _, _, _ in present])
        res = dict(scheme=sc, n=len(present), mean_pct=obs_mean, median_pct=float(np.median(pc)), top25=obs_top, exp_top=float(top_a.mean()), p_any=p_a, p_cat=p_b, exp_pct_any=0.5, exp_pct_cat=float(sim_b.mean()),
                   exp_top_cat=float(top_b.mean()), inner_mean=float(pc[inner].mean()) if inner.any() else float("nan"), edge_mean=float(pc[~inner].mean()) if (~inner).any() else float("nan"),
                   n_inner=int(inner.sum()), n_images=len(img_groups), hit1=hit1, hit3=hit3, near3=near3, mass=float(mass), uniform=float(uni))
        main_rows[sc] = res
        for k, (iid, tid, nb, tname) in enumerate(present):
            rows.append(dict(run=run, scheme=sc, image_id=iid, condition=iid.split("/")[1], tile_id=tname, x0=tid[0], y0=tid[1], neighbor_tile=("" if nb is None else f"r{nb[1] // 224:02d}_c{nb[0] // 224:02d}"), mask_frac=by[iid][tid]["mask_frac"], n_tiles=ntile[iid],
                             attention_ratio=round(rel[(iid, tid)], 3), rank=rnk[(iid, tid)], percentile=round(pc[k], 4), top25=int(top25[k]),
                             image_pred_prob_mean=round(float(np.mean(pred[sc][iid])), 3) if sc in pred and iid in pred[sc] else ""))
    out = out_dir / "m19_attention_vs_manual"; out.mkdir(exist_ok=True)
    with open(out / "overlap_tiles.csv", "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    lines.append("| scheme | 표시 타일 / 이미지 | percentile 평균 (중앙값) | 위치 통제 무작위 기대 | p (위치 무시) / p (위치 통제) | 상위 25% 이내 | 무작위 기대 (위치 통제) | attention 1위 / 3위 안에 표시 타일이 든 이미지 | 표시 타일이 받은 attention 질량 (균일 기대) |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for sc, r in main_rows.items():
        lines.append(f"| {sc} | {r['n']} / {r['n_images']} | {r['mean_pct']:.3f} ({r['median_pct']:.3f}) | {r['exp_pct_cat']:.3f} | {r['p_any']:.3f} / {r['p_cat']:.3f} | {r['top25']} / {r['n']} | {r['exp_top_cat']:.1f} | {r['hit1']} / {r['hit3']} / {r['n_images']} | {r['mass']:.3f} ({r['uniform']:.3f}) |")
    lines.append("\n안쪽(마스크 100%)/가장자리 타일별 percentile 평균 (표시 타일 중 안쪽 {n}개): ".format(n=main_rows[next(iter(main_rows))]["n_inner"]) + ", ".join(f"{sc} 안쪽 {r['inner_mean']:.3f} / 가장자리 {r['edge_mean']:.3f}" for sc, r in main_rows.items()))
    return rows, main_rows, by, idx, Z, counts


def figure(run, by, idx, Z, counts, rows):
    sc = "LOCO"
    agg1 = sorted({r["image_id"] for r in rows if r["scheme"] == sc})
    mk = {(r["image_id"], (int(r["x0"]), int(r["y0"]))) for r in rows if r["scheme"] == sc}
    fig, axes = plt.subplots(4, 4, figsize=(17, 14)); axes = axes.ravel()
    for ax in axes:
        ax.axis("off")
    for k, iid in enumerate(agg1):
        d = by[iid]; i = idx[iid]; a = np.nanmean(Z[f"att_{sc}"][i, 0, :, :counts[i]], axis=0) * len(d)
        ax = axes[k]; ax.set_xlim(-0.05, 8.05); ax.set_ylim(6.05, -0.05); ax.set_aspect("equal")
        for tid, rr in d.items():
            r_, c_ = tid[1] // 224, tid[0] // 224; v = float(a[int(rr["j"])])
            ax.add_patch(Rectangle((c_, r_), 1, 1, facecolor=plt.cm.inferno(min(v / 4.0, 1.0)), edgecolor="white", linewidth=0.8))
            ax.text(c_ + 0.5, r_ + 0.5, f"{v:.1f}", ha="center", va="center", fontsize=7, color="white" if v < 2.5 else "black")
            if (iid, tid) in mk:
                ax.add_patch(Rectangle((c_ + 0.04, r_ + 0.04), 0.92, 0.92, facecolor="none", edgecolor="#00e5ff", linewidth=3))
        ax.set_title(f"{iid.split('/')[0]} {iid.split('/')[1]}", fontsize=9)
    fig.suptitle(f"MIL attention 배율(x 타일 수) vs 직접 표시한 AGG 타일 (하늘색 테두리) - AGG1 이미지 14장, LOCO fold 모델, 시드 5개 평균 [{run}]", fontsize=12)
    fig.tight_layout(); fig.savefig(HERE / run / "m19_attention_vs_manual" / "fig_attention_vs_manual_LOCO.png", dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--runs", nargs="*", default=["outputs", "outputs_mf02", "outputs_full"])
    a = ap.parse_args()
    for run in a.runs:
        lines = [f"# MIL attention 과 직접 표시한 AGG 타일의 겹침 ({run})\n"]
        rows, main_rows, by, idx, Z, counts = analyse(run, lines)
        figure(run, by, idx, Z, counts, rows)
        (HERE / run / "m19_attention_vs_manual" / "overlap_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n".join(lines))


if __name__ == "__main__":
    main()
