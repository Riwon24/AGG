"""m20: 직접 표시한 AGG 타일 17개(AGG1 이미지 14장)가 MIL 의 attention 과 기여도에서 이미지 안 상위로 나오는지 직접 검증한다 (m10 이 저장한 배열만 읽음, 새 학습 없음; numpy, matplotlib).

점수 소스 (모두 원본 bag(k=0), 시드 5개 평균, 그 이미지를 시험으로 뺀 fold 모델의 값; 기본 scheme 은 LOCO):
  (a) gated 모델의 attention a_k              (outputs/, config.yaml)
  (b) additive 모델의 attention a_k           (outputs_additive/, config_additive.yaml, 참고용)
  (c) additive 모델의 기여도 a_k z_k          (outputs_additive/, 핵심 검증 대상; 부호가 있고 클수록 AGG1+2(양성) 쪽으로 더 많이 기여)
표시한 타일: ../exp_aggsim/outputs/c08_agg1_manual_labels/agg_tile_labels_filled.csv 의 has_agg == 1 (17개, AGG1 이미지 14장). 경계 부근이라고 적은 3개도 이름을 적은 타일만 쓴다 (c09 와 같음).
방식: exp_aggsim/c09_manual_prototype_sim.py 의 rank_pct 와 같다. 이미지 안 순위 (1 = 그 이미지에서 점수가 가장 큼), percentile = 1 - (rank - 1) / (n - 1) (1.0 = 이미지 안 최상위, 무작위 평균 0.5),
  이미지 안 상위 25% 이내 개수 (rank / n <= 0.25, 무작위 기대 약 4.3 / 17). 마스크 비율 1.0 인 "안쪽" 타일과 가장자리 타일을 나눠서 따로 보고한다 (안쪽 타일끼리만 순위를 낸 값도 함께).
c09 의 aggsim 은 17개 타일 자체로 prototype 을 만들어 순환성 보정(loo/loio)이 필요했지만, MIL 은 이미지 단위 라벨로만 학습되고 이 타일 위치를 보지 않으므로 보정 없이 그대로 쓴다.
무작위 대조 (m19 와 같은 방식, 20,000회): 같은 이미지에서 표시한 타일 수만큼 (a) 아무 타일, (b) 같은 종류(안쪽/가장자리) 타일을 무작위로 골라 percentile 평균을 비교한다.
타일은 번호가 아니라 좌표 (x0, y0) 로 맞춘다 (outputs 는 stride 224 전환 전에 돌려 옛 타일 번호를 쓰고 outputs_additive 는 새 번호를 쓴다. 두 run 의 타일 좌표와 순서가 같음을 assert 로 확인하고 출력의 tile_id 는 새 번호(표시 파일과 같은 체계)다).
출력: outputs_additive/m20_validate_manual_agg/manual_tile_scores.csv (AGG1 이미지 14장의 모든 타일), summary.csv, summary.md, fig_manual_tiles_LOCO.png, log.txt
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
SOURCES = [("attn_gated", "gated attention"), ("attn_additive", "additive attention"), ("contrib_additive", "additive 기여도 (a_k z_k)")]
rd = lambda p: list(csv.DictReader(open(p, encoding="utf-8-sig")))
lines = []


def log(m=""):
    print(m); lines.append(str(m))


def rank_pct(s, mask=None):
    """이미지 안 순위(1 = 점수가 가장 큼)와 percentile = 1 - (rank - 1) / (n - 1). mask 가 있으면 그 타일끼리만 (그 밖은 nan). s: 한 이미지의 점수 (n,)"""
    rk, pc = np.full(len(s), np.nan), np.full(len(s), np.nan)
    idx = np.where(mask if mask is not None else np.ones(len(s), bool))[0]
    order = idx[np.argsort(-s[idx], kind="stable")]
    for r_, j in enumerate(order, 1):
        rk[j] = r_; pc[j] = 1 - (r_ - 1) / (len(idx) - 1) if len(idx) > 1 else 1.0
    return rk, pc


def load_run(run, scheme_list):
    d = HERE / run / "m10_train_eval_mil"
    Z = np.load(d / "attention_arrays.npz")
    tiles = rd(d / "attention_tiles.csv")
    by = {}
    for r in tiles:
        by.setdefault(r["image_id"], []).append(r)
    idx = {r["image_id"]: int(r["image_idx"]) for r in tiles}
    return Z, by, idx


def seed_mean(arr, i, n):
    return np.nanmean(arr[i, 0, :, :n], axis=0)                                  # 원본 버전(k=0), 시드 평균, 타일 n개


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gated", default="outputs"); ap.add_argument("--additive", default="outputs_additive")
    a = ap.parse_args()
    out = HERE / a.additive / "m20_validate_manual_agg"; out.mkdir(parents=True, exist_ok=True)
    Zg, byg, idxg = load_run(a.gated, None)
    Za, bya, idxa = load_run(a.additive, None)
    assert "contrib_LOCO" in Za.files, "additive 결과에 기여도 배열이 없음 (config_additive.yaml 로 m10 을 먼저 실행)"
    marked = {(r["image_id"], (int(r["x0"]), int(r["y0"]))): r["tile_id"] for r in rd(LAB) if r["has_agg"] == "1"}
    agg1 = sorted({k[0] for k in marked})
    assert len(marked) == 17 and len(agg1) == 14
    # 두 run 의 타일 순서가 같은지 확인 (같은 타일 집합과 순서여야 같은 위치끼리 비교된다)
    for iid in agg1:
        cg = [(int(r["x0"]), int(r["y0"])) for r in byg[iid]]; ca = [(int(r["x0"]), int(r["y0"])) for r in bya[iid]]
        assert cg == ca, f"두 run 의 타일 순서가 다름: {iid}"
    log(f"gated: {a.gated} | additive: {a.additive} | AGG1 이미지 {len(agg1)}장, 표시한 AGG 타일 {len(marked)}개 (안쪽 {sum(1 for (i, c) in marked if float([r for r in byg[i] if (int(r['x0']), int(r['y0'])) == c][0]['mask_frac']) >= 0.999999)}개)")
    for (i, c) in marked:
        assert c in [(int(r["x0"]), int(r["y0"])) for r in byg[i]], f"표시 타일이 bag 에 없음: {i} {c}"
    schemes = [s for s in ("LOCO", "IMG", "LOMO") if f"contrib_{s}" in Za.files]

    all_rows, sum_rows = [], []
    md = ["# 직접 표시한 AGG 타일 17개의 이미지 안 순위 (attention / additive attention / additive 기여도)\n",
          f"AGG1 이미지 14장, 표시한 타일 17개 (안쪽 9, 가장자리 8). 이미지 안 percentile: 1.0 = 그 이미지에서 점수가 가장 큼, 무작위 평균 0.5. 상위 25% 이내 개수의 무작위 기대는 약 4.3 / 17. 시드 5개 평균, 그 이미지를 시험으로 뺀 fold 모델.\n"]
    for sc in schemes:
        score = {}                                                                # (iid) -> {source: (n,)}
        for iid in agg1:
            n = len(byg[iid])
            score[iid] = {"attn_gated": seed_mean(Zg[f"att_{sc}"], idxg[iid], n), "attn_additive": seed_mean(Za[f"att_{sc}"], idxa[iid], n), "contrib_additive": seed_mean(Za[f"contrib_{sc}"], idxa[iid], n)}
        res = {}
        for src, _ in SOURCES:
            for iid in agg1:
                s = score[iid][src]; n = len(s)
                coords = [(int(r["x0"]), int(r["y0"])) for r in byg[iid]]; mf = np.array([float(r["mask_frac"]) for r in byg[iid]]); inner = mf >= 0.999999
                rk, pc = rank_pct(s); rki, pci = rank_pct(s, inner)
                res.setdefault(src, {})[iid] = dict(rk=rk, pc=pc, rki=rki, pci=pci, inner=inner, coords=coords, n=n)
        # --- 표시 타일의 지표 ---
        tbl = {}
        rng = np.random.RandomState(0)
        for src, nm in SOURCES:
            R_ = res[src]
            items = [(iid, c) for (iid, c) in marked]
            pc = np.array([R_[iid]["pc"][R_[iid]["coords"].index(c)] for iid, c in items])
            rk = np.array([R_[iid]["rk"][R_[iid]["coords"].index(c)] for iid, c in items]); nn = np.array([R_[iid]["n"] for iid, c in items])
            inner = np.array([R_[iid]["inner"][R_[iid]["coords"].index(c)] for iid, c in items])
            pci = np.array([R_[iid]["pci"][R_[iid]["coords"].index(c)] if R_[iid]["inner"][R_[iid]["coords"].index(c)] else np.nan for iid, c in items])
            top25 = rk / nn <= 0.25
            groups = {}
            for k, (iid, c) in enumerate(items):
                groups.setdefault(iid, []).append(k)
            pool_all = {iid: R_[iid]["pc"] for iid in groups}
            pool_cat = {iid: {cat: R_[iid]["pc"][R_[iid]["inner"] == cat] for cat in (True, False)} for iid in groups}
            sa, sb = np.zeros(N_PERM), np.zeros(N_PERM)
            for t in range(N_PERM):
                va, vb = [], []
                for iid, ks in groups.items():
                    va += list(rng.choice(pool_all[iid], len(ks), replace=False))
                    for k in ks:
                        vb.append(rng.choice(pool_cat[iid][bool(inner[k])]))
                sa[t], sb[t] = np.mean(va), np.mean(vb)
            obs = float(pc.mean())
            tbl[src] = dict(mean=obs, median=float(np.median(pc)), top25=int(top25.sum()), inner_mean=float(pc[inner].mean()), edge_mean=float(pc[~inner].mean()), inner_pci_mean=float(np.nanmean(pci)),
                            inner_top25=int(top25[inner].sum()), n_inner=int(inner.sum()), p_any=float((sa >= obs).mean()), p_cat=float((sb >= obs).mean()), exp_cat=float(sb.mean()),
                            hit1=sum(1 for iid, ks in groups.items() if any(rk[k] == 1 for k in ks)), hit3=sum(1 for iid, ks in groups.items() if any(rk[k] <= 3 for k in ks)))
            sum_rows.append(dict(scheme=sc, source=src, n_tiles=len(items), **{k: round(v, 4) if isinstance(v, float) else v for k, v in tbl[src].items()}))
        md.append(f"\n## scheme = {sc}\n")
        md.append("| 점수 소스 | percentile 평균 (중앙값) | 상위 25% 이내 | 안쪽 표시 타일 (9개) percentile / 상위 25% | 가장자리 표시 타일 (8개) percentile | p (위치 무시) / p (위치 통제) | 이미지 1위 / 3위 안에 든 이미지 (14장 중) |")
        md.append("|---|---|---|---|---|---|---|")
        for src, nm in SOURCES:
            t = tbl[src]
            md.append(f"| {nm} | {t['mean']:.3f} ({t['median']:.3f}) | {t['top25']} / 17 | {t['inner_mean']:.3f} / {t['inner_top25']} ({t['inner_pci_mean']:.3f} 안쪽 타일끼리) | {t['edge_mean']:.3f} | {t['p_any']:.3f} / {t['p_cat']:.3f} | {t['hit1']} / {t['hit3']} |")
        log(f"\n[{sc}] 표시 타일 17개")
        for src, nm in SOURCES:
            t = tbl[src]
            log(f"  {nm:<24} percentile 평균 {t['mean']:.3f} (중앙값 {t['median']:.3f}), 상위 25% 이내 {t['top25']}/17 | 안쪽 {t['n_inner']}개 {t['inner_mean']:.3f} (안쪽끼리 {t['inner_pci_mean']:.3f}), 상위 25% {t['inner_top25']} | 가장자리 {t['edge_mean']:.3f} | p {t['p_any']:.3f}/{t['p_cat']:.3f} (위치 통제 기대 {t['exp_cat']:.3f}) | 1위/3위 {t['hit1']}/{t['hit3']}")
        for iid in agg1:
            for j, r in enumerate(byg[iid]):
                c = (int(r["x0"]), int(r["y0"]))
                row = dict(scheme=sc, image_id=iid, tile_id=bya[iid][j]["tile_id"], x0=c[0], y0=c[1], has_agg=int((iid, c) in marked), mask_frac=r["mask_frac"], n_tiles=len(byg[iid]))
                for src, _ in SOURCES:
                    R_ = res[src][iid]
                    row[src] = round(float(score[iid][src][j]), 6)
                    row[f"rank_{src}"] = int(R_["rk"][j]); row[f"percentile_{src}"] = round(float(R_["pc"][j]), 4)
                all_rows.append(row)
        if sc == "LOCO":
            fig, axes = plt.subplots(3, 14, figsize=(40, 8.5))
            for k, iid in enumerate(agg1):
                for r_, (src, nm) in enumerate(SOURCES):
                    ax = axes[r_, k]; ax.set_xlim(-0.05, 8.05); ax.set_ylim(6.05, -0.05); ax.set_aspect("equal"); ax.axis("off")
                    R_ = res[src][iid]; s = score[iid][src]
                    if src == "contrib_additive":
                        vmax = float(np.abs(s).max()) + 1e-9; cm_, norm = plt.cm.bwr, (lambda v, vm=vmax: 0.5 + 0.5 * v / vm)
                    else:
                        vmax = float(s.max()); cm_, norm = plt.cm.inferno, (lambda v, vm=vmax: v / vm)
                    for j, c in enumerate(R_["coords"]):
                        ax.add_patch(Rectangle((c[0] // 224, c[1] // 224), 1, 1, facecolor=cm_(norm(s[j])), edgecolor="white", linewidth=0.5))
                        if (iid, c) in marked:
                            ax.add_patch(Rectangle((c[0] // 224 + 0.05, c[1] // 224 + 0.05), 0.9, 0.9, facecolor="none", edgecolor="#00e5ff", linewidth=2.5))
                    if r_ == 0:
                        ax.set_title(f"{iid.split('/')[0]} {iid.split('/')[1]}", fontsize=9)
                    if k == 0:
                        ax.text(-0.4, 3, nm, rotation=90, va="center", ha="right", fontsize=10)
            fig.suptitle("직접 표시한 AGG 타일 (하늘색 테두리) 과 점수 (AGG1 이미지 14장, LOCO fold 모델, 시드 5개 평균; 위 두 줄 inferno = 이미지 안 최대 대비, 아래 bwr = 기여도, 빨강 = 양성 쪽, 파랑 = 음성 쪽)", fontsize=13)
            fig.savefig(out / "fig_manual_tiles_LOCO.png", dpi=70, bbox_inches="tight"); plt.close(fig)
    with open(out / "manual_tile_scores.csv", "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_rows[0])); w.writeheader(); w.writerows(all_rows)
    with open(out / "summary.csv", "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(sum_rows[0])); w.writeheader(); w.writerows(sum_rows)
    (out / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (out / "log.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"\n저장: {out}")


if __name__ == "__main__":
    main()
