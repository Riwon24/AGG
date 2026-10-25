"""c05: 안쪽 타일(마스크 100%, 시편만 있는 타일) 기준으로 aggsim 의 클래스별 분포를 비교한다.

입력 (읽기 전용): c03 의 tile_aggsim.csv, g00_tile_list/tiles.csv (mask_frac). 안쪽 타일 = mask_frac >= 0.999999 (결과를 보기 전에 정한 정의, 이전 MIL 의 full 타일과 같음), 1,309개.
test 타일의 aggsim 만 쓴다 (train 타일은 prototype 계산에 들어간 것이라 제외). 10 반복 x 5 fold 에서 각 타일은 반복마다 한 번씩 test 이므로 타일별로 10 반복의 test aggsim 을 평균한다.
  타일 단위: 안쪽 타일 1,309개 (이미지당 여러 개라 서로 독립이 아니다)
  이미지 단위: 이미지의 안쪽 타일 평균 aggsim, 60장 (독립 단위). 클래스 사이 분리도는 이미지 단위 AUROC 와 이미지 부트스트랩 95% CI 로 본다.
참고: 가장자리 타일(mask_frac < 1)의 같은 통계.
출력: outputs/c05_aggsim_inner_distribution/fig_inner_distribution.png, stats_by_class.csv, auroc_pairs.csv, image_means.csv, log.txt
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
OUT = HERE / "outputs"                                  # 이 폴더의 결과 (c03 -> c04, c05, c06 -> c08 -> c09 가 서로의 출력을 읽는다)
GRID = next((p for p in (HERE.parent / "exp_grid_aug" / "outputs", HERE.parent / "DINOv2_grid_augmentation" / "outputs") if p.exists()), HERE.parent / "DINOv2_grid_augmentation" / "outputs")
#                                                       # 입력: DINOv2_grid_augmentation 의 출력 (읽기 전용; 로컬 작업 폴더 구조 또는 저장소 구조 중 있는 쪽)
D = OUT / "c05_aggsim_inner_distribution"
D.mkdir(parents=True, exist_ok=True)
lines = []


def log(m):
    print(m); lines.append(m)


rd = lambda p: list(csv.DictReader(open(p, encoding="utf-8-sig")))
mf = {(r["image_id"], r["tile_id"]): float(r["mask_frac"]) for r in rd(GRID / "g00_tile_list/tiles.csv")}
acc, lab = {}, {}
for r in rd(OUT / "c03_tile_aggsim/tile_aggsim.csv"):
    if r["split"] != "test":
        continue
    k = (r["image_id"], r["tile_id"])
    acc.setdefault(k, []).append(float(r["aggsim"])); lab[k] = int(r["label"])
assert all(len(v) == 10 for v in acc.values()), "각 타일은 반복마다 한 번씩 test 여야 함"
keys = sorted(acc)
tile = [(k, lab[k], float(np.mean(acc[k])), mf[k]) for k in keys]
inner = [t for t in tile if t[3] >= 0.999999]; edge = [t for t in tile if t[3] < 0.999999]
log(f"test aggsim (타일별 10 반복 평균): 타일 {len(tile)}개 = 안쪽(mask_frac 100%) {len(inner)}개 + 가장자리 {len(edge)}개")


def auroc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


def by_class(rows):
    return {c: np.array([t[2] for t in rows if t[1] == c]) for c in (0, 1, 2)}


def img_means(rows):
    d = {}
    for t in rows:
        d.setdefault((t[0][0], t[1]), []).append(t[2])
    return {k: float(np.mean(v)) for k, v in d.items()}


stat_rows = []
for name, rows in (("안쪽", inner), ("가장자리", edge)):
    tc = by_class(rows); im = img_means(rows)
    for c in (0, 1, 2):
        v = tc[c]; iv = np.array([m for (i, l), m in im.items() if l == c])
        stat_rows.append(dict(tiles=name, label=f"AGG{c}", n_tiles=len(v), tile_mean=round(float(v.mean()), 4), tile_sd=round(float(v.std()), 4), tile_median=round(float(np.median(v)), 4),
                              tile_q25=round(float(np.percentile(v, 25)), 4), tile_q75=round(float(np.percentile(v, 75)), 4), n_images=len(iv), image_mean=round(float(iv.mean()), 4), image_sd=round(float(iv.std()), 4),
                              image_min=round(float(iv.min()), 4), image_max=round(float(iv.max()), 4)))
        log(f"  [{name}] AGG{c}: 타일 {len(v):4d}개 평균 {v.mean():.3f} (sd {v.std():.3f}, 중앙 {np.median(v):.3f}) | 이미지 {len(iv)}장 평균 {iv.mean():.3f} (sd {iv.std():.3f}, 범위 {iv.min():.3f}~{iv.max():.3f})")
with open(D / "stats_by_class.csv", "w", encoding="utf-8-sig", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(stat_rows[0])); w.writeheader(); w.writerows(stat_rows)

im_in = img_means(inner)
imgs = sorted(im_in)
ival = np.array([im_in[k] for k in imgs]); ilab = np.array([k[1] for k in imgs])
with open(D / "image_means.csv", "w", encoding="utf-8-sig", newline="") as fh:
    w = csv.writer(fh); w.writerow(["image_id", "label", "inner_aggsim_mean"]); [w.writerow([k[0], k[1], round(m, 4)]) for k, m in zip(imgs, ival)]
rng = np.random.RandomState(0)
pairs = [("AGG1+2 vs AGG0", lambda y: y >= 1, lambda y: y == 0), ("AGG1 vs AGG0", lambda y: y == 1, lambda y: y == 0), ("AGG2 vs AGG0", lambda y: y == 2, lambda y: y == 0), ("AGG2 vs AGG1", lambda y: y == 2, lambda y: y == 1)]
log("\n이미지 단위 AUROC (안쪽 타일 평균 aggsim, 첫 클래스가 양성; 0.5 = 구분 못 함), 이미지 부트스트랩 95% CI 2000회:")
arows = []
for nm, fp, fn in pairs:
    a = auroc(ival[fp(ilab)], ival[fn(ilab)])
    bs = []
    for _ in range(2000):
        ix = rng.randint(0, len(ival), len(ival)); y_, v_ = ilab[ix], ival[ix]
        if fp(y_).any() and fn(y_).any():
            bs.append(auroc(v_[fp(y_)], v_[fn(y_)]))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    ta = auroc([t[2] for t in inner if fp(np.array(t[1]))], [t[2] for t in inner if fn(np.array(t[1]))])
    arows.append(dict(pair=nm, image_auroc=round(a, 3), image_ci_lo=round(float(lo), 3), image_ci_hi=round(float(hi), 3), tile_auroc=round(ta, 3)))
    log(f"  {nm:<16} 이미지 AUROC {a:.3f} [{lo:.3f}, {hi:.3f}] | (참고) 타일 단위 AUROC {ta:.3f}, 타일은 이미지 안에서 묶여 있어 CI 를 주지 않음")
with open(D / "auroc_pairs.csv", "w", encoding="utf-8-sig", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(arows[0])); w.writeheader(); w.writerows(arows)

fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
cols = ["#4C78A8", "#F58518", "#54A24B"]
tc = by_class(inner)
vp = ax[0].violinplot([tc[c] for c in (0, 1, 2)], showmedians=True, showextrema=False)
for b, c in zip(vp["bodies"], cols):
    b.set_facecolor(c); b.set_alpha(0.6)
ax[0].set_xticks([1, 2, 3]); ax[0].set_xticklabels([f"AGG{c}\n(타일 {len(tc[c])}개)" for c in (0, 1, 2)])
ax[0].set_ylabel("aggsim (test 타일, 10 반복 평균)"); ax[0].set_title("타일 단위 분포 (안쪽 타일)", fontsize=10)
for c in (0, 1, 2):
    ax[0].text(c + 1, ax[0].get_ylim()[0] + 0.005, f"평균 {tc[c].mean():.3f}", ha="center", fontsize=8)
for c in (0, 1, 2):
    v = ival[ilab == c]
    ax[1].boxplot(v, positions=[c + 1], widths=0.5, showfliers=False, medianprops=dict(color="black"))
    ax[1].scatter(np.full(len(v), c + 1) + rng.uniform(-0.12, 0.12, len(v)), v, s=22, color=cols[c], alpha=0.85, zorder=3)
ax[1].set_xticks([1, 2, 3]); ax[1].set_xticklabels([f"AGG{c}\n(이미지 {(ilab == c).sum()}장)" for c in (0, 1, 2)])
ax[1].set_ylabel("이미지별 안쪽 타일 평균 aggsim"); ax[1].set_title("이미지 단위 분포 (점 하나 = 이미지 1장)", fontsize=10)
a0 = arows[0]
ax[1].text(0.02, 0.02, f"이미지 단위 AUROC (AGG1+2 vs AGG0) {a0['image_auroc']:.2f} [{a0['image_ci_lo']:.2f}, {a0['image_ci_hi']:.2f}]", transform=ax[1].transAxes, fontsize=8)
fig.suptitle("안쪽(마스크 100%) 타일의 aggsim 클래스별 분포 | test 타일만, prototype 은 각 fold 의 train 타일로 계산", fontsize=11)
fig.tight_layout(); fig.savefig(D / "fig_inner_distribution.png", dpi=130); plt.close(fig)
(D / "log.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
log(f"\n저장: {D}")
