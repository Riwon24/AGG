"""c09: 수동 라벨 AGG 타일 17개로 만든 prototype 과 모든 타일의 코사인 유사도.

입력 (읽기 전용): c08 agg_tile_labels_filled.csv (has_agg == 1 인 17개), e01 features.npz / tile_index.csv, g00_tile_list/tiles.csv (mask_frac).
방식: 원본(k=0) 타일 cls = tile_cls[tile_row_of[i, 0]] (2591, 768). AGG prototype = 17개 AGG 타일 cls 의 평균. 모든 2,591개 타일에 cos(tile_cls, prototype) = aggsim_manual.
이미지 안 순위는 같은 이미지 타일끼리 aggsim_manual 이 높은 순 (1위 = 가장 비슷), percentile = 1 - (rank - 1) / (n - 1) (1.0 = 이미지 안 최상위, 0.0 = 최하위; 무작위면 평균 0.5). 이전 aggsim(c03) 결과의 0.60 과 같은 정의.
주의 (순환성): 17개 AGG 타일은 prototype 자체를 만든 타일이라 자기 자신과 닮은 정도가 부풀려진다 (한 타일이 prototype 의 1/17). 그래서 부풀림을 제거한 두 변형을 같이 계산한다:
  loo   : 해당 AGG 타일 자신을 뺀 16개로 만든 prototype (AGG 타일에만 적용, 나머지 타일은 17개 prototype)
  loio  : 해당 타일과 같은 이미지에 속한 AGG 타일을 모두 뺀 prototype (이미지 단위 leave-one-out; 같은 이미지의 타일끼리 닮은 정도가 섞이는 것을 막는다)
검증 출력: 17개 AGG 타일의 이미지 안 percentile 평균/중앙값, 안쪽 타일(mask_frac == 1.0)만, 이미지 안 상위 25% (rank / n <= 0.25) 개수. 세 방식(전체 prototype, loo, loio) 모두.
출력: outputs/c09_manual_prototype_sim/tile_sim_manual.csv (image_id, tile_id, label, aggsim_manual, rank, percentile, has_agg + 추가 열), log.txt
"""
import csv
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"                                  # 이 폴더의 결과 (c03 -> c04, c05, c06 -> c08 -> c09 가 서로의 출력을 읽는다)
GRID = next((p for p in (HERE.parent / "exp_grid_aug" / "outputs", HERE.parent / "DINOv2_grid_augmentation" / "outputs") if p.exists()), HERE.parent / "DINOv2_grid_augmentation" / "outputs")
#                                                       # 입력: DINOv2_grid_augmentation 의 출력 (읽기 전용; 로컬 작업 폴더 구조 또는 저장소 구조 중 있는 쪽)
D = OUT / "c09_manual_prototype_sim"
D.mkdir(parents=True, exist_ok=True)
lines = []


def log(m):
    print(m); lines.append(m)


rd = lambda p: list(csv.DictReader(open(p, encoding="utf-8-sig")))
z = np.load(GRID / "e01_extract_features/features.npz")
tile_cls, row_of = z["tile_cls"].astype(np.float64), z["tile_row_of"]
tiles = rd(GRID / "e01_extract_features/tile_index.csv")
X = tile_cls[row_of[:, 0]]                                       # 원본 (k=0)
assert X.shape == (2591, 768) and len(tiles) == 2591
key = [(t["image_id"], t["tile_id"]) for t in tiles]
kidx = {k: i for i, k in enumerate(key)}
mf = {(r["image_id"], r["tile_id"]): float(r["mask_frac"]) for r in rd(GRID / "g00_tile_list/tiles.csv")}
lab = rd(OUT / "c08_agg1_manual_labels/agg_tile_labels_filled.csv")
agg_keys = [(r["image_id"], r["tile_id"]) for r in lab if r["has_agg"] == "1"]
assert len(agg_keys) == 17 and all(k in kidx for k in agg_keys)
agg_i = np.array([kidx[k] for k in agg_keys])
is_agg = np.zeros(len(tiles), bool); is_agg[agg_i] = True
log(f"원본 타일 특징 {X.shape}, AGG 타일 {len(agg_i)}개 (이미지 {len({k[0] for k in agg_keys})}장)")

unit = lambda M: M / np.linalg.norm(M, axis=-1, keepdims=True)
Xn = unit(X)
proto = X[agg_i].mean(0)
s_full = Xn @ (proto / np.linalg.norm(proto))                    # 전체 17개 prototype
s_loo, s_loio = s_full.copy(), s_full.copy()
timg = np.array([k[0] for k in key])
for i in agg_i:
    p = X[[j for j in agg_i if j != i]].mean(0); s_loo[i] = float(Xn[i] @ (p / np.linalg.norm(p)))     # 자기 자신 제외
for img in sorted({k[0] for k in agg_keys}):                     # 같은 이미지의 AGG 타일을 모두 제외한 prototype 으로 그 이미지의 모든 타일을 평가
    others = [j for j in agg_i if timg[j] != img]
    p = X[others].mean(0); m = timg == img
    s_loio[m] = Xn[m] @ (p / np.linalg.norm(p))


def rank_pct(s, mask=None):
    """이미지 안 순위(1 = 가장 비슷)와 percentile. mask 가 있으면 그 타일끼리만 순위를 낸다 (그 밖의 타일은 nan)."""
    rk, pc = np.full(len(s), np.nan), np.full(len(s), np.nan)
    for img in np.unique(timg):
        idx = np.where((timg == img) & (mask if mask is not None else True))[0]
        if len(idx) == 0:
            continue
        order = idx[np.argsort(-s[idx], kind="stable")]
        for r_, j in enumerate(order, 1):
            rk[j] = r_; pc[j] = 1 - (r_ - 1) / (len(idx) - 1) if len(idx) > 1 else 1.0
    return rk, pc


inner = np.array([mf[k] >= 0.999999 for k in key])
n_img = {img: int((timg == img).sum()) for img in np.unique(timg)}
variants = {"전체 prototype(17개)": s_full, "loo(자기 자신 제외)": s_loo, "loio(같은 이미지 AGG 타일 제외)": s_loio}
res = {}
for nm, s in variants.items():
    rk, pc = rank_pct(s); rki, pci = rank_pct(s, inner)
    res[nm] = (rk, pc, rki, pci)
    a = agg_i
    top25 = int(sum(rk[i] / n_img[timg[i]] <= 0.25 for i in a))
    ai = [i for i in a if inner[i]]
    log(f"\n[{nm}] AGG 타일 17개의 이미지 안 percentile: 평균 {np.nanmean(pc[a]):.3f}, 중앙값 {np.nanmedian(pc[a]):.3f} | 이미지 안 상위 25% 이내 {top25}/17 (무작위 기대 약 4.3)")
    log(f"    안쪽 타일(mask_frac == 1.0)인 AGG 타일 {len(ai)}개: percentile(이미지의 모든 타일 중) 평균 {np.mean(pc[ai]):.3f}, 중앙값 {np.median(pc[ai]):.3f}, 상위 25% 이내 "
        f"{sum(rk[i] / n_img[timg[i]] <= 0.25 for i in ai)}/{len(ai)} | percentile(안쪽 타일끼리만) 평균 {np.mean(pci[ai]):.3f}, 중앙값 {np.median(pci[ai]):.3f}, 상위 25% 이내 "
        f"{sum(rki[i] / inner[(timg == timg[i])].sum() <= 0.25 for i in ai)}/{len(ai)}")
    ae = [i for i in a if not inner[i]]
    log(f"    가장자리 타일인 AGG 타일 {len(ae)}개: percentile 평균 {np.mean(pc[ae]):.3f}, 중앙값 {np.median(pc[ae]):.3f}")
rk, pc, rki, pci = res["전체 prototype(17개)"]
_, pc_loo, _, _ = res["loo(자기 자신 제외)"]; rk_loio, pc_loio, _, _ = res["loio(같은 이미지 AGG 타일 제외)"]
rk_loo = res["loo(자기 자신 제외)"][0]
with open(D / "tile_sim_manual.csv", "w", encoding="utf-8-sig", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["image_id", "tile_id", "label", "aggsim_manual", "rank", "percentile", "has_agg", "mask_frac", "aggsim_manual_loo", "rank_loo", "percentile_loo", "aggsim_manual_loio", "rank_loio", "percentile_loio", "rank_inner_only", "percentile_inner_only"])
    for i, t in enumerate(tiles):
        w.writerow([t["image_id"], t["tile_id"], t["label"], round(s_full[i], 6), int(rk[i]), round(pc[i], 4), int(is_agg[i]), round(mf[key[i]], 4), round(s_loo[i], 6), int(rk_loo[i]), round(pc_loo[i], 4),
                    round(s_loio[i], 6), int(rk_loio[i]), round(pc_loio[i], 4), "" if np.isnan(rki[i]) else int(rki[i]), "" if np.isnan(pci[i]) else round(pci[i], 4)])
log("\n17개 AGG 타일 (전체 prototype / loo / loio 의 aggsim 과 이미지 안 순위):")
for i in agg_i:
    log(f"  {key[i][0][:32]:<32} {key[i][1]}  mask_frac {mf[key[i]]:.2f}  전체 {s_full[i]:.3f}(#{int(rk[i])}/{n_img[timg[i]]})  loo {s_loo[i]:.3f}(#{int(rk_loo[i])})  loio {s_loio[i]:.3f}(#{int(rk_loio[i])})")
(D / "log.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
log(f"\n저장: {D}")
