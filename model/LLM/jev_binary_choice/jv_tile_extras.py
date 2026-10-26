"""jv_tile_extras: JEVision 타일 추론 결과(jsonl)에서 타일별 점수 CSV 와 보조 분석을 만든다 (jv06, jv07, jv08 공통; numpy 만 필요).

입력: jv06/jv07/jv08 추론 스크립트의 jsonl (한 줄 = 타일 하나: image_id, label, tile_id, x0, y0, p_flat, p_A, p_B, choice_A, choice_B, extent).
      DINOv2_grid_augmentation 의 g00 tiles.csv (mask_frac), exp_aggsim/outputs/c08_agg1_manual_labels/agg_tile_labels_filled.csv (직접 표시한 AGG 타일 17개, has_agg).
출력 (--out-dir):
  tile_scores.csv   타일별 점수. 열: image_id, label, metal, condition, tile_id, x0, y0, mask_frac, has_agg(표시한 17개만 1, 그 외 AGG1 이미지는 0, 다른 이미지는 빈 칸),
                    p_flat(두 순서 평균), p_A, p_B, choice_A, choice_B, choice(평균 확률 기준), p_extent_NONE/PARTIAL/MOST, rank_in_image, n_tiles_in_image, percentile_in_image
  extras.md         (1) 선택 분포와 순서 효과 (2) 순서별 점수의 이미지 단위 AUROC (3) 안쪽 타일(mask_frac == 1)만으로 만든 이미지 점수의 AUROC (4) 표시한 AGG 타일 17개의 이미지 안 percentile
percentile = 1 - (rank - 1) / (n - 1) (1.0 = 이미지 안 최상위, 무작위 기대 0.5). 표시하지 않은 타일에 AGG 가 없는지는 확인되지 않았다.
"""
import argparse
import collections
import csv
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
G00 = ROOT / "exp_grid_aug/outputs/g00_tile_list/tiles.csv"
LAB = ROOT / "exp_aggsim/outputs/c08_agg1_manual_labels/agg_tile_labels_filled.csv"


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


def boot_auc(vals, y3, n=2000, seed=0):
    rng = np.random.RandomState(seed); N = len(vals); bs = []
    for _ in range(n):
        ix = rng.randint(0, N, N); yy = y3[ix] >= 1
        if yy.any() and (~yy).any():
            bs.append(auc(vals[ix][yy], vals[ix][~yy]))
    return np.percentile(bs, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl"); ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    R = [json.loads(l) for l in open(a.jsonl, encoding="utf-8") if l.strip()]
    mf = {(r["image_id"], r["tile_id"]): float(r["mask_frac"]) for r in csv.DictReader(open(G00, encoding="utf-8-sig"))}
    has = {}
    if LAB.exists():
        for r in csv.DictReader(open(LAB, encoding="utf-8-sig")):
            has[(r["image_id"], r["tile_id"])] = r["has_agg"]
    by = collections.defaultdict(list)
    for r in R:
        by[r["image_id"]].append(r)
    rows, md = [], [f"# 타일 점수 보조 분석 ({Path(a.jsonl).name})\n", f"타일 {len(R)}개, 이미지 {len(by)}장.\n"]
    for iid, ts in sorted(by.items()):
        order = sorted(range(len(ts)), key=lambda k: -ts[k]["p_flat"])
        rank = {k: i + 1 for i, k in enumerate(order)}; lab3 = int(ts[0]["label"][3:])
        for k, t in enumerate(ts):
            h = has.get((iid, t["tile_id"]), "")
            if lab3 == 1 and h == "":
                h = "0"
            rows.append(dict(image_id=iid, label=t["label"], metal=iid.split("/")[0], condition=iid.split("/")[1], tile_id=t["tile_id"], x0=t["x0"], y0=t["y0"],
                             mask_frac=round(mf[(iid, t["tile_id"])], 4), has_agg=h, p_flat=t["p_flat"], p_A=t.get("p_A", ""), p_B=t.get("p_B", ""), choice_A=t.get("choice_A", ""), choice_B=t.get("choice_B", ""),
                             choice="FLAT_REGION" if t["p_flat"] > 0.5 else "NO_FLAT_REGION", p_extent_NONE=t["extent"]["NONE"], p_extent_PARTIAL=t["extent"]["PARTIAL"], p_extent_MOST=t["extent"]["MOST"],
                             rank_in_image=rank[k], n_tiles_in_image=len(ts), percentile_in_image=round(1 - (rank[k] - 1) / (len(ts) - 1), 4) if len(ts) > 1 else 1.0))
    with open(out / "tile_scores.csv", "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    # (1) 선택 분포와 순서 효과
    n = len(R); pm = np.array([t["p_flat"] for t in R])
    md.append("## 1. 선택 분포와 순서 효과\n")
    if "p_A" in R[0] and R[0]["p_A"] != "":
        pa = np.array([t["p_A"] for t in R]); pb = np.array([t["p_B"] for t in R])
        ca = np.mean([t["choice_A"] == "FLAT_REGION" for t in R]); cb = np.mean([t["choice_B"] == "FLAT_REGION" for t in R])
        md += [f"- FLAT_REGION 을 고른 비율: 순서 A(NO 먼저) {ca:.1%}, 순서 B(FLAT 먼저) {cb:.1%}, 두 순서 평균 확률 기준 {np.mean(pm > 0.5):.1%}",
               f"- P(FLAT_REGION) 평균: A {pa.mean():.3f}, B {pb.mean():.3f}, 평균 {pm.mean():.3f}, 서로 다른 평균값 {len(set(pm.tolist()))}개",
               f"- |P_A - P_B| 평균 {np.abs(pa - pb).mean():.3f}, 최대 {np.abs(pa - pb).max():.2f}, 두 순서의 선택이 다른 타일 {np.mean([t['choice_A'] != t['choice_B'] for t in R]):.1%}\n"]
    else:
        md += [f"- 평균 P 기준 FLAT_REGION 비율 {np.mean(pm > 0.5):.1%}, P 평균 {pm.mean():.3f}\n"]

    # (2) 순서별 점수의 이미지 단위 AUROC, (3) 안쪽 타일만
    ids = sorted(by); y3 = np.array([int(by[i][0]["label"][3:]) for i in ids]); yb = y3 >= 1
    def img_score(key, inner=False):
        return np.array([np.mean([t[key] for t in by[i] if (not inner or mf[(i, t["tile_id"])] >= 0.999999)] or [np.nan]) for i in ids])
    md.append("## 2. 이미지 점수의 AUROC (AGG1+2 가 양성, 이미지 부트스트랩 95% CI, 0.5 = 구분 못 함)\n")
    md += ["| 점수 | 타일 | AUROC |", "|---|---|---|"]
    keys = [("p_flat (두 순서 평균)", "p_flat")] + ([("p_A (NO 먼저)", "p_A"), ("p_B (FLAT 먼저)", "p_B")] if "p_A" in R[0] and R[0]["p_A"] != "" else [])
    for nm, key in keys:
        for inner in (False, True):
            v = img_score(key, inner)
            if np.isnan(v).any():
                continue
            lo, hi = boot_auc(v, y3)
            md.append(f"| {nm} | {'안쪽(mask_frac 100%)' if inner else '전체'} | {auc(v[yb], v[~yb]):.3f} [{lo:.3f}, {hi:.3f}] |")

    # (4) 표시한 17개 타일
    md.append("\n## 3. 직접 표시한 AGG 타일 17개의 이미지 안 percentile (p_flat 기준)\n")
    marked = [(r["image_id"], r["tile_id"]) for r in rows if r["has_agg"] == "1"]
    if marked:
        pc = {(r["image_id"], r["tile_id"]): r["percentile_in_image"] for r in rows}
        v = np.array([pc[k] for k in marked])
        inner = np.array([mf[k] >= 0.999999 for k in marked])
        md += [f"- {len(marked)}개: percentile 평균 {v.mean():.3f}, 중앙값 {np.median(v):.3f} (무작위 기대 0.5), 이미지 안 상위 25% 이내 {int(np.sum(v >= 0.75))}개 (무작위 기대 약 {len(marked) * 0.25:.1f}개)",
               f"- 안쪽 타일인 {int(inner.sum())}개: 평균 {v[inner].mean():.3f}, 가장자리 타일인 {int((~inner).sum())}개: 평균 {v[~inner].mean():.3f}",
               "- 표시하지 않은 타일에 AGG 가 없는지는 확인되지 않았다. 점수는 JEVision 이 낸 것이라 표시로 만든 prototype 과 달리 순환성은 없지만, 예시 이미지를 프롬프트에 넣는 실행(jv08)에서는 예시가 이 표시에서 나오므로 독립이 아니다.\n"]
    else:
        md.append("- 표시 목록을 찾지 못함\n")
    (out / "extras.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
