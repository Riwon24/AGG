"""jv08 준비: few-shot 예시 타일 3개(AGG 예시 2개, 정상 예시 1개)를 정하고 확인용 그림을 만든다. JEVision 의 이미지 입력 제한(요청당 최대 4장) 때문에 AGG 2 + 정상 1 + 질의 1 = 4장이다.

선택 규칙 (결과를 보기 전에 정함, AGG 예시는 JEVision 점수를 쓰지 않는다):
  AGG 예시: 직접 표시한 AGG 타일 17개(c08 agg_tile_labels_filled.csv, has_agg == 1) 중 안쪽 타일(mask_frac == 1.0)에서, 타일 경계 부근이라고 적은 2개(01번 r01_c01, 14번 r04_c04)는 뺀다
            (AGG 가 타일 경계에 걸쳐 있어 타일 안에 있는지 불확실). 남은 후보 중 서로 다른 금속인 두 타일의 쌍 가운데, 사진의 겉모습 차이(평균 R,G,B 와 회색조 표준편차를 [0,1] 로 정규화한 4차원 벡터의
            유클리드 거리)가 가장 큰 쌍을 고른다. 겉모습 지표는 모델과 무관하다.
  정상 예시: AGG0 이미지(이미지 전체에 AGG 가 없음)의 안쪽 타일(mask_frac == 1.0) 중, 겉모습 벡터가 두 AGG 예시의 겉모습 평균에 가장 가까운 타일. 정상 예시와 AGG 예시의 밝기, 질감 차이가
            AGG 유무와 섞이지 않게 하려는 것이다. JEVision 점수는 쓰지 않는다. 상위 후보 5개를 출력한다.
출력: <out-dir>/examples.json (타일 3개: image_id, tile_id, x0, y0, 용도), examples_review.png (후보 7개와 선택 결과)
"""
import argparse
import csv
import itertools
import json
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LAB = ROOT / "exp_aggsim/outputs/c08_agg1_manual_labels/agg_tile_labels_filled.csv"
EXCLUDE = {("BASE/PH1180-1035/x10#1  - AGG1", "r01_c01"), ("REF-LN/PH1205-1050/x10#1 - AGG1", "r04_c04")}


def crop(iid, x0, y0, cache):
    if iid not in cache:
        cache.clear(); cache[iid] = cv2.imdecode(np.fromfile(str(ROOT / (iid + ".jpg")), np.uint8), cv2.IMREAD_COLOR)
    return cache[iid][y0:y0 + 224, x0:x0 + 224]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tile-scores", required=True); ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    sc = {(r["image_id"], r["tile_id"]): r for r in csv.DictReader(open(a.tile_scores, encoding="utf-8-sig"))}
    lab = list(csv.DictReader(open(LAB, encoding="utf-8-sig")))
    cache = {}
    cand = []
    for r in lab:
        k = (r["image_id"], r["tile_id"])
        if r["has_agg"] == "1" and float(sc[k]["mask_frac"]) >= 0.999999 and k not in EXCLUDE:
            t = crop(r["image_id"], int(r["x0"]), int(r["y0"]), cache)
            g = cv2.cvtColor(t, cv2.COLOR_BGR2GRAY).astype(np.float32)
            desc = np.array([t[..., 2].mean(), t[..., 1].mean(), t[..., 0].mean(), g.std()], np.float32) / 255.0
            cand.append(dict(image_id=r["image_id"], tile_id=r["tile_id"], x0=int(r["x0"]), y0=int(r["y0"]), metal=r["metal"], condition=r["condition"], desc=desc, p_flat=float(sc[k]["p_flat"])))
    print(f"AGG 예시 후보 {len(cand)}개 (안쪽 타일, 경계 표기 2개 제외)")
    best = None
    for i, j in itertools.combinations(range(len(cand)), 2):
        if cand[i]["metal"] == cand[j]["metal"]:
            continue
        d = float(np.linalg.norm(cand[i]["desc"] - cand[j]["desc"]))
        if best is None or d > best[0]:
            best = (d, i, j)
    d, i, j = best
    agg = [cand[i], cand[j]]
    for c in agg:
        print(f"  AGG 예시: {c['image_id']} {c['tile_id']} (금속 {c['metal']}, 조건 {c['condition']}) 겉모습 {np.round(c['desc'], 3).tolist()}")
    print(f"  두 예시의 겉모습 거리 {d:.3f}")
    target = (agg[0]["desc"] + agg[1]["desc"]) / 2
    ncand = []
    for (iid, tid), r in sc.items():
        if r["label"] == "AGG0" and float(r["mask_frac"]) >= 0.999999:
            tt = crop(iid, int(r["x0"]), int(r["y0"]), cache)
            g = cv2.cvtColor(tt, cv2.COLOR_BGR2GRAY).astype(np.float32)
            desc = np.array([tt[..., 2].mean(), tt[..., 1].mean(), tt[..., 0].mean(), g.std()], np.float32) / 255.0
            ncand.append((float(np.linalg.norm(desc - target)), iid, tid, int(r["x0"]), int(r["y0"]), desc))
    ncand.sort(key=lambda x: x[0])
    print(f"  정상 예시 후보 (AGG0 이미지 안쪽 타일 {len(ncand)}개, 두 AGG 예시 겉모습 평균과의 거리 순) 상위 5개:")
    for dd, iid, tid, x0, y0, desc in ncand[:5]:
        print(f"    거리 {dd:.3f}  {iid} {tid}")
    _, niid, ntid, nx0, ny0, _ = ncand[0]
    nr = dict(image_id=niid, tile_id=ntid, x0=nx0, y0=ny0)
    print(f"  정상 예시: {niid} {ntid}")
    ex = [dict(role="AGG", image_id=c["image_id"], tile_id=c["tile_id"], x0=c["x0"], y0=c["y0"]) for c in agg] + [dict(role="NORMAL", image_id=nr["image_id"], tile_id=nr["tile_id"], x0=nr["x0"], y0=nr["y0"])]
    (out / "examples.json").write_text(json.dumps(ex, ensure_ascii=False, indent=1), encoding="utf-8")

    # 확인용 그림: 후보 7개(선택은 굵은 테두리) + 정상 예시
    tiles = [(c, f"{c['metal']} {c['condition']} {c['tile_id']}", (c is agg[0] or c is agg[1])) for c in cand]
    S, cols, pad, cap = 320, 4, 6, 50
    items = [(crop(c["image_id"], c["x0"], c["y0"], cache), t, sel, "AGG candidate") for c, t, sel in tiles]
    for k_, (dd, iid, tid, x0, y0, desc) in enumerate(ncand[:5]):
        items.append((crop(iid, x0, y0, cache), f"{iid.split('/')[0]} {iid.split('/')[1]} {tid} d={dd:.3f}", k_ == 0, "NORMAL (AGG0 img)" + (" SELECTED" if k_ == 0 else " cand")))
    rows_n = (len(items) + cols - 1) // cols
    W, H = cols * (S + pad) + pad, rows_n * (S + cap + pad) + pad + 34
    im = np.full((H, W, 3), 255, np.uint8)
    cv2.putText(im, "few-shot example candidates (green border = selected)", (pad, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    for k, (t, cap_txt, sel, kind) in enumerate(items):
        cx, cy = pad + (k % cols) * (S + pad), 34 + pad + (k // cols) * (S + cap + pad)
        im[cy:cy + S, cx:cx + S] = cv2.resize(t, (S, S), interpolation=cv2.INTER_CUBIC)
        if sel:
            cv2.rectangle(im, (cx, cy), (cx + S - 1, cy + S - 1), (0, 160, 0), 6)
        cv2.putText(im, f"{k + 1}. {kind}", (cx, cy + S + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(im, cap_txt[:48], (cx, cy + S + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imencode(".png", im)[1].tofile(str(out / "examples_review.png"))
    print("저장:", out)


if __name__ == "__main__":
    main()
