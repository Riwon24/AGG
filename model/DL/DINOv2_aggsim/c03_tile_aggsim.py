"""c03: 타일별 AGG 유사도 점수.

입력 (읽기 전용): DINOv2_grid_augmentation/outputs/e01_extract_features/features.npz, tile_index.csv, DINOv2_grid_augmentation/outputs/e00_make_splits/splits.csv (열: repeat, fold, image_id, label, role).
방식:
  1. 원본(k=0) 타일 cls 특징만 쓴다: tile_cls[tile_row_of[i, 0]] -> (2591, 768)
  2. splits.csv 의 이미지 단위 분할(repeat x fold)을 tile_index.csv 의 image_id 로 타일에 옮긴다 (한 이미지의 타일은 모두 같은 쪽).
  3. (repeat, fold) 마다 train 이미지의 타일만으로 prototype 두 개를 만든다: AGG0 = label 0 타일 cls 평균, AGG1+2 = label 1 또는 2 타일 cls 평균.
  4. 모든 타일(train + test)에 대해 cosine_similarity(tile_cls, AGG1+2 prototype) = aggsim 을 계산한다.
타일 라벨은 그 타일이 속한 이미지의 라벨이다 (타일별 AGG 위치 라벨은 없다).
누수 방지: prototype 은 해당 (repeat, fold) 의 train 타일로만 계산하고 test 타일은 aggsim 계산에만 쓴다 (assert: train/test 이미지 서로소, 반복마다 모든 이미지가 정확히 한 번 test).
출력: outputs/c03_tile_aggsim/tile_aggsim.csv (열: tile_id, image_id, label, repeat, fold, split, aggsim), log.txt
  tile_id 는 이미지 안의 격자 번호(r00_c00 ...)라 이미지마다 같은 값이 있으므로 (image_id, tile_id) 가 타일의 키이다.
"""
import csv
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"                                  # 이 폴더의 결과 (c03 -> c04, c05, c06 -> c08 -> c09 가 서로의 출력을 읽는다)
GRID = next((p for p in (HERE.parent / "exp_grid_aug" / "outputs", HERE.parent / "DINOv2_grid_augmentation" / "outputs") if p.exists()), HERE.parent / "DINOv2_grid_augmentation" / "outputs")
#                                                       # 입력: DINOv2_grid_augmentation 의 출력 (읽기 전용; 로컬 작업 폴더 구조 또는 저장소 구조 중 있는 쪽)
D = OUT / "c03_tile_aggsim"
D.mkdir(parents=True, exist_ok=True)
lines = []


def log(*a):
    m = " ".join(str(x) for x in a); print(m); lines.append(m)


z = np.load(GRID / "e01_extract_features/features.npz")
tile_cls, row_of = z["tile_cls"].astype(np.float64), z["tile_row_of"]
tiles = list(csv.DictReader(open(GRID / "e01_extract_features/tile_index.csv", encoding="utf-8-sig")))
assert row_of.shape[0] == len(tiles) == 2591
X = tile_cls[row_of[:, 0]]                                   # 원본 (k=0) 타일 특징
assert X.shape == (2591, 768) and np.isfinite(X).all()
tl = np.array([int(t["label"]) for t in tiles]); timg = [t["image_id"] for t in tiles]
log(f"원본(k=0) 타일 특징 {X.shape}, 타일 {len(tiles)}개 (라벨 0/1/2 = {np.bincount(tl).tolist()})")

sp = list(csv.DictReader(open(GRID / "e00_make_splits/splits.csv", encoding="utf-8-sig")))
role, img_label = {}, {}
for r in sp:
    role[(int(r["repeat"]), int(r["fold"]), r["image_id"])] = r["role"]
    img_label[r["image_id"]] = int(r["label"])
assert all(img_label[timg[i]] == tl[i] for i in range(len(tiles))), "타일 라벨이 splits.csv 의 이미지 라벨과 다름"
reps = sorted({k[0] for k in role}); folds = sorted({k[1] for k in role})
log(f"splits.csv {len(sp)}행: 반복 {len(reps)}회 x fold {len(folds)}개, 이미지 {len(img_label)}장, role = train/test")

unit = lambda M: M / np.linalg.norm(M, axis=1, keepdims=True)
Xn = unit(X)
rows, test_count = [], {}
for r in reps:
    for f in folds:
        is_tr = np.array([role[(r, f, timg[i])] == "train" for i in range(len(tiles))])
        tr_imgs = {i for (rr, ff, i), v in role.items() if rr == r and ff == f and v == "train"}
        te_imgs = {i for (rr, ff, i), v in role.items() if rr == r and ff == f and v == "test"}
        assert tr_imgs.isdisjoint(te_imgs) and len(tr_imgs | te_imgs) == len(img_label)
        for i in te_imgs:
            test_count[(r, i)] = test_count.get((r, i), 0) + 1
        p0 = X[is_tr & (tl == 0)].mean(0)                    # AGG0 prototype (train 타일만)
        p12 = X[is_tr & (tl >= 1)].mean(0)                   # AGG1+2 prototype (train 타일만)
        assert (is_tr & (tl == 0)).any() and (is_tr & (tl >= 1)).any()
        assert not np.isin(np.array(timg), list(te_imgs))[is_tr].any(), "train 타일이 test 이미지에 속함"
        aggsim = Xn @ (p12 / np.linalg.norm(p12))            # cosine_similarity(tile_cls, AGG1+2 prototype), 모든 타일
        for i in range(len(tiles)):
            rows.append((tiles[i]["tile_id"], timg[i], int(tl[i]), r, f, "train" if is_tr[i] else "test", round(float(aggsim[i]), 6)))
assert all(test_count[(r, i)] == 1 for r in reps for i in img_label), "반복마다 모든 이미지가 정확히 한 번 test 여야 함"
assert len(rows) == len(reps) * len(folds) * len(tiles)
with open(D / "tile_aggsim.csv", "w", encoding="utf-8-sig", newline="") as fh:
    w = csv.writer(fh); w.writerow(["tile_id", "image_id", "label", "repeat", "fold", "split", "aggsim"]); w.writerows(rows)
log(f"저장: {D / 'tile_aggsim.csv'} ({len(rows):,}행 = 반복 {len(reps)} x fold {len(folds)} x 타일 {len(tiles)})")

A = np.array([r[6] for r in rows]); L_ = np.array([r[2] for r in rows]); S_ = np.array([r[5] for r in rows])
log("\n요약 (참고, 새 판단 기준 아님): aggsim 평균 (표준편차)")
for sname in ("train", "test"):
    log(f"  {sname:<5} " + " | ".join(f"AGG{c}: {A[(S_ == sname) & (L_ == c)].mean():.4f} ({A[(S_ == sname) & (L_ == c)].std():.4f})" for c in (0, 1, 2)))
(D / "log.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
