"""e00: 이미지 단위 층화 반복 K-fold 분할을 만든다 (K=5, 반복 10회, seed 고정).

- 입력: 전처리 outputs/s04_tile_grid/tiles.csv 에서 selected==1 인 타일 -> 이미지 60장 (읽기 전용)
- 처리 조건 파싱(R1): 2단계 폴더명 -> (처리 방법, 온도). 10개 조건 전부와 일치하는지 assert, conditions.csv 저장.
- 출력: splits.csv (repeat, fold, image_id, label, role[train/test]), conditions.csv, log.txt
- assert: 각 repeat 에서 모든 이미지가 정확히 한 번 test, 같은 이미지의 타일이 양쪽에 걸치지 않음(분할 단위가 이미지이므로 구조적으로 보장 + 검사)
"""
import argparse
from collections import Counter

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold

from common import (Logger, assert_conditions, load_config, load_image_table, load_selected_tiles, write_csv)

SCRIPT = "e00_make_splits"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--output", help="outputs 폴더 (기본: config paths.outputs_dir)")
    a = ap.parse_args()
    cfg = load_config(a.config)
    out = a.output or cfg["paths"]["outputs_dir"]
    log = Logger(out, SCRIPT)
    log("config:", cfg["_config_path"])
    cv = cfg["cv"]
    log(f"n_splits={cv['n_splits']} n_repeats={cv['n_repeats']} seed={cv['seed']}")

    tiles = load_selected_tiles(cfg)
    imgs = load_image_table(cfg)
    log(f"선택 타일 {len(tiles)}개 (기대 {cfg['data']['expected_tiles']}), 이미지 {len(imgs)}장 (기대 {cfg['data']['expected_images']})")
    assert len(tiles) == cfg["data"]["expected_tiles"] and len(imgs) == cfg["data"]["expected_images"]
    assert_conditions(imgs, cfg, log)
    write_csv(f"{log.dir}/conditions.csv", [dict(image_id=r["image_id"], metal=r["metal"], condition=r["condition"], method=r["method"], temp=r["temp"], label=r["label"]) for r in imgs])
    log(f"라벨별 이미지: {dict(sorted(Counter(r['label'] for r in imgs).items()))}")
    log("처리 조건 파싱 규칙(R1): 폴더명에 '-' 가 있으면 첫 '-' 로 (처리 방법, 온도), 없으면 ^(NPH)(\\d{4})$ -> (NPH, 온도)")

    y = np.array([r["label"] for r in imgs]); ids = [r["image_id"] for r in imgs]
    # 타일 -> 이미지 대응 확인
    tile_img = Counter(t["image_id"] for t in tiles)
    assert set(tile_img) == set(ids)

    rskf = RepeatedStratifiedKFold(n_splits=cv["n_splits"], n_repeats=cv["n_repeats"], random_state=cv["seed"])
    rows, fold_counts = [], []
    for k, (tr, te) in enumerate(rskf.split(np.zeros(len(y)), y)):
        rep, fold = divmod(k, cv["n_splits"])
        assert set(tr).isdisjoint(te)
        for i in range(len(y)):
            rows.append(dict(repeat=rep, fold=fold, image_id=ids[i], label=int(y[i]), role="test" if i in set(te.tolist()) else "train"))
        fold_counts.append((rep, fold, Counter(y[te].tolist()), Counter(y[tr].tolist())))
    write_csv(f"{log.dir}/splits.csv", rows)

    # ---- assert ----
    tile_by_image = {}
    for t in tiles:
        tile_by_image.setdefault(t["image_id"], []).append(t["tile_id"])
    for rep in range(cv["n_repeats"]):
        test_cnt = Counter()
        for fold in range(cv["n_splits"]):
            sel = [r for r in rows if r["repeat"] == rep and r["fold"] == fold]
            test = {r["image_id"] for r in sel if r["role"] == "test"}; train = {r["image_id"] for r in sel if r["role"] == "train"}
            assert test.isdisjoint(train) and len(test) + len(train) == len(ids)
            # tile_id 는 이미지마다 같은 이름이 반복되므로(r00_c00 등) (image_id, tile_id) 쌍으로 구분한다
            tile_test = {(i, tid) for i in test for tid in tile_by_image[i]}; tile_train = {(i, tid) for i in train for tid in tile_by_image[i]}
            assert tile_test.isdisjoint(tile_train), "같은 이미지의 타일이 train/test 양쪽에 있음"
            for i in test:
                test_cnt[i] += 1
        assert set(test_cnt) == set(ids) and all(v == 1 for v in test_cnt.values()), f"repeat {rep}: 모든 이미지가 정확히 한 번 test 이어야 함"
    log("assert 통과: 각 repeat 에서 모든 이미지가 정확히 한 번 test, train/test 이미지와 타일이 겹치지 않음")

    log(f"\nfold별 test 클래스 수 (repeat 0..{cv['n_repeats']-1}, 5 fold; 형식 AGG0/AGG1/AGG2)")
    summary = Counter()
    for rep, fold, tec, trc in fold_counts:
        s = "/".join(str(tec.get(c, 0)) for c in (0, 1, 2)); summary[s] += 1
        if rep < 2:
            log(f"  repeat {rep} fold {fold}: test {s} (이미지 {sum(tec.values())}장) | train {'/'.join(str(trc.get(c, 0)) for c in (0, 1, 2))}")
    log("  fold 구성(AGG0/AGG1/AGG2) 빈도 (50개 fold 전체):", dict(summary))
    log(f"  AGG1 이미지는 14장이라 fold 당 {min(c.get(1, 0) for _, _, c, _ in fold_counts)}~{max(c.get(1, 0) for _, _, c, _ in fold_counts)}장")
    log(f"\n저장: {log.dir}/splits.csv ({len(rows)}행), conditions.csv")


if __name__ == "__main__":
    main()
