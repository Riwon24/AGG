"""b00: 입력 점검. 마스크 60장, 타일 격자 전체(stride 224, 2,880개) + 분석용 타일 선택 규칙 점검, tiles.csv sha256, 이미지 해상도, 마스크 비율 재계산, 기존 분할(e00) 파일을 확인한다.

모든 입력은 읽기 전용. 하나라도 어긋나면 assert 로 멈춘다. 출력: outputs/b00_check_inputs/log.txt, inputs_sha256.csv
"""
import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np

from bg_common import Logger, inputs, load_config, read_mask, read_rgb, select_tiles, sha256_file, write_csv

SCRIPT = "b00_check_inputs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    log = Logger(cfg["paths"]["outputs_dir"], SCRIPT)
    d = cfg["data"]
    log("config:", cfg["_config_path"])
    imgs, tiles = inputs(cfg, extract=True)      # 특징 추출 대상: 격자 전체
    ids = [r["image_id"] for r in imgs]
    assert len(imgs) == d["expected_images"], len(imgs)
    assert len(tiles) == d["expected_tiles"], f"타일 {len(tiles)}개 (기대 {d['expected_tiles']})"
    cls = Counter(int(t["label"]) for t in tiles)
    assert {int(k): v for k, v in cls.items()} == {int(k): v for k, v in d["expected_class_tiles"].items()}, f"클래스별 타일 {dict(cls)}"
    log(f"이미지 {len(imgs)}장, 타일 {len(tiles)}개 (tiles.select={cfg['tiles']['select']}, 클래스별 {dict(sorted(cls.items()))}) -> 기대와 일치")
    sha_t = sha256_file(cfg["paths"]["tiles_csv"])
    assert sha_t.startswith(d["tiles_sha256_prefix"]), f"tiles.csv sha256 {sha_t[:12]} != {d['tiles_sha256_prefix']}"
    log(f"tiles.csv sha256 {sha_t} (앞 12자리 일치)")

    W, H = d["expected_resolution"]
    by_img = {}
    for t in tiles:
        by_img.setdefault(t["image_id"], []).append(t)
    assert set(by_img) == set(ids)
    maxdiff, frac_bg = 0.0, []
    n_full = n_part = n_none = 0
    rows = []
    for r in imgs:
        m = read_mask(r["mask"])
        img = read_rgb(r["path"])
        assert m.shape == (H, W) and img.shape == (H, W, 3), (r["image_id"], m.shape, img.shape)
        rows += [dict(kind="image", id=r["image_id"], sha256=sha256_file(r["path"])), dict(kind="mask", id=r["image_id"], sha256=sha256_file(r["mask"]))]
        for t in by_img[r["image_id"]]:
            x0, y0, w, h = int(t["x0"]), int(t["y0"]), int(t["w"]), int(t["h"])
            assert (w, h) == (224, 224) and x0 + w <= W and y0 + h <= H
            mf = float(m[y0:y0 + h, x0:x0 + w].mean())
            maxdiff = max(maxdiff, abs(mf - float(t["mask_frac"])))
            if mf >= 0.999999:
                n_full += 1
            elif mf > 0:
                n_part += 1
            else:
                n_none += 1
        frac_bg.append(1 - m.mean())
    assert maxdiff < 2e-5, f"tiles.csv 의 mask_frac 과 마스크 PNG 에서 다시 계산한 값의 최대 차이 {maxdiff}"
    log(f"마스크 PNG 에서 타일의 mask_frac 을 다시 계산: tiles.csv 와 최대 절대 차이 {maxdiff:.2e}")
    exp_m = d["expected_tiles_by_mask"]
    assert (n_full, n_part, n_none) == (exp_m["full"], exp_m["partial"], exp_m["none"]), (n_full, n_part, n_none)
    log(f"타일 구성(마스크 비율): 100% {n_full}개 ({n_full / len(tiles) * 100:.1f}%), 일부 {n_part}개 ({n_part / len(tiles) * 100:.1f}%), 0%(순수 배경) {n_none}개 ({n_none / len(tiles) * 100:.1f}%) -> 기대와 일치")
    log("  참고: ML_preproc 의 이전 격자(stride 112, 9,900개)와 선택(mask_frac >= 0.5, 7,921개)은 쓰지 않는다 (stride 224 격자로 교체, 사용자 결정)")
    log(f"이미지 배경 비율(1-마스크 면적): 최소 {min(frac_bg):.3f}, 중앙값 {float(np.median(frac_bg)):.3f}, 최대 {max(frac_bg):.3f}")

    # ---------- 분석용 타일 선택 규칙 점검 ----------
    sel, info = select_tiles(tiles, cfg["tiles"])
    log(f"\n[분석 타일 선택] 규칙: {cfg['tiles']['select']}, 시편 픽셀 0인 타일 제외: {cfg['tiles'].get('drop_zero_mask', False)}")
    spec_total = {r["image_id"]: int(read_mask(r["mask"]).sum()) for r in imgs}
    sel_by = {}
    for tt in sel:
        sel_by.setdefault(tt["image_id"], []).append(tt)
    assert set(sel_by) == set(ids)
    covered, rows_sel = 0, []
    for iid in ids:
        g = sel_by[iid]
        assert all(int(x["mask_px"]) > 0 for x in g) or not cfg["tiles"].get("drop_zero_mask", False), "시편 픽셀 0인 타일이 남음"
        if cfg["tiles"].get("expect_non_overlap", cfg["tiles"]["select"] == "nonoverlap_best_phase"):      # 타일끼리 겹치지 않는지 모든 쌍 검사
            xy = [(int(x["x0"]), int(x["y0"])) for x in g]
            for i in range(len(xy)):
                for j in range(i + 1, len(xy)):
                    assert abs(xy[i][0] - xy[j][0]) >= 224 or abs(xy[i][1] - xy[j][1]) >= 224, f"타일이 겹침: {iid} {xy[i]} {xy[j]}"
        px = sum(int(x["mask_px"]) for x in g)
        if cfg["tiles"].get("expect_non_overlap", cfg["tiles"]["select"] == "nonoverlap_best_phase"):
            assert px <= spec_total[iid]
        covered += px
        rows_sel.append(dict(image_id=iid, n_tiles=len(g), specimen_px_in_tiles=px, specimen_px_total=spec_total[iid], specimen_cov=round(px / spec_total[iid], 4),
                             phase=next((i_["phase"] for i_ in info if i_["image_id"] == iid), "-")))
    write_csv(f"{log.dir}/tile_selection.csv", rows_sel)
    cnt = Counter(int(x["label"]) for x in sel)
    nfull = sum(1 for x in sel if float(x["mask_frac"]) >= 0.999999); nzero = sum(1 for x in sel if int(x["mask_px"]) == 0)
    cov = np.array([r["specimen_cov"] for r in rows_sel]); nt = np.array([r["n_tiles"] for r in rows_sel])
    log(f"  선택 타일 {len(sel)}개 (이미지당 {nt.min()}~{nt.max()}개, 중앙값 {int(np.median(nt))}), 클래스별 {dict(sorted(cnt.items()))}")
    log(f"  구성: 마스크 100% {nfull}개, 일부 {len(sel) - nfull - nzero}개, 시편 픽셀 0 {nzero}개")
    log(f"  시편 픽셀이 선택 타일에 들어간 비율(타일이 겹치지 않으므로 중복 없음): 전체 {covered / sum(spec_total.values()) * 100:.2f}%, 이미지별 최소 {cov.min() * 100:.1f}%, 중앙값 {np.median(cov) * 100:.1f}%")
    if cfg["tiles"]["select"] == "nonoverlap_best_phase":
        ph = Counter(r["phase"] for r in rows_sel)
        log(f"  선택된 부분 격자(행 홀짝+열 홀짝): {dict(sorted(ph.items()))}")
    if cfg["tiles"].get("expect_non_overlap", cfg["tiles"]["select"] == "nonoverlap_best_phase"):
        log("  검증 통과: 선택 타일끼리 겹치지 않음 (모든 쌍), 시편 픽셀 0인 타일 없음")
    write_csv(f"{log.dir}/selected_tile_ids.csv", [dict(image_id=x["image_id"], tile_id=x["tile_id"]) for x in sel])
    log(f"  선택 목록 sha256 {sha256_file(f'{log.dir}/selected_tile_ids.csv')}")

    sp = list(csv.DictReader(open(cfg["paths"]["splits_csv"], encoding="utf-8-sig")))
    assert {r["image_id"] for r in sp} == set(ids), "e00 분할의 이미지 집합이 이미지 표와 다름"
    assert len(sp) == 10 * 5 * 60, len(sp)
    sha_s = sha256_file(cfg["paths"]["splits_csv"])
    rows.append(dict(kind="splits", id="e00_make_splits/splits.csv", sha256=sha_s))
    rows.append(dict(kind="tiles", id=Path(cfg["paths"]["tiles_csv"]).parent.name + "/tiles.csv", sha256=sha_t))
    log(f"기존 분할(e00) {len(sp)}행(반복 10 x fold 5 x 이미지 60), 이미지 집합 일치, sha256 {sha_s}")
    write_csv(f"{log.dir}/inputs_sha256.csv", rows)
    log(f"\n점검 통과. 해시 {len(rows)}개를 inputs_sha256.csv 에 기록: {log.dir}")


if __name__ == "__main__":
    main()
