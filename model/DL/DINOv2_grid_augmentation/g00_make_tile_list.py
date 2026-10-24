"""g00: ML_preproc 의 격자(m02, stride 224 2,880개)에서 분석 타일을 골라 e00~e11 이 읽는 tiles.csv 를 만든다 (모든 입력은 읽기 전용).

- 입력: ML_preproc m02 의 tiles.csv (이미지 60장 x 48개, 224px, stride 224 로 겹치지 않음, 마스크 비율 `mask_frac`, 시편 픽셀 수 `mask_px`)
- 규칙 (config g00, 모든 이미지 동일, 라벨 무관): select = all_grid 이면 시편 픽셀이 0인 타일을 뺀 나머지 전부 (격자가 이미 겹치지 않음). select = nonoverlap_best_phase 는 stride 112 격자 파일용:
  stride 112 격자에서 서로 겹치지 않는 stride 224 부분 격자 4개 중 시편 픽셀을 가장 많이 덮는 것을 이미지마다 하나 고르고(동률이면 앞 번호) 시편 픽셀 0인 타일을 뺀다.
  선택된 타일은 `selected=1`, 나머지는 0 (m02 의 원래 선택은 `selected_m02` 열에 보존).
- 검증: tiles.csv sha256, 격자 타일 수, 선택 타일끼리 겹치지 않음(모든 쌍), 시편 픽셀 0인 타일 없음, 선택 목록이 config 의 기준 sha256(배경 증강 실험의 b00 이 같은 규칙으로 고른 목록)과 같음,
  마스크 PNG 에서 시편 픽셀 커버리지 재계산.
출력: tiles.csv (e00~e11 의 입력), tile_selection.csv, selected_tile_ids.csv, log.txt
"""
import argparse
import csv
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from common import Logger, load_config, sha256_file, write_csv

SCRIPT = "g00_tile_list"
PHASES = [(0, 0), (0, 1), (1, 0), (1, 1)]     # (행 홀짝, 열 홀짝)


def rc(tile_id):
    a, b = tile_id.split("_c")
    return int(a[1:]), int(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    g = cfg["g00"]
    log = Logger(cfg["paths"]["outputs_dir"], SCRIPT)
    log("config:", cfg["_config_path"])
    src = next((Path(p).expanduser() for p in g["source_tiles_csv"] if Path(p).expanduser().exists()), None)
    assert src is not None, f"격자 tiles.csv 를 찾을 수 없음: {g['source_tiles_csv']}"
    sha = sha256_file(src)
    assert sha.startswith(g["source_tiles_sha256_prefix"]), f"tiles.csv sha256 {sha[:12]} != {g['source_tiles_sha256_prefix']}"
    rows = list(csv.DictReader(open(src, encoding="utf-8-sig")))
    assert len(rows) == g["expected_grid_tiles"], len(rows)
    log(f"입력 {src} | 격자 타일 {len(rows)}개, sha256 {sha} (앞 12자리 일치)")
    assert g["select"] in ("nonoverlap_best_phase", "all_grid"), g["select"]
    drop0 = bool(g["drop_zero_mask"])
    by = {}
    for r in rows:
        by.setdefault(r["image_id"], []).append(r)
    assert len(by) == cfg["data"]["expected_images"]
    chosen, info = set(), []
    for iid in sorted(by):
        gr = by[iid]
        if g["select"] == "all_grid":
            cand = [r for r in gr if (int(r["mask_px"]) > 0 or not drop0)]
            for r in cand:
                chosen.add((r["image_id"], r["tile_id"]))
            info.append(dict(image_id=iid, phase="grid", n_tiles=len(cand), specimen_px_in_tiles=sum(int(r["mask_px"]) for r in cand)))
            continue
        scores = []
        for pr, pc in PHASES:
            cand = [r for r in gr if rc(r["tile_id"])[0] % 2 == pr and rc(r["tile_id"])[1] % 2 == pc and (int(r["mask_px"]) > 0 or not drop0)]
            scores.append((sum(int(r["mask_px"]) for r in cand), cand))
        best = max(range(4), key=lambda i: (scores[i][0], -i))
        for r in scores[best][1]:
            chosen.add((r["image_id"], r["tile_id"]))
        info.append(dict(image_id=iid, phase=f"{PHASES[best][0]}{PHASES[best][1]}", n_tiles=len(scores[best][1]), specimen_px_in_tiles=scores[best][0]))
    out_rows = []
    for r in sorted(rows, key=lambda r: (r["image_id"], r["tile_id"])):
        rr = dict(r)
        rr["selected_m02"] = r["selected"]
        rr["selected"] = "1" if (r["image_id"], r["tile_id"]) in chosen else "0"
        out_rows.append(rr)
    write_csv(f"{log.dir}/tiles.csv", out_rows)
    sel = [r for r in out_rows if r["selected"] == "1"]
    assert len(sel) == len(chosen)
    # ---- 검증 ----
    by_sel = {}
    for r in sel:
        by_sel.setdefault(r["image_id"], []).append(r)
    for iid, gr in by_sel.items():
        xy = [(int(x["x0"]), int(x["y0"])) for x in gr]
        for i in range(len(xy)):
            for j in range(i + 1, len(xy)):
                assert abs(xy[i][0] - xy[j][0]) >= 224 or abs(xy[i][1] - xy[j][1]) >= 224, f"타일이 겹침: {iid} {xy[i]} {xy[j]}"
        assert all(int(x["mask_px"]) > 0 for x in gr) or not drop0
    masks_dir = Path(cfg["paths"]["masks_dir"])
    tot = cov = 0
    spec = {}
    for iid in sorted(by):
        m = cv2.imdecode(np.fromfile(str(masks_dir / (iid.replace("/", "__") + ".png")), dtype=np.uint8), cv2.IMREAD_GRAYSCALE) > 127
        spec[iid] = int(m.sum())
    for r in info:
        r["specimen_px_total"] = spec[r["image_id"]]
        r["specimen_cov"] = round(r["specimen_px_in_tiles"] / spec[r["image_id"]], 4)
        tot += spec[r["image_id"]]; cov += r["specimen_px_in_tiles"]
    write_csv(f"{log.dir}/tile_selection.csv", info)
    write_csv(f"{log.dir}/selected_tile_ids.csv", [dict(image_id=x["image_id"], tile_id=x["tile_id"]) for x in sel])
    sha_ids = sha256_file(f"{log.dir}/selected_tile_ids.csv")
    cnt = Counter(int(x["label"]) for x in sel)
    nt = np.array([r["n_tiles"] for r in info]); cv = np.array([r["specimen_cov"] for r in info])
    nfull = sum(1 for x in sel if float(x["mask_frac"]) >= 0.999999)
    log(f"\n선택 타일 {len(sel)}개 (이미지당 {nt.min()}~{nt.max()}개, 중앙값 {int(np.median(nt))}), 클래스별 {dict(sorted(cnt.items()))}, 마스크 100% {nfull}개 / 일부 {len(sel) - nfull}개")
    if g["select"] == "all_grid":
        log("격자: stride 224 격자 전체에서 시편 픽셀 0인 타일만 제외 (겹치지 않는 격자이므로 부분 격자 선택 없음)")
    else:
        log(f"선택된 부분 격자(행 홀짝+열 홀짝): {dict(sorted(Counter(r['phase'] for r in info).items()))}")
    log(f"시편 픽셀이 선택 타일에 들어간 비율(겹침 없음): 전체 {cov / tot * 100:.2f}%, 이미지별 최소 {cv.min() * 100:.1f}%, 중앙값 {np.median(cv) * 100:.1f}%")
    log("검증 통과: 선택 타일끼리 겹치지 않음 (모든 쌍), 시편 픽셀 0인 타일 없음")
    assert sha_ids == g["reference_selected_ids_sha256"], f"선택 목록 sha256 {sha_ids} != 기준 {g['reference_selected_ids_sha256']}"
    log(f"선택 목록 sha256 {sha_ids} == 배경 증강 실험(b00)이 같은 규칙으로 고른 목록의 sha256 (같은 선택)")
    assert len(sel) == cfg["data"]["expected_tiles"], f"config data.expected_tiles={cfg['data']['expected_tiles']} 와 선택 타일 수 {len(sel)} 가 다름"
    log(f"저장: {log.dir}")


if __name__ == "__main__":
    main()
