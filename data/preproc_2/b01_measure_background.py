"""b01: 이미지 60장의 배경을 측정해 증강에 쓸 분포(평균색, 저주파 변동, 중간 대역 질감, 미세 노이즈와 채널 공분산)를 기록한다.

- 배경 = 마스크 밖이고 경계에서 buffer+margin(14px) 보다 먼 픽셀, 스케일바 구역 제외.
- 라벨은 쓰지 않는다 (측정값은 60장 전체를 한 풀로 모아 둔다). 증강 때 이 풀에서 무작위로 한 이미지의 값 세트를 뽑는다.
- 사람이 값을 손으로 조정하지 않는다. 출력: aug_params.json, bg_stats.csv, log.txt
"""
import argparse
import json

import numpy as np

from bg_common import (Logger, bg_region, dist_to_mask, inputs, load_config, measure_background, read_mask, read_rgb, sha256_file, write_csv)

SCRIPT = "b01_measure_background"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    bg = dict(cfg["bgaug"])
    log = Logger(cfg["paths"]["outputs_dir"], SCRIPT)
    log("config:", cfg["_config_path"])
    imgs, _ = inputs(cfg)
    box = cfg["data"]["scalebar_exclude_box"]
    pool, rows = [], []
    for i, r in enumerate(imgs):
        m, img = read_mask(r["mask"]), read_rgb(r["path"])
        d = dist_to_mask(m)
        w = bg_region(m, d, bg["buffer_px"], bg["measure_margin_px"], box)
        ms = measure_background(img, w, bg)
        mean, lf, hf, n = ms["mean"], ms["lf"], ms["hf"], ms["n"]
        if n < bg["min_bg_px"]:
            log.warn(f"{r['image_id']}: 측정 가능한 배경 픽셀 {n}개 (기준 {bg['min_bg_px']})")
        pool.append(dict(idx=i, image_id=r["image_id"], mean=[round(float(x), 3) for x in mean], lf=[round(float(x), 3) for x in lf], hf=[round(float(x), 3) for x in hf],
                         hf_cov=[[round(float(x), 4) for x in row] for row in ms["hf_cov"]], mid=[round(float(x), 3) for x in ms["mid"]],
                         hf_corr_min=round(float(np.min(ms["hf_cov"] / np.sqrt(np.outer(np.diag(ms["hf_cov"]), np.diag(ms["hf_cov"]))))), 3),
                         mid_corr_min=round(float(ms["mid_corr"].min()), 3), n_bg_px=n))
        rows.append(dict(idx=i, image_id=r["image_id"], n_bg_px=n, **{f"mean_{c}": round(float(v), 2) for c, v in zip("RGB", mean)},
                         **{f"lf_{c}": round(float(v), 3) for c, v in zip("RGB", lf)}, **{f"hf_{c}": round(float(v), 3) for c, v in zip("RGB", hf)},
                         **{f"mid_{c}": round(float(v), 3) for c, v in zip("RGB", ms["mid"])}, hf_corr_min=pool[-1]["hf_corr_min"], mid_corr_min=pool[-1]["mid_corr_min"]))
    write_csv(f"{log.dir}/bg_stats.csv", rows)
    A = {k: np.array([p[k] for p in pool]) for k in ("mean", "lf", "mid", "hf")}
    log(f"\n이미지 {len(pool)}장 배경 측정 (RGB 0~255, 60장 전체 요약; 라벨별 요약은 하지 않음)")
    for k, name in (("mean", "평균색"), ("lf", "저주파 변동(σ=40px 흐린 값의 표준편차)"), ("mid", "중간 대역 질감(σ2 흐림 - σ8 흐림의 표준편차)"), ("hf", "미세 노이즈(원본 - σ2 흐림의 표준편차)")):
        log(f"  {name:<40} 채널별 중앙값 {np.round(np.median(A[k], 0), 2).tolist()} | 최소 {np.round(A[k].min(0), 2).tolist()} | 최대 {np.round(A[k].max(0), 2).tolist()}")
    log(f"  채널 간 상관(최소값) 미세 노이즈: 중앙값 {np.median([p['hf_corr_min'] for p in pool]):.3f}, 범위 [{min(p['hf_corr_min'] for p in pool):.3f}, {max(p['hf_corr_min'] for p in pool):.3f}] | 중간 대역 질감: 중앙값 {np.median([p['mid_corr_min'] for p in pool]):.3f}")
    npx = np.array([p["n_bg_px"] for p in pool])
    log(f"  측정에 쓴 배경 픽셀 수: 최소 {npx.min()}, 중앙값 {int(np.median(npx))}, 최대 {npx.max()}")
    params = dict(note="b01 측정값 풀. 증강은 이 풀에서 한 이미지의 (mean, lf, mid, hf, hf_cov) 세트를 무작위로 뽑는다. 라벨 미사용.", params={k: bg[k] for k in
                  ("seed", "buffer_px", "measure_margin_px", "hf_blur_sigma_px", "mid_hi_sigma_px", "lowfreq_sigma_px", "lowfreq_downscale")}, pool=pool)
    (log.dir / "aug_params.json").write_text(json.dumps(params, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"\n저장: {log.dir} | aug_params.json sha256 {sha256_file(log.dir / 'aug_params.json')}")


if __name__ == "__main__":
    main()
