"""b02: 배경 증강 이미지를 만들어 검증한다 (이미지 60장 x 증강본 K개 x feather 후보). 증강 이미지는 저장하지 않는다(b03 이 같은 함수로 다시 만든다).

검증 (모든 이미지, 모든 증강본, 모든 feather 후보):
  1. 마스크 안쪽과 buffer 이내(alpha=0) 픽셀은 원본과 완전히 같다 (uint8 비교).
  2. 증강본의 시드는 (seed, 이미지 번호, k) 로 정해지고 다시 만들면 같다 (sha 비교, 이미지 3장 x k=1).
링 점검 (증강에만 생기는 두 번째 경계): 링 = 완충 띠 바깥 경계 부근 [buffer-1, buffer+feather+3] 에서의 평균 기울기 크기.
  ring_ratio = 증강본 링의 기울기 / 원본의 같은 픽셀 기울기, step = 링 바깥쪽 평균 밝기 변화(회색 단계), 비교용으로 시편 경계(d<=3)와 일반 배경의 원본 기울기.
  판정 제안값(config ring_ratio_threshold)을 넘으면 링이 보인다고 본다 (제안값이며 데이터에서 유도한 값이 아니다).
출력: ring_metrics.csv, fig_zoom_ring.png, fig_overview_k1.png, log.txt
"""
import argparse
import hashlib
import json

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from bg_common import (Logger, augment_image, dist_to_mask, grad_mag, inputs, load_config, read_mask, read_rgb, sha256_file, write_csv)  # noqa: E402

SCRIPT = "b02_make_aug_images"
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    bg = dict(cfg["bgaug"])
    log = Logger(cfg["paths"]["outputs_dir"], SCRIPT)
    log("config:", cfg["_config_path"])
    pj = json.loads((log.dir.parent / "b01_measure_background" / "aug_params.json").read_text(encoding="utf-8"))
    pool = pj["pool"]
    for k_, v in pj["params"].items():
        assert bg[k_] == v, f"config 의 {k_}={bg[k_]} 가 b01 측정 때 값 {v} 와 다름 (b01 다시 실행 필요)"
    imgs, _ = inputs(cfg)
    assert [p["image_id"] for p in pool] == [r["image_id"] for r in imgs]
    K, b, feathers = int(bg["n_aug"]), float(bg["buffer_px"]), list(bg["feather_candidates_px"])
    log(f"이미지 {len(imgs)}장, 증강본 K={K}, buffer {b:g}px, feather 후보 {feathers}px, seed {bg['seed']}")

    rows, changed = [], {f: [] for f in feathers}
    for i, r in enumerate(imgs):
        m, img = read_mask(r["mask"]), read_rgb(r["path"])
        d = dist_to_mask(m)
        g0 = grad_mag(img)
        edge_ref = float(g0[(d > 0) & (d <= 3)].mean())
        far = d > 30
        far[cfg["data"]["scalebar_exclude_box"][1]:, cfg["data"]["scalebar_exclude_box"][0]:] = False
        bg_ref = float(g0[far].mean())
        for f in feathers:
            for k in range(1, K + 1):
                aug, alpha, src = augment_image(img, d, pool, bg, i, k, f)
                same = alpha == 0
                assert same[m].all(), "마스크 안쪽 픽셀의 alpha 가 0 이 아님"
                assert np.array_equal(aug[same], img[same]), f"원본이어야 하는 픽셀이 바뀜: {r['image_id']} k={k} feather={f}"
                changed[f].append(float((alpha > 0).mean()))
                lo, hi = b - 1, b + max(f, 0) + 3
                ring = (d > lo) & (d <= hi)
                g1 = grad_mag(aug)
                ratio = float(g1[ring].mean() / g0[ring].mean())
                inside = (d > b - 6) & (d <= b - 2)
                outside = (d > b + f + 2) & (d <= b + f + 6)
                gray = lambda x: x.astype(np.float32).mean(2)
                step = float(abs(gray(aug)[outside].mean() - gray(img)[inside].mean())) if inside.any() and outside.any() else float("nan")
                rows.append(dict(image_id=r["image_id"], feather_px=f, k=k, src=src, ring_ratio=round(ratio, 3), step_gray=round(step, 2),
                                 edge_grad_orig=round(edge_ref, 2), bg_grad_orig=round(bg_ref, 2), ring_grad_aug=round(float(g1[ring].mean()), 2),
                                 ring_grad_orig=round(float(g0[ring].mean()), 2), changed_frac=round(float((alpha > 0).mean()), 4)))
        print(f"  {i + 1}/{len(imgs)} {r['image_id']}")
    log("\n[1] 검증 통과: 모든 이미지, 모든 증강본, 모든 feather 후보에서 마스크 안쪽과 buffer 이내 픽셀이 원본과 완전히 같음")
    # 재현성
    for i in (0, 20, 40):
        r = imgs[i]
        img, m = read_rgb(r["path"]), read_mask(r["mask"])
        d = dist_to_mask(m)
        h1 = hashlib.sha256(augment_image(img, d, pool, bg, i, 1, feathers[-1])[0].tobytes()).hexdigest()
        h2 = hashlib.sha256(augment_image(img, d, pool, bg, i, 1, feathers[-1])[0].tobytes()).hexdigest()
        assert h1 == h2, "같은 시드인데 증강본이 다름"
    log("[2] 재현성: 이미지 3장 x k=1 을 두 번 만들어 sha256 이 같음")
    write_csv(f"{log.dir}/ring_metrics.csv", rows)

    thr = float(bg["ring_ratio_threshold"])
    log(f"\n[3] 링 점검 (제안 기준: ring_ratio > {thr:g} 이면 링이 보인다고 봄)")
    for f in feathers:
        rr = [x for x in rows if x["feather_px"] == f]
        ratio = np.array([x["ring_ratio"] for x in rr]); step = np.array([x["step_gray"] for x in rr])
        log(f"  feather {f:>2}px: 변경 픽셀 비율 평균 {np.mean(changed[f]) * 100:.1f}% | ring_ratio 평균 {ratio.mean():.3f}, 중앙값 {np.median(ratio):.3f}, 최대 {ratio.max():.3f} | "
            f"기준 초과 {int((ratio > thr).sum())}/{len(ratio)}개 | 밝기 단차(step) 중앙값 {np.nanmedian(step):.2f}, 90% {np.nanpercentile(step, 90):.2f}, 최대 {np.nanmax(step):.2f}")
    er = np.array([x["edge_grad_orig"] for x in rows if x["feather_px"] == feathers[0] and x["k"] == 1])
    bgm = np.array([x["bg_grad_orig"] for x in rows if x["feather_px"] == feathers[0] and x["k"] == 1])
    log(f"  참고(원본 기울기 크기): 시편 경계(d<=3px) 평균 {er.mean():.1f}, 일반 배경(d>30px) 평균 {bgm.mean():.1f}")
    hfm = np.median([p["hf"] for p in pool])
    log(f"  참고: 배경 미세 노이즈 표준편차 중앙값 {hfm:.1f} (밝기 단차 해석 기준)")

    # ---- 확인용 그림: 클래스별로 배경 비율이 중앙값에 가까운 이미지 1장 ----
    bgfrac = {}
    for i, r in enumerate(imgs):
        bgfrac[i] = 1 - float(read_mask(r["mask"]).mean())
    picks = []
    for c in (0, 1, 2):
        idx = [i for i, r in enumerate(imgs) if r["label"] == c]
        med = np.median([bgfrac[i] for i in idx])
        picks.append(min(idx, key=lambda i: abs(bgfrac[i] - med)))
    fig, axes = plt.subplots(3, 3, figsize=(13, 13))
    fig2, axes2 = plt.subplots(3, 3, figsize=(15, 10))
    for row, i in enumerate(picks):
        r = imgs[i]
        img, m = read_rgb(r["path"]), read_mask(r["mask"])
        d = dist_to_mask(m)
        augs = [None] + [augment_image(img, d, pool, bg, i, 1, f)[0] for f in feathers[:2]]
        ys, xs = np.where((d > b) & (d <= b + 1))
        cx, cy = 912, 684
        j = int(np.argmin((xs - cx) ** 2 + (ys - cy) ** 2))
        px, py = int(xs[j]), int(ys[j])
        x0, y0 = int(np.clip(px - 128, 0, img.shape[1] - 256)), int(np.clip(py - 128, 0, img.shape[0] - 256))
        titles = ["원본", f"증강 k=1, feather {feathers[0]}px", f"증강 k=1, feather {feathers[1]}px"]
        for col in range(3):
            im = img if augs[col] is None else augs[col]
            zoom = cv2.resize(im[y0:y0 + 256, x0:x0 + 256], None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
            axes[row, col].imshow(zoom); axes[row, col].axis("off")
            axes[row, col].set_title(f"AGG{r['label']} {r['image_id'].rsplit('/', 1)[0]}\n{titles[col]} (확대 x2)", fontsize=8)
            small = cv2.resize(im, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
            axes2[row, col].imshow(small); axes2[row, col].axis("off")
            axes2[row, col].set_title(f"AGG{r['label']} {r['image_id'].rsplit('/', 1)[0]} | {titles[col]}", fontsize=8)
    fig.suptitle(f"링 확인: 완충 띠 경계(buffer {b:g}px) 부근 확대", fontsize=11); fig.tight_layout(); fig.savefig(log.dir / "fig_zoom_ring.png", dpi=110); plt.close(fig)
    fig2.suptitle("증강 전체 모습 (k=1): 클래스별 배경 비율이 중앙값에 가까운 이미지 1장씩", fontsize=11); fig2.tight_layout(); fig2.savefig(log.dir / "fig_overview_k1.png", dpi=110); plt.close(fig2)
    log(f"\n그림 저장: fig_zoom_ring.png, fig_overview_k1.png (이미지: " + ", ".join(imgs[i]["image_id"] for i in picks) + ")")
    log(f"ring_metrics.csv sha256 {sha256_file(log.dir / 'ring_metrics.csv')}")


if __name__ == "__main__":
    main()
