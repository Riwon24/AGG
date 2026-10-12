"""s05: valid/선택 타일 수 분포와 클래스별 밝기/색 통계 (통계만 보고, 정규화는 적용하지 않는다).

명세 범위
[A] 이미지별/클래스별 valid 타일 수 분포 (+ 상한 적용 후 선택 타일 수)
[B] 클래스별 시편 영역(s02 마스크)과 선택 타일 영역의 평균 밝기/색 (RGB, HSV)  -> 밝기/색이 클래스의 지름길이 될 위험 확인용
추가 통계 (기술 통계, 예측/학습 아님)
[C-1] 분산설명비율 eta^2: 색 특징의 분산을 클래스 / 처리 조건(2단계 폴더) / 금속 종류(1단계 폴더)가 각각 얼마나 설명하는가
[C-2] 같은 처리 조건 안에서 클래스가 섞인 경우의 클래스별 평균
색은 OpenCV 기준(H 0-179, S/V 0-255), 픽셀 평균. H 는 파랑 대역(약 100~115)이라 선형 평균을 쓴다.
산출: image_stats.csv, class_counts_summary.csv, class_color_summary.csv, tile_level_summary.csv, variance_share.csv,
      within_sample_contrast.csv, counts_by_class.png, color_by_class.png, variance_share.png, log.txt
"""
import argparse
import csv
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sps

from common import Logger, PREPROC_DIR, load_config, safe_id, savefig_retry

SCRIPT = "s05_tile_stats"
FEATS = ["R", "G", "B", "H", "S", "V"]
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


def feat_means(img_bgr, hsv, region):
    px = img_bgr[region][:, ::-1].astype(np.float64)        # RGB
    ph = hsv[region].astype(np.float64)
    return dict(R=px[:, 0].mean(), G=px[:, 1].mean(), B=px[:, 2].mean(), H=ph[:, 0].mean(), S=ph[:, 1].mean(), V=ph[:, 2].mean(),
                V_std=ph[:, 2].std(), n_px=int(region.sum()))


def eta2(vals, groups):
    vals = np.asarray(vals, float)
    g = np.asarray(groups)
    gm = vals.mean()
    sst = ((vals - gm) ** 2).sum()
    ssb = sum((g == k).sum() * (vals[g == k].mean() - gm) ** 2 for k in set(g))
    return float(ssb / sst) if sst > 0 else 0.0


def pid_order(x):
    return tuple(int(p[1:]) for p in x.split("_"))


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", help="manifest_v1.csv (기본: outputs/s00b_meta_mag/manifest_v1.csv)")
    ap.add_argument("--input", help="이미지 루트 (기본: config paths.image_root)")
    ap.add_argument("--output", help="outputs 폴더 (기본: config paths.outputs_dir)")
    ap.add_argument("--config", default=str(PREPROC_DIR / "config.yaml"))
    a = ap.parse_args()

    cfg = load_config(a.config)
    root = Path(a.input or cfg["paths"]["image_root"])
    out = a.output or cfg["paths"]["outputs_dir"]
    manifest = Path(a.manifest or Path(out) / "s00b_meta_mag" / "manifest_v1.csv")
    use = cfg["experiment"]["use_mags"]
    W, H = cfg["naming"]["expected_resolution"]
    np.random.seed(int(cfg.get("seed", 0)))
    log = Logger(out, SCRIPT)
    o = Path(out) / SCRIPT
    s2d, s4d = Path(out) / "s02_specimen_mask", Path(out) / "s04_tile_grid"
    log("config:", cfg["_config_path"], "| 마스크 방법:", cfg["mask"]["method"], "| 정규화 없음 (통계만)")

    rows = [r for r in csv.DictReader(open(manifest, encoding="utf-8-sig")) if r["mag_final"] in use]
    counts = list(csv.DictReader(open(s4d / "tile_counts.csv", encoding="utf-8-sig")))
    tiles = defaultdict(list)    # (image_id, param) -> 선택된 타일
    for t in csv.DictReader(open(s4d / "tiles.csv", encoding="utf-8-sig")):
        if t["selected"] == "1":
            tiles[(t["image_id"], t["param_id"])].append(t)
    params = sorted({c["param_id"] for c in counts}, key=pid_order)
    labels = sorted({r["label"] for r in rows})
    log(f"대상 {len(rows)}장 | 라벨별 이미지: {dict(sorted(Counter(r['label'] for r in rows).items()))} | 조합 {params} | 상한 {cfg['tiling'].get('cap_per_image')}")

    # ================= [A] 타일 수 =================
    nvalid = {(c["image_id"], c["param_id"]): int(c["n_valid"]) for c in counts}
    nsel = {(c["image_id"], c["param_id"]): int(c["n_selected"]) for c in counts}
    log("\n[A] 클래스별 타일 수 분포 (이미지당)")
    crow = []
    for p in params:
        for kind, cnt in (("valid", nvalid), ("selected", nsel)):
            line, tot = [], sum(cnt[(r["image_id"], p)] for r in rows)
            for lab in labels:
                v = [cnt[(r["image_id"], p)] for r in rows if r["label"] == lab]
                crow.append(dict(param_id=p, kind=kind, label=lab, n_images=len(v), mean=round(statistics.mean(v), 2), sd=round(statistics.pstdev(v), 2),
                                 min=min(v), median=statistics.median(v), max=max(v), total=sum(v),
                                 image_share=round(len(v) / len(rows), 3), tile_share=round(sum(v) / tot, 3)))
                line.append(f"AGG{lab}: {statistics.mean(v):5.1f}±{statistics.pstdev(v):4.1f} [{min(v)}~{max(v)}] 이미지 {100*len(v)/len(rows):4.1f}% -> 타일 {100*sum(v)/tot:4.1f}% (합 {sum(v)})")
            log(f"    {p:<14} {kind:<8} " + " | ".join(line))
        kw = sps.kruskal(*[[nvalid[(r["image_id"], p)] for r in rows if r["label"] == lab] for lab in labels])
        log(f"    {p:<14} valid 타일 수의 클래스 간 Kruskal-Wallis p={float(kw.pvalue):.4f} (이미지 독립 가정, 참고용)")
    write_csv(o / "class_counts_summary.csv", crow)

    # ================= [B] 색/밝기 =================
    regions = ["mask"] + params
    img_rows = []
    tile_rows = defaultdict(list)    # (param, label) -> 타일별 특징
    for r in rows:
        img = cv2.imdecode(np.fromfile(str(root / r["path"]), dtype=np.uint8), cv2.IMREAD_COLOR)
        assert img.shape[:2] == (H, W)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        m = cv2.imdecode(np.fromfile(str(s2d / "masks" / f"{safe_id(r['image_id'])}.png"), dtype=np.uint8), cv2.IMREAD_GRAYSCALE) > 127
        row = dict(image_id=r["image_id"], label=int(r["label"]), treatment=r["treatment"], sample=r["sample"])
        for p in params:
            row[f"valid_{p}"], row[f"selected_{p}"] = nvalid[(r["image_id"], p)], nsel[(r["image_id"], p)]
        f = feat_means(img, hsv, m)
        row.update({f"mask_{k}": v for k, v in f.items()})
        for p in params:
            reg = np.zeros((H, W), bool)
            for t in tiles[(r["image_id"], p)]:
                x0, y0, w_, h_ = int(t["x0"]), int(t["y0"]), int(t["w"]), int(t["h"])
                reg[y0:y0 + h_, x0:x0 + w_] = True
                tile_rows[(p, r["label"])].append(feat_means(img[y0:y0 + h_, x0:x0 + w_], hsv[y0:y0 + h_, x0:x0 + w_], np.ones((h_, w_), bool)))
            f = feat_means(img, hsv, reg) if reg.any() else {k: float("nan") for k in FEATS + ["V_std", "n_px"]}
            row.update({f"sel_{p}_{k}": v for k, v in f.items()})
        img_rows.append(row)
    write_csv(o / "image_stats.csv", [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in rr.items()} for rr in img_rows])

    lab_arr = np.array([r["label"] for r in img_rows])
    samp = np.array([r["sample"] for r in img_rows]); trt = np.array([r["treatment"] for r in img_rows])
    log("\n[B] 클래스별 시편 영역(s02 마스크) 평균 색/밝기 — 이미지 단위 (평균±표준편차 [min~max])")
    ccrow = []
    for k in FEATS + ["V_std"]:
        vals = np.array([r[f"mask_{k}"] for r in img_rows])
        parts = []
        for lab in sorted(set(lab_arr)):
            v = vals[lab_arr == lab]
            ccrow.append(dict(region="mask", feature=k, label=lab, n=len(v), mean=round(v.mean(), 2), sd=round(v.std(), 2), min=round(v.min(), 2), max=round(v.max(), 2)))
            parts.append(f"AGG{lab} {v.mean():6.1f}±{v.std():4.1f} [{v.min():5.1f}~{v.max():5.1f}]")
        log(f"    {k:<6} " + " | ".join(parts))
    for p in params:
        log(f"\n    선택 타일 영역({p})의 이미지 단위 평균 (클래스별 평균±표준편차)")
        for k in FEATS:
            vals = np.array([r[f"sel_{p}_{k}"] for r in img_rows])
            parts = []
            for lab in sorted(set(lab_arr)):
                v = vals[lab_arr == lab]
                ccrow.append(dict(region=p, feature=k, label=lab, n=len(v), mean=round(v.mean(), 2), sd=round(v.std(), 2), min=round(v.min(), 2), max=round(v.max(), 2)))
                parts.append(f"AGG{lab} {v.mean():6.1f}±{v.std():4.1f}")
            log(f"      {k:<2} " + " | ".join(parts))
    write_csv(o / "class_color_summary.csv", ccrow)

    log("\n    타일 단위 (선택 타일 하나하나의 평균을 클래스별로 집계: 평균±표준편차, 타일 수)")
    trow = []
    for p in params:
        for k in FEATS:
            parts = []
            for lab in labels:
                v = np.array([t[k] for t in tile_rows[(p, lab)]])
                trow.append(dict(param_id=p, feature=k, label=lab, n_tiles=len(v), mean=round(v.mean(), 2), sd=round(v.std(), 2)))
                parts.append(f"AGG{lab} {v.mean():6.1f}±{v.std():4.1f} (n={len(v)})")
            if k in ("V", "S"):
                log(f"    {p} {k:<2} " + " | ".join(parts))
    write_csv(o / "tile_level_summary.csv", trow)

    # ================= [C] 추가 통계 (기술 통계) =================
    log("\n[C-1] 분산설명비율 eta^2 (시편 영역 평균, 이미지 단위; 0~1) — 추가 통계")
    log("      클래스와 처리 조건은 강하게 얽혀 있어서(처리 조건의 다수 라벨로 90% 일치, s07) 서로 분리되지 않는다. 금속 종류(6종)별 라벨 비율은 대체로 비슷하지만 BASE-LCSI 는 AGG0 가 70%로 높다(s07).")
    vs = []
    for k in FEATS:
        vals = [r[f"mask_{k}"] for r in img_rows]
        e = dict(feature=k, eta2_class=round(eta2(vals, lab_arr), 3), eta2_condition=round(eta2(vals, samp), 3), eta2_metal=round(eta2(vals, trt), 3),
                 kruskal_p_class=round(float(sps.kruskal(*[np.array(vals)[lab_arr == l] for l in sorted(set(lab_arr))]).pvalue), 4))
        vs.append(e)
        log(f"    {k:<2} 클래스 {e['eta2_class']:.3f} | 처리 조건 {e['eta2_condition']:.3f} | 금속 {e['eta2_metal']:.3f} | 클래스 간 Kruskal p={e['kruskal_p_class']}")
    write_csv(o / "variance_share.csv", vs)

    log("\n[C-2] 같은 처리 조건 안에서 클래스가 섞인 경우의 클래스별 평균 (각 칸이 이미지 1~5장이라 결론용 아님)")
    wsrow = []
    for s in sorted(set(samp)):
        idx = samp == s
        cl = sorted(set(lab_arr[idx]))
        if len(cl) < 2:
            continue
        log(f"    {s}: 클래스별 이미지 수 {dict(sorted(Counter(int(c) for c in lab_arr[idx]).items()))}")
        for k in ("V", "S"):
            vals = np.array([r[f"mask_{k}"] for r in img_rows])
            wsrow.append(dict(sample=s, feature=k, **{f"AGG{c}": round(float(vals[idx & (lab_arr == c)].mean()), 2) for c in cl}))
            log(f"        {k}: " + " | ".join(f"AGG{c} {vals[idx & (lab_arr == c)].mean():.1f}" for c in cl))
    if wsrow:
        keys = sorted({k for r in wsrow for k in r}, key=lambda x: (x not in ("sample", "feature"), x))
        with open(o / "within_sample_contrast.csv", "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=keys, restval="")
            w.writeheader(); w.writerows(wsrow)

    # ================= 그림 =================
    col = {"0": "#4c78a8", "1": "#f58518", "2": "#54a24b"}
    n = len(params); nc = min(3, n); nr = (n + nc - 1) // nc
    fig, axs = plt.subplots(nr, nc, figsize=(5.5 * nc, 4.2 * nr), squeeze=False)
    for ax, p in zip(axs.ravel(), params):
        for off, kind, cnt in ((-0.18, "valid", nvalid), (0.18, "선택", nsel)):
            data = [[cnt[(r["image_id"], p)] for r in rows if r["label"] == lab] for lab in labels]
            pos = [i + off for i in range(1, len(labels) + 1)]
            bp = ax.boxplot(data, positions=pos, widths=0.3, showfliers=False)
            for i, d in enumerate(data):
                ax.scatter(pos[i] + np.random.uniform(-0.06, 0.06, len(d)), d, s=10, color=col[labels[i]], alpha=0.7, marker="o" if kind == "valid" else "s")
            ax.plot([], [], marker="o" if kind == "valid" else "s", color="gray", ls="", label=kind)
        ax.set_xticks(range(1, len(labels) + 1)); ax.set_xticklabels([f"AGG{l}\n(n={sum(1 for r in rows if r['label'] == l)})" for l in labels])
        ax.set_title(p, fontsize=10); ax.set_ylabel("타일 수/이미지"); ax.set_ylim(bottom=0); ax.legend(fontsize=8)
    for ax in axs.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("클래스별 타일 수 분포 (10x): 왼쪽 valid, 오른쪽 상한 적용 후 선택"); fig.tight_layout(); savefig_retry(fig, o / "counts_by_class.png"); plt.close(fig)

    fig, axs = plt.subplots(2, 3, figsize=(13, 7))
    sam_list = sorted(set(samp)); cmap = plt.get_cmap("tab10")
    for ax, k in zip(axs.ravel(), FEATS):
        vals = np.array([r[f"mask_{k}"] for r in img_rows])
        ax.boxplot([vals[lab_arr == l] for l in sorted(set(lab_arr))], widths=0.5, showfliers=False)
        for i, l in enumerate(sorted(set(lab_arr)), 1):
            for j, s in enumerate(sam_list):
                v = vals[(lab_arr == l) & (samp == s)]
                ax.scatter(i + np.random.uniform(-0.15, 0.15, len(v)), v, s=14, color=cmap(j), alpha=0.85, label=s if i == 1 else None)
        ax.set_xticks(range(1, len(set(lab_arr)) + 1)); ax.set_xticklabels([f"AGG{l}" for l in sorted(set(lab_arr))]); ax.set_title(f"{k} (시편 영역 평균)")
    h, l_ = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l_, loc="lower center", ncol=5, fontsize=8, title="처리 조건(2단계 폴더)")
    fig.suptitle("클래스별 밝기/색 (점 색 = 처리 조건)"); fig.tight_layout(rect=(0, 0.08, 1, 0.95)); savefig_retry(fig, o / "color_by_class.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    xx = np.arange(len(FEATS)); bw = 0.27
    for i, (key, lab_) in enumerate((("eta2_class", "클래스"), ("eta2_condition", "처리 조건"), ("eta2_metal", "금속 종류"))):
        ax.bar(xx + (i - 1) * bw, [e[key] for e in vs], bw, label=lab_)
    ax.set_xticks(xx); ax.set_xticklabels(FEATS); ax.set_ylabel("분산설명비율 η²"); ax.set_ylim(0, 1); ax.legend()
    ax.set_title("색 특징 분산을 설명하는 그룹 (이미지 단위, 시편 영역 평균)"); fig.tight_layout(); savefig_retry(fig, o / "variance_share.png"); plt.close(fig)
    log(f"\n저장: {o}")


if __name__ == "__main__":
    main()
