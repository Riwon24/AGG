"""s06: 최종 결과(선택된 타일)를 사람이 볼 수 있는 이미지로 만든다 (시각화 전용, 데이터셋 저장 아님).

- tiles_overlay_AGG<클래스>_p<k>.png : 전체 이미지(10x 60장)에 선택 타일을 그려 넣은 contact sheet (페이지당 12장)
- pipeline_example_AGG<클래스>.png   : 클래스별 대표 1장의 단계별 모습 (원본 / 마스크+침식선 / 선택 타일)
- tile_samples_AGG<클래스>.png       : 모델이 보게 될 선택 타일 24개 샘플 (224px 원본 크기 그대로, 고정 seed)
- final_overview.png                  : 요약 (클래스별 이미지/타일 수, 이미지별 valid 타일 수와 상한, 파라미터)
원본 이미지는 읽기만 하며, 타일 샘플 그림은 시각화이고 학습용 타일 파일을 만들지 않는다.
"""
import argparse
import csv
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import s04_tile_grid as s4
from common import Logger, PREPROC_DIR, load_config, safe_id, savefig_retry

SCRIPT = "s06_make_figures"
PER_PAGE, COLS, TILE_W = 12, 4, 380
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


def imread(path, flag=cv2.IMREAD_COLOR):
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), flag)


def imwrite(path, img, retries=6):
    buf = cv2.imencode(".png", img)[1]
    for i in range(retries):
        try:
            buf.tofile(str(path)); return
        except OSError:
            if i == retries - 1:
                raise
            time.sleep(0.5 * (i + 1))


def titled(img, title, w, color=(255, 255, 255)):
    h0, w0 = img.shape[:2]
    t = cv2.resize(img, (w, int(h0 * w / w0)), interpolation=cv2.INTER_AREA)
    bar = np.zeros((18, w, 3), np.uint8)
    cv2.putText(bar, title, (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
    return np.vstack([bar, t])


def sheet_of(tiles, cols):
    th, tw = max(t.shape[0] for t in tiles), max(t.shape[1] for t in tiles)
    rws = (len(tiles) + cols - 1) // cols
    sheet = np.zeros((rws * th, cols * tw, 3), np.uint8)
    for i, t in enumerate(tiles):
        sheet[(i // cols) * th:(i // cols) * th + t.shape[0], (i % cols) * tw:(i % cols) * tw + t.shape[1]] = t
    return sheet


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
    use, tc = cfg["experiment"]["use_mags"], cfg["tiling"]
    ex = cfg["scalebar"]["exclude_box"]
    assert len(tc["stride"]) == 1 and len(tc["erosion_margin_px"]) == 1 and len(tc["tile_size"]) == 1, "s06 은 최종 조합 1개만 지원"
    p = f"t{tc['tile_size'][0]}_s{tc['stride'][0]}_m{tc['erosion_margin_px'][0]}"
    margin, cap = tc["erosion_margin_px"][0], tc.get("cap_per_image")
    seed = int(cfg.get("seed", 0))
    log = Logger(out, SCRIPT)
    o = Path(out) / SCRIPT
    s2d, s4d = Path(out) / "s02_specimen_mask" / "masks", Path(out) / "s04_tile_grid"
    log("config:", cfg["_config_path"], "| 최종 조합:", p, "| 상한:", cap, "| seed:", seed)

    rows = sorted([r for r in csv.DictReader(open(manifest, encoding="utf-8-sig")) if r["mag_final"] in use], key=lambda r: r["image_id"])
    meta = {r["image_id"]: r for r in rows}
    sel, allv = defaultdict(list), Counter()
    for t in csv.DictReader(open(s4d / "tiles.csv", encoding="utf-8-sig")):
        if t["param_id"] != p:
            continue
        if t["selected"] == "1":
            sel[t["image_id"]].append(t)
        if t["valid"] == "1":
            allv[t["image_id"]] += 1
    npool = {c["image_id"]: int(c["n_pool"]) for c in csv.DictReader(open(s4d / "tile_counts.csv", encoding="utf-8-sig")) if c["param_id"] == p}
    labels = sorted({r["label"] for r in rows})
    log(f"이미지 {len(rows)}장 | 선택 타일 합계 {sum(len(v) for v in sel.values())} | 이미지당 선택 수 {sorted(set(len(v) for v in sel.values()))}")
    col = {"0": (255, 150, 60), "1": (60, 150, 255), "2": (90, 200, 90)}

    # ---------- 1. 전체 overlay contact sheet ----------
    files = []
    for lab in labels:
        rr = [r for r in rows if r["label"] == lab]
        for pg in range(0, len(rr), PER_PAGE):
            tl = []
            for i, r in enumerate(rr[pg:pg + PER_PAGE], pg + 1):
                img = imread(root / r["path"])
                for t in sel[r["image_id"]]:
                    x0, y0, w, h = int(t["x0"]), int(t["y0"]), int(t["w"]), int(t["h"])
                    cv2.rectangle(img, (x0, y0), (x0 + w - 1, y0 + h - 1), (0, 220, 0), 3)
                cv2.rectangle(img, (ex[0], ex[1]), (ex[2] - 1, ex[3] - 1), (255, 0, 255), 4)
                tl.append(titled(img, f"{i:02d} {r['treatment']}/{r['sample']} 선택 {len(sel[r['image_id']])}", TILE_W))
            fn = f"tiles_overlay_AGG{lab}_p{pg // PER_PAGE + 1}.png"
            imwrite(o / fn, sheet_of(tl, COLS)); files.append(fn)

    # ---------- 2. 파이프라인 예시 (클래스별 면적 중앙값에 가까운 1장) ----------
    ar = {}
    for r in rows:
        ar[r["image_id"]] = float((imread(s2d / f"{safe_id(r['image_id'])}.png", cv2.IMREAD_GRAYSCALE) > 127).mean())
    for lab in labels:
        cand = [r for r in rows if r["label"] == lab]
        med = statistics.median(ar[r["image_id"]] for r in cand)
        r = min(cand, key=lambda r: abs(ar[r["image_id"]] - med))
        img = imread(root / r["path"])
        m = imread(s2d / f"{safe_id(r['image_id'])}.png", cv2.IMREAD_GRAYSCALE) > 127
        e = s4.erode(m, margin, bool(tc["erode_frame_as_boundary"]))
        a1 = img.copy()
        a2 = img.copy()
        ov = np.zeros_like(img); ov[m] = (255, 120, 0)
        a2 = cv2.addWeighted(a2, 0.65, ov, 0.35, 0)
        for mm, c in ((m, (0, 255, 0)), (e, (0, 255, 255))):
            cn, _ = cv2.findContours(mm.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(a2, cn, -1, c, 4)
        a3 = img.copy()
        for t in sel[r["image_id"]]:
            x0, y0, w, h = int(t["x0"]), int(t["y0"]), int(t["w"]), int(t["h"])
            cv2.rectangle(a3, (x0, y0), (x0 + w - 1, y0 + h - 1), (0, 220, 0), 3)
        for a_ in (a1, a2, a3):
            cv2.rectangle(a_, (ex[0], ex[1]), (ex[2] - 1, ex[3] - 1), (255, 0, 255), 4)
        w_ = 600
        pan = [titled(a1, f"1. 원본 ({r['image_id'][-34:]})", w_), titled(a2, f"2. 마스크 otsu_v(파랑 음영, 초록=경계) + 침식 {margin}px(노랑)", w_),
               titled(a3, f"3. 선택 타일 {len(sel[r['image_id']])}개 (겹치지 않는 부분 격자, valid {allv[r['image_id']]}개 중)", w_)]
        fn = f"pipeline_example_AGG{lab}.png"
        imwrite(o / fn, np.hstack(pan)); files.append(fn)

    # ---------- 3. 타일 샘플 (224 원본 크기 그대로) ----------
    rng = np.random.RandomState(seed)
    for lab in labels:
        pool = [t for r in rows if r["label"] == lab for t in sel[r["image_id"]]]
        pick = [pool[i] for i in sorted(rng.choice(len(pool), size=min(24, len(pool)), replace=False))]
        cache, tl = {}, []
        for t in pick:
            if t["image_id"] not in cache:
                cache[t["image_id"]] = imread(root / meta[t["image_id"]]["path"])
            x0, y0, w, h = int(t["x0"]), int(t["y0"]), int(t["w"]), int(t["h"])
            crop = cache[t["image_id"]][y0:y0 + h, x0:x0 + w].copy()
            assert crop.shape[:2] == (h, w) == (224, 224)
            bar = np.zeros((16, w, 3), np.uint8)
            m_ = meta[t["image_id"]]
            cv2.putText(bar, f"{m_['treatment']}/{m_['sample']}", (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
            tl.append(np.vstack([bar, crop]))
        fn = f"tile_samples_AGG{lab}.png"
        imwrite(o / fn, sheet_of(tl, 6)); files.append(fn)

    # ---------- 4. 요약 ----------
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.8), gridspec_kw=dict(width_ratios=[1, 1.3, 1.1]))
    ni = [sum(1 for r in rows if r["label"] == l) for l in labels]
    nt = [sum(len(sel[r["image_id"]]) for r in rows if r["label"] == l) for l in labels]
    x = np.arange(len(labels))
    div = max(1, round(sum(nt) / sum(ni)))   # 막대 높이를 이미지 수와 비슷하게 맞추는 배율 (이미지당 평균 타일 수)
    axs[0].bar(x - 0.2, ni, 0.4, label="이미지 수"); b = axs[0].bar(x + 0.2, [v / div for v in nt], 0.4, label=f"선택 타일 수 ÷ {div}")
    for xi, v in zip(x, nt):
        axs[0].text(xi + 0.2, v / div + 0.4, f"{v}", ha="center", fontsize=9)
    for xi, v in zip(x, ni):
        axs[0].text(xi - 0.2, v + 0.4, f"{v}", ha="center", fontsize=9)
    axs[0].set_xticks(x); axs[0].set_xticklabels([f"AGG{l}" for l in labels]); axs[0].set_title("클래스별 이미지 / 선택 타일"); axs[0].legend(fontsize=8)
    for i, lab in enumerate(labels):
        v = [npool[r["image_id"]] for r in rows if r["label"] == lab]
        axs[1].scatter(np.full(len(v), i) + rng.uniform(-0.15, 0.15, len(v)), v, s=14, alpha=0.8)
    if cap is not None:
        axs[1].axhline(cap, color="red", ls="--", lw=1); axs[1].text(len(labels) - 0.6, cap + 0.4, f"상한 {cap}", color="red", fontsize=9)
    axs[1].set_xticks(range(len(labels))); axs[1].set_xticklabels([f"AGG{l}" for l in labels]); axs[1].set_ylabel("선택 타일 수/이미지")
    axs[1].set_ylim(0, max(max(npool.values()), cap or 0) * 1.1); axs[1].set_title("이미지별 선택 타일 수 (겹치지 않는 부분 격자" + (f", 상한 {cap})" if cap is not None else ", 상한 없음)"))
    axs[2].axis("off")
    txt = (f"최종 조합 (겹치지 않는 타일 선택)\n\n타일 크기   {tc['tile_size'][0]} px (resize 없음)\nstride        {tc['stride'][0]} px\n침식 마진    {margin} px\n"
           f"선택 방식   {tc.get('selection_method')}\n이미지당 상한 {cap if cap is not None else '없음'}\n마스크        {cfg['mask']['method']} (전 이미지 동일, override 없음)\n"
           f"스케일바 제외 {ex}\n배율          {'/'.join(use)}x ({len(rows)}장)\n\n선택 타일 합계 {sum(nt)}\n"
           f"이미지당 선택 {sorted(set(len(v) for v in sel.values()))}")
    axs[2].text(0.02, 0.98, txt, va="top", fontsize=11, family="Malgun Gothic")
    fig.suptitle("전처리 최종 결과 요약"); fig.tight_layout(); savefig_retry(fig, o / "final_overview.png"); plt.close(fig)
    files.append("final_overview.png")

    log(f"\n생성된 그림 {len(files)}장:")
    for f in files:
        log("   ", f)
    log(f"\n클래스별 이미지/선택 타일: " + ", ".join(f"AGG{l}: {i}장/{t}개" for l, i, t in zip(labels, ni, nt)))
    log(f"저장: {o}")


if __name__ == "__main__":
    main()
