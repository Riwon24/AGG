"""s03: 마스크 QC (사람이 눈으로 확인하는 단계용 자료 생성).

- 클래스별 contact sheet (페이지당 12장): 윤곽선 overlay, 타일 제목에 금속/처리 조건, 면적 비율, 의심 사유.
- 의심 목록 CSV (qc_flags.csv 전체 60장 + review_list.csv 의심만):
    우선순위 A: 면적 이상치 / 여러 덩어리 / 방법 간 불일치(otsu_v vs otsu_sv) / 스케일바 제외 구역 겹침
    우선순위 B: 프레임에 잘림(마스크가 이미지 경계에 닿음)만 해당
  프레임 잘림은 변(위/아래/왼/오른)마다 경계 픽셀 중 마스크 비율도 기록한다 (안쪽 마진 결정 참고).
- 침식 마진 후보(config tiling.erosion_margin_px)를 샘플 이미지에 그려 보여 준다.
  침식은 이미지 프레임을 시편 경계로 보지 않는다 (프레임 밖은 마스크 안으로 간주; 타일이 프레임 안이어야 한다는 조건은 s04 가 따로 확인).
- 이 스크립트는 판정을 바꾸지 않는다. 실패 이미지는 overrides/ 로 보정한다 (overrides/README.txt 참조).
산출: contact_<class>_p<k>.png, evidence_margin_preview.png, qc_flags.csv, review_list.csv, log.txt
"""
import argparse
import csv
import statistics
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from common import Logger, PREPROC_DIR, load_config, safe_id

SCRIPT = "s03_mask_qc"
PER_PAGE, COLS, TILE_W = 12, 4, 380

README = """overrides/ 사용법 (s02_specimen_mask.py 가 읽는다; 키 = safe_id = image_id 의 '/' 를 '__' 로 바꾼 것)

1) 개별 임계값 보정:  overrides/mask_<safe_id>.json
   {"method": "manual", "s": 40, "v": 70}            # S>=s & V>=v
   {"method": "otsu_v", "open_px": 41, "close_px": 31}  # 방법은 유지하고 모폴로지만 변경
2) 수동 마스크:       overrides/mask_<safe_id>.png    # 흰색(>127)=시편, 원본과 같은 1824x1368
   (png 가 있으면 json 보다 우선한다)

적용 후 s02 -> s03 을 다시 실행한다. s02 의 mask_stats.csv 의 mask_source 열에 override 여부가 남는다.
파일명은 outputs/s03_mask_qc/review_list.csv 의 safe_id 열을 그대로 쓰면 된다.
"""


def imread(path, flag=cv2.IMREAD_COLOR):
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), flag)


def imwrite(path, img, retries=6):
    """PNG 저장. 같은 파일을 덮어쓸 때 간헐적으로 OSError(22)가 나서(원인 미확인: 뷰어/백신 잠금 추정) 재시도한다."""
    buf = cv2.imencode(".png", img)[1]
    for i in range(retries):
        try:
            buf.tofile(str(path)); return
        except OSError:
            if i == retries - 1:
                raise
            time.sleep(0.5 * (i + 1))


def tile(img, title, color):
    h, w = img.shape[:2]
    t = cv2.resize(img, (TILE_W, int(h * TILE_W / w)), interpolation=cv2.INTER_AREA)
    bar = np.zeros((18, TILE_W, 3), np.uint8)
    cv2.putText(bar, title, (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return np.vstack([bar, t])


def sheet_of(tiles, cols):
    th, tw = tiles[0].shape[:2]
    rws = (len(tiles) + cols - 1) // cols
    sheet = np.zeros((rws * th, cols * tw, 3), np.uint8)
    for i, t in enumerate(tiles):
        sheet[(i // cols) * th:(i // cols + 1) * th, (i % cols) * tw:(i % cols + 1) * tw] = t
    return sheet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", help="이미지 루트 (기본: config paths.image_root)")
    ap.add_argument("--stats", help="s02 mask_stats.csv (기본: outputs/s02_specimen_mask/mask_stats.csv)")
    ap.add_argument("--output", help="outputs 폴더 (기본: config paths.outputs_dir)")
    ap.add_argument("--config", default=str(PREPROC_DIR / "config.yaml"))
    a = ap.parse_args()

    cfg = load_config(a.config)
    root = Path(a.input or cfg["paths"]["image_root"])
    out = a.output or cfg["paths"]["outputs_dir"]
    s02 = Path(out) / "s02_specimen_mask"
    stats_path = Path(a.stats or s02 / "mask_stats.csv")
    mc = cfg["mask"]
    margins = cfg["tiling"]["erosion_margin_px"]
    W, H = cfg["naming"]["expected_resolution"]
    ex = cfg["scalebar"]["exclude_box"]
    log = Logger(out, SCRIPT)
    o = Path(out) / SCRIPT
    log("config:", cfg["_config_path"], "| stats:", stats_path, "| 침식 마진 후보:", margins)
    rows = list(csv.DictReader(open(stats_path, encoding="utf-8-sig")))
    log(f"대상 {len(rows)}장")
    ovd = Path(cfg["paths"]["overrides_dir"]); ovd.mkdir(exist_ok=True)
    (ovd / "README.txt").write_text(README, encoding="utf-8")

    # ---------- 변별 경계 접촉 비율 + 플래그 ----------
    ar = np.array([float(r["area_ratio"]) for r in rows])
    q1, q3 = np.percentile(ar, [25, 75]); k = float(mc["area_ratio_outlier_iqr"])
    lo, hi = q1 - k * (q3 - q1), q3 + k * (q3 - q1)
    log(f"\n[1] 면적 비율 이상치 범위(IQR x{k}): [{lo:.3f}, {hi:.3f}] (Q1={q1:.3f}, Q3={q3:.3f})")
    for r in rows:
        sid = safe_id(r["image_id"]); r["safe_id"] = sid
        m = imread(s02 / "masks" / f"{sid}.png", cv2.IMREAD_GRAYSCALE) > 127
        assert m.shape == (H, W), f"마스크 크기 이상: {sid}"
        r["frac_top"], r["frac_bottom"] = round(float(m[0].mean()), 3), round(float(m[-1].mean()), 3)
        r["frac_left"], r["frac_right"] = round(float(m[:, 0].mean()), 3), round(float(m[:, -1].mean()), 3)
        reasons = []
        if not lo <= float(r["area_ratio"]) <= hi:
            reasons.append("area_outlier")
        if int(r["n_components"]) > 1 and float(r["second_ratio"]) >= 0.05:
            reasons.append("multi_blob")
        if float(r["iou_otsu_sv_vs_default"]) < float(mc["min_iou_agree"]):
            reasons.append(f"method_disagree(iou_sv={r['iou_otsu_sv_vs_default']})")
        if int(r["mask_px_in_exclude"]) > 0:
            reasons.append(f"overlap_scalebar_zone({r['mask_px_in_exclude']}px)")
        a_reasons = list(reasons)
        if r["touches_border"] == "True":
            sides = [n for n, c in (("T", "touch_top"), ("B", "touch_bottom"), ("L", "touch_left"), ("R", "touch_right")) if r[c] == "True"]
            reasons.append("frame_cut(" + "".join(sides) + ")")
        r["reasons"] = ";".join(reasons)
        r["priority"] = "A" if a_reasons else ("B" if reasons else "")

    cnt = Counter(r["priority"] for r in rows)
    log(f"\n[2] 우선순위: A(눈으로 먼저 확인)={cnt['A']}장, B(프레임 잘림만)={cnt['B']}장, 문제 없음={cnt['']}장")
    for r in rows:
        if r["priority"] == "A":
            log(f"    A {r['path']} (AGG{r['label']}): {r['reasons']}")
    fc = [r for r in rows if "frame_cut" in r["reasons"]]
    log(f"\n[3] 프레임 잘림 {len(fc)}장 — 변별 경계 픽셀 중 마스크 비율 (평균/최대)")
    for side in ("top", "bottom", "left", "right"):
        v = [float(r[f"frac_{side}"]) for r in rows]
        log(f"    {side:<6} 닿은 장수={sum(1 for x in v if x > 0)} | 평균={statistics.mean(v):.3f} | 최대={max(v):.3f}")
    full = [r for r in rows if max(float(r['frac_top']), float(r['frac_bottom'])) > 0.5]
    log(f"    위/아래 경계의 50% 넘게를 마스크가 덮는 이미지: {len(full)}장 (시편이 세로로 프레임을 꽉 채움)")

    for fn, rr in (("qc_flags.csv", rows), ("review_list.csv", [r for r in rows if r["priority"]])):
        keys = ["safe_id", "image_id", "label", "priority", "reasons", "area_ratio", "n_components", "second_ratio",
                "frac_top", "frac_bottom", "frac_left", "frac_right", "iou_otsu_sv_vs_default", "mask_px_in_exclude", "mask_source"]
        with open(o / fn, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
            w.writeheader(); w.writerows(sorted(rr, key=lambda r: (r["priority"] or "Z", r["label"], r["image_id"])))

    # ---------- 클래스별 contact sheet ----------
    col = {"A": (0, 0, 255), "B": (0, 165, 255), "": (255, 255, 255)}
    log("\n[4] contact sheet (제목: 번호 금속/처리 조건 면적 [A/B]; 빨강=A, 주황=B, 흰색=문제 없음)")
    for lab in sorted({r["label"] for r in rows}):
        rr = sorted([r for r in rows if r["label"] == lab], key=lambda r: r["image_id"])
        for p in range(0, len(rr), PER_PAGE):
            page = rr[p:p + PER_PAGE]
            tl = []
            for i, r in enumerate(page, start=p + 1):
                img = imread(s02 / "overlays" / f"{r['safe_id']}.png")
                parts = r["image_id"].split("/")
                tl.append(tile(img, f"{i:02d} {parts[0]}/{parts[1]} a={float(r['area_ratio']):.2f} {r['priority']}", col[r["priority"]]))
            fn = f"contact_AGG{lab}_p{p // PER_PAGE + 1}.png"
            imwrite(o / fn, sheet_of(tl, COLS))
            log(f"    {fn}: {len(page)}장 (번호 {p + 1}~{p + len(page)})")

    # ---------- 침식 마진 미리보기 (클래스별 2장: 면적 중앙값에 가까운 것, 프레임 잘림이 가장 심한 것) ----------
    tl = []
    for lab in sorted({r["label"] for r in rows}):
        rr = [r for r in rows if r["label"] == lab]
        med = statistics.median(float(r["area_ratio"]) for r in rr)
        picks = [min(rr, key=lambda r: abs(float(r["area_ratio"]) - med)),
                 max(rr, key=lambda r: float(r["frac_top"]) + float(r["frac_bottom"]))]
        for r in picks:
            img = imread(root / r["path"]); m = (imread(s02 / "masks" / f"{r['safe_id']}.png", cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
            vis = img.copy()
            for mg, c in zip(margins, [(0, 255, 0), (0, 255, 255), (0, 0, 255)]):
                e = m if mg == 0 else cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * mg + 1,) * 2),
                                                borderType=cv2.BORDER_CONSTANT, borderValue=1)
                cnts, _ = cv2.findContours(e, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis, cnts, -1, c, 3)
            cv2.rectangle(vis, (ex[0], ex[1]), (ex[2] - 1, ex[3] - 1), (255, 0, 255), 3)
            tl.append(tile(vis, f"AGG{lab} {r['image_id'].split('/')[0]}/{r['image_id'].split('/')[1]} 마진 {margins} = 초록/노랑/빨강", (255, 255, 255)))
    imwrite(o / "evidence_margin_preview.png", sheet_of(tl, 2))
    names_ = ["초록", "노랑", "빨강"]
    log(f"\n[5] 침식 마진 미리보기: evidence_margin_preview.png ({', '.join(f'{names_[i]}={m}px' for i, m in enumerate(margins))}, 자홍=스케일바 제외 구역)")
    log("    최종 마진은 사람이 위 그림과 contact sheet 를 보고 정한다. 지금은 단정하지 않음.")
    log(f"\n저장: {o} | 보정 방법: {ovd / 'README.txt'}")


if __name__ == "__main__":
    main()
