"""m01_masks: 최종 시편 마스크를 만든다 (10x 60장). preproc 는 읽기 전용, 원본 이미지는 읽기만 한다.

마스크 = 점수 > 0 -> 최대 연결 영역 -> 구멍 메우기.
점수 = A + lambda_blue*(B-1) + lambda_prox*(P-1)  (감점 전용; B, P 는 0~1 이라 점수는 A 보다 커질 수 없다)
  A = tanh((V - Otsu 임계값)/sigma_v)  밝기 | B = 시편 색상(H) 가우시안 x 채도 문  파랑 | P = exp(-max(d - d0, 0)/tau)  hue_sv 핵심부와의 근접
이미지별 변형 선택 규칙 (모든 이미지에 동일, 수동 보정 없음):
    otsu_v 마스크 안쪽 V(명도) 중앙값 >= final_mask.v_med_threshold  ->  plateau  (핵심부에서 d0 까지 감점 없음: 밝고 파란기 약한 시편의 흰 띠 보존)
    그 미만                                                          ->  baseline
산출:
    masks/<safe_id>.png            흰색(255) = 시편, 원본과 같은 크기 (확인용 PNG 와 별개; 이후 단계가 읽을 마스크)
    selection.csv                  이미지별 선택된 변형, V 중앙값, 면적, otsu_v 대비 IoU
    check_01.png ... check_10.png  확인용: 한 장에 6개(3열). 마스크 밖은 어둡게, 윤곽선 빨강, 스케일바 제외 구역 노랑
"""
import argparse
import csv
import statistics
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy import ndimage as ndi

ML_DIR = Path(__file__).resolve().parent
SCRIPT = "m01_masks"


def load_ml_config(path):
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg["paths"] = {k: str((path.parent / v).resolve()) for k, v in cfg["paths"].items()}
    cfg["_config_path"] = str(path)
    return cfg


def iou(a, b):
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ML_DIR / "config.yaml"))
    ap.add_argument("--output", help="outputs 폴더 (기본: config paths.outputs_dir)")
    a = ap.parse_args()

    cfg = load_ml_config(a.config)
    pre_cfg_path = Path(cfg["preproc"]["config"])
    sys.path.insert(0, str(pre_cfg_path.parent))
    from common import Logger, imwrite_png, load_config, safe_id
    import s03_mask_qc as s03
    from s02_specimen_mask import imread, imwrite, postprocess, raw_mask

    pre = load_config(pre_cfg_path)
    base_p, fm = cfg["score_map"], cfg["final_mask"]
    s03.TILE_W = int(fm["tile_w"])
    root, pre_out = Path(pre["paths"]["image_root"]), Path(pre["paths"]["outputs_dir"])
    out = a.output or cfg["paths"]["outputs_dir"]
    mc = pre["mask"]
    W, H = pre["naming"]["expected_resolution"]
    ex = pre["scalebar"]["exclude_box"]
    thr, per_file, cols = float(fm["v_med_threshold"]), int(fm["per_file"]), int(fm["cols"])
    log = Logger(out, SCRIPT)
    o = Path(out) / SCRIPT
    (o / "masks").mkdir(parents=True, exist_ok=True)
    log("config:", cfg["_config_path"], "| preproc config:", pre["_config_path"])
    log(f"선택 규칙: otsu_v 마스크 안쪽 V 중앙값 >= {thr:g} -> plateau, 아니면 baseline")
    prm = lambda ov: {**base_p, **ov}
    variants = {"baseline": prm(fm["variants"]["baseline"]), "plateau": prm(fm["variants"]["plateau"])}
    log("변형 파라미터:", variants)
    assert variants["baseline"].get("penalty_only") and variants["plateau"].get("penalty_only"), "감점 전용 변형이어야 함"

    def make_mask(p, hsv, h0, t_otsu, d_core):
        h, sat, v = (hsv[..., k].astype(np.float32) for k in range(3))
        A = np.tanh((v - t_otsu) / p["sigma_v"])
        gate = np.clip((sat - p["s_lo"]) / (p["s_hi"] - p["s_lo"]), 0, 1)
        B = np.exp(-((h - h0) ** 2) / (2 * p["sigma_h"] ** 2)) * gate
        if p.get("blue_smooth_px", 0) > 0:
            B = cv2.GaussianBlur(B, (0, 0), float(p["blue_smooth_px"]))
        P = np.exp(-np.maximum(d_core - float(p.get("prox_plateau_px", 0)), 0) / p["tau_px"])
        m = (A + p["lambda_blue"] * (B - 1) + p["lambda_prox"] * (P - 1)) > 0
        n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
        assert n > 1, "마스크가 비었음"
        m = lab == (1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])))
        return ndi.binary_fill_holes(m)

    stats = list(csv.DictReader(open(pre_out / "s02_specimen_mask" / "mask_stats.csv", encoding="utf-8-sig")))
    log(f"대상 {len(stats)}장")
    rows, tiles = [], []
    for n_, s in enumerate(stats, 1):
        iid, sid = s["image_id"], safe_id(s["image_id"])
        base = cv2.imdecode(np.fromfile(str(pre_out / "s02_specimen_mask" / "masks" / f"{sid}.png"), dtype=np.uint8), cv2.IMREAD_GRAYSCALE) > 127
        img = imread(root / s["path"])
        assert img is not None and img.shape[:2] == (H, W), f"읽기 실패/해상도 이상: {s['path']}"
        hsv = cv2.cvtColor(cv2.GaussianBlur(img, (int(mc["blur_ksize"]) | 1,) * 2, 0), cv2.COLOR_BGR2HSV)
        t_otsu, _ = cv2.threshold(hsv[..., 2], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        core, _, _ = postprocess(raw_mask(hsv, "hue_sv", mc["hue"])[0], mc["close_px"], mc["open_px"])
        h0 = float(np.median(hsv[..., 0][core]))
        d_core = cv2.distanceTransform((~core).astype(np.uint8), cv2.DIST_L2, 5)
        v_med = float(np.median(hsv[..., 2][base].astype(np.float32)))
        choice = "plateau" if v_med >= thr else "baseline"
        m = make_mask(variants[choice], hsv, h0, float(t_otsu), d_core)
        imwrite_png(o / "masks" / f"{sid}.png", (m * 255).astype(np.uint8))
        rows.append(dict(num=n_, image_id=iid, label=s["label"], variant=choice, v_med=round(v_med, 1), area=round(float(m.mean()), 4),
                         area_otsu=round(float(base.mean()), 4), iou_vs_otsu=round(iou(m, base), 3), mask_px_in_scalebar_zone=int(m[ex[1]:ex[3], ex[0]:ex[2]].sum()),
                         touches_border=bool(m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any())))

        vis = img.copy()
        vis[~m] = (vis[~m] * 0.35).astype(np.uint8)
        cn, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cn, -1, (0, 0, 255), 4)
        cv2.rectangle(vis, (ex[0], ex[1]), (ex[2] - 1, ex[3] - 1), (0, 255, 255), 3)
        vis = cv2.resize(vis, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        p_ = iid.split("/")
        tiles.append(s03.tile(vis, f"{n_:02d} {p_[0]}/{p_[1]} AGG{s['label']} {choice} Vmed={v_med:.0f}", (255, 255, 255)))

    with open(o / "selection.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    n_files = (len(tiles) + per_file - 1) // per_file
    for k in range(n_files):
        chunk = tiles[k * per_file:(k + 1) * per_file]
        sheet = s03.sheet_of(chunk, cols)
        imwrite(o / f"check_{k + 1:02d}.png", sheet)
    assert len(list((o / "masks").glob("*.png"))) == len(stats), "마스크 파일 수가 이미지 수와 다름"

    cnt = Counter(r["variant"] for r in rows)
    log(f"\n[1] 마스크 PNG {len(stats)}장 저장: {o / 'masks'}")
    log(f"    확인용 PNG {n_files}개 (한 장에 최대 {per_file}개, {cols}열): check_01.png ~ check_{n_files:02d}.png")
    log(f"\n[2] 변형 선택: {dict(cnt)}")
    for v in ("plateau", "baseline"):
        vm = [r["v_med"] for r in rows if r["variant"] == v]
        if vm:
            log(f"    {v:<8} {len(vm)}장, V 중앙값 범위 {min(vm):.0f}~{max(vm):.0f}")
    near = sorted(rows, key=lambda r: abs(r["v_med"] - thr))[:5]
    log(f"    기준 {thr:g} 에 가장 가까운 5장: " + ", ".join(f"{r['image_id'].rsplit('/', 1)[0]}({r['v_med']}->{r['variant']})" for r in near))
    iv = [r["iou_vs_otsu"] for r in rows]
    log(f"\n[3] otsu_v 대비 IoU: median={statistics.median(iv):.3f} min={min(iv):.3f} | 0.99 미만 {sum(1 for x in iv if x < 0.99)}장: "
        + ", ".join(f"{r['image_id'].rsplit('/', 1)[0]}({r['iou_vs_otsu']})" for r in sorted(rows, key=lambda r: r['iou_vs_otsu'])[:8] if r["iou_vs_otsu"] < 0.99))
    log(f"    마스크가 스케일바 제외 구역과 겹치는 이미지: {sum(1 for r in rows if r['mask_px_in_scalebar_zone'] > 0)}장 (s02 의 otsu_v 기준: {sum(1 for s in stats if int(s['mask_px_in_exclude']) > 0)}장)")
    log(f"    마스크가 이미지 경계에 닿는 이미지(프레임에 잘린 시편): {sum(1 for r in rows if r['touches_border'])}장 (otsu_v 기준: {sum(1 for s in stats if s['touches_border'] == 'True')}장)")
    log(f"\n저장: {o}")


if __name__ == "__main__":
    main()
