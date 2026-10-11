"""s00b: Leica 메타데이터(.xlif)로 배율을 확인하고, 비정상 파일명의 배율을 판정한다 (읽기 전용).

- 이미지마다 원본 폴더의 leicametadata/<파일명(AGG 표기 제외)>.xlif 를 찾아 시야 가로 길이(FOV, mm)와
  Magnification/SameZoom 값을 읽는다.  px/mm = 가로 픽셀수 / FOV(mm).
- 정상 이름 파일로 배율별 FOV 분포를 만들어 메타데이터 방법이 맞는지 검증한다.
- 비정상 이름 파일은 FOV 가 가장 가까운 배율로 판정하되, 차이가 config naming.meta_tol_rel 이내이고
  이름에서 추정한 배율과도 일치할 때만 확정한다. 아니면 unresolved/conflict 로 남긴다.
- '#번호' 유무와 오타(#!, 101, 7.51 등)는 무시한다 (사용자 지시). 파일명은 바꾸지 않는다.
- 산출: meta_mag.csv, manifest_v1.csv (s00 manifest + mag_final/mag_source/px_per_mm)
"""
import argparse
import csv
import re
import statistics
import xml.etree.ElementTree as ET
from pathlib import Path

from common import Logger, PREPROC_DIR, load_config

SCRIPT = "s00b_meta_mag"
AGG_TAIL_RE = re.compile(r"\s*-\s*AGG\s*[012]\s*$", re.I)


def read_xlif(path):
    root = ET.parse(path).getroot()
    fov_mm = n_px = None
    for d in root.iter("DimensionDescription"):
        if d.get("DimID") == "1":
            fov_mm = float(d.get("Length")) * 1000.0
            n_px = int(d.get("NumberOfElements"))
            break
    mag = zoom = None
    for el in root.iter():
        if mag is None and el.get("Magnification") is not None:
            mag = float(el.get("Magnification"))
        if zoom is None and el.get("SameZoom") is not None:
            zoom = float(el.get("SameZoom"))
        if mag is not None and zoom is not None:
            break
    return dict(fov_x_mm=fov_mm, n_px_x=n_px, magnification=mag, same_zoom=zoom,
                element_name=next((e.get("Name") for e in root.iter("Element")), None))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", help="s00 manifest_draft.csv (기본: outputs/s00_inventory/manifest_draft.csv)")
    ap.add_argument("--original", help="원본 마스터 폴더 (기본: config paths.original_root)")
    ap.add_argument("--output", help="outputs 폴더 (기본: config paths.outputs_dir)")
    ap.add_argument("--config", default=str(PREPROC_DIR / "config.yaml"))
    a = ap.parse_args()

    cfg = load_config(a.config)
    out = a.output or cfg["paths"]["outputs_dir"]
    orig = Path(a.original or cfg["paths"]["original_root"])
    manifest = Path(a.manifest or Path(out) / "s00_inventory" / "manifest_draft.csv")
    known = cfg["naming"]["known_mags"]
    tol = float(cfg["naming"]["meta_tol_rel"])
    log = Logger(out, SCRIPT)
    log("config:", cfg["_config_path"], "| manifest:", manifest, "| original:", orig, "| tol_rel:", tol)

    rows = list(csv.DictReader(open(manifest, encoding="utf-8-sig")))
    log(f"manifest rows: {len(rows)}")

    # ---------- 1. xlif 읽기 ----------
    for r in rows:
        p = orig / Path(r["path"])
        stem = AGG_TAIL_RE.sub("", p.stem).strip()
        cands = [p.parent / "leicametadata" / f"{stem}.xlif"]
        x = next((c for c in cands if c.exists()), None)
        r["xlif_found"] = bool(x)
        r["xlif_path"] = str(x.relative_to(orig)).replace("\\", "/") if x else ""
        meta = read_xlif(x) if x else dict(fov_x_mm=None, n_px_x=None, magnification=None, same_zoom=None, element_name=None)
        r.update(meta)
        r["px_per_mm"] = (meta["n_px_x"] / meta["fov_x_mm"]) if meta["fov_x_mm"] else None
    missing = [r for r in rows if not r["xlif_found"]]
    log(f"\n[1] xlif 매칭 실패: {len(missing)}장")
    for r in missing:
        log("    ", r["path"])

    # ---------- 2. 정상 이름 파일로 배율별 FOV 분포 ----------
    log("\n[2] 정상 이름 파일 기준 배율별 시야 가로 길이(FOV, mm) 분포")
    ref = {}
    for m in known:
        v = [r["fov_x_mm"] for r in rows if r["parse_flag"] == "ok" and r["mag"] == m and r["fov_x_mm"]]
        ref[m] = statistics.median(v)
        log(f"    {m}x: n={len(v)} median={ref[m]:.3f} min={min(v):.3f} max={max(v):.3f} "
            f"(범위 {100*(max(v)-min(v))/ref[m]:.2f}%) | px/mm median={rows[0]['n_px_x']/ref[m]:.1f}")
    ks = sorted(known, key=float)
    for i in range(len(ks) - 1):
        gap = abs(ref[ks[i]] - ref[ks[i + 1]]) / ref[ks[i + 1]]
        log(f"    배율 간격 {ks[i]}x↔{ks[i+1]}x: 중앙값 차이 {100*gap:.1f}%")

    # ---------- 3. 정상 파일 검증: 메타 판정 vs 이름 ----------
    def nearest(fov):
        d = sorted(((abs(fov - ref[m]) / ref[m], m) for m in known))
        return d[0][1], d[0][0], d[1][0]

    wrong = 0
    for r in rows:
        r["meta_check"] = "no_xlif"
        if r["fov_x_mm"]:
            m, d, _ = nearest(r["fov_x_mm"])
            r["meta_check"] = "agree" if m == (r["mag"] or r["mag_guess"]) else "DISAGREE"
            if r["parse_flag"] == "ok" and m != r["mag"]:
                wrong += 1
                log(f"    [불일치] 정상 이름인데 메타 판정이 다름: {r['path']} 이름={r['mag']} 메타={m}")
    log(f"\n[3] 정상 이름 파일 중 메타 판정이 이름과 다른 파일: {wrong}장 (0이면 메타 방법이 신뢰할 만함)")

    # ---------- 4. 비정상 이름 판정 ----------
    log("\n[4] 비정상 이름 파일 판정")
    n_ok = 0
    for r in rows:
        if r["parse_flag"] == "ok":
            r["mag_final"], r["mag_source"], r["rel_diff_pct"] = r["mag"], "name", ""
            continue
        if not r["fov_x_mm"]:
            r["mag_final"], r["mag_source"], r["rel_diff_pct"] = "", "unresolved(no_xlif)", ""
            log(f"    {r['path']}: xlif 없음 -> unresolved (이름 추정 {r['mag_guess']})")
            continue
        m, d, d2 = nearest(r["fov_x_mm"])
        r["rel_diff_pct"] = f"{100*d:.2f}"
        name_guess = r["mag_guess"]
        if d <= tol and d < d2 / 2 and (not name_guess or name_guess == m):
            r["mag_final"], r["mag_source"] = m, ("meta+name_agree" if name_guess == m else "meta_only")
            n_ok += 1
        else:
            r["mag_final"] = ""
            r["mag_source"] = "conflict" if (name_guess and name_guess != m) else "unresolved(diff>tol)"
        log(f"    {r['path']}: 이름추정={name_guess} 메타판정={m} 차이={100*d:.2f}% "
            f"(다음 후보까지 {100*d2:.1f}%) -> {r['mag_final'] or '-'} [{r['mag_source']}]")
    n_irr = sum(1 for r in rows if r["parse_flag"] != "ok")
    log(f"    확정: {n_ok}/{n_irr}장")

    # ---------- 4b. 사용자 확인 override ----------
    ov_path = Path(cfg["paths"]["overrides_dir"]) / "mag_overrides.csv"
    if ov_path.exists():
        ov = {r["path"]: r for r in csv.DictReader(open(ov_path, encoding="utf-8-sig"))}
        by_path = {r["path"]: r for r in rows}
        log(f"\n[4b] 사용자 확인 override {len(ov)}건 ({ov_path})")
        for p, o in ov.items():
            assert p in by_path, f"override 대상이 manifest에 없음: {p}"
            assert o["mag_final"] in known, f"알 수 없는 배율: {o['mag_final']}"
            r = by_path[p]
            log(f"    {p}: {r['mag_final'] or '-'}[{r['mag_source']}] -> {o['mag_final']} [user_override] (meta_check={r['meta_check']})")
            r["mag_final"], r["mag_source"] = o["mag_final"], "user_override"

    # ---------- 5. 저장 ----------
    keys = ["image_id", "path", "xlif_found", "xlif_path", "element_name", "fov_x_mm", "px_per_mm",
            "magnification", "same_zoom", "parse_flag", "mag", "mag_guess", "mag_final", "mag_source", "meta_check", "rel_diff_pct"]
    o = Path(out) / SCRIPT
    with open(o / "meta_mag.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    base_keys = list(csv.DictReader(open(manifest, encoding="utf-8-sig")).fieldnames)
    full = base_keys + ["mag_final", "mag_source", "meta_check", "fov_x_mm", "px_per_mm"]
    with open(o / "manifest_v1.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=full, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    use = cfg["experiment"]["use_mags"]
    sel = [r for r in rows if r["mag_final"] in use]
    lab = {}
    for r in sel:
        lab[int(r["label"])] = lab.get(int(r["label"]), 0) + 1
    log(f"\n[5] 저장: {o/'meta_mag.csv'}, {o/'manifest_v1.csv'}")
    log(f"    1차 실험 대상({'/'.join(use)}x): {len(sel)}장, 라벨별 {dict(sorted(lab.items()))}")
    gc = {}
    for r in rows:
        if r["mag_final"]:
            gc.setdefault(r["group"], {}).setdefault(r["mag_final"], 0)
            gc[r["group"]][r["mag_final"]] += 1
    log("    10x가 그룹당 1장이 아닌 그룹:",
        [g for g in sorted({r['group'] for r in rows}) if gc.get(g, {}).get(use[0], 0) != 1] or "없음")


if __name__ == "__main__":
    main()
