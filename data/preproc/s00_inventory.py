"""s00: 이미지 인벤토리 + manifest_draft.csv (읽기 전용).

- 파일명/폴더에서 금속 종류(열 이름 treatment), 처리 조건(열 이름 sample), 배율, 라벨을 파싱한다. (용어는 common.py 참조)
- 규칙에 안 맞는 이름은 parse_flag=irregular 로 표시하고, 추정 배율(mag_guess)을 따로 둔다 (확정 아님).
- 원본 마스터 폴더와 대조해 누락/이름 차이를 보고한다 (원본은 읽기만 함).
- 이 환경은 pandas/skimage 가 정책으로 차단되어 표준 csv + numpy 계열만 사용한다.
"""
import argparse
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from common import Logger, PREPROC_DIR, list_images, load_config, parse_name

SCRIPT = "s00_inventory"
AGG_RE = re.compile(r"AGG\s*([012])", re.I)
MAG_PREFIX_RE = re.compile(r"^[xX](\d+(?:\.\d+)?)")
FIELDS = ["image_id", "path", "treatment", "sample", "group", "mag", "mag_guess", "mag_used", "rep",
          "label", "fixed_marker", "parse_flag", "width", "height", "mode", "format", "size_bytes"]


def guess_mag(stem, known):
    m = MAG_PREFIX_RE.match(stem)
    if not m:
        return None
    tok = m.group(1)
    for k in sorted(known, key=len, reverse=True):
        if tok.startswith(k):
            return k
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", help="이미지 루트 (기본: config paths.image_root)")
    ap.add_argument("--original", help="원본 마스터 폴더 (기본: config paths.original_root)")
    ap.add_argument("--output", help="outputs 폴더 (기본: config paths.outputs_dir)")
    ap.add_argument("--config", default=str(PREPROC_DIR / "config.yaml"))
    a = ap.parse_args()

    cfg = load_config(a.config)
    root = Path(a.input or cfg["paths"]["image_root"]).resolve()
    orig = Path(a.original or cfg["paths"]["original_root"]).resolve()
    out = a.output or cfg["paths"]["outputs_dir"]
    known = cfg["naming"]["known_mags"]
    exp_wh = tuple(cfg["naming"]["expected_resolution"])
    log = Logger(out, SCRIPT)
    log("config:", cfg["_config_path"], "| root:", root, "| original:", orig)

    # ---------- 1. 스캔 ----------
    rows = []
    for f in list_images(root, exclude_dirs=[PREPROC_DIR]):
        rel = f.relative_to(root)
        if len(rel.parts) != 3:
            log.warn("폴더 깊이가 3이 아님 (건너뜀):", rel.as_posix()); continue
        treatment, sample = rel.parts[0], rel.parts[1]
        p = parse_name(f.stem)
        flag, mag, rep, label, fixed = "ok", None, None, None, False
        if p is not None and p["mag"] in known:
            mag, rep, label, fixed = p["mag"], p["rep"], p["label"], p["fixed"]
        else:
            flag = "irregular"
            am = AGG_RE.search(f.stem)
            label = int(am.group(1)) if am else None
        mg = mag or guess_mag(f.stem, known)
        with Image.open(f) as im:
            w, h, mode, fmt = im.width, im.height, im.mode, im.format
        rows.append(dict(
            image_id=f"{treatment}/{sample}/{f.stem}", path=rel.as_posix(),
            treatment=treatment, sample=sample, group=f"{treatment}/{sample}",
            mag=mag or "", mag_guess="" if mag else (mg or ""), mag_used=mg or "",
            rep="" if rep is None else rep, label="" if label is None else label,
            fixed_marker=fixed, parse_flag=flag, width=w, height=h, mode=mode, format=fmt,
            size_bytes=f.stat().st_size))
    mp = Path(out) / SCRIPT / "manifest_draft.csv"
    with open(mp, "w", newline="", encoding="utf-8-sig") as fh:
        wr = csv.DictWriter(fh, fieldnames=FIELDS)
        wr.writeheader()
        wr.writerows(rows)
    log(f"\n[1] 총 이미지: {len(rows)}장 -> {mp}")
    log("    형식:", dict(Counter(r["format"] for r in rows)), "| 모드(채널):", dict(Counter(r["mode"] for r in rows)))
    log("    (라벨은 폴더명이 아니라 파일명의 AGG<n>에서만 파싱함)")

    # ---------- 2. 해상도 ----------
    bad_res = [r for r in rows if (r["width"], r["height"]) != exp_wh]
    log(f"\n[2] 해상도 {exp_wh}가 아닌 파일: {len(bad_res)}장")
    for r in bad_res:
        log("    ", r["path"], r["width"], r["height"])

    # ---------- 3. 파싱 모호 ----------
    irr = [r for r in rows if r["parse_flag"] == "irregular"]
    log(f"\n[3] 파일명 규칙에 안 맞는 파일(irregular): {len(irr)}장  (mag_guess는 추정, 확정 아님)")
    for r in irr:
        log(f"     {r['path']} | label={r['label']} | mag_guess={r['mag_guess']}")
    log(f"    라벨 파싱 실패: {sum(1 for r in rows if r['label'] == '')}장")

    # ---------- 4. 그룹당 배율 ----------
    groups = defaultdict(list)
    for r in rows:
        groups[r["group"]].append(r)
    log("\n[4] 묶음(금속/처리 조건)별 배율 개수 — 기대: 그룹 수 60, 그룹당 3장(7.5/10/12.5 각 1장)")
    log(f"    그룹 수: {len(groups)}")
    bad_groups = []
    for g in sorted(groups):
        sub = groups[g]
        c = Counter(r["mag_used"] for r in sub)
        if len(sub) != 3 or any(c.get(m, 0) != 1 for m in known):
            bad_groups.append(g)
            strict = Counter(r["mag"] for r in sub if r["mag"])
            log(f"     {g}: n={len(sub)} 배율(추정 포함)={dict(c)} 배율(엄격 파싱만)={dict(strict)}")
    log(f"    기대와 다른 그룹: {len(bad_groups)}개")
    log("    배율별 개수(추정 포함):", dict(Counter(r["mag_used"] for r in rows)),
        "| 엄격 파싱만:", dict(Counter(r["mag"] for r in rows if r["mag"])))

    # ---------- 5. 라벨 ----------
    def lc(rs):
        return dict(sorted(Counter(int(r["label"]) for r in rs if r["label"] != "").items()))
    use = cfg["experiment"]["use_mags"]
    r10 = [r for r in rows if r["mag_used"] in use]
    log("\n[5] 라벨별 개수 (전체):", lc(rows))
    log(f"    라벨별 개수 ({'/'.join(use)}x, 추정 포함 {len(r10)}장):", lc(r10))
    mixed = [g for g, s in groups.items() if len({r['label'] for r in s}) > 1]
    log("    같은 묶음(금속/처리 조건) 안에서 라벨이 다른 그룹:", mixed or "없음")
    by_sample = defaultdict(list)
    for r in rows:
        by_sample[r["sample"]].append(r)
    log("    처리 조건(2단계 폴더명)별 라벨:")
    for s in sorted(by_sample):
        log(f"      {s}: {lc(by_sample[s])}")

    # ---------- 6. 원본 대조 (읽기만) ----------
    log("\n[6] 원본 마스터 폴더 대조 (읽기만)")
    orows = []
    for f in list_images(orig, exclude_dirs=[root]):
        rel = f.relative_to(orig)
        orows.append(dict(path=rel.as_posix(), has_agg=bool(AGG_RE.search(f.stem)), depth=len(rel.parts)))
    n_agg = sum(r["has_agg"] for r in orows)
    log(f"    원본 이미지: {len(orows)}장 (AGG 라벨 있음 {n_agg}, 없음 {len(orows) - n_agg})")
    og = Counter("/".join(r["path"].split("/")[:2]) for r in orows if r["depth"] == 3)
    log(f"    원본 그룹 수(깊이 3): {len(og)}, 그룹당 장수 분포: {dict(Counter(og.values()))}")
    for g, n in sorted(og.items()):
        if n != 3:
            log(f"     원본 그룹 {g}: {n}장")
    mine = {r["path"] for r in rows}
    opaths = {r["path"] for r in orows}
    only_orig = [r["path"] for r in orows if r["has_agg"] and r["path"] not in mine]
    only_mine = [r["path"] for r in rows if r["path"] not in opaths]
    log(f"    원본(AGG 라벨 있음) 중 AGG_labeled_only에 같은 경로로 없는 파일: {len(only_orig)}장")
    log(f"    AGG_labeled_only 중 원본에 같은 경로로 없는 파일: {len(only_mine)}장 (이름이 바뀐 파일 후보)")
    for p in only_orig:
        log("     원본에만:", p)
    for p in only_mine:
        log("     복사본에만:", p)
    noagg = [r["path"] for r in orows if not r["has_agg"] and r["depth"] == 3]
    log(f"    원본에서 AGG 라벨이 없는 이미지: {len(noagg)}장")
    for p in noagg:
        log("     라벨 없음:", p)

    log("\n요약: 총", len(rows), "장 | irregular", len(irr), "| 해상도 이상", len(bad_res),
        "| 기대와 다른 그룹", len(bad_groups))


if __name__ == "__main__":
    main()
