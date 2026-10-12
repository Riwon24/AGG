"""run_all: 전처리 전체(s00 -> s07)를 순서대로 실행하고, 원본 불변과 재현성을 검증한다.

실행:  .venv\\Scripts\\python.exe run_all.py [--config config.yaml]
- 단계: s00_inventory, s00b_meta_mag, s01_scalebar_locate, s02_specimen_mask, s03_mask_qc, s04_tile_grid, s05_tile_stats, s06_make_figures, s07_label_crosstab
  (결정 과정에서 쓴 진단 스크립트는 이 파이프라인에 포함하지 않는다)
- 검증 1: AGG_labeled_only 의 이미지(jpg) sha256 을 실행 전/후로 비교 (원본 읽기 전용 확인) + 원본 마스터 폴더의 같은 경로 파일과 비교
- 검증 2: s02, s04 를 한 번 더 실행해서 마스크 PNG 와 tiles.csv 의 sha256 이 같은지 비교 (같은 입력 -> 같은 출력)
로그: outputs/run_all/log.txt
"""
import argparse
import hashlib
import subprocess
import sys
import time
from pathlib import Path

from common import Logger, PREPROC_DIR, list_images, load_config

STEPS = ["s00_inventory", "s00b_meta_mag", "s01_scalebar_locate", "s02_specimen_mask", "s03_mask_qc",
         "s04_tile_grid", "s05_tile_stats", "s06_make_figures", "s07_label_crosstab"]


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_images(root, exclude):
    return {f.relative_to(root).as_posix(): sha(f) for f in list_images(root, exclude_dirs=exclude)}


def hash_dir(d):
    return {f.name: sha(f) for f in sorted(Path(d).glob("*.png"))}


def run_step(name, config, log):
    t0 = time.time()
    r = subprocess.run([sys.executable, str(PREPROC_DIR / f"{name}.py"), "--config", config], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", cwd=str(PREPROC_DIR))
    dt = time.time() - t0
    log(f"  {name:<22} exit={r.returncode} {dt:5.1f}s")
    if r.returncode != 0:
        log("  --- stderr (끝 15줄) ---")
        for line in r.stderr.strip().splitlines()[-15:]:
            log("   ", line)
        raise SystemExit(f"{name} 실패")
    return dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(PREPROC_DIR / "config.yaml"))
    ap.add_argument("--output", help="outputs 폴더 (기본: config paths.outputs_dir)")
    a = ap.parse_args()
    cfg = load_config(a.config)
    out = a.output or cfg["paths"]["outputs_dir"]
    root, orig = Path(cfg["paths"]["image_root"]), Path(cfg["paths"]["original_root"])
    log = Logger(out, "run_all")
    log("config:", cfg["_config_path"], "| python:", sys.executable)
    log(f"조합: tile={cfg['tiling']['tile_size']} stride={cfg['tiling']['stride']} margin={cfg['tiling']['erosion_margin_px']} "
        f"선택={cfg['tiling'].get('selection_method', 'random_seeded')} cap={cfg['tiling'].get('cap_per_image')} | 마스크={cfg['mask']['method']} | seed={cfg.get('seed')}")

    log("\n[0] 이미지 해시(실행 전)")
    before = hash_images(root, [PREPROC_DIR])
    log(f"    AGG_labeled_only 이미지 {len(before)}장")
    same = diff = missing = 0
    for rel, h in before.items():
        op = orig / rel
        if not op.exists():
            missing += 1
        elif sha(op) == h:
            same += 1
        else:
            diff += 1
    log(f"    원본 마스터 폴더의 같은 경로 파일과 비교: 동일 {same}, 다름 {diff}, 원본에 없음 {missing}")

    log("\n[1] 단계 실행")
    total = 0.0
    for s in STEPS:
        total += run_step(s, a.config, log)
    log(f"    전체 소요 {total:.0f}s")

    log("\n[2] 원본 불변 확인 (실행 후 해시)")
    after = hash_images(root, [PREPROC_DIR])
    changed = [k for k in before if after.get(k) != before[k]] + [k for k in after if k not in before]
    log(f"    이미지 {len(after)}장, 변경/추가/삭제된 파일: {len(changed)}장 {'(없음: 원본 불변 확인됨)' if not changed else changed[:5]}")
    assert not changed, "원본 이미지가 바뀌었음"

    log("\n[3] 재현성 확인: s02, s04 를 한 번 더 실행해 결과 해시 비교")
    mdir = Path(out) / "s02_specimen_mask" / "masks"
    m1, t1 = hash_dir(mdir), sha(Path(out) / "s04_tile_grid" / "tiles.csv")
    run_step("s02_specimen_mask", a.config, log)
    run_step("s04_tile_grid", a.config, log)
    m2, t2 = hash_dir(mdir), sha(Path(out) / "s04_tile_grid" / "tiles.csv")
    log(f"    마스크 PNG {len(m1)}장: {'모두 동일' if m1 == m2 else '다름!'} | tiles.csv sha256: {'동일' if t1 == t2 else '다름!'} ({t1[:12]}…)")
    assert m1 == m2 and t1 == t2, "같은 입력인데 결과가 다름"
    log("\n완료: 전 단계 성공, 원본 불변, 재현성 확인")


if __name__ == "__main__":
    main()
