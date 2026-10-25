"""c08: 사용자가 PPT 를 보고 직접 정한 "AGG 가 있는 타일" 목록(AGG1 이미지 14장)을 CSV 로 기록하고 검증한다.

입력: c06 의 agg_tile_labels_template.csv (14장의 모든 타일과 aggsim, has_agg 빈 칸), 사용자가 준 목록 (아래 MANUAL).
- 목록의 번호는 그림 파일 이름 앞의 번호(NN)이고, c06/c07 과 같은 순서(금속, 처리 조건, 이미지 이름 정렬)로 이미지에 대응시킨다.
- "경계 부근" 이라고 적은 3건은 사용자가 이름을 적은 타일을 has_agg = 1 로, 같이 적은 이웃 타일은 neighbor_tile 열에만 기록하고 has_agg 는 비워 둔다 (이웃에도 걸쳐 있는지는 사용자가 정하지 않았다).
검증: 모든 타일 번호가 그 이미지의 분석 타일에 존재하는지.
출력: outputs/c08_agg1_manual_labels/agg_tile_labels_filled.csv (590행, has_agg 채움), agg_tiles_list.csv (AGG 타일 목록, aggsim 과 이미지 안 순위), log.txt
"""
import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"                                  # 이 폴더의 결과 (c03 -> c04, c05, c06 -> c08 -> c09 가 서로의 출력을 읽는다)
GRID = next((p for p in (HERE.parent / "exp_grid_aug" / "outputs", HERE.parent / "DINOv2_grid_augmentation" / "outputs") if p.exists()), HERE.parent / "DINOv2_grid_augmentation" / "outputs")
#                                                       # 입력: DINOv2_grid_augmentation 의 출력 (읽기 전용; 로컬 작업 폴더 구조 또는 저장소 구조 중 있는 쪽)
D = OUT / "c08_agg1_manual_labels"
D.mkdir(parents=True, exist_ok=True)
lines = []


def log(m):
    print(m); lines.append(m)


# (번호, [AGG 타일], 경계 이웃 {타일: 이웃})  사용자 목록 그대로
MANUAL = {
    1: (["r01_c01"], {"r01_c01": "r02_c01"}), 2: (["r02_c01", "r02_c05"], {}), 3: (["r04_c03"], {}),
    4: (["r05_c02"], {"r05_c02": "r05_c03"}), 5: (["r01_c02"], {}), 6: (["r01_c05"], {}),
    7: (["r05_c03", "r04_c05"], {}), 8: (["r04_c01"], {}), 9: (["r00_c05"], {}), 10: (["r04_c03"], {}),
    11: (["r00_c03"], {}), 12: (["r05_c04"], {}), 13: (["r02_c01"], {}), 14: (["r04_c06", "r04_c04"], {"r04_c04": "r04_c03"}),
}
rd = lambda p: list(csv.DictReader(open(p, encoding="utf-8-sig")))
cond = sorted(rd(GRID / "e00_make_splits/conditions.csv"), key=lambda r: (r["metal"], r["condition"], r["image_id"]))
cond = [r for r in cond if int(r["label"]) == 1]
assert len(cond) == 14
num_img = {n: c["image_id"] for n, c in enumerate(cond, 1)}
tmpl = rd(OUT / "c06_aggsim_agg1_heatmaps/agg_tile_labels_template.csv")
by = {}
for r in tmpl:
    by.setdefault(r["image_id"], {})[r["tile_id"]] = r
ok = True
agg_rows = []
for n, (tl, nb) in MANUAL.items():
    iid = num_img[n]
    for t in tl:
        if t not in by[iid]:
            log(f"[오류] {n:02d} {iid}: 타일 {t} 가 분석 타일에 없음"); ok = False
    for t, nt in nb.items():
        if nt not in by[iid]:
            log(f"[오류] {n:02d} {iid}: 이웃 타일 {nt} 가 분석 타일에 없음"); ok = False
assert ok, "타일 번호 검증 실패"
for r in tmpl:
    r["neighbor_tile"] = ""
    r["has_agg"] = ""
for n, (tl, nb) in MANUAL.items():
    iid = num_img[n]
    sv = sorted(by[iid].values(), key=lambda r: -float(r["aggsim"]))
    rank = {r["tile_id"]: k for k, r in enumerate(sv, 1)}
    for t in tl:
        by[iid][t]["has_agg"] = "1"
        by[iid][t]["neighbor_tile"] = nb.get(t, "")
        v = float(by[iid][t]["aggsim"])
        agg_rows.append(dict(no=n, image_id=iid, metal=by[iid][t]["metal"], condition=by[iid][t]["condition"], tile_id=t, neighbor_tile=nb.get(t, ""), aggsim=v, rank_in_image=rank[t], n_tiles=len(sv),
                             image_mean_aggsim=round(sum(float(x["aggsim"]) for x in sv) / len(sv), 4)))
with open(D / "agg_tile_labels_filled.csv", "w", encoding="utf-8-sig", newline="") as fh:
    cols = ["image_id", "metal", "condition", "tile_id", "x0", "y0", "aggsim", "has_agg", "neighbor_tile"]
    w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows([{k: r[k] for k in cols} for r in tmpl])
with open(D / "agg_tiles_list.csv", "w", encoding="utf-8-sig", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(agg_rows[0])); w.writeheader(); w.writerows(agg_rows)
n_agg = sum(1 for r in tmpl if r["has_agg"] == "1")
log(f"이미지 14장, 타일 {len(tmpl)}개 중 AGG 타일 {n_agg}개 (경계 이웃 표기 {sum(1 for r in agg_rows if r['neighbor_tile'])}건, 이웃 타일은 has_agg 를 채우지 않음). 모든 타일 번호가 존재함을 확인.")
log("\nAGG 타일 목록 (aggsim 은 10 반복 평균 test 값, 순위는 그 이미지의 타일 중 높은 순):")
for r in agg_rows:
    log(f"  {r['no']:02d} {r['metal']:<10} {r['condition']:<12} {r['tile_id']}" + (f" (이웃 {r['neighbor_tile']})" if r["neighbor_tile"] else "") + f"  aggsim {r['aggsim']:.3f}  이미지 안 순위 {r['rank_in_image']}/{r['n_tiles']}  (이미지 평균 {r['image_mean_aggsim']:.3f})")
(D / "log.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
