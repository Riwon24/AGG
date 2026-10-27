"""jv_tile_scores_all_runs: JEVision 6가지 실행의 타일별 점수를 타일 하나당 한 줄로 합친 CSV 를 만든다 (표준 라이브러리만 사용).

실행 (모두 DINOv2_grid_augmentation 의 분석 타일 2,591개, 타일 하나를 이미지 1장으로 보냄):
  v1   yes/no 확률(noul), run_tiles.py 프롬프트           outputs/jv01_binary_agg2_grid/input/tiles_grid_v1.jsonl
  v2   yes/no 확률, 설명문을 사용자가 정한 새 AGG 정의로     outputs/jv04_grid_prompt_v2/input/tiles_grid_v2.jsonl
  v3   yes/no 확률, 설명문과 flat 질문 문구를 모두 새로      outputs/jv05_grid_prompt_v3/input/tiles_grid_v3.jsonl
  jv06 선택형(choice) + 선택지 순서 교차, run_tiles.py 프롬프트   outputs/jv06_tiles_choice/input/tiles_grid_choice.jsonl
  jv07 jv06 + 설명문에 제외 항목 보강                       outputs/jv07_tiles_choice_v2/input/tiles_grid_choice_v2.jsonl
  jv08 jv06 + 예시 이미지 3장 함께 전송                      outputs/jv08_tiles_fewshot/input/tiles_grid_fewshot.jsonl
v1~v3 의 jsonl 은 옛 타일 번호(stride 112 격자)로 저장돼 있어서 (image_id, x0, y0) 로 맞춘다. 타일 번호(tile_id)는 g00 의 현재 번호(stride 224 격자, r행_c열)를 쓴다.
열: image_id, label(AGG0/1/2), metal, condition, tile_id, x0, y0, mask_frac(시편 픽셀 비율), tile_type(안쪽 = 마스크 100% / 가장자리),
    has_agg(직접 표시한 AGG 타일이면 1, 표시 안 한 AGG1 이미지의 타일이면 0, 그 밖에는 빈 칸),
    실행마다 p_<실행> (타일이 "AGG 영역이 있다/평평한 영역이 있다" 쪽일 확률, 두 순서를 평균한 값) 과 pct_<실행> (같은 이미지 안에서 점수가 높은 순위의 백분위, 1.0 = 최상위, 0.5 = 중간),
    jv06~jv08 은 순서별 확률 pA_<실행> (NO 먼저), pB_<실행> (FLAT 먼저) 도.
사용: python jv_tile_scores_all_runs.py [--runs jv06 jv07 jv08] [--name 출력이름]
출력: outputs/<name>/<name>.csv (utf-8-sig, 엑셀에서 바로 열림), README.txt (열 설명, 들어 있는 실행)
주의: 점수가 서로 다른 프롬프트와 질문 형식에서 나온 것이라 실행 사이에 값의 크기를 그대로 비교할 수 없다 (순위, 백분위는 이미지 안에서 비교).
"""
import argparse
import collections
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUTS = HERE / "outputs"
RUNS = [("v1", "jv01_binary_agg2_grid/input/tiles_grid_v1.jsonl"), ("v2", "jv04_grid_prompt_v2/input/tiles_grid_v2.jsonl"), ("v3", "jv05_grid_prompt_v3/input/tiles_grid_v3.jsonl"),
        ("jv06", "jv06_tiles_choice/input/tiles_grid_choice.jsonl"), ("jv07", "jv07_tiles_choice_v2/input/tiles_grid_choice_v2.jsonl"), ("jv08", "jv08_tiles_fewshot/input/tiles_grid_fewshot.jsonl")]
G00 = ROOT / "exp_grid_aug/outputs/g00_tile_list/tiles.csv"
LAB = ROOT / "exp_aggsim/outputs/c08_agg1_manual_labels/agg_tile_labels_filled.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=[n for n, _ in RUNS], help="합칠 실행 이름 (v1 v2 v3 jv06 jv07 jv08)")
    ap.add_argument("--name", default="tile_scores_all_runs", help="출력 폴더와 파일 이름")
    a = ap.parse_args()
    runs = [(n, rel) for n, rel in RUNS if n in a.runs]
    assert [n for n, _ in runs] == [n for n in [x for x, _ in RUNS] if n in a.runs] and len(runs) == len(a.runs), f"알 수 없는 실행 이름: {a.runs}"
    tiles = [r for r in csv.DictReader(open(G00, encoding="utf-8-sig")) if r["selected"] == "1"]
    assert len(tiles) == 2591
    key = lambda r: (r["image_id"], int(r["x0"]), int(r["y0"]))
    rows = collections.OrderedDict((key(r), dict(image_id=r["image_id"], label=f"AGG{r['label']}", metal=r["image_id"].split("/")[0], condition=r["image_id"].split("/")[1], tile_id=r["tile_id"],
                                                 x0=int(r["x0"]), y0=int(r["y0"]), mask_frac=round(float(r["mask_frac"]), 4),
                                                 tile_type="안쪽" if float(r["mask_frac"]) >= 0.999999 else "가장자리")) for r in tiles)
    has = {(r["image_id"], int(r["x0"]), int(r["y0"])): (r["has_agg"] or "0") for r in csv.DictReader(open(LAB, encoding="utf-8-sig"))}      # c08 파일은 AGG1 이미지 14장의 모든 타일을 담고, 표시한 타일만 1 (나머지는 빈 칸 -> 0)
    for k, r in rows.items():
        r["has_agg"] = has.get(k, "")
    order = list(rows)
    for name, rel in runs:
        data = {}
        for line in open(OUTS / rel, encoding="utf-8"):
            if line.strip():
                d = json.loads(line); data[(d["image_id"], d["x0"], d["y0"])] = d
        assert set(data) == set(rows), f"{name}: 타일 집합이 g00 과 다름 ({len(data)}개)"
        by_img = collections.defaultdict(list)
        for k in order:
            rows[k][f"p_{name}"] = data[k]["p_flat"]
            by_img[k[0]].append(k)
            if "p_A" in data[k]:
                rows[k][f"pA_{name}"] = data[k]["p_A"]; rows[k][f"pB_{name}"] = data[k]["p_B"]
        for iid, ks in by_img.items():
            ordered = sorted(ks, key=lambda k: -rows[k][f"p_{name}"])
            for i, k in enumerate(ordered, 1):
                rows[k][f"pct_{name}"] = round(1 - (i - 1) / (len(ks) - 1), 4) if len(ks) > 1 else 1.0
        print(f"{name:5s} 타일 {len(data)}개 읽음, p 평균 {sum(rows[k][f'p_{name}'] for k in order) / len(order):.3f}")
    cols = ["image_id", "label", "metal", "condition", "tile_id", "x0", "y0", "mask_frac", "tile_type", "has_agg"]
    for name, _ in runs:
        cols += [f"p_{name}", f"pct_{name}"] + ([f"pA_{name}", f"pB_{name}"] if name.startswith("jv") else [])
    out = OUTS / a.name; out.mkdir(exist_ok=True)
    with open(out / f"{a.name}.csv", "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(rows[k] for k in order)
    (out / "README.txt").write_text(__doc__.strip() + "\n\n이 파일에 들어 있는 실행: " + ", ".join(n for n, _ in runs) + "\n", encoding="utf-8")
    print(f"저장: {out / (a.name + '.csv')} ({len(rows)}행, 열 {len(cols)}개), 표시한 AGG 타일 {sum(1 for r in rows.values() if r['has_agg'] == '1')}개")


if __name__ == "__main__":
    main()
