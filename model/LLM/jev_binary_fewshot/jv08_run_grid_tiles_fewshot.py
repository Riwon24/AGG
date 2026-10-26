"""jv08: jv06_run_grid_tiles_choice.py 와 설계가 같고, 요청마다 예시 이미지 3장(AGG 예시 2, 정상 예시 1)을 판단할 타일 앞에 함께 보낸다 (few-shot). JEVision 은 요청당 이미지 4장까지 받는다.

- 이미지 순서: [AGG 예시 1, AGG 예시 2, 정상 예시, 판단할 타일]. 예시 타일은 224px 를 256x256 으로 리사이즈한 PNG, 판단할 타일은 jv06 과 같이 224px.
- 설명문: jv06(run_tiles.py)의 STATE 그대로 + 이미지의 순서를 알려 주는 한 문단 (이미지에는 이름이 없어서 필요하다). 질문 문구, 선택지 순서 교차(A: NO 먼저, B: FLAT 먼저), 타일 2,591개, 출력 형식은 jv06 과 같다.
- 예시: jv08_select_examples.py 가 정한 outputs/jv08_tiles_fewshot/examples.json (표시한 AGG 타일 중 안쪽 타일에서 이미지 다양성 기준, 정상 예시는 AGG1 이미지의 has_agg == 0 안쪽 타일 중 jv06 p_flat 최저).
- 판단할 타일이 예시 타일과 같은 경우(3개)도 그대로 추론한다 (보조 분석에서 따로 다룬다).
사용: python jv08_run_grid_tiles_fewshot.py [--out jsonl] [--limit N]   /   --summary jsonl (jv06 과 같은 요약)
"""
import argparse, base64, csv, io, json, sys
from pathlib import Path

import jv06_run_grid_tiles_choice as base

ROOT = base.ROOT
EX = json.loads((Path(__file__).resolve().parent / "outputs/jv08_tiles_fewshot/examples.json").read_text(encoding="utf-8"))
EXTRA = ("Four images are attached, in this order: images 1 and 2 are example crops that contain AGG, "
         "image 3 is an example crop of a normal surface, and image 4 is the crop to judge. Answer only about image 4.")
STATE = base.STATE + "\n\n" + EXTRA


def example_url(img, x0, y0):
    from PIL import Image
    buf = io.BytesIO(); img.crop((x0, y0, x0 + 224, y0 + 224)).resize((256, 256), Image.BICUBIC).save(buf, "PNG")
    return {"data_url": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path.home() / "jevision_work/results/tiles_grid_fewshot.jsonl"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--summary", default=None)
    a = ap.parse_args()
    if a.summary:
        base.summary(a.summary); return
    sys.path.insert(0, str(Path.home() / "jevision_work" / "JEVision"))
    from jevvision import JEVision
    from PIL import Image
    ex_urls = [example_url(Image.open(ROOT / (e["image_id"] + ".jpg")).convert("RGB"), e["x0"], e["y0"]) for e in EX]
    assert [e["role"] for e in EX] == ["AGG", "AGG", "NORMAL"]
    rows = [r for r in csv.DictReader(open(base.TILES, encoding="utf-8-sig")) if r["selected"] == "1"]
    if a.limit:
        rows = rows[::max(1, len(rows) // a.limit)][:a.limit]
    out = Path(a.out); done = set()
    if out.exists():
        done = {(r["image_id"], r["tile_id"]) for r in map(json.loads, out.open())}
    todo = [r for r in rows if (r["image_id"], r["tile_id"]) not in done]
    print(f"{len(rows)} tiles, {len(todo)} to do, 예시 {[(e['role'], e['image_id'][:24], e['tile_id']) for e in EX]}", flush=True)
    model = JEVision.from_pretrained(str(Path.home() / "jevision_work/JEVision"), device="cuda:0")
    cache = {}
    with out.open("a") as fh:
        for i, r in enumerate(todo, 1):
            if r["image_id"] not in cache:
                cache = {r["image_id"]: Image.open(ROOT / (r["image_id"] + ".jpg")).convert("RGB")}
            img = cache[r["image_id"]]
            urls = ex_urls + [base.tile_url(img, int(r["x0"]), int(r["y0"]), int(r["w"]), int(r["h"]))]
            res = {o: model.system_one(state=STATE, questions=base.questions(o), images=urls)["answers"] for o in ("A", "B")}
            pA, pB = res["A"]["flat"]["probabilities"]["FLAT_REGION"], res["B"]["flat"]["probabilities"]["FLAT_REGION"]
            ext = {k: (res["A"]["extent"]["probabilities"][k] + res["B"]["extent"]["probabilities"][k]) / 2 for k in ("NONE", "PARTIAL", "MOST")}
            rec = {"image_id": r["image_id"], "label": f"AGG{r['label']}", "tile_id": r["tile_id"], "x0": int(r["x0"]), "y0": int(r["y0"]),
                   "p_flat": round((pA + pB) / 2, 4), "p_A": pA, "p_B": pB, "choice_A": res["A"]["flat"]["choice"], "choice_B": res["B"]["flat"]["choice"],
                   "extent": {k: round(v, 4) for k, v in ext.items()}}
            fh.write(json.dumps(rec) + "\n"); fh.flush()
            if i % 100 == 0 or i == len(todo):
                print(i, len(todo), flush=True)


if __name__ == "__main__":
    main()
