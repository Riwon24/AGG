"""jv06: JEVision 타일 추론, 선택(choice) 형식 + 선택지 순서 교차 (설계안 B). 프롬프트 문구는 model/LLM/jev/scripts/run_tiles.py 의 것을 그대로 쓴다.

- 타일: DINOv2_grid_augmentation 의 분석 타일 2,591개 (exp_grid_aug g00 tiles.csv, selected == 1), 타일 하나를 이미지 1장으로 보낸다 (jv02, jv04, jv05 와 같음).
- 설명문(STATE)과 `flat`, `extent` 질문의 문구와 선택지 설명은 run_tiles.py 와 글자 그대로 같다.
- 바뀐 것은 `flat` 질문의 형식 하나다: run_tiles.py 는 yes/no 확률(noul)로 묻고, 여기서는 같은 문구와 같은 두 설명을 두 선택지(NO_FLAT_REGION / FLAT_REGION) 중 하나를 고르는 choice 로 묻는다.
  `extent` 질문은 run_tiles.py 와 같고 이미지 점수에는 쓰지 않는다.
- 선택지 순서: 타일마다 2번 묻는다 (A: NO_FLAT_REGION 먼저, B: FLAT_REGION 먼저). 선택지 확률을 평균한 값이 p_flat, 직접 선택은 평균 확률이 큰 쪽이다 (p_flat > 0.5 이면 FLAT_REGION).
- 출력 jsonl 한 줄 = 타일 하나: image_id, label, tile_id, x0, y0, p_flat(두 순서 평균 P(FLAT_REGION)), p_A, p_B, choice_A, choice_B, extent(두 순서 평균). 라벨은 평가를 위해 기록할 뿐 추론에는 쓰지 않는다.
- --summary <jsonl>: 라벨을 쓰지 않고 선택의 쏠림과 순서 효과만 요약한다 (본 실행 전 시험용).
사용: python jv06_run_grid_tiles_choice.py [--out jsonl] [--limit N]   /   python jv06_run_grid_tiles_choice.py --summary jsonl
"""
import argparse, base64, csv, io, json, sys, time
from pathlib import Path

ROOT = Path("/mnt/c/Users/<USER>/Downloads/260929_images_for_AGG - manual labeling/AGG_labeled_only")
TILES = ROOT / "exp_grid_aug/outputs/g00_tile_list/tiles.csv"

# ---- run_tiles.py 의 STATE 와 같다 (글자 그대로) ----
STATE = ("The attached image is a small crop (about 2 mm wide) of the surface of a metal sample after heat treatment. "
         "A normal surface shows a uniform fine granular texture. "
         "Abnormal grain growth (AGG) shows up as regions that differ from the surrounding texture, "
         "typically flat, smooth patches within the fine texture.")
FLAT_INSTR = "Does this crop contain a flat, smooth region that differs from the surrounding fine texture?"   # run_tiles.py 의 flat 질문 문구
NO_TXT = "No: the texture is uniform across the crop."                                                       # run_tiles.py 의 criteria false
YES_TXT = "Yes: there is a flat, smooth region that differs from the surrounding texture."                  # run_tiles.py 의 criteria true
EXTENT = {"type": "choice",                                                                                  # run_tiles.py 의 extent 질문 (그대로)
          "instructions": "How much of this crop shows abnormal grain growth (flat regions differing from the surrounding texture)?",
          "criteria": {"NONE": "None: uniform fine texture everywhere.",
                       "PARTIAL": "Partial: a flat region covers a small part of the crop; the rest is normal fine texture.",
                       "MOST": "Most: flat regions cover most of the crop."}}


def questions(order):
    crit = {"NO_FLAT_REGION": NO_TXT, "FLAT_REGION": YES_TXT}
    if order == "B":
        crit = {"FLAT_REGION": YES_TXT, "NO_FLAT_REGION": NO_TXT}
    return {"flat": {"type": "choice", "instructions": FLAT_INSTR, "criteria": crit}, "extent": EXTENT}


def tile_url(img, x0, y0, w, h):
    buf = io.BytesIO(); img.crop((x0, y0, x0 + w, y0 + h)).save(buf, "PNG")
    return {"data_url": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()}


def summary(path):
    R = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    n = len(R)
    pa = [r["p_A"] for r in R]; pb = [r["p_B"] for r in R]; pm = [r["p_flat"] for r in R]
    mean = lambda v: sum(v) / len(v)
    ca = sum(r["choice_A"] == "FLAT_REGION" for r in R) / n; cb = sum(r["choice_B"] == "FLAT_REGION" for r in R) / n
    cm = sum(p > 0.5 for p in pm) / n
    d = [abs(a - b) for a, b in zip(pa, pb)]
    print(f"타일 {n}개 (라벨은 쓰지 않음)")
    print(f"  FLAT_REGION 을 고른 비율: 순서 A {ca:.1%}, 순서 B {cb:.1%}, 두 순서 평균 확률 기준 {cm:.1%}")
    print(f"  P(FLAT_REGION) 평균: A {mean(pa):.3f}, B {mean(pb):.3f}, 평균 {mean(pm):.3f} | 서로 다른 평균값 {len(set(pm))}개")
    print(f"  순서 효과: |P_A - P_B| 평균 {mean(d):.3f}, 최대 {max(d):.2f}, 0.05 초과 {sum(x > 0.05 for x in d) / n:.1%}, 두 순서의 선택이 다른 타일 {sum(r['choice_A'] != r['choice_B'] for r in R) / n:.1%}")
    big = max(cm, 1 - cm)
    print(f"  선택 쏠림 (평균 확률 기준 더 많이 고른 쪽 비율): {big:.1%} -> " + ("95% 초과, 쏠림 (중단 규칙 해당)" if big > 0.95 else "95% 이하, 쏠림 아님"))
    return big


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path.home() / "jevision_work/results/tiles_grid_choice.jsonl"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--summary", default=None)
    a = ap.parse_args()
    if a.summary:
        summary(a.summary); return
    sys.path.insert(0, str(Path.home() / "jevision_work" / "JEVision"))
    from jevvision import JEVision
    from PIL import Image
    rows = [r for r in csv.DictReader(open(TILES, encoding="utf-8-sig")) if r["selected"] == "1"]
    if a.limit:
        rows = rows[::max(1, len(rows) // a.limit)][:a.limit]            # 라벨과 무관한 고정 간격 추출
    out = Path(a.out); done = set()
    if out.exists():
        done = {(r["image_id"], r["tile_id"]) for r in map(json.loads, out.open())}
    todo = [r for r in rows if (r["image_id"], r["tile_id"]) not in done]
    print(f"{len(rows)} tiles, {len(todo)} to do", flush=True)
    model = JEVision.from_pretrained(str(Path.home() / "jevision_work/JEVision"), device="cuda:0")
    cache = {}
    with out.open("a") as fh:
        for i, r in enumerate(todo, 1):
            if r["image_id"] not in cache:
                cache = {r["image_id"]: Image.open(ROOT / (r["image_id"] + ".jpg")).convert("RGB")}
            img = cache[r["image_id"]]
            url = [tile_url(img, int(r["x0"]), int(r["y0"]), int(r["w"]), int(r["h"]))]
            res = {o: model.system_one(state=STATE, questions=questions(o), images=url)["answers"] for o in ("A", "B")}
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
