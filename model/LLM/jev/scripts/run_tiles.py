"""Tile-level JEVision scoring on preproc selected tiles (10x, 224px, native resolution). Resumable jsonl."""
import argparse, base64, csv, io, json, sys, time
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path.home() / "jevision_work" / "JEVision"))
from jevvision import JEVision

ROOT = Path("/mnt/c/Users/admin/Downloads/260929_images_for_AGG - manual labeling/AGG_labeled_only")
TILES = ROOT / "preproc/outputs/s04_tile_grid/tiles.csv"

STATE = ("The attached image is a small crop (about 2 mm wide) of the surface of a metal sample after heat treatment. "
         "A normal surface shows a uniform fine granular texture. "
         "Abnormal grain growth (AGG) shows up as regions that differ from the surrounding texture, "
         "typically flat, smooth patches within the fine texture.")
QUESTIONS = {
    "flat": {"type": "noul",
             "instructions": "Does this crop contain a flat, smooth region that differs from the surrounding fine texture?",
             "criteria": {"false": "No: the texture is uniform across the crop.",
                          "true": "Yes: there is a flat, smooth region that differs from the surrounding texture."}},
    "extent": {"type": "choice",
               "instructions": "How much of this crop shows abnormal grain growth (flat regions differing from the surrounding texture)?",
               "criteria": {"NONE": "None: uniform fine texture everywhere.",
                            "PARTIAL": "Partial: a flat region covers a small part of the crop; the rest is normal fine texture.",
                            "MOST": "Most: flat regions cover most of the crop."}},
}

def tile_url(img, x0, y0, w, h):
    buf = io.BytesIO(); img.crop((x0, y0, x0 + w, y0 + h)).save(buf, "PNG")
    return {"data_url": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path.home() / "jevision_work/results/tiles_v1.jsonl"))
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    rows = [r for r in csv.DictReader(open(TILES, encoding="utf-8-sig")) if r["selected"] == "1"]
    if a.limit: rows = rows[::max(1, len(rows) // a.limit)][:a.limit]
    out = Path(a.out); done = set()
    if out.exists(): done = {(r["image_id"], r["tile_id"]) for r in map(json.loads, out.open())}
    todo = [r for r in rows if (r["image_id"], r["tile_id"]) not in done]
    print(f"{len(rows)} tiles, {len(todo)} to do", flush=True)
    model = JEVision.from_pretrained(str(Path.home() / "jevision_work/JEVision"), device="cuda:0")
    cache = {}
    with out.open("a") as fh:
        for i, r in enumerate(todo, 1):
            if r["image_id"] not in cache:
                cache = {r["image_id"]: Image.open(ROOT / (r["image_id"] + ".jpg")).convert("RGB")}
            img = cache[r["image_id"]]
            res = model.system_one(state=STATE, questions=QUESTIONS,
                                   images=[tile_url(img, int(r["x0"]), int(r["y0"]), int(r["w"]), int(r["h"]))])["answers"]
            rec = {"image_id": r["image_id"], "label": f"AGG{r['label']}", "tile_id": r["tile_id"],
                   "x0": int(r["x0"]), "y0": int(r["y0"]),
                   "p_flat": res["flat"]["noul"], "extent": res["extent"]["probabilities"]}
            fh.write(json.dumps(rec) + "\n"); fh.flush()
            if i % 100 == 0 or i == len(todo): print(i, len(todo), flush=True)

if __name__ == "__main__":
    main()
