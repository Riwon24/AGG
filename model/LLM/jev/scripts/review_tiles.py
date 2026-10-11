import json, sys
from pathlib import Path
from PIL import Image, ImageDraw
ROOT = Path("/mnt/c/Users/admin/Downloads/260929_images_for_AGG - manual labeling/AGG_labeled_only")
OUT = ROOT / "llm_exp/tile_review"; OUT.mkdir(exist_ok=True)
R = [json.loads(l) for l in open(Path.home() / "jevision_work/results/tiles_v1.jsonl")]
cache = {}
def tile(r):
    if r["image_id"] not in cache: cache.clear(); cache[r["image_id"]] = Image.open(ROOT / (r["image_id"] + ".jpg")).convert("RGB")
    return cache[r["image_id"]].crop((r["x0"], r["y0"], r["x0"] + 224, r["y0"] + 224))
def sheet(rows, name, cols=6):
    cw, ch = 224, 224 + 26
    sh = Image.new("RGB", (cols * cw, ((len(rows) + cols - 1) // cols) * ch), (30, 30, 30))
    d = ImageDraw.Draw(sh)
    for i, r in enumerate(rows):
        x, y = (i % cols) * cw, (i // cols) * ch
        sh.paste(tile(r), (x, y + 26))
        m, c = r["image_id"].split("/")[:2]
        d.text((x + 3, y + 2), f"p={r['p_flat']:.2f} {r['label']}", fill=(255, 255, 0))
        d.text((x + 3, y + 13), f"{m} {c}", fill=(200, 200, 200))
    sh.save(OUT / name, quality=92)
srt = sorted(R, key=lambda r: -r["p_flat"])
sheet(srt[:24], "A_top24_p_flat.jpg")
sheet(srt[-24:], "B_bottom24_p_flat.jpg")
sheet([r for r in srt if r["label"] == "AGG0"][:24], "C_AGG0_images_highest_p_flat.jpg")
sheet([r for r in srt if r["label"] == "AGG2"][-24:], "D_AGG2_images_lowest_p_flat.jpg")
import collections
print("top24 labels:", collections.Counter(r["label"] for r in srt[:24]), "bottom24:", collections.Counter(r["label"] for r in srt[-24:]))
print("top24 conds:", collections.Counter(r["image_id"].split("/")[1] for r in srt[:24]))
print("p range top24", srt[0]["p_flat"], srt[23]["p_flat"], "bottom24", srt[-24]["p_flat"], srt[-1]["p_flat"])
