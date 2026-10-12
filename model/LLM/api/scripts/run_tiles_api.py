"""Tile-level scoring with a hosted Gemini model (same 1,514 preproc tiles and same state text as the JEVision tile run).
Resumable jsonl (only successful rows count as done), per-minute rate limit, seeded shuffle, stops on daily quota.
Keys are read from ~/.agg_keys.env (never printed)."""
import argparse, base64, csv, io, json, os, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
from PIL import Image

sys.path.insert(0, str(Path.home() / "jevision_work" / "scripts"))
from run_api import load_keys, parse_delay

ROOT = Path("/mnt/c/Users/admin/Downloads/260929_images_for_AGG - manual labeling/AGG_labeled_only")
TILES = ROOT / "preproc/outputs/s04_tile_grid/tiles.csv"

PROMPT = ("The attached image is a small crop (about 2 mm wide) of the surface of a metal sample after heat treatment. "
          "A normal surface shows a uniform fine granular texture. "
          "Abnormal grain growth (AGG) shows up as regions that differ from the surrounding texture, "
          "typically flat, smooth patches within the fine texture.\n\n"
          "Question: does this crop contain a flat, smooth region that differs from the surrounding fine texture?\n"
          "Answer as JSON with: \"evidence\" (one short sentence), \"flat\" (true or false), "
          "\"probability\" (a number from 0 to 1: how likely such a region is present).")

SCHEMA = {"type": "OBJECT", "propertyOrdering": ["evidence", "flat", "probability"],
          "properties": {"evidence": {"type": "STRING"}, "flat": {"type": "BOOLEAN"}, "probability": {"type": "NUMBER"}},
          "required": ["evidence", "flat", "probability"]}


def tile_png(img, x0, y0, w, h):
    buf = io.BytesIO(); img.crop((x0, y0, x0 + w, y0 + h)).save(buf, "PNG"); return buf.getvalue()


def call_gemini_tile(model, png, client):
    body = {"contents": [{"parts": [{"text": PROMPT}, {"inline_data": {"mime_type": "image/png", "data": base64.b64encode(png).decode()}}]}],
            "generationConfig": {"temperature": 0, "responseMimeType": "application/json", "responseSchema": SCHEMA, "maxOutputTokens": 2048}}
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    for attempt in range(12):
        r = client.post(url, headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"]}, json=body, timeout=180)
        if r.status_code in (429, 500, 503):
            d = parse_delay(r.text)
            if r.status_code == 429 and d is not None and d > 300:
                raise RuntimeError("DAILY_QUOTA: retry in %.0fs" % d)
            time.sleep(min(d + 2, 120) if d is not None else min(5 * 2 ** attempt, 120)); continue
        r.raise_for_status(); break
    else:
        raise RuntimeError(f"gave up after retries: HTTP {r.status_code}")
    j = r.json(); cand = j["candidates"][0]
    text = "".join(p.get("text", "") for p in cand["content"]["parts"] if not p.get("thought"))
    return json.loads(text), j.get("usageMetadata", {}), cand.get("finishReason")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--rpm", type=float, default=13)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(); load_keys()
    rows = [r for r in csv.DictReader(open(TILES, encoding="utf-8-sig")) if r["selected"] == "1"]
    import random; random.Random(0).shuffle(rows)
    if a.limit: rows = rows[:a.limit]
    out = Path(a.out); out.parent.mkdir(exist_ok=True); done = set()
    if out.exists(): done = {(r["image_id"], r["tile_id"]) for r in map(json.loads, out.open()) if "error" not in r}
    todo = [r for r in rows if (r["image_id"], r["tile_id"]) not in done]
    print(f"{a.model}: {len(rows)} tiles, {len(todo)} to do", flush=True)
    lock = threading.Lock(); n = [0]; client = httpx.Client(); stop = [False]
    slot = {"next": 0.0}; slot_lock = threading.Lock(); imgs = {}; img_lock = threading.Lock()

    def wait_slot():
        with slot_lock:
            now = time.time(); wait = slot["next"] - now
            slot["next"] = max(now, slot["next"]) + 60.0 / a.rpm
        if wait > 0: time.sleep(wait)

    def get_img(image_id):
        with img_lock:
            if image_id not in imgs:
                if len(imgs) > 8: imgs.clear()
                imgs[image_id] = Image.open(ROOT / (image_id + ".jpg")).convert("RGB")
            return imgs[image_id]

    def work(r):
        if stop[0]: return
        wait_slot(); t = time.time()
        base = {"image_id": r["image_id"], "label": f"AGG{r['label']}", "tile_id": r["tile_id"], "x0": int(r["x0"]), "y0": int(r["y0"]), "model": a.model}
        try:
            png = tile_png(get_img(r["image_id"]), int(r["x0"]), int(r["y0"]), int(r["w"]), int(r["h"]))
            ans, usage, fin = call_gemini_tile(a.model, png, client)
            rec = {**base, "flat": ans["flat"], "p": ans["probability"], "evidence": ans["evidence"], "finish": fin,
                   "tokens_in": usage.get("promptTokenCount"), "tokens_out": usage.get("candidatesTokenCount"), "sec": round(time.time() - t, 1)}
        except Exception as e:
            rec = {**base, "error": f"{type(e).__name__}: {str(e)[:200]}"}
            if "DAILY_QUOTA" in str(e): stop[0] = True
        with lock:
            out.open("a").write(json.dumps(rec, ensure_ascii=False) + "\n"); n[0] += 1
            if n[0] % 25 == 0 or "error" in rec: print(n[0], len(todo), rec.get("error", "ok"), flush=True)

    with ThreadPoolExecutor(a.workers) as ex: list(ex.map(work, todo))
    print("finished", n[0], "rows", flush=True)


if __name__ == "__main__":
    main()
