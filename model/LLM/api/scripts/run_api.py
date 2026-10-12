"""Zero-shot AGG0/1/2 classification of whole images with hosted multimodal APIs (Gemini). Resumable jsonl.
Keys are read from ~/.agg_keys.env (never printed). Same prompt as JEVision v2 (user definition of AGG)."""
import argparse, base64, json, os, re, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import httpx

ROOT = Path("/mnt/c/Users/admin/Downloads/260929_images_for_AGG - manual labeling/AGG_labeled_only")
METALS = ["BASE", "BASE-LC", "BASE-LCSI", "BASE-LSi", "REF", "REF-LN"]
LABELS = ["AGG0", "AGG1", "AGG2"]

PROMPT = """You are inspecting a low-magnification surface photograph of a metal sample after heat treatment (scale bar 1 mm in the lower right corner).
A normal sample surface shows a uniform fine granular texture.
Abnormal grain growth (AGG) shows up as regions that differ from the surrounding texture, typically flat, smooth patches within the fine texture.
Judge only the sample surface; ignore the dark background and the sample edges.

Classify the amount of AGG:
- AGG0 (normal): the whole surface has a uniform fine texture; no flat patches that differ from the surrounding texture.
- AGG1 (partial AGG): some flat patches that differ from the surrounding texture appear in parts of the surface; the rest keeps the normal fine texture.
- AGG2 (severe AGG): flat patches that differ from the surrounding texture cover a large part of the surface.

Answer as JSON with: "evidence" (one or two short sentences describing what you see), "label" (AGG0, AGG1 or AGG2), "severity" (a number from 0 = clearly normal to 1 = clearly severe AGG)."""

def load_keys():
    for line in (Path.home() / ".agg_keys.env").read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1); os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

def parse(p: Path):
    m = re.search(r"AGG(\d)", p.stem); mag = re.match(r"[xX](12\.5|10|7\.5)", p.stem)
    return {"metal": p.parts[len(ROOT.parts)], "cond": p.parent.name, "mag": mag.group(1) if mag else None,
            "file": p.name, "label": f"AGG{m.group(1)}" if m else None}

def parse_delay(text):
    m = re.search(r'retryDelay[^0-9]{1,6}([0-9.]+)s', text)
    if m: return float(m.group(1))
    m = re.search(r'retry in ((?:[0-9.]+h)?(?:[0-9.]+m)?(?:[0-9.]+s)?)', text)
    if not m or not m.group(1): return None
    h = re.search(r'([0-9.]+)h', m.group(1)); mi = re.search(r'([0-9.]+)m', m.group(1)); se = re.search(r'([0-9.]+)s', m.group(1))
    return (float(h.group(1)) * 3600 if h else 0) + (float(mi.group(1)) * 60 if mi else 0) + (float(se.group(1)) if se else 0)

def call_gemini(model, img_bytes, client):
    schema = {"type": "OBJECT", "propertyOrdering": ["evidence", "label", "severity"],
              "properties": {"evidence": {"type": "STRING"}, "label": {"type": "STRING", "enum": LABELS},
                             "severity": {"type": "NUMBER"}},
              "required": ["evidence", "label", "severity"]}
    body = {"contents": [{"parts": [{"text": PROMPT},
                                    {"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(img_bytes).decode()}}]}],
            "generationConfig": {"temperature": 0, "responseMimeType": "application/json", "responseSchema": schema,
                                 "maxOutputTokens": 4096}}
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    for attempt in range(12):
        r = client.post(url, headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"]}, json=body, timeout=180)
        if r.status_code in (429, 500, 503):
            d = parse_delay(r.text)
            if r.status_code == 429 and d is not None and d > 300:
                raise RuntimeError('DAILY_QUOTA: retry in %.0fs' % d)
            time.sleep(min(d + 2, 120) if d is not None else min(5 * 2 ** attempt, 120)); continue
        r.raise_for_status(); break
    else:
        raise RuntimeError(f"gave up after retries: HTTP {r.status_code}")
    j = r.json()
    cand = j["candidates"][0]
    text = "".join(p.get("text", "") for p in cand["content"]["parts"] if not p.get("thought"))
    return json.loads(text), j.get("usageMetadata", {}), cand.get("finishReason")

NV_URL = 'https://integrate.api.nvidia.com/v1/chat/completions'
# per-model settings follow the examples given by the user (provider-recommended); llama is non-reasoning, so temperature 0.
NV_CFG = {
    'moonshotai/kimi-k3': {'max_tokens': 16384, 'seed': 0, 'temperature': 1, 'reasoning_effort': 'max'},
    'nvidia/nemotron-3-nano-omni-30b-a3b-reasoning': {'max_tokens': 65536, 'reasoning_budget': 16384, 'temperature': 0.6, 'top_p': 0.95},
    'meta/llama-3.2-90b-vision-instruct': {'max_tokens': 512, 'temperature': 0, 'top_p': 1},
}

def shrink_jpeg(img_bytes, max_b64):
    # re-encode as JPEG so that base64 length <= max_b64 (reduces size by lowering quality, then scale)
    import io
    from PIL import Image
    im = Image.open(io.BytesIO(img_bytes)).convert('RGB'); scale = 1.0
    while True:
        for q in (92, 85, 75, 65, 50):
            buf = io.BytesIO(); (im if scale == 1.0 else im.resize((int(im.width * scale), int(im.height * scale)), Image.LANCZOS)).save(buf, 'JPEG', quality=q)
            if len(buf.getvalue()) * 4 // 3 <= max_b64: return buf.getvalue()
        scale *= 0.85

def extract_json(text):
    import re as _re
    m = _re.search(r'[{].*[}]', text, _re.S)
    if m:
        try: return json.loads(m.group(0))
        except Exception: pass
    lab = _re.findall(r'AGG[012]', text)
    if lab: return {'label': lab[-1], 'severity': None, 'evidence': text[-200:], '_fallback': True}
    raise ValueError('no JSON/label in: ' + text[-200:])

def call_nvidia(model, img_bytes, client, max_b64=0):
    if max_b64: img_bytes = shrink_jpeg(img_bytes, max_b64)
    url = 'data:image/jpeg;base64,' + base64.b64encode(img_bytes).decode()
    body = {'model': model, 'stream': False,
            'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': PROMPT}, {'type': 'image_url', 'image_url': {'url': url}}]}],
            **NV_CFG[model]}
    for attempt in range(6):
        r = client.post(NV_URL, headers={'Authorization': 'Bearer ' + os.environ['NVIDIA_API_KEY'], 'Accept': 'application/json'}, json=body, timeout=900)
        if r.status_code in (429, 500, 502, 503, 504): time.sleep(5 * 2 ** attempt); continue
        if r.status_code >= 400: raise RuntimeError(f'HTTP {r.status_code}: {r.text[:300]}')
        break
    else:
        raise RuntimeError(f'gave up after retries: HTTP {r.status_code}')
    j = r.json(); ch = j['choices'][0]; msg = ch['message']
    text = msg.get('content') or ''
    ans = extract_json(text)
    u = j.get('usage', {})
    return ans, {'promptTokenCount': u.get('prompt_tokens'), 'candidatesTokenCount': u.get('completion_tokens'), 'thoughtsTokenCount': None, 'sent_bytes': len(img_bytes)}, ch.get('finish_reason')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="gemini", choices=["gemini", "nvidia"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--rpm", type=float, default=0, help="max requests per minute for this process (0 = unlimited)")
    ap.add_argument("--shuffle", action="store_true", help="process images in a seeded random order (partial results stay unbiased)")
    ap.add_argument("--list-models", action="store_true")
    ap.add_argument("--max-b64", type=int, default=0, help="nvidia: shrink image so base64 <= this many bytes (0 = send original)")
    a = ap.parse_args(); load_keys()
    if a.list_models:
        r = httpx.get("https://generativelanguage.googleapis.com/v1beta/models?pageSize=200",
                      headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"]}, timeout=60); r.raise_for_status()
        for m in r.json().get("models", []):
            if "generateContent" in m.get("supportedGenerationMethods", []): print(m["name"].split("/")[-1])
        return
    files = sorted(f for m in METALS for f in (ROOT / m).rglob("*.jpg"))
    if a.shuffle:
        import random; random.Random(0).shuffle(files)
    if a.limit:
        import random; random.Random(0).shuffle(files); files = files[:a.limit]
    out = Path(a.out); out.parent.mkdir(exist_ok=True); done = set()
    if out.exists(): done = {(r["metal"], r["cond"], r["file"]) for r in map(json.loads, out.open()) if "error" not in r}
    todo = [f for f in files if (parse(f)["metal"], parse(f)["cond"], f.name) not in done]
    print(f"{a.model}: {len(files)} images, {len(todo)} to do", flush=True)
    lock = threading.Lock(); n = [0]; client = httpx.Client()
    slot = {'next': 0.0}; slot_lock = threading.Lock(); stop = [False]
    def wait_slot():
        if not a.rpm: return
        with slot_lock:
            now = time.time(); wait = slot['next'] - now
            slot['next'] = max(now, slot['next']) + 60.0 / a.rpm
        if wait > 0: time.sleep(wait)
    def work(f):
        if stop[0]: return
        wait_slot()
        meta = parse(f); t = time.time()
        try:
            ans, usage, fin = call_gemini(a.model, f.read_bytes(), client) if a.provider == "gemini" else call_nvidia(a.model, f.read_bytes(), client, a.max_b64)
            rec = {**meta, "pred": ans["label"], "score": ans.get("severity"), "evidence": ans.get("evidence"), "finish": fin,
                   "tokens_in": usage.get("promptTokenCount"), "tokens_out": usage.get("candidatesTokenCount"),
                   "tokens_think": usage.get("thoughtsTokenCount"), "sent_bytes": usage.get("sent_bytes"), "provider": a.provider, "sec": round(time.time() - t, 1),
                   "model": a.model, "prompt": "api_v2"}
        except Exception as e:
            rec = {**meta, "error": f"{type(e).__name__}: {str(e)[:200]}", "model": a.model, "prompt": "api_v2"}
            if "DAILY_QUOTA" in str(e): stop[0] = True
        with lock:
            out.open("a").write(json.dumps(rec, ensure_ascii=False) + "\n"); n[0] += 1
            print(n[0], len(todo), meta["metal"], meta["cond"], meta["file"], meta["label"], "->", rec.get("pred", rec.get("error")), flush=True)
    with ThreadPoolExecutor(a.workers) as ex: list(ex.map(work, todo))

if __name__ == "__main__":
    main()
