"""Zero-shot AGG0/1/2 classification with JEVision (visual route). Appends to a jsonl; skips already-done images."""
import argparse, json, re, sys, time
from pathlib import Path

sys.path.insert(0, str(Path.home() / "jevision_work" / "JEVision"))
from jevvision import JEVision, image_file_as_data_url

ROOT = Path("/mnt/c/Users/admin/Downloads/260929_images_for_AGG - manual labeling/AGG_labeled_only")
METALS = ["BASE", "BASE-LC", "BASE-LCSI", "BASE-LSi", "REF", "REF-LN"]

PROMPT_V1 = {
    "state": "The attached image is a microscope image of a metal sample after heat treatment. "
             "Decide whether abnormal grain growth (AGG) occurred.",
    "questions": {"agg": {
        "type": "choice",
        "instructions": "Which AGG level does this microstructure image show?",
        "criteria": {
            "AGG0": "Normal: no abnormal grain growth.",
            "AGG1": "Partial AGG: abnormal grain growth in part of the sample.",
            "AGG2": "Severe AGG: abnormal grain growth is extensive.",
        }}},
}

PROMPT_V2 = {
    "state": "The attached image is a low-magnification surface photograph of a metal sample after heat treatment "
             "(scale bar 1 mm). A normal sample surface shows a uniform fine granular texture. "
             "Abnormal grain growth (AGG) shows up as regions that differ from the surrounding texture, "
             "typically flat, smooth patches within the fine texture. "
             "Judge only the sample surface; ignore the dark background and the sample edges.",
    "questions": {"agg": {
        "type": "choice",
        "instructions": "How much abnormal grain growth (AGG) does this sample surface show?",
        "criteria": {
            "AGG0": "Normal: the whole surface has a uniform fine texture; no flat patches that differ from the surrounding texture.",
            "AGG1": "Partial AGG: some flat patches that differ from the surrounding texture appear in parts of the surface; the rest keeps the normal fine texture.",
            "AGG2": "Severe AGG: flat patches that differ from the surrounding texture cover a large part of the surface.",
        }}},
}
PROMPTS = {"v1": PROMPT_V1, "v2": PROMPT_V2}

def parse(p: Path):
    m = re.search(r"AGG(\d)", p.stem)
    mag = re.match(r"[xX](12\.5|10|7\.5)", p.stem)
    return {"metal": p.parts[len(ROOT.parts)], "cond": p.parent.name,
            "mag": mag.group(1) if mag else None, "file": p.name,
            "label": f"AGG{m.group(1)}" if m else None}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="v1", choices=["v1", "v2"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=0, help="stratified-ish: take first N of a shuffled list")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    P = PROMPTS[args.prompt]
    if args.out is None:
        args.out = str(Path.home() / f"jevision_work/results/zeroshot_{args.prompt}.jsonl")

    files = sorted(f for m in METALS for f in (ROOT / m).rglob("*.jpg"))
    if args.limit:
        import random; random.Random(args.seed).shuffle(files); files = files[:args.limit]
    out = Path(args.out); out.parent.mkdir(exist_ok=True)
    done = set()
    if out.exists():
        done = {(r["metal"], r["cond"], r["file"]) for r in map(json.loads, out.open())}
    todo = [f for f in files if (parse(f)["metal"], parse(f)["cond"], f.name) not in done]
    print(f"{len(files)} images, {len(todo)} to do", flush=True)

    model = JEVision.from_pretrained(str(Path.home() / "jevision_work/JEVision"), device="cuda:0")
    with out.open("a") as fh:
        for i, f in enumerate(todo, 1):
            t = time.time()
            res = model.system_one(state=P["state"], questions=P["questions"],
                                   images=[image_file_as_data_url(f)])
            a = res["answers"]["agg"]
            rec = {**parse(f), "pred": a["choice"], "conf": a["confidence"], "probs": a["probabilities"],
                   "sec": round(time.time() - t, 2), "prompt": args.prompt}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n"); fh.flush()
            print(i, len(todo), rec["metal"], rec["cond"], rec["file"], rec["label"], "->", rec["pred"], rec["probs"], rec["sec"], flush=True)

if __name__ == "__main__":
    main()
