"""Aggregate tile-level JEVision outputs to image level and evaluate (10x, 60 images)."""
import json, sys, collections, statistics as st
R = [json.loads(l) for l in open(sys.argv[1])]
L = ["AGG0", "AGG1", "AGG2"]
by = collections.defaultdict(list)
for r in R: by[r["image_id"]].append(r)
imgs = []
for k, ts in by.items():
    imgs.append({"id": k, "label": ts[0]["label"], "n": len(ts), "cond": k.split("/")[1], "metal": k.split("/")[0],
                 "mean_flat": st.mean(t["p_flat"] for t in ts),
                 "frac_flat": st.mean(t["p_flat"] > 0.5 for t in ts),
                 "mean_most": st.mean(t["extent"]["MOST"] for t in ts),
                 "mean_part": st.mean(t["extent"]["PARTIAL"] for t in ts)})
print(f"tiles={len(R)} images={len(imgs)}")
def auc(pos, neg):
    return sum(1 if p > n else 0.5 if p == n else 0 for p in pos for n in neg) / (len(pos) * len(neg))
g = lambda lab: [i for i in imgs if i["label"] == lab]
for score in ("mean_flat", "frac_flat", "mean_most"):
    print(f"AUROC {score:10s}: AGG2>AGG0 {auc([i[score] for i in g('AGG2')],[i[score] for i in g('AGG0')]):.3f} | "
          f"AGG1>AGG0 {auc([i[score] for i in g('AGG1')],[i[score] for i in g('AGG0')]):.3f} | "
          f"AGG2>AGG1 {auc([i[score] for i in g('AGG2')],[i[score] for i in g('AGG1')]):.3f}")
print("\nmean score by true label (image level):")
for lab in L:
    ims = g(lab)
    print(f"  {lab} (n={len(ims)}): mean_flat={st.mean(i['mean_flat'] for i in ims):.3f} frac_flat={st.mean(i['frac_flat'] for i in ims):.3f} mean_most={st.mean(i['mean_most'] for i in ims):.3f}")
# tile-level: is p_flat higher in tiles of AGG2 images?
print("\ntile-level mean p_flat by image label:", {lab: round(st.mean(t["p_flat"] for t in R if t["label"] == lab), 3) for lab in L})
# 3-class via 2 thresholds on mean_flat, fit on these same images (optimistic, shown only as upper bound)
best = (0, None)
vals = sorted(set(i["mean_flat"] for i in imgs))
for a in vals:
    for b in vals:
        if b < a: continue
        pred = lambda s: "AGG0" if s < a else ("AGG1" if s < b else "AGG2")
        acc = st.mean(pred(i["mean_flat"]) == i["label"] for i in imgs)
        if acc > best[0]: best = (acc, (a, b))
base = max(collections.Counter(i["label"] for i in imgs).values()) / len(imgs)
print(f"\nbest 2-threshold accuracy on mean_flat (fit on same data, optimistic): {best[0]:.3f} | majority baseline {base:.3f}")
# control: does it just recover the treatment condition? within-condition AUROC
print("\nwithin-condition check (AGG labels differ inside the same condition only for a few conditions) -> see per-condition mean_flat:")
c = collections.defaultdict(list)
for i in imgs: c[i["cond"]].append(i)
for k in sorted(c): print(f"  {k:12s} mean_flat={st.mean(i['mean_flat'] for i in c[k]):.3f}  labels={dict(collections.Counter(i['label'] for i in c[k]))}")

# ---- 3-class confusion for the (optimistic) best thresholds + macro-F1 ----
a, b = best[1]
pred = lambda s: "AGG0" if s < a else ("AGG1" if s < b else "AGG2")
cm = {x: {y: 0 for y in L} for x in L}
for i in imgs: cm[i["label"]][pred(i["mean_flat"])] += 1
print(f"\nthresholds on mean_flat: AGG0 < {a:.3f} <= AGG1 < {b:.3f} <= AGG2")
print("true/pred   " + "  ".join(L))
for x in L: print(f"{x:9s}  " + "  ".join(f"{cm[x][y]:4d}" for y in L))
f1 = []
for y in L:
    tp = cm[y][y]; pp = sum(cm[x][y] for x in L); tt = sum(cm[y].values())
    p = tp / pp if pp else 0; r = tp / tt if tt else 0
    f1.append(2 * p * r / (p + r) if p + r else 0)
    print(f"  {y}: precision={p:.3f} recall={r:.3f} f1={f1[-1]:.3f}")
print(f"  macro-F1={sum(f1)/3:.3f}")
# ---- same 60 images, whole-image zero-shot (v1/v2) for apples-to-apples ----
import os
ids = {i["id"] for i in imgs}
for v in ("v1", "v2"):
    p = os.path.join(os.path.dirname(sys.argv[1]), f"zeroshot_{v}.jsonl")
    Z = [json.loads(l) for l in open(p)]
    Z = [z for z in Z if f"{z['metal']}/{z['cond']}/{z['file'][:-4]}" in ids]
    print(f"\nwhole-image zero-shot {v} on the same {len(Z)} images: acc={st.mean(z['pred']==z['label'] for z in Z):.3f}, preds={dict(collections.Counter(z['pred'] for z in Z))}")
    sev = lambda z: z["probs"]["AGG1"] + 2 * z["probs"]["AGG2"]
    zg = lambda lab: [sev(z) for z in Z if z["label"] == lab]
    print(f"  AUROC severity: AGG2>AGG0 {auc(zg('AGG2'), zg('AGG0')):.3f} | AGG1>AGG0 {auc(zg('AGG1'), zg('AGG0')):.3f} | AGG2>AGG1 {auc(zg('AGG2'), zg('AGG1')):.3f}")
