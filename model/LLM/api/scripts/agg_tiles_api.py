"""Aggregate API tile outputs (flat / p per tile) to image level and evaluate (10x images). Handles partial coverage."""
import collections, json, statistics as st, sys

R = [json.loads(l) for l in open(sys.argv[1])]
errs = [r for r in R if "error" in r]
R = [r for r in R if "error" not in r]
L = ["AGG0", "AGG1", "AGG2"]
print(f"tiles ok={len(R)} errors={len(errs)}" + (f" (e.g. {errs[0]['error']})" if errs else ""))
by = collections.defaultdict(list)
for r in R: by[r["image_id"]].append(r)
imgs = []
for k, ts in by.items():
    imgs.append({"id": k, "label": ts[0]["label"], "n": len(ts), "cond": k.split("/")[1],
                 "mean_p": st.mean(t["p"] for t in ts), "frac_flat": st.mean(1.0 if t["flat"] else 0.0 for t in ts)})
print(f"images={len(imgs)} | tiles per image: min {min(i['n'] for i in imgs)} median {st.median(i['n'] for i in imgs)} max {max(i['n'] for i in imgs)}")
print("images per label:", dict(collections.Counter(i["label"] for i in imgs)))


def auc(pos, neg):
    if not pos or not neg: return float("nan")
    return sum(1 if p > n else 0.5 if p == n else 0 for p in pos for n in neg) / (len(pos) * len(neg))


g = lambda lab: [i for i in imgs if i["label"] == lab]
for score in ("mean_p", "frac_flat"):
    print(f"AUROC {score:9s}: AGG2>AGG0 {auc([i[score] for i in g('AGG2')], [i[score] for i in g('AGG0')]):.3f} | "
          f"AGG1>AGG0 {auc([i[score] for i in g('AGG1')], [i[score] for i in g('AGG0')]):.3f} | "
          f"AGG2>AGG1 {auc([i[score] for i in g('AGG2')], [i[score] for i in g('AGG1')]):.3f}")
print("\nimage-level mean score by true label:")
for lab in L:
    ims = g(lab)
    if ims: print(f"  {lab} (n={len(ims)}): mean_p={st.mean(i['mean_p'] for i in ims):.3f} frac_flat={st.mean(i['frac_flat'] for i in ims):.3f}")
print("\ntile-level: flat=True rate by image label:", {lab: round(st.mean(1.0 if t["flat"] else 0.0 for t in R if t["label"] == lab), 3) for lab in L if any(t["label"] == lab for t in R)})
print("tile-level mean p by image label:", {lab: round(st.mean(t["p"] for t in R if t["label"] == lab), 3) for lab in L if any(t["label"] == lab for t in R)})
print("tile p distribution:", dict(collections.Counter(round(t["p"], 2) for t in R).most_common(8)))
print("\nper-condition mean_p:")
c = collections.defaultdict(list)
for i in imgs: c[i["cond"]].append(i)
for k in sorted(c): print(f"  {k:12s} mean_p={st.mean(i['mean_p'] for i in c[k]):.3f}  labels={dict(collections.Counter(i['label'] for i in c[k]))}")
