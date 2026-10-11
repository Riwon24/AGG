import json, sys, collections, re
recs = [json.loads(l) for l in open(sys.argv[1])]
for r in recs:
    m = re.match(r"[xX](12\.5|10|7\.5)", r["file"]); r["mag"] = m.group(1) if m else "?"
L = ["AGG0", "AGG1", "AGG2"]
def cm(rs):
    m = {a: {b: 0 for b in L} for a in L}
    for r in rs: m[r["label"]][r["pred"]] += 1
    return m
def report(rs, name):
    m = cm(rs); n = len(rs); acc = sum(m[a][a] for a in L) / n
    f1s = []
    print(f"== {name}  n={n}  acc={acc:.3f}")
    print("true/pred  " + "  ".join(L) + "   recall")
    for a in L:
        tot = sum(m[a].values()); rec = m[a][a] / tot if tot else float("nan")
        print(f"{a:9s} " + "  ".join(f"{m[a][b]:4d}" for b in L) + f"   {rec:.3f} ({tot})")
    for b in L:
        tp = m[b][b]; pp = sum(m[a][b] for a in L); tot = sum(m[b].values())
        p = tp / pp if pp else 0; r = tp / tot if tot else 0
        f1s.append(2 * p * r / (p + r) if p + r else 0)
        print(f"  {b}: precision={p:.3f} recall={r:.3f} f1={f1s[-1]:.3f}")
    print(f"  macro-F1={sum(f1s)/3:.3f}\n")
report(recs, "ALL")
for key in ("mag", "metal"):
    g = collections.defaultdict(list)
    for r in recs: g[r[key]].append(r)
    print(f"-- accuracy by {key}")
    for k, rs in sorted(g.items(), key=lambda kv: str(kv[0])): print(f"  {k:10s} {sum(r['label']==r['pred'] for r in rs)/len(rs):.3f} (n={len(rs)})")
    print()
g = collections.defaultdict(list)
for r in recs: g[r["cond"]].append(r)
print("-- accuracy by condition")
for k, rs in sorted(g.items(), key=lambda kv: str(kv[0])): print(f"  {k:12s} {sum(r['label']==r['pred'] for r in rs)/len(rs):.3f} (n={len(rs)})")
print("\nmean probs by true label:")
for a in L:
    rs = [r for r in recs if r["label"] == a]
    print(" ", a, {b: round(sum(r["probs"][b] for r in rs)/len(rs), 3) for b in L})
