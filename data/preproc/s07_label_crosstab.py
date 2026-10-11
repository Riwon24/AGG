"""s07: 라벨과 금속 종류 / 처리 조건(처리 방법 x 온도)의 얽힘을 교차표로 확인한다 (읽기 전용, 통계만; 분류기/예측 없음).

용어 (사용자 확인, 2026-10-12)
  금속 종류  = 1단계 폴더 (BASE, BASE-LC, BASE-LCSI, BASE-LSi, REF, REF-LN; 코드/CSV 의 `treatment` 열)
  처리 조건  = 2단계 폴더 (NPH1035, PH1180-1050 ...; 코드/CSV 의 `sample` 열) = 처리 방법(NPH, PH1180, PH1205, PH1230, PH1255) x 온도(1035, 1050)
산출: crosstab_*.csv, label_confounding.png, split_options.csv, log.txt
"""
import argparse
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sps

from common import Logger, PREPROC_DIR, load_config, savefig_retry

SCRIPT = "s07_label_crosstab"
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


def split_condition(name):
    """'PH1180-1035' -> ('PH1180', '1035'), 'NPH1035' -> ('NPH', '1035')"""
    if "-" in name:
        a, b = name.split("-", 1)
        return a, b
    m = re.match(r"^([A-Za-z]+)(\d{4})$", name)
    assert m, f"처리 조건 이름을 해석할 수 없음: {name}"
    return m.group(1), m.group(2)


def crosstab(rows, key, labels):
    t = defaultdict(Counter)
    for r in rows:
        t[r[key]][r["label"]] += 1
    return {k: [t[k][l] for l in labels] for k in sorted(t)}


def cramers_v(tab):
    m = np.array(list(tab.values()), float)
    chi2, p, dof, exp = sps.chi2_contingency(m, correction=False)
    return float(chi2), int(dof), float(p), float(np.sqrt(chi2 / (m.sum() * (min(m.shape) - 1)))), float((exp < 5).mean())


def write_tab(path, tab, labels, first):
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh); w.writerow([first] + [f"AGG{l}" for l in labels] + ["total"])
        for k, v in tab.items():
            w.writerow([k] + v + [sum(v)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", help="manifest_v1.csv (기본: outputs/s00b_meta_mag/manifest_v1.csv)")
    ap.add_argument("--output", help="outputs 폴더 (기본: config paths.outputs_dir)")
    ap.add_argument("--config", default=str(PREPROC_DIR / "config.yaml"))
    a = ap.parse_args()
    cfg = load_config(a.config)
    out = a.output or cfg["paths"]["outputs_dir"]
    manifest = Path(a.manifest or Path(out) / "s00b_meta_mag" / "manifest_v1.csv")
    use = cfg["experiment"]["use_mags"]
    log = Logger(out, SCRIPT)
    o = Path(out) / SCRIPT
    log("config:", cfg["_config_path"], "| manifest:", manifest)
    log("용어: 금속 종류 = 1단계 폴더(6종), 처리 조건 = 2단계 폴더(10종) = 처리 방법(5종) x 온도(2종)")
    rows = [r for r in csv.DictReader(open(manifest, encoding="utf-8-sig")) if r["mag_final"] in use]
    for r in rows:
        r["metal"], r["cond"] = r["treatment"], r["sample"]
        r["method"], r["temp"] = split_condition(r["cond"])
        r["method_temp"] = f"{r['method']}/{r['temp']}"
    labels = sorted({r["label"] for r in rows}); N = len(rows)
    log(f"대상 {N}장 ({'/'.join(use)}x), 금속 {len({r['metal'] for r in rows})}종 x 처리 조건 {len({r['cond'] for r in rows})}종, "
        f"처리 방법 {sorted({r['method'] for r in rows})}, 온도 {sorted({r['temp'] for r in rows})}")
    log(f"라벨: {dict(sorted(Counter(r['label'] for r in rows).items()))}")
    sel = Counter()
    for c in csv.DictReader(open(Path(out) / "s04_tile_grid" / "tile_counts.csv", encoding="utf-8-sig")):
        sel[c["image_id"]] += int(c["n_selected"])
    for r in rows:
        r["n_sel"] = sel[r["image_id"]]

    # ---------- [1] 처리 조건 x 라벨 ----------
    tc_ = crosstab(rows, "cond", labels)
    log("\n[1] 처리 조건(방법/온도) x 라벨 — 이미지 수 (조건당 금속 6종 x 1장 = 6장)")
    log(f"    {'처리 조건':<14} " + " ".join(f"AGG{l:<4}" for l in labels) + " | 다수 라벨")
    maj = {}
    for k, v in tc_.items():
        maj[k] = labels[int(np.argmax(v))]
        log(f"    {k:<14} " + " ".join(f"{x:<8}" for x in v) + f"| AGG{maj[k]} ({max(v)}/{sum(v)})")
    write_tab(o / "crosstab_condition_label.csv", tc_, labels, "condition")

    # ---------- [2] 처리 방법 x 온도 x 라벨 ----------
    log("\n[2] 처리 방법 x 온도 → 라벨 (이미지 수: AGG0/AGG1/AGG2)")
    methods = sorted({r["method"] for r in rows}); temps = sorted({r["temp"] for r in rows})
    mt = crosstab(rows, "method_temp", labels)
    log(f"    {'처리 방법':<10} " + " ".join(f"{t:>14}" for t in temps))
    for m in methods:
        log(f"    {m:<10} " + " ".join(f"{'/'.join(str(x) for x in mt[f'{m}/{t}']):>14}" for t in temps))
    log("    (표를 읽는 법: 'PH1180  0/5/0' = PH1180-1035 에서 AGG0 0장, AGG1 5장, AGG2 0장)")
    write_tab(o / "crosstab_method_temp_label.csv", mt, labels, "method/temp")
    write_tab(o / "crosstab_method_label.csv", crosstab(rows, "method", labels), labels, "method")
    write_tab(o / "crosstab_temp_label.csv", crosstab(rows, "temp", labels), labels, "temp")
    dom = {}
    for m in methods:
        for t in temps:
            v = mt[f"{m}/{t}"]; dom[(m, t)] = labels[int(np.argmax(v))]
    log("    온도 1035 -> 1050 에서 다수 라벨 변화: " + ", ".join(f"{m}: AGG{dom[(m, temps[0])]}->AGG{dom[(m, temps[1])]}" for m in methods))

    # ---------- [3] 금속 x 라벨 ----------
    tm = crosstab(rows, "metal", labels)
    log("\n[3] 금속 종류 x 라벨 — 이미지 수 (금속당 처리 조건 10종 x 1장 = 10장)")
    for k, v in tm.items():
        log(f"    {k:<10} " + " ".join(f"{x:<5}" for x in v) + f"합계 {sum(v)} | 비율 " + " / ".join(f"{100*x/sum(v):.0f}%" for x in v))
    write_tab(o / "crosstab_metal_label.csv", tm, labels, "metal")

    # ---------- [4] 처리 조건 x 금속 칸별 라벨 ----------
    conds = sorted({r["cond"] for r in rows}); metals = sorted({r["metal"] for r in rows})
    cell = {(r["cond"], r["metal"]): r["label"] for r in rows}
    log("\n[4] 처리 조건 x 금속 칸별 라벨 (칸당 이미지 1장)")
    log(f"    {'처리 조건':<14} " + " ".join(f"{m:<9}" for m in metals))
    for c in conds:
        log(f"    {c:<14} " + " ".join(f"AGG{cell.get((c, m), '-'):<6}" for m in metals))
    with open(o / "crosstab_condition_metal_label.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh); w.writerow(["condition"] + metals)
        for c in conds:
            w.writerow([c] + [cell.get((c, m), "") for m in metals])
    log("    같은 처리 조건 안에서 금속에 따라 라벨이 달라지는 경우 (다수 라벨과 다른 금속):")
    for c in conds:
        odd = [(m, cell[(c, m)]) for m in metals if (c, m) in cell and cell[(c, m)] != maj[c]]
        if odd:
            log(f"      {c} (다수 AGG{maj[c]}): " + ", ".join(f"{m}=AGG{l}" for m, l in odd))
    lower = Counter()
    for c in conds:
        for m in metals:
            if (c, m) in cell and cell[(c, m)] != maj[c]:
                lower[(m, "낮음" if int(cell[(c, m)]) < int(maj[c]) else "높음")] += 1
    log("    금속별 다수 라벨 대비 낮음/높음 횟수: " + ", ".join(f"{m} {d} {n}회" for (m, d), n in sorted(lower.items())))

    # ---------- [5] 요약 ----------
    base = max(Counter(r["label"] for r in rows).values()) / N
    log("\n[5] 요약 — 각 요인의 다수 라벨로 라벨을 맞히는 일치율 (전체 다수 클래스 기준선 {:.1f}%)".format(100 * base))
    log(f"    {'요인':<16} {'일치율':<14} Cramér's V (기대 빈도 5 미만 칸 비율; 표본이 작아 p 는 참고용)")
    for name, key in (("처리 조건(방법x온도)", "cond"), ("처리 방법", "method"), ("온도", "temp"), ("금속 종류", "metal")):
        tab = crosstab(rows, key, labels)
        pur = sum(max(v) for v in tab.values()) / N
        chi2, dof, p, v, low = cramers_v(tab)
        log(f"    {name:<16} {100*pur:5.1f}% ({sum(max(x) for x in tab.values())}/{N})  V={v:.3f} (chi2={chi2:.1f}, dof={dof}, p={p:.2g}, 5미만 칸 {100*low:.0f}%)")
    log("    (일치율이 높을수록 그 요인을 알면 라벨이 거의 정해진다)")

    # ---------- [6] 분할 단위 후보 ----------
    log("\n[6] 분할 단위 후보별: 그룹 하나를 통째로 test 로 뺄 때 (leave-one-group-out) 구성")
    options = {"처리 조건(10종, 방법x온도)": "cond", "처리 방법(5종, 온도 둘 다 묶음)": "method", "금속 종류(6종)": "metal"}
    opt_rows = []
    for oname, key in options.items():
        groups = sorted({r[key] for r in rows})
        single = 0; missing = 0
        log(f"  [{oname}] 그룹 {len(groups)}개")
        for g in groups:
            te = [r for r in rows if r[key] == g]; tr = [r for r in rows if r[key] != g]
            tel, trl = Counter(r["label"] for r in te), Counter(r["label"] for r in tr)
            miss = [l for l in labels if trl[l] == 0]
            single += len(tel) == 1; missing += bool(miss)
            opt_rows.append(dict(option=oname, test_group=g, test_images=len(te), test_labels=dict(sorted(tel.items())),
                                 train_images=len(tr), train_labels=dict(sorted(trl.items())), classes_missing_in_train=miss))
            log(f"      test={g:<12} test {len(te):>2}장 {dict(sorted(tel.items()))} | train {len(tr)}장 {dict(sorted(trl.items()))}" + (f" | train 에 없는 클래스 {miss}" if miss else ""))
        log(f"      -> test 가 한 클래스뿐인 그룹 {single}/{len(groups)}, train 에 클래스가 빠지는 그룹 {missing}/{len(groups)}")
    with open(o / "split_options.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(opt_rows[0].keys())); w.writeheader(); w.writerows(opt_rows)

    # ---------- 그림 ----------
    fig, axs = plt.subplots(1, 3, figsize=(17, 5), gridspec_kw=dict(width_ratios=[1, 1, 1.25]))
    for ax, tab, title, cm in ((axs[0], tc_, "처리 조건(방법/온도) x 라벨 (이미지 수)", "Blues"), (axs[1], tm, "금속 종류 x 라벨 (이미지 수)", "Oranges")):
        m = np.array(list(tab.values())); ax.imshow(m, cmap=cm, aspect="auto")
        ax.set_xticks(range(len(labels))); ax.set_xticklabels([f"AGG{l}" for l in labels]); ax.set_yticks(range(len(tab))); ax.set_yticklabels(list(tab))
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                ax.text(j, i, m[i, j], ha="center", va="center", color="white" if m[i, j] > m.max() / 2 else "black")
        ax.set_title(title)
    ax = axs[2]
    grid = np.array([[int(cell[(c, m)]) if (c, m) in cell else -1 for m in metals] for c in conds])
    cmap = matplotlib.colors.ListedColormap(["#dddddd", "#4c78a8", "#f58518", "#54a24b"])
    ax.imshow(grid + 1, cmap=cmap, vmin=0, vmax=3, aspect="auto")
    ax.set_xticks(range(len(metals))); ax.set_xticklabels(metals, rotation=30, ha="right"); ax.set_yticks(range(len(conds))); ax.set_yticklabels(conds)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.text(j, i, f"AGG{grid[i, j]}" if grid[i, j] >= 0 else "-", ha="center", va="center", color="white", fontsize=9)
    ax.set_title("처리 조건 x 금속: 칸별 라벨 (파랑 AGG0 / 주황 AGG1 / 초록 AGG2)")
    fig.suptitle("라벨과 처리 조건(방법x온도) / 금속 종류의 관계 (10x 60장)"); fig.tight_layout(); savefig_retry(fig, o / "label_confounding.png"); plt.close(fig)
    log(f"\n저장: {o}")


if __name__ == "__main__":
    main()
