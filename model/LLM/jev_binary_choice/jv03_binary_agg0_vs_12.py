"""jv03: JEVision 타일 점수(p_flat)로 AGG0 vs AGG1+AGG2 이진 분류를 평가하고 혼동행렬을 그린다 (JEVision 추론은 다시 하지 않음, numpy 와 matplotlib 만 필요).

JEVision 은 학습하지 않는 zero-shot 모델이다. 이미 계산된 타일 점수 p_flat (평평한 부분이 있을 확률)을 이미지별로 합쳐 점수 하나를 만들고, 임계값 하나로 분류한다.
- 이미지 점수: 타일 p_flat 의 평균(mean_flat, primary) 또는 p_flat > 0.5 인 타일의 비율(frac_flat, 보조). 점수 >= 임계값 이면 양성(AGG1 또는 AGG2), 아니면 음성(AGG0).
- 임계값: 학습 fold 이미지의 점수값 중 학습 accuracy 가 처음으로 최대가 되는 값 (jv01 과 같은 규칙). test 점수는 예측에만 쓴다.
- 평가: DINOv2 이진 실험(exp_binary)과 같은 분할: 이미지 단위 층화 5-fold x 10 반복(e00), LOCO(처리 조건 1개 제외), LOMO(처리 방법 1개 제외). IMG 는 반복마다 구해 평균 +- SD.
- 기준선: 다수 클래스(학습 fold), 같은 처리 조건 다수 라벨(처리 조건 암기, 이미지 단위 분할만), 같은 온도 다수 라벨(LOCO/LOMO), 항상 음성. 이미지 단위/처리 조건 단위 부트스트랩 CI, primary 와 기준선의 쌍 차이, 라벨 순열 검사(20회).
- 비교: DINOv2 이진 primary (exp_binary, 학습한 logreg) 의 같은 평가 값을 표에 함께 싣는다. 학습 방식이 달라(zero-shot 임계값 하나 대 학습한 분류기) 같은 선상의 비교가 아니다.
- 혼동행렬: primary 점수의 예측으로 IMG(반복 10회 평균), LOCO, LOMO 를 한 장에 (칸 안에 행 비율과 괄호로 이미지 수).
출력: image_scores.csv, results_table.csv, bootstrap_ci.csv, paired_diff.csv, predictions.csv, fold_thresholds.csv, permutation_runs.csv, confusion_counts.csv, fig_confusion_jevision_binary.png, report.md, log.txt
사용: python jv03_binary_agg0_vs_12.py [--config config_agg0_vs_12.yaml | config_agg0_vs_12_1514.yaml]
"""
import argparse
import collections
import csv
import json
import os
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager as _fm  # noqa: E402

for _f in ("C:/Windows/Fonts/malgun.ttf", "/mnt/c/Windows/Fonts/malgun.ttf"):
    if os.path.exists(_f):
        _fm.fontManager.addfont(_f)
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "exp"))
from common import Logger, load_config, metrics, write_csv  # noqa: E402  (exp/common.py, 읽기 전용 import)

SCH = {"IMG": "이미지 단위 분할", "LOCO": "LOCO (조건 1개 제외)", "LOMO": "LOMO (방법 1개 제외)"}
LAB = ["AGG0", "AGG1+2"]


def fit_threshold(s, y):
    """후보 = 학습 이미지의 서로 다른 점수값(오름차순), 점수 >= 임계값 이면 양성. 학습 accuracy 가 엄격히 더 좋아질 때만 갱신 (처음 최대)."""
    best, thr = -1.0, None
    for a in np.unique(s):
        acc = float(np.mean((s >= a).astype(int) == y))
        if acc > best:
            best, thr = acc, float(a)
    return thr, best


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


def acc_mean(y, P, idx):
    return float(np.mean([np.mean(P[idx, r] == y[idx]) for r in range(P.shape[1])]))


def boot(y, P, groups, n, seed, others):
    """P: (N,R) primary 예측, others: {이름: (N,R)} 기준선. 같은 재표본으로 accuracy CI 와 primary - 기준선 쌍 차이 CI (이미지 단위, 처리 조건 단위)."""
    uniq = sorted(set(groups.tolist())); members = {g: np.where(groups == g)[0] for g in uniq}
    rng = np.random.RandomState(seed); N = len(y); out = {}
    for mode in ("image", "cluster"):
        a_p, a_o = [], {k: [] for k in others}
        for _ in range(n):
            if mode == "image":
                idx = rng.randint(0, N, N)
            else:
                pick = rng.randint(0, len(uniq), len(uniq)); idx = np.concatenate([members[uniq[i]] for i in pick])
            a_p.append(acc_mean(y, P, idx))
            for k, Q in others.items():
                a_o[k].append(acc_mean(y, Q, idx))
        out[mode] = dict(p=tuple(np.percentile(a_p, [2.5, 97.5])),
                         diff={k: (float(np.mean(np.array(a_p) - np.array(v))), *np.percentile(np.array(a_p) - np.array(v), [2.5, 97.5])) for k, v in a_o.items()})
    return out


def verdict(lo, hi):
    return "primary 가 높다 (CI 가 0 초과)" if lo > 0 else ("primary 가 낮다 (CI 가 0 미만)" if hi < 0 else "구분되지 않는다 (CI 가 0 포함)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config or HERE / "config_agg0_vs_12.yaml")
    log = Logger(cfg["paths"]["outputs_dir"], "")
    out = log.dir
    log("config:", cfg["_config_path"])

    meta = {r["image_id"]: r for r in csv.DictReader(open(cfg["paths"]["conditions_csv"], encoding="utf-8-sig"))}
    sp = list(csv.DictReader(open(cfg["paths"]["splits_csv"], encoding="utf-8-sig")))
    ids = sorted({r["image_id"] for r in sp}); pos = {i: k for k, i in enumerate(ids)}; N = len(ids)
    assert N == cfg["data"]["expected_images"] and set(ids) == set(meta)
    y3 = np.array([int(meta[i]["label"]) for i in ids]); y = np.isin(y3, cfg["data"]["positive_labels"]).astype(int)
    cond = np.array([meta[i]["condition"] for i in ids]); meth = np.array([meta[i]["method"] for i in ids]); temp = np.array([meta[i]["temp"] for i in ids])
    log(f"이미지 {N}장: AGG0 {(y3 == 0).sum()} / AGG1 {(y3 == 1).sum()} / AGG2 {(y3 == 2).sum()} -> 음성(AGG0) {(y == 0).sum()} / 양성(AGG1+2) {(y == 1).sum()}, 다수 클래스 {max((y == 0).mean(), (y == 1).mean()):.3f}")

    T = [json.loads(l) for l in open(cfg["paths"]["tiles"], encoding="utf-8")]
    assert len(T) == cfg["data"]["expected_tiles"], len(T)
    by = collections.defaultdict(list)
    for t in T:
        by[t["image_id"]].append(t)
    assert set(by) == set(ids)
    for iid, ts in by.items():
        assert all(int(t["label"][3:]) == y3[pos[iid]] for t in ts), f"라벨 불일치: {iid}"
    S = {"mean_flat": np.array([np.mean([t["p_flat"] for t in by[i]]) for i in ids]), "frac_flat": np.array([np.mean([t["p_flat"] > 0.5 for t in by[i]]) for i in ids])}
    log(f"타일 {len(T)}개 (이미지당 {min(len(v) for v in by.values())}~{max(len(v) for v in by.values())}개), JEVision 추론은 다시 하지 않고 저장된 p_flat 을 사용")
    write_csv(out / "image_scores.csv", [dict(image_id=i, label=int(y3[k]), binary=int(y[k]), condition=cond[k], method=meth[k], n_tiles=len(by[i]), mean_flat=round(S["mean_flat"][k], 4), frac_flat=round(S["frac_flat"][k], 4)) for k, i in enumerate(ids)])

    # ---------- 임계값 무관 분리도 (AUROC, 이미지 부트스트랩) ----------
    rng = np.random.RandomState(cfg["bootstrap"]["seed"]); auc_rows = {}
    log("\n[임계값 무관 AUROC (이미지 점수, 이미지 부트스트랩 95% CI)]")
    for sc in cfg["scores"]:
        for nm, fp, fn in (("AGG1+2 vs AGG0", lambda v: v >= 1, lambda v: v == 0), ("AGG1 vs AGG0", lambda v: v == 1, lambda v: v == 0),
                           ("AGG2 vs AGG0", lambda v: v == 2, lambda v: v == 0), ("AGG2 vs AGG1", lambda v: v == 2, lambda v: v == 1)):
            a0 = auc(S[sc][fp(y3)], S[sc][fn(y3)]); bs = []
            for _ in range(cfg["bootstrap"]["n"]):
                ix = rng.randint(0, N, N); v3 = y3[ix]
                if fp(v3).any() and fn(v3).any():
                    bs.append(auc(S[sc][ix][fp(v3)], S[sc][ix][fn(v3)]))
            lo, hi = np.percentile(bs, [2.5, 97.5]); auc_rows[(sc, nm)] = (a0, lo, hi)
            log(f"  {sc:<9} {nm:<16} {a0:.3f} [{lo:.3f}, {hi:.3f}]")

    # ---------- 분할 ----------
    folds = {"IMG": []}
    fd = collections.defaultdict(lambda: {"train": [], "test": []})
    for r in sp:
        fd[(int(r["repeat"]), int(r["fold"]))][r["role"]].append(pos[r["image_id"]])
    reps = sorted({k[0] for k in fd}); nf = sorted({k[1] for k in fd}); assert len(reps) == cfg["cv"]["n_repeats"] and len(nf) == cfg["cv"]["n_splits"]
    for r in reps:
        folds["IMG"].append([(f"r{r}f{f}", np.array(fd[(r, f)]["train"]), np.array(fd[(r, f)]["test"]), r) for f in nf])
    for nm, arr in (("LOCO", cond), ("LOMO", meth)):
        folds[nm] = [[(u, np.where(arr != u)[0], np.where(arr == u)[0], 0) for u in sorted(set(arr.tolist()))]]
    for nm, rs in folds.items():
        for fl in rs:
            assert sorted(np.concatenate([f[2] for f in fl]).tolist()) == list(range(N)), nm
            for _, tr, te, _ in fl:
                assert set(tr.tolist()).isdisjoint(te.tolist()) and len(tr) + len(te) == N
                assert len(set(y[tr].tolist())) == 2, "학습 fold 에 한 클래스만 있음"
    log("\n누수 방지: 임계값은 각 fold 의 학습 이미지의 점수와 라벨로만 고르고 test 점수는 예측에만 쓴다. 모든 fold 에서 train/test 서로소, 모든 이미지가 정확히 한 번 test, 학습에 두 클래스 존재를 assert 로 확인했다.")

    def run(sc, scheme):
        Rn = len(folds[scheme]); P = np.full((N, Rn), -1, int); rows = []
        for ri, fl in enumerate(folds[scheme]):
            for name, tr, te, rep in fl:
                thr, tacc = fit_threshold(S[sc][tr], y[tr])
                P[te, ri] = (S[sc][te] >= thr).astype(int)
                rows.append(dict(scheme=scheme, score=sc, repeat=ri, fold=name, threshold=round(thr, 4), train_acc=round(tacc, 4), test_acc=round(float(np.mean(P[te, ri] == y[te])), 4), n_test=len(te)))
        assert (P >= 0).all()
        return P, rows

    pred, thr_rows = {}, []
    for sc in cfg["scores"]:
        for scheme in folds:
            pred[(sc, scheme)], rows = run(sc, scheme); thr_rows += rows
    write_csv(out / "fold_thresholds.csv", thr_rows)

    # ---------- 기준선 ----------
    base = {}
    for scheme in folds:
        Rn = len(folds[scheme]); Pm, Pc, Pt = (np.full((N, Rn), -1, int) for _ in range(3))
        for ri, fl in enumerate(folds[scheme]):
            for name, tr, te, rep in fl:
                gm = int(y[tr].mean() >= 0.5); Pm[te, ri] = gm
                for i in te:
                    same = tr[cond[tr] == cond[i]]; Pc[i, ri] = (int(y[same].mean() > 0.5) if y[same].mean() != 0.5 else gm) if len(same) else gm
                    st = tr[temp[tr] == temp[i]]; Pt[i, ri] = int(y[st].mean() >= 0.5)
        base[scheme] = {"다수 클래스 (학습 fold)": Pm, "같은 온도 다수 라벨": Pt}
        if scheme == "IMG":
            base[scheme]["같은 처리 조건 다수 라벨 (처리 조건 암기)"] = Pc
    const = {"항상 AGG0 로 예측": np.zeros((N, 1), int)}
    nb, sd = cfg["bootstrap"]["n"], cfg["bootstrap"]["seed"]; sc0 = cfg["primary_score"]

    rows_t, rows_b, rows_d, rows_p, res = [], [], [], [], {}
    for scheme in folds:
        methods = [(f"JEVision 타일 {sc}", pred[(sc, scheme)]) for sc in cfg["scores"]] + list(base[scheme].items()) + list(const.items())
        others = {k: v for k, v in methods if not k.startswith("JEVision")}
        for name, Pm_ in methods:
            ms = [metrics(y, Pm_[:, r], labels=(0, 1)) for r in range(Pm_.shape[1])]
            rec = np.mean([m["recall"] for m in ms], 0)
            res[(scheme, name)] = dict(acc=float(np.mean([m["accuracy"] for m in ms])), sd=float(np.std([m["accuracy"] for m in ms])), f1=float(np.mean([m["macro_f1"] for m in ms])),
                                       bal=float(np.mean([(m["recall"][0] + m["recall"][1]) / 2 for m in ms])), rec=rec)
            for r in range(Pm_.shape[1]):
                for i in range(N):
                    rows_p.append(dict(scheme=scheme, method=name, repeat=r, image_id=ids[i], label=int(y3[i]), binary=int(y[i]), pred=int(Pm_[i, r])))
        for sc in cfg["scores"]:
            nm = f"JEVision 타일 {sc}"; b_ = boot(y, pred[(sc, scheme)], cond, nb, sd, others); res[(scheme, nm)]["ci"] = b_
            rows_b += [dict(scheme=scheme, method=nm, mode=m, acc_lo=round(b_[m]["p"][0], 4), acc_hi=round(b_[m]["p"][1], 4)) for m in ("image", "cluster")]
            for k in others:
                rows_d.append(dict(scheme=scheme, method=nm, baseline=k, diff=round(b_["image"]["diff"][k][0], 4), image_lo=round(b_["image"]["diff"][k][1], 4), image_hi=round(b_["image"]["diff"][k][2], 4),
                                   cluster_lo=round(b_["cluster"]["diff"][k][1], 4), cluster_hi=round(b_["cluster"]["diff"][k][2], 4), verdict_image=verdict(b_["image"]["diff"][k][1], b_["image"]["diff"][k][2])))
        for name, Pm_ in methods:
            if not name.startswith("JEVision"):
                res[(scheme, name)]["ci"] = boot(y, Pm_, cond, nb, sd, {})
    write_csv(out / "bootstrap_ci.csv", rows_b); write_csv(out / "paired_diff.csv", rows_d); write_csv(out / "predictions.csv", rows_p)
    for scheme in folds:
        for name in [k for (s, k) in res if s == scheme]:
            v = res[(scheme, name)]
            rows_t.append(dict(scheme=scheme, method=name, accuracy=round(v["acc"], 4), accuracy_sd=round(v["sd"], 4), balanced_acc=round(v["bal"], 4), macro_f1=round(v["f1"], 4),
                               recall_neg=round(float(v["rec"][0]), 4), recall_pos=round(float(v["rec"][1]), 4), ci_image=f"[{v['ci']['image']['p'][0]:.3f}, {v['ci']['image']['p'][1]:.3f}]",
                               ci_cluster=f"[{v['ci']['cluster']['p'][0]:.3f}, {v['ci']['cluster']['p'][1]:.3f}]"))
    write_csv(out / "results_table.csv", rows_t)
    log("\n[결과] accuracy (IMG: 반복 10회 평균 ± SD), 이미지 단위 95% CI")
    for scheme in folds:
        log(f" {SCH[scheme]}")
        for name in [k for (s, k) in res if s == scheme]:
            v = res[(scheme, name)]; ci = v["ci"]["image"]["p"]
            log(f"   {name:<42} acc {v['acc']:.3f}" + (f" ± {v['sd']:.3f}" if scheme == "IMG" else "") + f" [{ci[0]:.3f}, {ci[1]:.3f}] | balanced {v['bal']:.3f} | macro-F1 {v['f1']:.3f} | recall 음성/양성 {v['rec'][0]:.2f}/{v['rec'][1]:.2f}")
    log("\n[primary 와 기준선의 쌍 차이 (이미지 단위 CI)]")
    for r in rows_d:
        if r["method"].endswith(sc0):
            log(f"   {r['scheme']:5s} - {r['baseline']:<42} {r['diff']:+.3f} [{r['image_lo']:+.3f}, {r['image_hi']:+.3f}] -> {r['verdict_image']}")
    tp = [r for r in thr_rows if r["score"] == sc0 and r["scheme"] == "IMG"]
    log(f"\n[primary 임계값 (이미지 단위 분할 fold 별)] 평균 {np.mean([r['threshold'] for r in tp]):.3f} (범위 {min(r['threshold'] for r in tp):.3f}~{max(r['threshold'] for r in tp):.3f}), 학습 acc 평균 {np.mean([r['train_acc'] for r in tp]):.3f} vs test acc 평균 {np.mean([r['test_acc'] for r in tp]):.3f}")

    # ---------- 순열 검사 ----------
    npm, ps = cfg["permutation"]["n_permutations"], cfg["permutation"]["seed"]; prm = []
    for scheme in folds:
        for k in range(npm):
            yp = np.random.RandomState(ps + k).permutation(y)
            if scheme == "IMG":
                rs = np.random.RandomState(ps + k); fid = np.zeros(N, int)
                for c in (0, 1):
                    idx = rs.permutation(np.where(yp == c)[0]); fid[idx] = np.arange(len(idx)) % cfg["cv"]["n_splits"]
                fl = [(f, np.where(fid != f)[0], np.where(fid == f)[0], 0) for f in range(cfg["cv"]["n_splits"])]
            else:
                fl = folds[scheme][0]
            pr = np.full(N, -1)
            for _, tr, te, _ in fl:
                if len(set(yp[tr].tolist())) < 2:
                    pr[te] = int(yp[tr].mean() >= 0.5); continue
                thr, _ = fit_threshold(S[sc0][tr], yp[tr]); pr[te] = (S[sc0][te] >= thr).astype(int)
            assert (pr >= 0).all()
            prm.append(dict(scheme=scheme, permutation=k, accuracy=metrics(yp, pr, labels=(0, 1))["accuracy"]))
    write_csv(out / "permutation_runs.csv", prm); perm = {}
    for scheme in folds:
        v = np.array([r["accuracy"] for r in prm if r["scheme"] == scheme]); real = res[(scheme, f"JEVision 타일 {sc0}")]["acc"]
        perm[scheme] = (real, v.mean(), v.min(), v.max(), bool(real > v.max()))
        log(f"[순열 검사 {scheme}] 실제 accuracy {real:.3f}, 순열 {npm}회 평균 {v.mean():.3f} (범위 {v.min():.3f}~{v.max():.3f}), 실제가 순열 최댓값 초과: {'예' if real > v.max() else '아니오'}")

    # ---------- 혼동행렬 (primary) ----------
    cms, crow = {}, []
    for scheme in folds:
        P = pred[(sc0, scheme)]; cm = np.zeros((2, 2))
        for r in range(P.shape[1]):
            for i in range(N):
                cm[y[i], P[i, r]] += 1
        cms[scheme] = cm / P.shape[1]
        for i_ in range(2):
            crow.append(dict(scheme=scheme, true=LAB[i_], pred_AGG0=round(cms[scheme][i_, 0], 2), pred_AGG1_2=round(cms[scheme][i_, 1], 2)))
        log(f"[혼동행렬 {scheme}] (행 = 실제, 열 = 예측, " + ("반복 평균" if P.shape[1] > 1 else "60장 pooled") + f")  AGG0: {cms[scheme][0, 0]:.1f} {cms[scheme][0, 1]:.1f} | AGG1+2: {cms[scheme][1, 0]:.1f} {cms[scheme][1, 1]:.1f}")
    write_csv(out / "confusion_counts.csv", crow)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.0))
    for ax, scheme in zip(axes, folds):
        cm = cms[scheme]; rate = cm / cm.sum(1, keepdims=True); dec = 1 if len(folds[scheme]) > 1 else 0
        ax.imshow(rate, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{rate[i, j]:.0%}\n({cm[i, j]:.{dec}f}장)", ha="center", va="center", fontsize=14, color="white" if rate[i, j] > 0.5 else "black")
        ax.set_xticks(range(2)); ax.set_yticks(range(2)); ax.set_xticklabels(LAB); ax.set_yticklabels([f"{l}\n(n={int(cm[i].sum().round())})" for i, l in enumerate(LAB)])
        v = res[(scheme, f"JEVision 타일 {sc0}")]
        ax.set_xlabel("예측"); ax.set_ylabel("실제"); ax.set_title(f"{SCH[scheme]}\naccuracy {v['acc']:.3f}" + (f" ± {v['sd']:.3f}" if scheme == "IMG" else ""), fontsize=10)
    fig.suptitle(f"JEVision (zero-shot, 타일 {sc0} 평균 + 임계값 1개), 이진 AGG0 vs AGG1+2, 타일 {len(T):,}개 | IMG 는 반복 10회 평균", fontsize=11)
    fig.tight_layout(); fig.savefig(out / "fig_confusion_jevision_binary.png", dpi=130); plt.close(fig)

    # ---------- DINOv2 이진 primary 와 비교, 처리 조건별 ----------
    dino = {}
    dp = Path(cfg["paths"]["dinov2_binary_results"])
    if dp.exists():
        for r in csv.DictReader(open(dp, encoding="utf-8-sig")):
            if r["primary"] == "1":
                dino[r["scheme"]] = r
    wrong = (pred[(sc0, "IMG")] != y[:, None]).mean(1); grows = []
    for c in sorted(set(cond)):
        sel = cond == c
        grows.append(dict(condition=c, n_images=int(sel.sum()), n_AGG0=int((y3[sel] == 0).sum()), n_AGG1=int((y3[sel] == 1).sum()), n_AGG2=int((y3[sel] == 2).sum()),
                          score_mean=round(float(S[sc0][sel].mean()), 4), wrong_rate_img=round(float(wrong[sel].mean()), 4)))
    write_csv(out / "by_condition.csv", grows)
    log("\n[처리 조건별 mean_flat 평균과 오분류율 (이미지 단위 분할, primary)]")
    for r in grows:
        log(f"   {r['condition']:12s} AGG0/1/2 = {r['n_AGG0']}/{r['n_AGG1']}/{r['n_AGG2']}  score {r['score_mean']:.3f}  오분류율 {r['wrong_rate_img']:.2f}")

    # ---------- 보고서 ----------
    g = lambda s, n: res[(s, n)]
    md = [f"# JEVision 이진 평가 (AGG0 vs AGG1+AGG2), 타일 {len(T):,}개\n",
          f"JEVision 은 학습하지 않는 zero-shot 모델이다. 저장된 타일 점수 p_flat 의 이미지별 평균(`{sc0}`)을 점수로 쓰고 임계값 하나(학습 fold 이미지에서 고른 값)로 분류했다. 분할은 DINOv2 이진 실험과 같다. 이미지 {N}장: AGG0 {(y == 0).sum()} / AGG1+2 {(y == 1).sum()}.\n",
          "## 1. 결과 (accuracy)\n", "| 평가 | JEVision (primary) | 이미지 단위 95% CI | 처리 조건 단위 95% CI | balanced acc | recall AGG0 / AGG1+2 | DINOv2 이진 primary |", "|---|---|---|---|---|---|---|"]
    for s in folds:
        v = g(s, f"JEVision 타일 {sc0}"); d = dino.get(s)
        md.append(f"| {SCH[s]} | {v['acc']:.3f}" + (f" ± {v['sd']:.3f}" if s == "IMG" else "") + f" | [{v['ci']['image']['p'][0]:.3f}, {v['ci']['image']['p'][1]:.3f}] | [{v['ci']['cluster']['p'][0]:.3f}, {v['ci']['cluster']['p'][1]:.3f}] | {v['bal']:.3f} | {v['rec'][0]:.2f} / {v['rec'][1]:.2f} | " + (f"{float(d['accuracy']):.3f}" if d else "-") + " |")
    md += ["\n### 기준선 (모델 없음, 라벨만 쓰는 규칙)\n", "| 평가 | 방법 | accuracy |", "|---|---|---|"]
    for s in folds:
        for k in [k for (s_, k) in res if s_ == s and not k.startswith("JEVision")]:
            md.append(f"| {SCH[s]} | {k} | {g(s, k)['acc']:.3f} |")
    md += ["\n## 2. primary 와 기준선의 쌍 차이 (accuracy, 이미지 단위 CI)\n", "| 평가 | 기준선 | 차이 | 이미지 단위 95% CI | 판정 |", "|---|---|---|---|---|"]
    for r in rows_d:
        if r["method"].endswith(sc0):
            md.append(f"| {SCH[r['scheme']]} | {r['baseline']} | {r['diff']:+.3f} | [{r['image_lo']:+.3f}, {r['image_hi']:+.3f}] | {r['verdict_image']} |")
    md += ["\n## 3. 임계값 무관 AUROC (이미지 점수, 이미지 부트스트랩 95% CI)\n", "| 점수 | 비교 | AUROC |", "|---|---|---|"]
    for (sc, nm), (a0, lo, hi) in auc_rows.items():
        md.append(f"| {sc} | {nm} | {a0:.3f} [{lo:.3f}, {hi:.3f}] |")
    md += ["\n## 4. 라벨 순열 검사 (primary 점수, 임계값 선택부터 다시, 20회)\n", "| 평가 | 실제 accuracy | 순열 평균 | 순열 범위 | 실제가 순열 최댓값 초과 |", "|---|---|---|---|---|"]
    for s, (real, mu, lo, hi, ok) in perm.items():
        md.append(f"| {SCH[s]} | {real:.3f} | {mu:.3f} | {lo:.3f}~{hi:.3f} | {'예' if ok else '아니오'} |")
    md += ["\n## 5. 혼동행렬 (primary, 칸 안은 행 비율과 이미지 수)\n", "| 평가 | 실제 \\ 예측 | AGG0 | AGG1+2 |", "|---|---|---|---|"]
    for s in folds:
        cm = cms[s]; rt = cm / cm.sum(1, keepdims=True)
        for i_ in range(2):
            md.append(f"| {SCH[s] if i_ == 0 else ''} | {LAB[i_]} | {rt[i_, 0]:.0%} ({cm[i_, 0]:.1f}) | {rt[i_, 1]:.0%} ({cm[i_, 1]:.1f}) |")
    md += ["\n그림: `fig_confusion_jevision_binary.png`. IMG 는 반복 10회 평균, LOCO/LOMO 는 60장 pooled (소수는 반복 평균 때문).\n",
           "## 6. 처리 조건별 (이미지 단위 분할, primary)\n", "| 처리 조건 | 이미지 | AGG0 / AGG1 / AGG2 | 평균 점수 | 오분류율 |", "|---|---|---|---|---|"]
    for r in grows:
        md.append(f"| {r['condition']} | {r['n_images']} | {r['n_AGG0']} / {r['n_AGG1']} / {r['n_AGG2']} | {r['score_mean']:.3f} | {r['wrong_rate_img']:.2f} |")
    md += ["\n## 7. 한계\n",
           "- 학습 방식이 달라 DINOv2 이진 primary(학습한 logreg)와 같은 선상의 비교가 아니다. JEVision 쪽은 학습 fold 에서 고른 임계값 하나뿐이다.",
           "- 라벨이 처리 조건과 얽혀 있어(처리 조건 암기 기준선이 높다) 점수가 AGG 를 읽은 결과인지 처리 조건의 외형을 읽은 결과인지 이 평가로는 가르지 못한다.",
           "- 이미지 60장이라 신뢰구간이 넓다. 반복 10회는 독립이 아니다.",
           "- JEVision 의 p_flat 은 \"평평한 부분이 있을 확률\"이라는 프롬프트의 답이지 AGG 판정이 아니다. 조명 줄무늬나 스크래치 같은 아티팩트에도 반응한다는 이전 육안 관찰이 있다.\n"]
    (out / "report.md").write_text("\n".join(md), encoding="utf-8")
    log(f"\n저장: {out}")


if __name__ == "__main__":
    main()
