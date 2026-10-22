"""jv01: (config_bin.yaml 은 run_tiles.py 재실행 1,514타일, config_grid.yaml 은 grid_mode 로 DINOv2_grid_augmentation 타일 2,591개) JEVision 타일 점수(run_tiles.py 를 그대로 다시 실행한 결과)를 이진(AGG0+AGG1 vs AGG2)으로 평가한다 (numpy 와 matplotlib 만 필요, 모델 실행 없음).

JEVision 추론은 model/LLM/jev/scripts/run_tiles.py 와 같다 (프롬프트, 질문, 타일 1,514개, 모델). 바뀐 것은 분류 기준뿐이다.
- 이미지 점수: 타일 p_flat 의 평균(mean_flat, primary) 또는 p_flat > 0.5 인 타일의 비율(frac_flat, 보조). jev/scripts/agg_tiles.py 와 같은 정의.
- 분류: 점수 >= 임계값 이면 AGG2(양성), 아니면 AGG0 또는 AGG1(음성). 임계값은 학습 이미지의 점수값 중 학습 accuracy 가 처음으로 최대가 되는 값이며 학습 fold 이미지만으로 고른다.
- 평가: 이미지 단위 분할(e00, 층화 5-fold x 10 반복), LOCO(처리 조건 1개 제외, 10 fold), LOMO(처리 방법 1개 제외, 5 fold). 재학습은 없고 임계값 하나만 학습 fold 에서 고른다.
- 재실행 확인: 재실행한 p_flat 이 저장소의 기존 tiles_v1.jsonl 과 같은지 타일 단위로 비교한다.
- 기준선: 다수 클래스(학습 fold 기준), 처리 조건 암기(이미지 단위 분할), 처리 방법 규칙(AGG2 = NPH), 같은 온도 다수 라벨(LOCO/LOMO, 사후 추가).
- 이미지 단위/처리 조건 단위 부트스트랩 CI(accuracy), 기준선과의 쌍 차이, 라벨 순열 검사.
출력: tile_repro.csv, image_scores.csv, results_table.csv, bootstrap_ci.csv, paired_diff.csv, predictions.csv, fold_thresholds.csv, permutation_runs.csv, report.md, fig1_scores.png, fig2_accuracy.png, log.txt
"""
import argparse
import collections
import csv
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "exp"))
from common import Logger, load_config, metrics, sha256_file, write_csv  # noqa: E402  (exp/common.py, 읽기 전용 import)

SCRIPT = "jv01_binary_agg2"
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
SCH = {"IMG": "이미지 단위 분할", "LOCO": "LOCO (조건 1개 제외)", "LOMO": "LOMO (방법 1개 제외)"}


def fit_threshold(s, y):
    """후보 = 학습 이미지의 서로 다른 점수값(오름차순), 점수 >= 임계값 이면 양성. 학습 accuracy 가 엄격히 더 좋아질 때만 갱신 (처음 최대)."""
    best, thr = -1.0, None
    for a in np.unique(s):
        acc = float(np.mean((s >= a).astype(int) == y))
        if acc > best:
            best, thr = acc, float(a)
    return thr, best


def auc(pos, neg):
    return sum(1 if p > n else 0.5 if p == n else 0 for p in pos for n in neg) / (len(pos) * len(neg))


def mbin(y, p):
    return metrics(y, p, labels=(0, 1))


def acc_mean(y, P, idx):
    return float(np.mean([np.mean(P[idx, r] == y[idx]) for r in range(P.shape[1])]))


def boot(y, P, groups, n, seed, others):
    """P: (N,R) primary 예측, others: {이름: (N,R)} 기준선. 같은 재표본으로 accuracy CI 와 primary - 기준선 쌍 차이 CI (이미지 단위, 처리 조건 단위)."""
    uniq = sorted(set(groups.tolist())); members = {g: np.where(groups == g)[0] for g in uniq}
    rng = np.random.RandomState(seed); N = len(y); out = {}
    for mode in ("image", "cluster"):
        a_p, a_o, miss = [], {k: [] for k in others}, 0
        for _ in range(n):
            if mode == "image":
                idx = rng.randint(0, N, N)
            else:
                pick = rng.randint(0, len(uniq), len(uniq)); idx = np.concatenate([members[uniq[i]] for i in pick])
            if len(set(y[idx].tolist())) < 2:
                miss += 1
            a_p.append(acc_mean(y, P, idx))
            for k, Q in others.items():
                a_o[k].append(acc_mean(y, Q, idx))
        out[mode] = dict(p=tuple(np.percentile(a_p, [2.5, 97.5])), o={k: tuple(np.percentile(v, [2.5, 97.5])) for k, v in a_o.items()},
                         diff={k: (float(np.mean(np.array(a_p) - np.array(v))), *np.percentile(np.array(a_p) - np.array(v), [2.5, 97.5])) for k, v in a_o.items()}, missing=miss)
    return out


def verdict(lo, hi):
    return "primary 가 높다 (CI 가 0 초과)" if lo > 0 else ("primary 가 낮다 (CI 가 0 미만)" if hi < 0 else "구분되지 않는다 (CI 가 0 포함)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config or HERE / "config_bin.yaml")
    log = Logger(cfg["paths"]["outputs_dir"], "")
    out = log.dir
    log("config:", cfg["_config_path"])

    # ---------- 이미지, 라벨, 메타 ----------
    meta = {r["image_id"]: r for r in csv.DictReader(open(cfg["paths"]["conditions_csv"], encoding="utf-8-sig"))}
    sp = list(csv.DictReader(open(cfg["paths"]["splits_csv"], encoding="utf-8-sig")))
    ids = sorted({r["image_id"] for r in sp}); pos = {i: k for k, i in enumerate(ids)}; N = len(ids)
    assert N == cfg["data"]["expected_images"] and set(ids) == set(meta)
    y3 = np.array([int(meta[i]["label"]) for i in ids])
    y = np.isin(y3, cfg["data"]["positive_labels"]).astype(int)
    cond = np.array([meta[i]["condition"] for i in ids]); meth = np.array([meta[i]["method"] for i in ids]); temp = np.array([meta[i]["temp"] for i in ids])
    log(f"이미지 {N}장: AGG0 {(y3 == 0).sum()} / AGG1 {(y3 == 1).sum()} / AGG2 {(y3 == 2).sum()} -> 음성(AGG0+AGG1) {(y == 0).sum()} / 양성(AGG2) {(y == 1).sum()}, 다수 클래스 {max((y == 0).mean(), (y == 1).mean()):.3f}")
    pos_by_meth = {m: (int(y[meth == m].sum()), int((meth == m).sum())) for m in sorted(set(meth))}
    log("  처리 방법별 양성(AGG2) 수 / 이미지 수:", pos_by_meth)

    # ---------- 타일 점수와 재실행 확인 ----------
    T = [json.loads(l) for l in open(cfg["paths"]["tiles_rerun"], encoding="utf-8")]
    grid_mode = bool(cfg.get("grid_mode", False))
    if grid_mode:
        assert len(T) == cfg["data"]["expected_tiles"], len(T)
        diffs = np.zeros(1); repro_ok = True; tol = 0.0; R = None
        log(f"\n타일 {len(T)}개 (DINOv2_grid_augmentation 분석 타일, 기존 JEVision 결과와 같은 타일이 아니어서 재실행 비교는 하지 않음)")
    else:
        R = [json.loads(l) for l in open(cfg["paths"]["tiles_reference"], encoding="utf-8")]
        assert len(T) == cfg["data"]["expected_tiles"] == len(R), (len(T), len(R))
        key = lambda t: (t["image_id"], t["tile_id"])
        rr = {key(t): t for t in T}; assert len(rr) == len(T)
        d_rows, diffs, ext_diff = [], [], []
        for r in R:
            t = rr[key(r)]
            d = abs(t["p_flat"] - r["p_flat"]); diffs.append(d)
            ext_diff.append(max(abs(t["extent"][k] - r["extent"][k]) for k in r["extent"]))
            d_rows.append(dict(image_id=r["image_id"], tile_id=r["tile_id"], p_flat_reference=r["p_flat"], p_flat_rerun=t["p_flat"], abs_diff=round(d, 4)))
        write_csv(out / "tile_repro.csv", d_rows)
        diffs = np.array(diffs); tol = cfg["tile_repro_tolerance"]
        log(f"\n[재실행 확인] 타일 {len(T)}개, 기존 tiles_v1.jsonl(sha256 {sha256_file(cfg['paths']['tiles_reference'])[:12]}) 과 비교: p_flat 이 다른 타일 {(diffs > 0).sum()}개, 최대 차이 {diffs.max():.3f}, 허용({tol}) 초과 {(diffs > tol).sum()}개, extent 확률 최대 차이 {max(ext_diff):.3f}")
        repro_ok = bool((diffs <= tol).all())
    by = collections.defaultdict(list)
    for t in T:
        by[t["image_id"]].append(t)
    assert set(by) == set(ids)
    for iid, ts in by.items():
        assert all(int(t["label"][3:]) == y3[pos[iid]] for t in ts), f"라벨 불일치: {iid}"
    S = {"mean_flat": np.array([np.mean([t["p_flat"] for t in by[i]]) for i in ids]), "frac_flat": np.array([np.mean([t["p_flat"] > 0.5 for t in by[i]]) for i in ids])}
    if not grid_mode:
        S_ref = {"mean_flat": np.array([np.mean([t["p_flat"] for t in R if t["image_id"] == i]) for i in ids])}
        log(f"  이미지 점수(mean_flat) 최대 차이(재실행 - 기존): {np.abs(S['mean_flat'] - S_ref['mean_flat']).max():.4f}")
    au = dict(agg2_gt_agg0=auc(S["mean_flat"][y3 == 2], S["mean_flat"][y3 == 0]), agg2_gt_agg1=auc(S["mean_flat"][y3 == 2], S["mean_flat"][y3 == 1]))
    if not grid_mode:
        ref = cfg["data"]["readme_auroc_mean_flat"]
        log(f"  AUROC mean_flat: AGG2>AGG0 {au['agg2_gt_agg0']:.3f} (README {ref['agg2_gt_agg0']}), AGG2>AGG1 {au['agg2_gt_agg1']:.3f} (README {ref['agg2_gt_agg1']})")
        assert abs(au["agg2_gt_agg0"] - ref["agg2_gt_agg0"]) < 0.0015 and abs(au["agg2_gt_agg1"] - ref["agg2_gt_agg1"]) < 0.0015, "README 의 AUROC 가 재현되지 않음 -> 멈춤"
    log(f"  AUROC mean_flat: AGG2>AGG0 {au['agg2_gt_agg0']:.3f}, AGG2>AGG1 {au['agg2_gt_agg1']:.3f}")
    auc_bin = {sc: auc(S[sc][y == 1], S[sc][y == 0]) for sc in S}
    log("  이진 AUROC (AGG2 vs AGG0+1, 임계값 무관):", {k: round(v, 3) for k, v in auc_bin.items()})
    write_csv(out / "image_scores.csv", [dict(image_id=i, label=int(y3[k]), binary=int(y[k]), condition=cond[k], method=meth[k], n_tiles=len(by[i]), mean_flat=round(S["mean_flat"][k], 4), frac_flat=round(S["frac_flat"][k], 4)) for k, i in enumerate(ids)])

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
            allte = np.concatenate([f[2] for f in fl]); assert sorted(allte.tolist()) == list(range(N)), nm
            for _, tr, te, _ in fl:
                assert set(tr.tolist()).isdisjoint(te.tolist()) and len(tr) + len(te) == N
                assert len(set(y[tr].tolist())) == 2, "학습 fold 에 한 클래스만 있음"
    log("\n누수 방지: 임계값은 각 fold 의 학습 이미지의 점수와 라벨로만 고르고 test 점수는 예측에만 쓴다. 모든 fold 에서 train/test 서로소, 모든 이미지가 정확히 한 번 test, 학습에 두 클래스 존재를 assert 로 확인했다.")

    def run(sc, yy, scheme):
        """반환: P (N,R) 예측, thr 행 목록"""
        Rn = len(folds[scheme]); P = np.full((N, Rn), -1, int); rows = []
        for ri, fl in enumerate(folds[scheme]):
            for name, tr, te, rep in fl:
                thr, tacc = fit_threshold(S[sc][tr], yy[tr])
                P[te, ri] = (S[sc][te] >= thr).astype(int)
                rows.append(dict(scheme=scheme, score=sc, repeat=ri, fold=name, threshold=round(thr, 4), train_acc=round(tacc, 4), test_acc=round(float(np.mean(P[te, ri] == yy[te])), 4), n_test=len(te)))
        assert (P >= 0).all()
        return P, rows

    pred, thr_rows = {}, []
    for sc in cfg["scores"]:
        for scheme in folds:
            pred[(sc, scheme)], rows = run(sc, y, scheme)
            thr_rows += rows
    write_csv(out / "fold_thresholds.csv", thr_rows)

    # ---------- 기준선 ----------
    base = {s: {} for s in folds}
    for scheme in folds:
        Rn = len(folds[scheme])
        Pm, Pc, Pt, Pn = (np.full((N, Rn), -1, int) for _ in range(4))
        for ri, fl in enumerate(folds[scheme]):
            for name, tr, te, rep in fl:
                gm = int(y[tr].mean() >= 0.5); Pm[te, ri] = gm                                    # 학습 fold 다수 클래스
                for i in te:
                    same = tr[cond[tr] == cond[i]]; Pc[i, ri] = (int(y[same].mean() > 0.5) if y[same].mean() != 0.5 else gm) if len(same) else gm   # 같은 처리 조건 학습 이미지의 다수 라벨 (없으면 전체 다수)
                    st = tr[temp[tr] == temp[i]]; Pt[i, ri] = int(y[st].mean() >= 0.5)           # 같은 온도 학습 이미지의 다수 라벨
                    sm = tr[meth[tr] == meth[i]]; Pn[i, ri] = (int(y[sm].mean() > 0.5) if len(sm) and y[sm].mean() != 0.5 else int(meth[i] == "NPH"))   # 처리 방법 규칙: 같은 방법 학습 이미지의 다수 라벨 (방법을 통째로 뺀 경우 NPH 여부)
        base[scheme] = {"다수 클래스 (학습 fold)": Pm, "같은 온도 다수 라벨": Pt, "같은 처리 조건 다수 라벨 (처리 조건 암기)": Pc, "같은 처리 방법 다수 라벨 (방법 암기)": Pn}
        if scheme != "IMG":
            del base[scheme]["같은 처리 조건 다수 라벨 (처리 조건 암기)"]
        if scheme == "LOMO":
            del base[scheme]["같은 처리 방법 다수 라벨 (방법 암기)"]
        if scheme == "LOCO":
            pass
    const = {"항상 AGG0/AGG1 로 예측": np.zeros((N, 1), int)}
    nb, sd = cfg["bootstrap"]["n"], cfg["bootstrap"]["seed"]
    sc0 = cfg["primary_score"]

    rows_t, rows_b, rows_d, rows_p = [], [], [], []
    res = {}
    for scheme in folds:
        methods = [(f"JEVision 타일 {sc}", pred[(sc, scheme)]) for sc in cfg["scores"]] + list(base[scheme].items()) + list(const.items())
        others = {k: v for k, v in methods if not k.startswith("JEVision")}
        for name, Pm_ in methods:
            ms = [mbin(y, Pm_[:, r]) for r in range(Pm_.shape[1])]
            accs = [m["accuracy"] for m in ms]; f1 = [m["macro_f1"] for m in ms]; rec = np.mean([m["recall"] for m in ms], 0)
            bal = float(np.mean([(m["recall"][0] + m["recall"][1]) / 2 for m in ms]))
            res[(scheme, name)] = dict(acc=float(np.mean(accs)), sd=float(np.std(accs)), f1=float(np.mean(f1)), bal=bal, rec=rec)
            for r in range(Pm_.shape[1]):
                for i in range(N):
                    rows_p.append(dict(scheme=scheme, method=name, repeat=r, image_id=ids[i], label=int(y3[i]), binary=int(y[i]), pred=int(Pm_[i, r])))
        for sc in cfg["scores"]:
            nm = f"JEVision 타일 {sc}"
            b_ = boot(y, pred[(sc, scheme)], cond, nb, sd, others)
            res[(scheme, nm)]["ci"] = b_
            rows_b += [dict(scheme=scheme, method=nm, mode=m, acc_lo=round(b_[m]["p"][0], 4), acc_hi=round(b_[m]["p"][1], 4), missing_class_resamples=b_[m]["missing"]) for m in ("image", "cluster")]
            for k in others:
                rows_d.append(dict(scheme=scheme, method=nm, baseline=k, diff=round(b_["image"]["diff"][k][0], 4), image_lo=round(b_["image"]["diff"][k][1], 4), image_hi=round(b_["image"]["diff"][k][2], 4),
                                   cluster_lo=round(b_["cluster"]["diff"][k][1], 4), cluster_hi=round(b_["cluster"]["diff"][k][2], 4), verdict_image=verdict(b_["image"]["diff"][k][1], b_["image"]["diff"][k][2])))
        for name, _ in methods:
            if name.startswith("JEVision"):
                continue
            b_ = boot(y, dict(methods)[name], cond, nb, sd, {})
            res[(scheme, name)]["ci"] = b_
    write_csv(out / "bootstrap_ci.csv", rows_b); write_csv(out / "paired_diff.csv", rows_d); write_csv(out / "predictions.csv", rows_p)
    for scheme in folds:
        for name in [k for (s, k) in res if s == scheme]:
            v = res[(scheme, name)]
            rows_t.append(dict(scheme=scheme, method=name, accuracy=round(v["acc"], 4), accuracy_sd=round(v["sd"], 4), balanced_acc=round(v["bal"], 4), macro_f1=round(v["f1"], 4),
                               recall_neg=round(float(v["rec"][0]), 4), recall_pos=round(float(v["rec"][1]), 4),
                               ci_image=f"[{v['ci']['image']['p'][0]:.3f}, {v['ci']['image']['p'][1]:.3f}]", ci_cluster=f"[{v['ci']['cluster']['p'][0]:.3f}, {v['ci']['cluster']['p'][1]:.3f}]"))
    write_csv(out / "results_table.csv", rows_t)

    log("\n[결과] accuracy (IMG: 반복 10회 평균 ± SD), 이미지 단위 95% CI")
    for scheme in folds:
        log(f" {SCH[scheme]}")
        for name in [k for (s, k) in res if s == scheme]:
            v = res[(scheme, name)]; ci = v["ci"]["image"]["p"]
            log(f"   {name:<40} acc {v['acc']:.3f}" + (f" ± {v['sd']:.3f}" if scheme == "IMG" else "") + f" [{ci[0]:.3f}, {ci[1]:.3f}] | balanced {v['bal']:.3f} | macro-F1 {v['f1']:.3f} | recall 음성/양성 {v['rec'][0]:.2f}/{v['rec'][1]:.2f}")
    log("\n[primary 와 기준선의 쌍 차이 (이미지 단위 CI)]")
    for r in rows_d:
        if r["method"].endswith(sc0):
            log(f"   {r['scheme']:5s} - {r['baseline']:<40} {r['diff']:+.3f} [{r['image_lo']:+.3f}, {r['image_hi']:+.3f}] -> {r['verdict_image']}")
    tp = [r for r in thr_rows if r["score"] == sc0 and r["scheme"] == "IMG"]
    log(f"\n[primary 임계값 (이미지 단위 분할 fold 별)] 평균 {np.mean([r['threshold'] for r in tp]):.3f} (범위 {min(r['threshold'] for r in tp):.3f}~{max(r['threshold'] for r in tp):.3f}), 학습 acc 평균 {np.mean([r['train_acc'] for r in tp]):.3f} vs test acc 평균 {np.mean([r['test_acc'] for r in tp]):.3f}")

    # ---------- 순열 검사 (primary 점수, 임계값 선택부터 다시) ----------
    npm, ps = cfg["permutation"]["n_permutations"], cfg["permutation"]["seed"]
    prm = []
    for scheme in folds:
        for k in range(npm):
            yp = np.random.RandomState(ps + k).permutation(y)
            if scheme == "IMG":      # 층화 5-fold 를 새로 만든다 (순열 라벨 기준)
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
            m = mbin(yp, pr)
            prm.append(dict(scheme=scheme, permutation=k, accuracy=m["accuracy"], macro_f1=m["macro_f1"]))
    write_csv(out / "permutation_runs.csv", prm)
    perm = {}
    for scheme in folds:
        v = np.array([r["accuracy"] for r in prm if r["scheme"] == scheme]); real = res[(scheme, f"JEVision 타일 {sc0}")]["acc"]
        perm[scheme] = (real, v.mean(), v.min(), v.max(), bool(real > v.max()))
        log(f"[순열 검사 {scheme}] 실제 accuracy {real:.3f}, 순열 {npm}회 평균 {v.mean():.3f} (범위 {v.min():.3f}~{v.max():.3f}), 실제가 순열 최댓값 초과: {'예' if real > v.max() else '아니오'}")

    # ---------- 처리 조건별 ----------
    grows = []
    wrong = (pred[(sc0, "IMG")] != y[:, None]).mean(1)
    for c in sorted(set(cond)):
        sel = cond == c
        grows.append(dict(condition=c, n_images=int(sel.sum()), n_AGG0=int((y3[sel] == 0).sum()), n_AGG1=int((y3[sel] == 1).sum()), n_AGG2=int((y3[sel] == 2).sum()),
                          mean_flat_mean=round(float(S[sc0][sel].mean()), 4), wrong_rate_img=round(float(wrong[sel].mean()), 4)))
    write_csv(out / "by_condition.csv", grows)
    log("\n[처리 조건별 mean_flat 평균과 오분류율 (이미지 단위 분할, primary)]")
    for r in grows:
        log(f"   {r['condition']:12s} AGG0/1/2 = {r['n_AGG0']}/{r['n_AGG1']}/{r['n_AGG2']}  mean_flat {r['mean_flat_mean']:.3f}  오분류율 {r['wrong_rate_img']:.2f}")
    within = {}
    for m in ("PH1180-1050", "PH1205-1050"):
        sel = cond == m
        within[m] = [(round(float(S[sc0][k]), 3), int(y[k])) for k in np.where(sel)[0]]

    # ---------- 그림 ----------
    fig, ax = plt.subplots(figsize=(9, 4.2)); rs = np.random.RandomState(0)
    order = sorted(set(cond), key=lambda c: (c[:3] != "NPH", c))
    for xi, c in enumerate(order):
        for k in np.where(cond == c)[0]:
            ax.scatter(xi + rs.uniform(-0.2, 0.2), S[sc0][k], c=["#4c78a8", "#f2b134", "#c0392b"][y3[k]], s=26, alpha=0.85)
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=40, ha="right", fontsize=8); ax.set_ylabel("mean_flat (이미지 평균 p_flat)")
    for lab, col in zip(("AGG0", "AGG1", "AGG2"), ("#4c78a8", "#f2b134", "#c0392b")):
        ax.scatter([], [], c=col, label=lab)
    ax.legend(fontsize=8); ax.set_title("JEVision 이미지 점수를 처리 조건별로 (점 = 이미지 1장)", fontsize=10); fig.tight_layout(); fig.savefig(out / "fig1_scores.png", dpi=140); plt.close(fig)
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8), sharex=True)
    for ax, scheme in zip(axs, folds):
        names = [k for (s, k) in res if s == scheme]; ys = np.arange(len(names))[::-1]
        for y_, n in zip(ys, names):
            v = res[(scheme, n)]; ci = v["ci"]["image"]["p"]
            ax.barh(y_, v["acc"], color="#c0392b" if n.endswith(sc0) else ("#777" if not n.startswith("JEVision") else "#4c78a8"), xerr=[[v["acc"] - ci[0]], [ci[1] - v["acc"]]], capsize=2, ecolor="#333")
            ax.text(0.01, y_, f"{v['acc']:.2f}", va="center", fontsize=7, color="white")
        ax.set_yticks(ys); ax.set_yticklabels([n.replace("같은 ", "").replace(" 다수 라벨", "") for n in names], fontsize=7); ax.set_xlim(0, 1.05); ax.set_title(SCH[scheme], fontsize=9)
    fig.suptitle("JEVision 이진 (AGG0+1 vs AGG2) accuracy, 막대 끝선 = 이미지 단위 95% CI (빨강 = primary)", fontsize=10); fig.tight_layout(); fig.savefig(out / "fig2_accuracy.png", dpi=140); plt.close(fig)

    # ---------- 보고서 ----------
    P_ = lambda s: res[(s, f"JEVision 타일 {sc0}")]
    if grid_mode:
        md = ["# JEVision 이진 평가 (AGG0+AGG1 vs AGG2), DINOv2_grid_augmentation 의 분석 타일 2,591개 사용\n",
              "JEVision 추론 코드는 `model/LLM/jev/scripts/run_tiles.py` 와 같고 (`jv02_run_grid_tiles.py`) 타일 목록만 DINOv2_grid_augmentation 의 분석 타일(원본 이미지 60장, 224px 겹치지 않는 타일)로 바꿨다. 분류 기준은 `mean_flat` 에 임계값 하나를 학습 fold 이미지에서만 골라 적용하는 것이다. 재학습은 하지 않았다.\n",
              "## 1. 사실 (측정값)\n", "### 1.1 입력\n",
              f"- 타일 {len(T)}개 (이미지당 {min(len(v) for v in by.values())}~{max(len(v) for v in by.values())}개, 마스크 100% 타일과 가장자리 타일 포함).",
              f"- 이진 AUROC (AGG2 vs AGG0+1, 임계값 무관): mean_flat {auc_bin['mean_flat']:.3f}, frac_flat {auc_bin['frac_flat']:.3f}.\n",
              f"### 1.2 라벨과 처리 방법의 관계 (이 표가 해석의 핵심 전제)\n",
              "| 처리 방법 | AGG2 이미지 | 전체 이미지 |", "|---|---|---|"] + [f"| {m} | {a_} | {b_} |" for m, (a_, b_) in pos_by_meth.items()] + [
              "\n- 양성(AGG2) 18장 중 NPH 가 12장, PH1180-1050 이 5장, PH1205-1050 이 1장이다. 양성 라벨이 처리 방법(NPH)과 거의 겹친다.\n",
              "### 1.3 점수 (accuracy)\n",
              "| 평가 | JEVision mean_flat (primary) | JEVision frac_flat | 다수 클래스 (학습 fold) | 같은 온도 다수 라벨 | 처리 조건 암기 | 방법 암기 |", "|---|---|---|---|---|---|---|"]
    else:
        md = ["# JEVision 이진 평가 (AGG0+AGG1 vs AGG2), 분류 기준만 바꿈\n",
              "JEVision 추론은 `model/LLM/jev/scripts/run_tiles.py` 를 수정 없이 다시 실행한 결과다 (같은 프롬프트, 질문, 타일 1,514개, 모델). 바뀐 것은 분류 기준뿐이다: 이미지 점수 `mean_flat` 에 임계값 하나를 두고 **학습 fold 이미지에서만** 골랐다. 재학습은 하지 않았다.\n",
              "## 1. 사실 (측정값)\n", "### 1.1 재실행 확인\n",
              f"- 재실행한 타일 {len(T)}개의 `p_flat` 이 기존 `tiles_v1.jsonl` 과 다른 타일 {int((diffs > 0).sum())}개, 최대 차이 {diffs.max():.3f}. 허용 오차({tol}) {'이내' if repro_ok else '초과 있음'}.",
              f"- 이미지 점수 `mean_flat` 의 최대 차이 {np.abs(S['mean_flat'] - S_ref['mean_flat']).max():.4f}. AUROC 는 README 의 값이 재현된다 (AGG2>AGG0 {au['agg2_gt_agg0']:.3f}, AGG2>AGG1 {au['agg2_gt_agg1']:.3f}).",
              f"- 이진 AUROC (AGG2 vs AGG0+1, 임계값 무관): mean_flat {auc_bin['mean_flat']:.3f}, frac_flat {auc_bin['frac_flat']:.3f}.\n",
              f"### 1.2 라벨과 처리 방법의 관계 (이 표가 해석의 핵심 전제)\n",
              "| 처리 방법 | AGG2 이미지 | 전체 이미지 |", "|---|---|---|"] + [f"| {m} | {a_} | {b_} |" for m, (a_, b_) in pos_by_meth.items()] + [
              "\n- 양성(AGG2) 18장 중 NPH 가 12장, PH1180-1050 이 5장, PH1205-1050 이 1장이다. 양성 라벨이 처리 방법(NPH)과 거의 겹친다.\n",
              "### 1.3 점수 (accuracy)\n",
              "| 평가 | JEVision mean_flat (primary) | JEVision frac_flat | 다수 클래스 (학습 fold) | 같은 온도 다수 라벨 | 처리 조건 암기 | 방법 암기 |", "|---|---|---|---|---|---|---|"]
    def g(s, n):
        return f"{res[(s, n)]['acc']:.3f}" + (f" ± {res[(s, n)]['sd']:.3f}" if s == "IMG" and res[(s, n)]['sd'] > 0 else "") if (s, n) in res else "-"
    for s in folds:
        md.append(f"| {SCH[s]} | **{g(s, 'JEVision 타일 mean_flat')}** | {g(s, 'JEVision 타일 frac_flat')} | {g(s, '다수 클래스 (학습 fold)')} | {g(s, '같은 온도 다수 라벨')} | {g(s, '같은 처리 조건 다수 라벨 (처리 조건 암기)')} | {g(s, '같은 처리 방법 다수 라벨 (방법 암기)')} |")
    md += [f"\n- 항상 AGG0/AGG1 로 예측하면 accuracy {res[('IMG', '항상 AGG0/AGG1 로 예측')]['acc']:.3f} (전체 다수 클래스). LOCO/LOMO 의 \"학습 fold 다수 클래스\" 는 시험 단위가 많이 가진 쪽이 학습에서 소수가 되는 구조라 낮게 나오므로 기준으로 쓰지 않고 위 항상 예측과 같은 온도 규칙을 같이 본다.",
           "- 같은 온도 다수 라벨, 처리 조건 암기, 방법 암기는 이미지를 보지 않고 라벨만으로 맞히는 규칙이다 (사후 추가, 사전 지정 아님).\n",
           "### 1.4 신뢰구간과 기준선과의 쌍 차이 (accuracy, primary)\n",
           "| 평가 | primary accuracy 이미지 단위 95% CI | 처리 조건 단위 95% CI | 비교 대상 | primary − 비교 | 이미지 단위 CI | 판정 |", "|---|---|---|---|---|---|---|"]
    for s in folds:
        ci = P_(s)["ci"]
        for r in rows_d:
            if r["scheme"] == s and r["method"].endswith(sc0):
                md.append(f"| {SCH[s]} | [{ci['image']['p'][0]:.3f}, {ci['image']['p'][1]:.3f}] | [{ci['cluster']['p'][0]:.3f}, {ci['cluster']['p'][1]:.3f}] | {r['baseline']} | {r['diff']:+.3f} | [{r['image_lo']:+.3f}, {r['image_hi']:+.3f}] | {r['verdict_image']} |")
    md += [f"\n### 1.5 라벨 순열 검사 (primary, {npm}회, 임계값 선택부터 다시)\n", "| 평가 | 실제 accuracy | 순열 평균 (범위) | 실제가 순열 최댓값 초과 |", "|---|---|---|---|"]
    for s, (re_, mn, lo_, hi_, ex) in perm.items():
        md.append(f"| {SCH[s]} | {re_:.3f} | {mn:.3f} ({lo_:.3f}~{hi_:.3f}) | {'예' if ex else '아니오'} |")
    md += ["\n### 1.6 클래스별 재현율 (primary)\n", "| 평가 | balanced accuracy | recall 음성 (AGG0+1) | recall 양성 (AGG2) | macro-F1 |", "|---|---|---|---|---|"]
    for s in folds:
        v = P_(s); md.append(f"| {SCH[s]} | {v['bal']:.3f} | {v['rec'][0]:.2f} | {v['rec'][1]:.2f} | {v['f1']:.3f} |")
    md += ["\n### 1.7 같은 처리 조건 안에서 (양성과 음성이 섞인 조건)\n", "| 처리 조건 | (mean_flat, 양성 여부) 이미지별 |", "|---|---|"] + [f"| {m} | {v} |" for m, v in within.items()]
    md += [f"\n(처리 조건별 mean_flat 평균과 오분류율은 `by_condition.csv`, 그림은 `fig1_scores.png`.)\n",
           "## 2. 해석 (가능한 설명이며 확정이 아님)\n",
           "- 양성 라벨이 NPH 에 거의 겹치므로 \"AGG2 를 가려낸다\" 와 \"NPH 처리 이미지의 외형을 가려낸다\" 를 이 점수로는 구분할 수 없다. 방법 암기 규칙 (NPH 이면 AGG2) 이 라벨만으로 얼마나 맞히는지가 JEVision 의 점수를 읽는 기준선이다.",
           "- LOMO 에서 NPH 를 통째로 빼면 학습 fold 에는 AGG2 가 6장 (PH 두 조건)뿐이어서 임계값이 그 6장에 맞춰진다. 이 fold 의 점수는 일반화 평가라기보다 극단적으로 불균형한 조건의 시험이다.",
           "- 같은 처리 조건 안의 양성/음성 구분 (1.7) 은 이미지가 6장씩이라 근거가 매우 약하다.",
           "- JEVision 의 타일 점수는 이전 점검에서 조명 줄무늬, 스크래치, 이물질에도 반응했다 (`model/LLM/jev` README 5절). 이 점수가 높은 것이 결정립 변화를 본 것인지는 확인되지 않았다.\n",
           "## 3. 한계\n",
           "1. 이미지가 60장 (양성 18장) 이고 양성이 소수 조건에 몰려 있어 CI 가 넓다. 이미지 단위 CI 는 이미지를 독립으로 보아 처리 조건 단위 CI 보다 좁다.",
           "2. 임계값 선택, 점수 정의 (mean_flat), 프롬프트는 기존 `jev` 의 것을 그대로 썼고 이 평가 안에서 바꾸지 않았다. 점수 정의와 프롬프트는 60장 전체를 본 뒤 정해졌다.",
           ("3. 타일은 DINOv2_grid_augmentation 의 분석 타일 2,591개이고, 가장자리·배경이 섞인 타일을 포함한다. 이전 JEVision 평가(1,514개, 침식 마진 안쪽 타일)와 타일 집합이 다르다." if grid_mode else "3. 타일 점수는 preproc 의 선택 타일 1,514개에 대해서만 있다 (DINOv2_grid 의 2,591개와 다름)."),
           "4. 같은 온도, 처리 조건 암기, 방법 암기 기준선은 결과를 본 뒤 추가했다.",
           "5. 이 이진 라벨 (AGG0+1 vs AGG2) 로는 DINOv2 를 돌린 적이 없어 DINOv2 와 직접 비교하지 않았다. 확장(재학습, 다른 프롬프트 등)은 하지 않았다.\n",
           "## 4. 파일\n", "`results_table.csv`, `bootstrap_ci.csv`, `paired_diff.csv`, `predictions.csv`, `fold_thresholds.csv`, `permutation_runs.csv`, `by_condition.csv`, `image_scores.csv`, `tile_repro.csv`, `fig1_scores.png`, `fig2_accuracy.png`, `log.txt`"]
    (out / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log(f"\n보고서 저장: {out / 'report.md'}")
    bad = []
    for f in [Path(__file__), HERE / "config_bin.yaml", out / "report.md", out / "log.txt"]:
        for i, line in enumerate(f.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if "forbidden_terms" in line:
                continue
            if any(t in line for t in cfg.get("forbidden_terms", [])):
                bad.append((f.name, i))
    log(f"금지어 검사: {'없음' if not bad else bad}")
    assert not bad


if __name__ == "__main__":
    main()
