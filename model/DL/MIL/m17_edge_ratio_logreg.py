"""m17: DINOv2 특징과 MIL 없이, 타일 구성(시편 가장자리/배경이 섞인 타일 대 시편만 있는 타일)에서 계산한 값만 feature 로 쓴 로지스틱 회귀 (이진: AGG0 vs AGG1+2). WSL, sklearn 필요.

목적: MIL(0.862 등)이 사실 "가장자리 타일 비율을 센 것"이었는지 가른다. 이 통제가 MIL 이나 baseline 의 점수에 가까우면 가장자리 구성만으로 설명이 가능하고, 훨씬 낮으면 MIL 이 그 이상을 보고 있다.
- 가장자리 타일 = 마스크 비율(mask_frac)이 100% 가 아닌 타일 (시편 가장자리와 배경이 들어 있음), 안쪽 타일 = 마스크 100% 타일. 분석 타일 2,591개(이미지당 36~48개) 기준.
- feature (결과를 보기 전에 고정): E1 (primary) = [가장자리 타일 비율]. E2 = E1 + [타일 수] (bag 크기), E3 = [가장자리 타일 비율, 평균 마스크 비율, 타일 수, 안쪽 타일 수] (구성 전체). E2, E3 는 시편 크기 같은 교란까지 포함한 탐색용.
- 분류기: baseline 과 같은 정의 (표준화 + L2 로지스틱 회귀, C=1, 클래스 균형, max_iter 1000). 이 feature 는 픽셀 내용이 아니라 마스크 기하에서 나오므로 배경 증강 버전과 무관하다 (버전마다 같다).
- 분할과 평가: 이미지 단위 e00 (5-fold x 10 반복), LOCO, LOMO, MIL 과 같은 지표, 이미지 단위/처리 조건 단위 부트스트랩 CI, MIL 과 baseline 대비 쌍 차이, 라벨 순열 검사 (E1, 20회).
해석 규칙 (사전 지정): E1 − MIL 의 이미지 단위 95% CI 가 0 을 포함하면 "가장자리 비율만으로 MIL 과 구분되지 않는다"(= 그 정도를 가장자리 구성으로 설명할 수 있다), 상한이 0 미만이면 "MIL 이 가장자리 비율 이상을 본다", 하한이 0 초과이면 "가장자리 비율이 MIL 보다 높다".
출력: edge_features.csv, edge_summary.csv, edge_predictions.csv, edge_permutation_runs.csv, report_edge_ratio.md, log.txt
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

from mcommon import Logger, check_folds, load_config, load_images, make_folds, metrics_bin, to_binary, write_csv   # (exp_binary 경로를 sys.path 에 넣는다)
from bcommon import fit_predict_proba_bin  # noqa: E402

SCRIPT = "m17_edge_ratio_logreg"
SETS = {"E1": ["edge_fraction"], "E2": ["edge_fraction", "n_tiles"], "E3": ["edge_fraction", "mean_mask_frac", "n_tiles", "n_full_tiles"]}
SCH = {"IMG": "이미지 단위 분할", "LOCO": "LOCO", "LOMO": "LOMO"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    out = Path(cfg["paths"]["outputs_dir"])
    log = Logger(out, SCRIPT)
    log("config:", cfg["_config_path"])
    img = load_images(cfg)
    ids = img["ids"]; y3 = img["y"]; y = to_binary(y3, cfg); N = len(ids); pos = {i: k for k, i in enumerate(ids)}
    conds = img["condition"]; cond_list = sorted(set(conds))
    # ---------- feature: 타일 구성만 (마스크 비율), 픽셀 내용과 DINOv2 특징은 쓰지 않는다 ----------
    mf = defaultdict(list)
    for r in csv.DictReader(open(cfg["paths"]["tiles_csv"], encoding="utf-8-sig")):
        if r["selected"] == "1":
            mf[r["image_id"]].append(float(r["mask_frac"]))
    feat = {}
    for iid in ids:
        m = np.array(mf[iid]); feat[iid] = dict(edge_fraction=float((m < 0.999999).mean()), mean_mask_frac=float(m.mean()), n_tiles=float(len(m)), n_full_tiles=float((m >= 0.999999).sum()))
    X_all = {k: np.array([[feat[i][c] for c in cols] for i in ids]) for k, cols in SETS.items()}
    write_csv(f"{log.dir}/edge_features.csv", [dict(image_id=i, label3=int(y3[k]), label_bin=int(y[k]), **{c: round(feat[i][c], 4) for c in SETS["E3"]}) for k, i in enumerate(ids)])
    log(f"이미지 {N}장, 분석 타일 {sum(len(v) for v in mf.values())}개. feature 는 마스크 비율에서만 계산 (DINOv2 특징, 픽셀 내용 미사용)")
    log("\n[라벨별 feature 요약] (이미지 평균 ± SD)")
    for lab, nm in ((0, "AGG0"), (1, "AGG1"), (2, "AGG2")):
        s = [k for k in range(N) if y3[k] == lab]
        log(f"  {nm} (n={len(s)}): 가장자리 타일 비율 {np.mean([feat[ids[k]]['edge_fraction'] for k in s]):.3f} ± {np.std([feat[ids[k]]['edge_fraction'] for k in s]):.3f} | "
            f"타일 수 {np.mean([feat[ids[k]]['n_tiles'] for k in s]):.1f} ± {np.std([feat[ids[k]]['n_tiles'] for k in s]):.1f} | 안쪽 타일 수 {np.mean([feat[ids[k]]['n_full_tiles'] for k in s]):.1f} | 평균 마스크 비율 {np.mean([feat[ids[k]]['mean_mask_frac'] for k in s]):.3f}")
    ef = X_all["E1"][:, 0]
    pos_ = ef[y == 1]; neg_ = ef[y == 0]
    auc = float(np.mean([(1.0 if p > n else 0.5 if p == n else 0.0) for p in pos_ for n in neg_]))
    log(f"  가장자리 타일 비율의 AUROC (양성 AGG1+2 대 음성 AGG0, 임계값과 무관): {auc:.3f} (0.5 = 구분 못 함, 1 이하이면 클수록 양성이 낮은 쪽에서도 구분됨 -> 방향 무관 |AUROC-0.5| 로 읽는다)")

    # ---------- 분할 ----------
    sp = list(csv.DictReader(open(cfg["paths"]["splits_csv"], encoding="utf-8-sig")))
    byk = defaultdict(lambda: {"train": [], "test": []})
    for r in sp:
        byk[(int(r["repeat"]), int(r["fold"]))][r["role"]].append(pos[r["image_id"]])
    reps = sorted({k[0] for k in byk}); nf = sorted({k[1] for k in byk})
    plan = {"IMG": [[(np.array(byk[(r, f)]["train"]), np.array(byk[(r, f)]["test"])) for f in nf] for r in reps]}
    for sch, unit in cfg["schemes"].items():
        folds = make_folds(img, img[unit]); check_folds(img, folds, sch)
        plan[sch] = [[(tr, te) for u, tr, te in folds]]
    guard = defaultdict(int)
    pred_rows, sum_rows = [], []
    P_all = {}
    for fs, X in X_all.items():
        for sch, reps_ in plan.items():
            R = len(reps_); P = np.full((N, R), -1); S = np.zeros((N, R))
            for r, folds in enumerate(reps_):
                for tr, te in folds:
                    assert set(tr.tolist()).isdisjoint(te.tolist()) and set(np.unique(y[tr]).tolist()) == {0, 1}
                    proba = fit_predict_proba_bin("logreg", X[tr], y[tr], X[te], cfg, guard)          # 학습 이미지의 feature 로만 표준화와 학습
                    P[te, r] = np.argmax(proba, 1); S[te, r] = proba[:, 1]
            assert (P >= 0).all()
            P_all[(fs, sch)] = P
            ms = [metrics_bin(y, P[:, r], S[:, r]) for r in range(R)]
            row = dict(feature_set=fs, features="+".join(SETS[fs]), scheme=sch, n_repeats=R)
            for k in ("accuracy", "balanced_acc", "macro_f1", "recall_neg", "recall_pos", "auroc"):
                row[k] = round(float(np.mean([m[k] for m in ms])), 4); row[k + "_sd"] = round(float(np.std([m[k] for m in ms])), 4)
            sum_rows.append(row)
            for r in range(R):
                for i in range(N):
                    pred_rows.append(dict(feature_set=fs, scheme=sch, repeat=r, image_id=ids[i], label=int(y[i]), pred=int(P[i, r]), prob=round(float(S[i, r]), 6)))
            log(f"[{fs} {'+'.join(SETS[fs])}] {sch}: accuracy {row['accuracy']:.3f}" + (f" ± {row['accuracy_sd']:.3f}" if R > 1 else "") + f" | balanced {row['balanced_acc']:.3f} | recall AGG0 {row['recall_neg']:.2f} / AGG1+2 {row['recall_pos']:.2f} | AUROC {row['auroc']:.3f}")
    write_csv(f"{log.dir}/edge_summary.csv", sum_rows)
    write_csv(f"{log.dir}/edge_predictions.csv", pred_rows)

    # ---------- 라벨 순열 검사 (E1) ----------
    cnt = {c: int((y == c).sum()) for c in (0, 1)}
    prm = []
    for sch in plan:
        for k in range(20):
            yp = np.random.RandomState(k).permutation(y)
            folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=k).split(np.zeros(N), yp)) if sch == "IMG" else plan[sch][0]
            Pp = np.full(N, -1)
            for tr, te in folds:
                assert set(np.unique(yp[tr]).tolist()) == {0, 1}
                Pp[te] = np.argmax(fit_predict_proba_bin("logreg", X_all["E1"][tr], yp[tr], X_all["E1"][te], cfg, guard), 1)
            m = metrics_bin(yp, Pp)
            prm.append(dict(scheme=sch, permutation=k, accuracy=round(m["accuracy"], 4), balanced_acc=round(m["balanced_acc"], 4)))
        pa = np.array([r["accuracy"] for r in prm if r["scheme"] == sch])
        log(f"[순열 E1 {sch}] 20회: accuracy 평균 {pa.mean():.3f} (범위 {pa.min():.3f}~{pa.max():.3f})")
    write_csv(f"{log.dir}/edge_permutation_runs.csv", prm)

    # ---------- MIL, baseline 과 비교 (쌍 차이) ----------
    def rd(p):
        return list(csv.DictReader(open(p, encoding="utf-8-sig")))
    def load_pred(rows_, sel=lambda r: True):
        Pm = {}
        for r in rows_:
            if sel(r):
                R = 10 if r["scheme"] == "IMG" else 1
                Pm.setdefault(r["scheme"], np.full((N, R), -1, int))[pos[r["image_id"]], int(r["repeat"])] = int(r["pred"])
        return Pm
    PM = load_pred(rd(out / "m10_train_eval_mil" / "mil_predictions.csv"))
    bset = cfg["primary_baseline_setting"]
    PB = load_pred(rd(cfg["paths"]["baseline_predictions"]), lambda r: r["setting"] == bset)
    SM = {r["scheme"]: r for r in rd(out / "m10_train_eval_mil" / "mil_summary.csv")}
    SB = {r["scheme"]: r for r in rd(cfg["paths"]["baseline_summary"]) if r["setting"] == bset}
    nb = cfg["bootstrap"]["n"]; rng = np.random.RandomState(cfg["bootstrap"]["seed"])
    members = {c: np.where(np.array(conds) == c)[0] for c in cond_list}
    idx_img = rng.randint(0, N, (nb, N))
    idx_clu = np.array([np.concatenate([members[cond_list[k]] for k in rng.randint(0, len(cond_list), len(cond_list))]) for _ in range(nb)])

    def accvec(Pm, idx):
        C = (Pm.reshape(N, -1) == y[:, None]).astype(float)
        return C[idx].mean(axis=1).mean(axis=1)

    def ci(v):
        return np.percentile(v, [2.5, 97.5])
    S_ = {(r["feature_set"], r["scheme"]): r for r in sum_rows}
    maj = {"IMG": 0.533, "LOCO": float("nan"), "LOMO": float("nan")}
    L = ["# 가장자리 비율만 feature 로 쓴 로지스틱 회귀 (DINOv2 특징과 MIL 없음) vs MIL, baseline (이진: AGG0 vs AGG1+AGG2)\n",
         "> feature 는 마스크 비율에서 나온 값뿐이다 (가장자리 타일 = 마스크 비율 100% 미만인 타일, 안쪽 타일 = 100% 타일). 픽셀 내용, DINOv2 특징은 쓰지 않았다. 목적은 MIL 의 점수가 \"가장자리 타일 비율을 센 것\"으로 설명되는지 가르는 것이다. "
         "분류기, 분할(이미지 단위 e00, LOCO, LOMO), 지표는 MIL 과 같다. feature 세트는 결과를 보기 전에 고정했다 (E1 = 가장자리 비율 하나가 primary).\n",
         "## 1. 관찰 (측정값)\n",
         "### 1.1 라벨별 feature (이미지 평균 ± SD)\n", "| 라벨 | 이미지 수 | 가장자리 타일 비율 | 타일 수 | 안쪽 타일 수 | 평균 마스크 비율 |\n|---|---|---|---|---|---|"]
    for lab, nm in ((0, "AGG0"), (1, "AGG1"), (2, "AGG2")):
        s = [k for k in range(N) if y3[k] == lab]
        L.append(f"| {nm} | {len(s)} | {np.mean([feat[ids[k]]['edge_fraction'] for k in s]):.3f} ± {np.std([feat[ids[k]]['edge_fraction'] for k in s]):.3f} | {np.mean([feat[ids[k]]['n_tiles'] for k in s]):.1f} ± {np.std([feat[ids[k]]['n_tiles'] for k in s]):.1f} | "
                 f"{np.mean([feat[ids[k]]['n_full_tiles'] for k in s]):.1f} ± {np.std([feat[ids[k]]['n_full_tiles'] for k in s]):.1f} | {np.mean([feat[ids[k]]['mean_mask_frac'] for k in s]):.3f} ± {np.std([feat[ids[k]]['mean_mask_frac'] for k in s]):.3f} |")
    L.append(f"\n가장자리 타일 비율의 AUROC (양성 대 음성): {auc:.3f}.\n")
    L.append("### 1.2 점수 (accuracy, 이진)\n")
    L.append("| 평가 | 가장자리 비율만 (E1, primary) | E2 (+ 타일 수) | E3 (구성 전체) | MIL (전체 타일) | baseline (패치 평균 + logreg) | 다수 클래스 |\n|---|---|---|---|---|---|---|")
    for sc in ("IMG", "LOCO", "LOMO"):
        sd = lambda r: f" ± {float(r['accuracy_sd']):.3f}" if sc == "IMG" else ""
        L.append(f"| {SCH[sc]} | {float(S_[('E1', sc)]['accuracy']):.3f}{sd(S_[('E1', sc)])} | {float(S_[('E2', sc)]['accuracy']):.3f} | {float(S_[('E3', sc)]['accuracy']):.3f} | {float(SM[sc]['accuracy']):.3f}{sd(SM[sc])} | {float(SB[sc]['accuracy']):.3f}{sd(SB[sc])} | "
                 + {"IMG": "0.533", "LOCO": "0.100 (구조적으로 낮음)", "LOMO": "0.167 (구조적으로 낮음)"}[sc] + " |")
    L.append("\n(IMG 는 반복 10회 평균 ± SD. LOCO/LOMO 의 학습 fold 다수 클래스 기준선은 시험 단위가 많이 가진 클래스가 학습에서 소수가 되는 구조라 낮다. 같은 온도 다수 라벨 기준선은 0.667.)\n")
    L.append("### 1.3 E1 의 신뢰구간과 쌍 차이 (accuracy, 같은 재표본 2000회)\n")
    L.append("| 평가 | E1 이미지 단위 95% CI | E1 − MIL | 이미지 단위 CI | 처리 조건 단위 CI | 판정 (사전 규칙) | E1 − baseline | 이미지 단위 CI |\n|---|---|---|---|---|---|---|---|")
    verdict = {}
    for sc in ("IMG", "LOCO", "LOMO"):
        P1 = P_all[("E1", sc)]
        c1 = ci(accvec(P1, idx_img)); dm = accvec(P1, idx_img) - accvec(PM[sc], idx_img); dmc = accvec(P1, idx_clu) - accvec(PM[sc], idx_clu); db = accvec(P1, idx_img) - accvec(PB[sc], idx_img)
        lo, hi = ci(dm); lc, hc = ci(dmc); bl, bh = ci(db)
        d0 = float(S_[("E1", sc)]["accuracy"]) - float(SM[sc]["accuracy"]); db0 = float(S_[("E1", sc)]["accuracy"]) - float(SB[sc]["accuracy"])
        v = "가장자리 비율이 MIL 보다 높다" if lo > 0 else ("MIL 이 가장자리 비율 이상을 본다" if hi < 0 else "MIL 과 구분되지 않는다 (CI 가 0 포함)")
        verdict[sc] = (d0, lo, hi, v)
        L.append(f"| {SCH[sc]} | [{c1[0]:.3f}, {c1[1]:.3f}] | {d0:+.3f} | [{lo:+.3f}, {hi:+.3f}] | [{lc:+.3f}, {hc:+.3f}] | {v} | {db0:+.3f} | [{bl:+.3f}, {bh:+.3f}] |")
        log(f"  E1 {sc}: {float(S_[('E1', sc)]['accuracy']):.3f} vs MIL {float(SM[sc]['accuracy']):.3f} 차이 {d0:+.3f} 이미지 CI [{lo:+.3f}, {hi:+.3f}] -> {v}")
    L.append("\n### 1.4 라벨 순열 검사 (E1, 20회)\n")
    L.append("| 평가 | 실제 E1 accuracy | 순열 accuracy 평균 (범위) | 실제가 순열 최댓값 초과 |\n|---|---|---|---|")
    for sc in ("IMG", "LOCO", "LOMO"):
        pa = np.array([r["accuracy"] for r in prm if r["scheme"] == sc]); real = float(S_[("E1", sc)]["accuracy"])
        L.append(f"| {SCH[sc]} | {real:.3f} | {pa.mean():.3f} ({pa.min():.3f}~{pa.max():.3f}) | {'예' if real > pa.max() else '아니오'} |")
    L.append("\n## 2. 해석 (가능한 설명이며 확정이 아님)\n")
    e1 = {sc: float(S_[("E1", sc)]["accuracy"]) for sc in ("IMG", "LOCO", "LOMO")}
    L.append("- 사전 규칙 적용: " + "; ".join(f"{SCH[sc]} E1 {e1[sc]:.3f} vs MIL {float(SM[sc]['accuracy']):.3f} → {verdict[sc][3]}" for sc in ("IMG", "LOCO", "LOMO")) + ".")
    L.append("- 가장자리 비율 하나(E1)만으로 얻은 점수가 MIL 의 점수에 얼마나 가까운지가 이 통제의 핵심이다. 위 표에서 E1 이 MIL 보다 확연히 낮으면 MIL 은 가장자리 구성 이상의 정보를 쓰고 있고, 비슷하면 MIL 의 점수를 가장자리 구성만으로 설명할 수 있다.")
    L.append("- E2, E3 는 시편 크기(타일 수, 안쪽 타일 수)까지 포함한 탐색이다. 이 값이 올라가면 점수의 일부가 시편이 프레임에서 차지하는 크기와 위치(촬영 조건과 처리 조건에 얽힐 수 있는 값)로도 설명된다는 뜻이다.")
    L.append("- 이 통제는 \"가장자리 구성이라는 한 가지 교란\"만 점검한다. 처리 조건 혼동, 시편 색과 조명 등 다른 교란은 이 결과로 배제되지 않는다.\n")
    L.append("## 3. 한계\n")
    L.append("- 이미지가 60장이라 CI 가 넓다. 가장자리 타일은 '마스크 비율 100% 미만'으로 정의했고 타일 격자가 이미지 모서리에서 시작하므로, 이 비율은 시편 윤곽 길이와 격자 위치에도 영향을 받는다.")
    L.append("- feature 가 1~4개인 단순한 모델이라 과적합 위험은 작지만 비선형 관계는 잡지 못한다 (로지스틱 회귀).")
    L.append("- 사전 규칙의 'MIL 과 구분되지 않는다'는 같은 점수라는 증명이 아니라 CI 가 넓어서 구분되지 않는다는 뜻이다.")
    (out / "m17_edge_ratio_logreg" / "report_edge_ratio.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    assert "시편 종류" not in "\n".join(L)
    log(f"\n누수 방지: 표준화와 학습은 학습 이미지의 feature 로만 (scaler assert {guard['scaler_checked']}회 통과), test 이미지는 변환과 예측에만 사용. 저장: {log.dir}")


if __name__ == "__main__":
    main()
