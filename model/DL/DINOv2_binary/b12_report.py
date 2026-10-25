"""b12: 이진 라벨(AGG0 vs AGG1+2) 결과를 정확도 중심으로 정리한다 (새 학습 없음, numpy 와 matplotlib 만 필요).

- 평가 3가지(IMG = 이미지 단위 분할, LOCO, LOMO) x 16개 설정의 accuracy, balanced accuracy, macro-F1, 클래스별 재현율, AUROC.
- primary 의 이미지 단위/처리 조건 단위 부트스트랩 95% CI, 기준선과의 쌍 차이, LOCO/LOMO - 이미지 단위 분할 쌍 차이, 라벨 순열 검사.
- 사전 지정 해석 규칙: (1) primary - 다수 클래스(학습 fold) 의 이미지 단위 CI 가 0 을 제외하면 기준선보다 낫다고 적고 아니면 단정하지 않는다.
  (2) IMG 에서 primary - 처리 조건 암기(이진) 의 CI 가 0 을 제외하며 양수일 때만 "처리 조건 암기보다 낫다"고 적는다. (3) LOCO/LOMO - IMG 의 CI 가 0 을 제외하면 이미지 단위 분할이 낙관적이었다고 적는다.
출력: results_table.csv, paired_diff.csv, report.md, fig1_primary_binary.png, log.txt
"""
import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
import sys  # noqa: E402
sys.path.insert(0, str(HERE.parent / "exp_group_cv"))
from gcommon import Logger, load_images, write_csv  # noqa: E402
from gcommon import load_config as _lc  # noqa: E402

SCRIPT = "b12_report"
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
SCH_NAME = {"IMG": "이미지 단위 분할", "LOCO": "LOCO (조건 1개 제외)", "LOMO": "LOMO (방법 1개 제외)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = _lc(a.config or HERE / "config.yaml")
    out = Path(cfg["paths"]["outputs_dir"])
    log = Logger(out, SCRIPT)
    log("config:", cfg["_config_path"])
    img = load_images(cfg)
    ids = img["ids"]; pos = {i: k for k, i in enumerate(ids)}; N = len(ids)
    y = np.isin(img["y"], cfg["positive_labels"]).astype(int)
    conds = img["condition"]; cond_list = sorted(set(conds))
    pid = "tiles|" + "|".join([cfg["experiment"]["primary"][k] for k in ("features", "aggregator", "classifier")])

    def rd(p):
        return list(csv.DictReader(open(p, encoding="utf-8-sig")))
    summ = rd(out / "b10_run_binary_cv" / "cv_bin_summary.csv")
    foldacc = rd(out / "b10_run_binary_cv" / "cv_bin_fold_accuracy.csv")
    preds = rd(out / "b10_run_binary_cv" / "cv_bin_predictions.csv")
    bpred = rd(out / "b11_baselines_permutation" / "baseline_bin_predictions.csv")
    bsum = rd(out / "b11_baselines_permutation" / "baseline_bin.csv")
    perm = rd(out / "b11_baselines_permutation" / "bin_permutation_runs.csv")
    P = {}
    for r in preds:
        R = 10 if r["scheme"] == "IMG" else 1
        P.setdefault((r["scheme"], r["setting"]), np.full((N, R), -1, int))[pos[r["image_id"]], int(r["repeat"])] = int(r["pred"])
    assert all((v >= 0).all() for v in P.values())
    B = {}
    for r in bpred:
        R = 10 if (r["scheme"] == "IMG" and r["rule"] == "majority") else 1
        B.setdefault((r["scheme"], r["rule"]), np.full((N, R), -1, int))[pos[r["image_id"]], int(r["repeat"])] = int(r["pred"])
    assert all((v >= 0).all() for v in B.values())
    S = {(r["scheme"], r["setting"]): r for r in summ}
    BS = {(r["scheme"], r["rule"]): r for r in bsum}
    settings = [r["setting"] for r in summ if r["scheme"] == "IMG"]
    nb = cfg["bootstrap"]["n"]; rng = np.random.RandomState(cfg["bootstrap"]["seed"])
    members = {c: np.where(np.array(conds) == c)[0] for c in cond_list}
    idx_img = rng.randint(0, N, (nb, N))
    idx_clu = np.array([np.concatenate([members[cond_list[k]] for k in rng.randint(0, len(cond_list), len(cond_list))]) for _ in range(nb)])
    assert idx_clu.shape == (nb, N)

    def accvec(Pm, idx):
        C = (Pm.reshape(N, -1) == y[:, None]).astype(float)
        return C[idx].mean(axis=1).mean(axis=1)

    def ci(v):
        return np.percentile(v, [2.5, 97.5])
    f = lambda r, k: float(r[k])
    L = ["# 이진분류 확인: AGG0 vs AGG1+AGG2 (DINOv2_grid_augmentation 의 특징, 배경 증강 학습)\n",
         f"라벨을 AGG0 (음성 {int((y == 0).sum())}장) 대 AGG1+AGG2 (양성 {int((y == 1).sum())}장) 로 바꿔 같은 16개 설정을 학습했다. 이진 라벨의 다수 클래스(양성) 기준선은 {max(int((y == 0).sum()), int((y == 1).sum())) / N:.3f} 이다 (3분류는 0.467 이므로 3분류 accuracy 와 직접 비교할 수 없다). "
         "학습은 DINOv2_grid_augmentation 과 같고(이미지 단위 설정은 원본 + 증강 5개 버전, SL 은 타일마다 무작위 한 버전) 평가는 원본으로만 했다. 모델은 학습하지 않았다 (DINOv2 고정).\n"]
    L.append("## 1. 정확도 (primary = cls · mean · logreg 은 굵게; IMG 는 반복 10회 평균 ± SD)\n")
    L.append("| 설정 | IMG accuracy | LOCO | LOMO | IMG AUROC | LOCO AUROC | LOMO AUROC |\n|---|---|---|---|---|---|---|")
    for s in settings:
        nm = f"**{s}**" if s == pid else s
        L.append(f"| {nm} | {f(S[('IMG', s)], 'accuracy'):.3f} ± {f(S[('IMG', s)], 'accuracy_sd'):.3f} | {f(S[('LOCO', s)], 'accuracy'):.3f} | {f(S[('LOMO', s)], 'accuracy'):.3f} | "
                 f"{f(S[('IMG', s)], 'auroc'):.3f} | {f(S[('LOCO', s)], 'auroc'):.3f} | {f(S[('LOMO', s)], 'auroc'):.3f} |")
    L.append("\n(AUROC 는 양성 확률의 순위 지표로 임계값과 무관하다. LOCO/LOMO 는 fold 마다 모델이 달라 fold 사이 점수 비교가 정확하지 않아 참고용이다.)\n")
    L.append("### 기준선 (라벨만 쓰는 단순 규칙, 모델 없음)\n")
    L.append("| 규칙 | IMG | LOCO | LOMO |\n|---|---|---|---|")
    for rn, nm in (("majority", "학습 fold 다수 클래스"), ("condition_memorization", "같은 처리 조건의 다른 5장 다수 라벨 (처리 조건 암기)"), ("metal", "같은 금속 종류 다수 라벨"), ("temp", "같은 온도 다수 라벨"), ("sibling_temp", "같은 처리 방법의 다른 온도 조건 다수 라벨")):
        c = [f"{f(BS[(sc, rn)], 'accuracy'):.3f}" if (sc, rn) in BS else "-" for sc in ("IMG", "LOCO", "LOMO")]
        L.append(f"| {nm} | " + " | ".join(c) + " |")
    L.append("")

    # ---------- primary CI 와 쌍 차이 ----------
    log("[primary] 이진 accuracy 와 95% CI (이미지 단위 / 처리 조건 단위)")
    L.append("## 2. primary 의 CI와 쌍 차이\n")
    L.append("| 평가 | accuracy | 이미지 단위 95% CI | 처리 조건 단위 95% CI | balanced accuracy | recall AGG0 / AGG1+2 | AUROC |\n|---|---|---|---|---|---|---|")
    for sc in ("IMG", "LOCO", "LOMO"):
        Pm = P[(sc, pid)]
        ci_i, ci_c = ci(accvec(Pm, idx_img)), ci(accvec(Pm, idx_clu)); r = S[(sc, pid)]
        L.append(f"| {SCH_NAME[sc]} | {f(r, 'accuracy'):.3f} | [{ci_i[0]:.3f}, {ci_i[1]:.3f}] | [{ci_c[0]:.3f}, {ci_c[1]:.3f}] | {f(r, 'balanced_acc'):.3f} | {f(r, 'recall_neg'):.2f} / {f(r, 'recall_pos'):.2f} | {f(r, 'auroc'):.3f} |")
        log(f"  {sc}: {f(r, 'accuracy'):.3f} 이미지 [{ci_i[0]:.3f}, {ci_i[1]:.3f}] 조건 [{ci_c[0]:.3f}, {ci_c[1]:.3f}]")
    drows, rule = [], {}

    def pair(name, va, vb, d0):
        o = {}
        for mode, idx in (("image", idx_img), ("cluster", idx_clu)):
            o[mode] = ci(accvec(va, idx) - accvec(vb, idx))
        drows.append(dict(comparison=name, diff=round(d0, 4), image_lo=round(o["image"][0], 4), image_hi=round(o["image"][1], 4), cluster_lo=round(o["cluster"][0], 4), cluster_hi=round(o["cluster"][1], 4)))
        return o
    L.append("\n### 쌍 차이 (accuracy, 같은 재표본 2000회)\n")
    L.append("| 비교 | 차이 | 이미지 단위 95% CI | 처리 조건 단위 95% CI |\n|---|---|---|---|")
    pairs = []
    for sc in ("IMG", "LOCO", "LOMO"):
        pairs.append((f"규칙 1: {sc} primary - 다수 클래스", P[(sc, pid)], B[(sc, "majority")], f(S[(sc, pid)], "accuracy") - f(BS[(sc, "majority")], "accuracy"), ("r1", sc)))
    pairs.append(("규칙 2: IMG primary - 처리 조건 암기", P[("IMG", pid)], B[("IMG", "condition_memorization")], f(S[("IMG", pid)], "accuracy") - f(BS[("IMG", "condition_memorization")], "accuracy"), ("r2", "IMG")))
    for sc in ("LOCO", "LOMO"):
        pairs.append((f"규칙 3: {sc} primary - IMG primary", P[(sc, pid)], P[("IMG", pid)], f(S[(sc, pid)], "accuracy") - f(S[("IMG", pid)], "accuracy"), ("r3", sc)))
    for sc in ("LOCO",):
        pairs.append((f"참고: {sc} primary - 같은 처리 방법의 다른 온도 조건 규칙", P[(sc, pid)], B[(sc, "sibling_temp")], f(S[(sc, pid)], "accuracy") - f(BS[(sc, "sibling_temp")], "accuracy"), ("ref", sc)))
    for sc in ("LOCO", "LOMO"):          # 학습 fold 다수 클래스가 구조적으로 매우 낮아져서(아래 주의 문장) 가장 센 단순 기준선인 같은 온도 다수 라벨과도 비교한다 (사후 추가, 사전 지정 아님)
        pairs.append((f"사후 추가: {sc} primary - 같은 온도 다수 라벨", P[(sc, pid)], B[(sc, "temp")], f(S[(sc, pid)], "accuracy") - f(BS[(sc, "temp")], "accuracy"), ("temp", sc)))
    for name, va, vb, d0, key in pairs:
        o = pair(name, va, vb, d0); rule[key] = (d0, o)
        L.append(f"| {name} | {d0:+.3f} | [{o['image'][0]:+.3f}, {o['image'][1]:+.3f}] | [{o['cluster'][0]:+.3f}, {o['cluster'][1]:+.3f}] |")
        log(f"  {name}: {d0:+.3f} 이미지 [{o['image'][0]:+.3f}, {o['image'][1]:+.3f}] 조건 [{o['cluster'][0]:+.3f}, {o['cluster'][1]:+.3f}]")
    write_csv(f"{log.dir}/paired_diff.csv", drows)
    L.append("")

    # ---------- fold 별 ----------
    L.append("## 3. primary 의 fold 별 accuracy (LOCO, LOMO)\n")
    for sc in ("LOCO", "LOMO"):
        L.append(f"**{sc}**\n\n| 시험 단위 | 장수 | 시험 단위의 AGG0 / AGG1+2 장수 | accuracy |\n|---|---|---|---|")
        for r in [r for r in foldacc if r["scheme"] == sc and r["setting"] == pid]:
            m = np.array([(c == r["unit"]) if sc == "LOCO" else (mt == r["unit"]) for c, mt in zip(conds, img["method"])])
            L.append(f"| {r['unit']} | {r['n_test']} | {int((y[m] == 0).sum())} / {int((y[m] == 1).sum())} | {float(r['accuracy']):.3f} |")
        L.append("")

    # ---------- 순열 ----------
    L.append("## 4. 라벨 순열 검사 (20회, 섞은 라벨 기준 accuracy)\n")
    L.append("| 평가 | 설정 | 실제 accuracy | 순열 accuracy 평균 (범위) | 실제가 순열 최댓값 초과 |\n|---|---|---|---|---|")
    for sc in ("IMG", "LOCO", "LOMO"):
        for sid in cfg["permutation"]["settings"]:
            pa = np.array([float(r["accuracy"]) for r in perm if r["scheme"] == sc and r["setting"] == sid]); real = f(S[(sc, sid)], "accuracy")
            L.append(f"| {SCH_NAME[sc]} | {sid} | {real:.3f} | {pa.mean():.3f} ({pa.min():.3f}~{pa.max():.3f}) | {'예' if real > pa.max() else '아니오'} |")
            log(f"  순열 {sc} {sid}: 실제 {real:.3f} vs 순열 평균 {pa.mean():.3f} (최대 {pa.max():.3f})")
    L.append("\n- 이진 라벨은 다수 클래스가 0.533 이라 순열 accuracy 의 평균이 0.5 부근이다 (3분류의 1/3 과 다르다).\n")
    L.append("- **주의 (기준선)**: LOCO/LOMO 의 학습 fold 다수 클래스 기준선(0.100, 0.167)은 구조적으로 매우 낮다. 시험 단위(조건/방법)를 빼면 그 단위가 많이 가진 클래스가 학습에서는 소수가 되어, 다수 클래스 예측이 시험 단위의 다수와 반대가 되기 쉽기 때문이다. 이 기준선 대비 차이(+0.75, +0.67)는 크게 해석하면 안 되고, 더 센 단순 기준선(같은 온도 다수 라벨 0.667)과의 비교를 같이 본다.\n")

    # ---------- 규칙 적용 ----------
    def ex(k):
        d0, o = rule[k]; return (o["image"][0] > 0 or o["image"][1] < 0), d0, o
    L.append("## 5. 사전 지정 규칙의 적용\n")
    for sc in ("IMG", "LOCO", "LOMO"):
        e, d0, o = ex(("r1", sc))
        L.append(f"- 규칙 1 ({sc}): primary − 다수 클래스 {d0:+.3f}, 이미지 단위 CI [{o['image'][0]:+.3f}, {o['image'][1]:+.3f}] → **{'0 을 제외한다: 기준선보다 낫다' if e else '0 을 포함한다: 단정하지 않는다'}**")
    e, d0, o = ex(("r2", "IMG"))
    L.append(f"- 규칙 2 (IMG): primary − 처리 조건 암기 {d0:+.3f}, CI [{o['image'][0]:+.3f}, {o['image'][1]:+.3f}] → **{'0 을 제외하며 양수다: 처리 조건 암기보다 낫다' if (e and d0 > 0) else ('0 을 제외하며 음수다: 처리 조건 암기보다 낮다' if e else '0 을 포함한다: 처리 조건 암기보다 낫다고 말할 수 없다')}**")
    for sc in ("LOCO", "LOMO"):
        e, d0, o = ex(("r3", sc))
        L.append(f"- 규칙 3 ({sc}): {sc} − IMG {d0:+.3f}, CI [{o['image'][0]:+.3f}, {o['image'][1]:+.3f}] → **{'0 을 제외한다: 이미지 단위 분할이 낙관적이었다' if e else '0 을 포함한다: 낙관적이었다고 단정하지 않는다'}**")
    L.append("\n## 6. 한계\n")
    L.append("1. 이진 라벨로 합치면 AGG1 과 AGG2 의 구분은 평가하지 않는다. AGG1 은 3분류에서 가장 약했던 클래스이므로 이진 점수가 올라가도 AGG1 을 잘 구분한다는 뜻이 아니다.")
    L.append("2. 처리 조건 혼동은 그대로다. 이진 라벨도 처리 조건과 강하게 얽혀 있을 수 있어(처리 조건 암기 이진 기준선 참조) 이미지 단위 분할 점수는 처리 조건을 알아본 결과와 구분되지 않는다. LOCO, LOMO 가 이를 점검한다.")
    L.append("3. IMG 분할은 3분류 라벨로 층화된 기존 분할을 그대로 썼다 (이진 라벨 기준으로는 정확히 층화되지 않음). 이미지 60장이라 CI 가 넓다.")
    L.append("4. 모델은 학습하지 않았다 (DINOv2 고정, 분류기만 학습). 설정은 결과를 보기 전에 고정했다.")
    (out / "b12_report" / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    vals = [f(S[(sc, pid)], "accuracy") for sc in ("IMG", "LOCO", "LOMO")]
    cis = [ci(accvec(P[(sc, pid)], idx_img)) for sc in ("IMG", "LOCO", "LOMO")]
    ax.bar(range(3), vals, 0.5, color="#c0392b", yerr=[[v - c[0] for v, c in zip(vals, cis)], [c[1] - v for v, c in zip(vals, cis)]], capsize=4, ecolor="#333", label="primary (이진, 배경 증강 학습)")
    for i, v in enumerate(vals):
        ax.text(i, 0.03, f"{v:.3f}", ha="center", color="white", fontsize=10)
    for i, sc in enumerate(("IMG", "LOCO", "LOMO")):
        m_ = f(BS[(sc, "majority")], "accuracy"); ax.plot([i - 0.35, i + 0.35], [m_, m_], color="#777", ls="--"); ax.text(i + 0.36, m_, f"다수 클래스 {m_:.3f}", fontsize=7, va="center", color="#555")
    cm_ = f(BS[("IMG", "condition_memorization")], "accuracy"); ax.plot([-0.35, 0.35], [cm_, cm_], color="k", ls=":"); ax.text(0.36, cm_ + 0.01, f"처리 조건 암기 {cm_:.3f}", fontsize=7)
    ax.set_xticks(range(3)); ax.set_xticklabels([SCH_NAME[s] for s in ("IMG", "LOCO", "LOMO")]); ax.set_ylim(0, 1.0); ax.set_ylabel("이진 accuracy (막대 끝선 = 이미지 단위 95% CI)")
    ax.legend(fontsize=8, loc="upper right"); ax.set_title("AGG0 vs AGG1+2 이진분류: primary (cls · mean · logreg)", fontsize=10)
    fig.tight_layout(); fig.savefig(log.dir / "fig1_primary_binary.png", dpi=140); plt.close(fig)
    write_csv(f"{log.dir}/results_table.csv", summ)
    bad = []
    for fp in [Path(__file__), HERE / "config.yaml", out / "b12_report" / "report.md"]:
        for i, line in enumerate(fp.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if "forbidden_terms" in line:
                continue
            if any(t in line for t in cfg["forbidden_terms"]):
                bad.append((fp.name, i))
    log(f"\n금지어 검사: {'없음' if not bad else bad}\n저장: {out / 'b12_report'}")
    assert not bad


if __name__ == "__main__":
    main()
