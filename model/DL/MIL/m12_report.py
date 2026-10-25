"""m12: MIL 과 기존 이진 baseline(패치 평균 + logreg, cls · mean · logreg)을 같은 표에 놓고 비교한다 (새 학습 없음, numpy 와 matplotlib 만 필요).

- 세 평가(IMG = 이미지 단위 분할, LOCO, LOMO)의 accuracy, balanced accuracy, macro-F1, 재현율, AUROC 를 baseline 과 나란히 정리.
- MIL 과 baseline 각각의 이미지 단위/처리 조건 단위 부트스트랩 95% CI, 둘의 쌍 차이(MIL - baseline)의 CI, 라벨 순열 검사, fold 별 accuracy.
- 사전 지정 판정 (정확도는 향상이 목표가 아니고 baseline 대비 떨어지지 않는지만 확인): 쌍 차이의 이미지 단위 95% CI 가 0 을 포함하면 "구분되지 않는다", 상한이 0 미만이면 "낮다", 하한이 0 초과이면 "높다".
  낮게 나오면 숨기거나 포장하지 않고 해석 가능성 트레이드오프로 그대로 보고한다.
- 누수 점검 목록은 m10 의 assert 와 설계에서 가져와 정리한다.
출력: results_table.csv, paired_diff.csv, report_scores.md, fig1_mil_vs_baseline.png, log.txt
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

SCRIPT = "m12_report"
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
SCH = {"IMG": "이미지 단위 분할", "LOCO": "LOCO (조건 1개 제외)", "LOMO": "LOMO (방법 1개 제외)"}


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
    bset = cfg["primary_baseline_setting"]

    def rd(p):
        return list(csv.DictReader(open(p, encoding="utf-8-sig")))
    msum = {r["scheme"]: r for r in rd(out / "m10_train_eval_mil" / "mil_summary.csv")}
    mpred = rd(out / "m10_train_eval_mil" / "mil_predictions.csv")
    fold = rd(out / "m10_train_eval_mil" / "mil_fold_accuracy.csv")
    stats = rd(out / "m10_train_eval_mil" / "mil_train_stats.csv")
    full_mode = cfg.get("tile_filter", "all") in ("full", "min_frac")
    if full_mode:       # 마스크 100% 타일만 쓰는 판: baseline 도 같은 타일 집합으로 다시 학습한 것(m15)을 쓴다. 이전(모든 타일) 결과는 참고용
        bsum = {r["scheme"]: r for r in rd(out / "m15_baseline_pooled_logreg" / "baseline_summary.csv")}
        bpred = rd(out / "m15_baseline_pooled_logreg" / "baseline_predictions.csv")
        ref_out = Path(cfg["paths"]["reference_outputs"])
        ref_m = {r["scheme"]: r for r in rd(ref_out / "m10_train_eval_mil" / "mil_summary.csv")}
        ref_b = {r["scheme"]: r for r in rd(cfg["paths"]["baseline_summary"]) if r["setting"] == bset}
    else:
        bsum = {r["scheme"]: r for r in rd(cfg["paths"]["baseline_summary"]) if r["setting"] == bset}
        bpred = [r for r in rd(cfg["paths"]["baseline_predictions"]) if r["setting"] == bset]
    perm = rd(out / "m11_permutation_mil" / "mil_permutation_runs.csv") if (out / "m11_permutation_mil" / "mil_permutation_runs.csv").exists() else []
    PM, PB, PRM = {}, {}, {}
    for r in mpred:
        R = 10 if r["scheme"] == "IMG" else 1
        PM.setdefault(r["scheme"], np.full((N, R), -1, int))[pos[r["image_id"]], int(r["repeat"])] = int(r["pred"])
    for r in bpred:
        R = 10 if r["scheme"] == "IMG" else 1
        PB.setdefault(r["scheme"], np.full((N, R), -1, int))[pos[r["image_id"]], int(r["repeat"])] = int(r["pred"])
    assert all((v >= 0).all() for v in list(PM.values()) + list(PB.values()))
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
    _tt = rd(out / "m10_train_eval_mil" / "attention_tiles.csv")
    _cnt = {}
    for r_ in _tt:
        _cnt[r_["image_idx"]] = _cnt.get(r_["image_idx"], 0) + 1
    if cfg.get("tile_filter", "all") == "min_frac":
        tf = f"마스크 비율 {cfg['min_mask_frac']} 이상 타일만"
        tdesc = f"마스크 비율 {cfg['min_mask_frac']} 이상 타일 (이미지당 {min(_cnt.values())}~{max(_cnt.values())}개, 전체 {len(_tt):,}개; 배경이 섞인 타일 중 시편 비율이 낮은 타일은 뺐다), baseline 도 같은 타일의 평균으로 다시 학습"
    else:
        tf = "마스크 100% 타일(시편만 있는 타일)만" if full_mode else "분석 타일 전체"
        tdesc = "마스크 100% 타일 (이미지당 16~33개, 전체 1,309개; 시편만 있어 배경 증강이 이 타일의 픽셀을 바꾸지 않는다), baseline 도 같은 타일의 평균으로 다시 학습"
    L = [f"# Attention-based MIL vs 패치 평균 + logreg (이진: AGG0 vs AGG1+AGG2), 점수 비교 ({tf} 사용)\n",
         "bag = 이미지 1장, instance = 그 이미지의 " + (tdesc if full_mode else "분석 타일") + "(DINOv2_grid_augmentation 의 cls 특징 768차원, 겹치지 않는 224px 타일). 같은 특징, 같은 분할(이미지 단위 분할 e00, LOCO, LOMO), 같은 학습/평가 방식(학습: 원본 + 배경 증강 5개 버전, 평가: 원본)이다. "
         "모델은 Ilse et al. (2018) 의 gated attention MIL (타일 FC 64 → gated attention 64 → softmax 가중합 → FC → sigmoid, dropout 0.25, L2 1e-3, 40 epoch 고정, 시드 5개 앙상블). 하이퍼파라미터는 결과를 보기 전에 고정했고 조정하지 않았다.\n",
         "## 1. 점수 (baseline = 기존 패치 평균 + logreg)\n"]
    L.append("| 평가 | 방법 | accuracy | balanced accuracy | macro-F1 | recall AGG0 / AGG1+2 | AUROC |\n|---|---|---|---|---|---|---|")
    rows = []
    for sc in ("IMG", "LOCO", "LOMO"):
        for nm, S in (("baseline (패치 평균 + logreg)", bsum[sc]), ("MIL (gated attention)", msum[sc])):
            sd = f" ± {f(S, 'accuracy_sd'):.3f}" if (sc == "IMG") else ""
            L.append(f"| {SCH[sc]} | {nm} | {f(S, 'accuracy'):.3f}{sd} | {f(S, 'balanced_acc'):.3f} | {f(S, 'macro_f1'):.3f} | {f(S, 'recall_neg'):.2f} / {f(S, 'recall_pos'):.2f} | {f(S, 'auroc'):.3f} |")
            rows.append(dict(scheme=sc, method=nm, accuracy=f(S, "accuracy"), balanced_acc=f(S, "balanced_acc"), macro_f1=f(S, "macro_f1"), recall_neg=f(S, "recall_neg"), recall_pos=f(S, "recall_pos"), auroc=f(S, "auroc")))
    write_csv(f"{log.dir}/results_table.csv", rows)
    L.append("\n(IMG 는 반복 10회 평균 ± SD, LOCO/LOMO 는 fold 를 합친 60장 예측 한 번. AUROC 는 양성 확률의 순위 지표이며 LOCO/LOMO 는 fold 마다 모델이 달라 참고용이다.)\n")
    if full_mode:
        L.append("### 참고: 분석 타일 전체를 쓴 이전 결과 (같은 모델과 절차)\n")
        L.append("| 평가 | 방법 | accuracy | balanced accuracy | macro-F1 | recall AGG0 / AGG1+2 | AUROC |\n|---|---|---|---|---|---|---|")
        for sc in ("IMG", "LOCO", "LOMO"):
            for nm, S_ in (("baseline (패치 평균 + logreg), 전체 타일", ref_b[sc]), ("MIL (gated attention), 전체 타일", ref_m[sc])):
                sd = f" ± {f(S_, 'accuracy_sd'):.3f}" if sc == "IMG" else ""
                L.append(f"| {SCH[sc]} | {nm} | {f(S_, 'accuracy'):.3f}{sd} | {f(S_, 'balanced_acc'):.3f} | {f(S_, 'macro_f1'):.3f} | {f(S_, 'recall_neg'):.2f} / {f(S_, 'recall_pos'):.2f} | {f(S_, 'auroc'):.3f} |")
        L.append("")

    L.append("## 2. 신뢰구간과 쌍 차이 (accuracy, 같은 재표본 2000회)\n")
    L.append("| 평가 | baseline 이미지 단위 95% CI | MIL 이미지 단위 95% CI | MIL − baseline | 이미지 단위 CI | 처리 조건 단위 CI | 판정 (이미지 단위 CI 기준) |\n|---|---|---|---|---|---|---|")
    drows, verdict = [], {}
    for sc in ("IMG", "LOCO", "LOMO"):
        cb, cm = ci(accvec(PB[sc], idx_img)), ci(accvec(PM[sc], idx_img))
        d = accvec(PM[sc], idx_img) - accvec(PB[sc], idx_img); dc = accvec(PM[sc], idx_clu) - accvec(PB[sc], idx_clu)
        lo, hi = ci(d); lc, hc = ci(dc); d0 = f(msum[sc], "accuracy") - f(bsum[sc], "accuracy")
        v = "높다 (CI 하한 > 0)" if lo > 0 else ("낮다 (CI 상한 < 0)" if hi < 0 else "구분되지 않는다 (CI 가 0 포함)")
        verdict[sc] = (d0, lo, hi, v)
        L.append(f"| {SCH[sc]} | [{cb[0]:.3f}, {cb[1]:.3f}] | [{cm[0]:.3f}, {cm[1]:.3f}] | {d0:+.3f} | [{lo:+.3f}, {hi:+.3f}] | [{lc:+.3f}, {hc:+.3f}] | {v} |")
        drows.append(dict(scheme=sc, diff=round(d0, 4), image_lo=round(lo, 4), image_hi=round(hi, 4), cluster_lo=round(lc, 4), cluster_hi=round(hc, 4), verdict=v))
        log(f"  {sc}: baseline {f(bsum[sc], 'accuracy'):.3f} MIL {f(msum[sc], 'accuracy'):.3f} 차이 {d0:+.3f} 이미지 CI [{lo:+.3f}, {hi:+.3f}] 조건 CI [{lc:+.3f}, {hc:+.3f}] -> {v}")
    write_csv(f"{log.dir}/paired_diff.csv", drows)
    L.append("")

    L.append("## 3. fold 별 accuracy (MIL, LOCO/LOMO)\n")
    for sc in ("LOCO", "LOMO"):
        base_fold = {}
        for r in bpred:
            pass
        L.append(f"**{sc}**\n\n| 시험 단위 | 장수 | MIL accuracy | baseline accuracy |\n|---|---|---|---|")
        for r in [r for r in fold if r["scheme"] == sc]:
            unit = r["unit"]
            m = np.array([(c == unit) if sc == "LOCO" else (mt == unit) for c, mt in zip(conds, img["method"])])
            bacc = float(np.mean(PB[sc][m, 0] == y[m]))
            L.append(f"| {unit} | {r['n_test']} | {float(r['accuracy']):.3f} | {bacc:.3f} |")
        L.append("")

    L.append("## 4. 라벨 순열 검사 (MIL, 20회, 섞은 라벨 기준 accuracy)\n")
    if perm:
        L.append("| 평가 | 실제 MIL accuracy | 순열 accuracy 평균 (범위) | 실제가 순열 최댓값 초과 |\n|---|---|---|---|")
        for sc in ("IMG", "LOCO", "LOMO"):
            pa = np.array([float(r["accuracy"]) for r in perm if r["scheme"] == sc]); real = f(msum[sc], "accuracy")
            L.append(f"| {SCH[sc]} | {real:.3f} | {pa.mean():.3f} ({pa.min():.3f}~{pa.max():.3f}) | {'예' if real > pa.max() else '아니오'} |")
            log(f"  순열 {sc}: 실제 {real:.3f} vs 순열 평균 {pa.mean():.3f} (최대 {pa.max():.3f})")
        L.append("\n- 순열마다 시드 1개로 학습했고(본 평가는 시드 5개 앙상블), 이진 라벨의 다수 클래스가 0.533 이라 순열 accuracy 평균이 0.5 부근이다.\n")
    else:
        L.append("(순열 검사 결과 없음)\n")

    L.append("## 5. 누수 점검 (m10 의 assert 와 설계)\n")
    tr = [r for r in stats]
    L.append("- 분할: 기존 e00 이미지 단위 분할, LOCO, LOMO 를 그대로 사용했고 모든 repeat 에서 모든 이미지가 정확히 한 번 test, train/test 이미지 서로소, train 에 두 클래스가 있음을 assert 로 확인했다.")
    L.append("- 학습 bag 은 학습 이미지의 모든 버전(원본 + 배경 증강 5개)에서만 만들었고, test 이미지의 어떤 버전도 학습에 들어가지 않았음을 assert 로 확인했다.")
    L.append("- 표준화 평균/표준편차는 학습 bag 의 타일로만 구했다 (타일 수가 학습 bag 타일 수와 같음을 assert). test 타일은 변환에만 쓰였다.")
    L.append("- 하이퍼파라미터(임베딩 64, attention 64, dropout 0.25, L2 1e-3, lr 1e-3, 40 epoch)는 결과를 보기 전에 config 에 고정했고 test 결과나 검증 손실로 조정하거나 조기 종료하지 않았다.")
    L.append("- test 점수는 원본(k=0) bag 만으로 계산했다. 다른 버전 bag 의 attention 은 해석 분석용으로만 저장했고 예측에 쓰지 않았다.")
    L.append(f"- 라벨 순열 검사: 라벨을 섞으면 우연 수준으로 떨어지는지 확인했다 (4절). 학습 bag 에서의 마지막 epoch 평균 학습 bag accuracy 는 {np.mean([float(r['train_acc']) for r in tr]):.3f} 로 학습 데이터에는 거의 맞춰진다 (참고, 평가 지표 아님).")
    L.append("- 처리 조건 혼동은 이 점검으로 가려지지 않는다. LOCO/LOMO 가 그 부분을 점검하는 평가다.\n")

    L.append("## 6. 요약\n")
    for sc in ("IMG", "LOCO", "LOMO"):
        d0, lo, hi, v = verdict[sc]
        L.append(f"- {SCH[sc]}: MIL {f(msum[sc], 'accuracy'):.3f} vs baseline {f(bsum[sc], 'accuracy'):.3f} (차이 {d0:+.3f}, 이미지 단위 CI [{lo:+.3f}, {hi:+.3f}]) → **{v}**")
    L.append("- 이 비교는 점수만 본다. MIL 의 목적은 정확도 향상이 아니라 attention 해석 가능성이므로, 점수가 baseline 보다 낮더라도 숨기거나 포장하지 않고 이 표대로 보고한다. attention 의 관찰과 해석은 별도 보고서 `report_attention.md` 에 있다.")
    (out / "m12_report" / "report_scores.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    w = 0.36
    for k, (nm, col, S, Pm) in enumerate((("baseline (패치 평균 + logreg)", "#4c78a8", bsum, PB), ("MIL (gated attention)", "#c0392b", msum, PM))):
        vals = [f(S[sc], "accuracy") for sc in ("IMG", "LOCO", "LOMO")]
        cis = [ci(accvec(Pm[sc], idx_img)) for sc in ("IMG", "LOCO", "LOMO")]
        xs = np.arange(3) + (k - 0.5) * w
        ax.bar(xs, vals, w, color=col, label=nm, yerr=[[v - c[0] for v, c in zip(vals, cis)], [c[1] - v for v, c in zip(vals, cis)]], capsize=3, ecolor="#333")
        for x_, v_ in zip(xs, vals):
            ax.text(x_, 0.03, f"{v_:.3f}", ha="center", color="white", fontsize=9)
    ax.axhline(0.533, color="#777", ls="--", lw=1); ax.text(2.45, 0.54, "다수 클래스 0.533", fontsize=7, ha="right", color="#555")
    ax.set_xticks(range(3)); ax.set_xticklabels([SCH[s] for s in ("IMG", "LOCO", "LOMO")]); ax.set_ylim(0, 1.0); ax.set_ylabel("이진 accuracy (막대 끝선 = 이미지 단위 95% CI)")
    ax.legend(fontsize=8, loc="upper right"); ax.set_title("이진 분류 (AGG0 vs AGG1+2): MIL 과 기존 baseline", fontsize=10)
    fig.tight_layout(); fig.savefig(log.dir / "fig1_mil_vs_baseline.png", dpi=140); plt.close(fig)
    bad = []
    for fp in [Path(__file__), HERE / "config.yaml", out / "m12_report" / "report_scores.md"]:
        for i, line in enumerate(fp.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if "forbidden_terms" in line:
                continue
            if any(t in line for t in cfg["forbidden_terms"]):
                bad.append((fp.name, i))
    log(f"\n금지어 검사: {'없음' if not bad else bad}\n저장: {out / 'm12_report'}")
    assert not bad


if __name__ == "__main__":
    main()
