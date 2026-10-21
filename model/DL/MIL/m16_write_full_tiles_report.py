"""m16: 마스크 100% 타일만으로 MIL 을 다시 돌린 결과(outputs_full)를 모든 타일 결과(outputs)와 비교하고 관찰과 해석을 분리해서 report_full_tiles.md 로 쓴다 (새 학습 없음).

비교: (1) 같은 방법(baseline, MIL)에서 타일 집합을 모든 타일 → 마스크 100% 타일로 줄였을 때 점수가 달라지는 정도(같은 재표본 쌍 차이), (2) 마스크 100% 타일에서 MIL 대 baseline, (3) attention 의 성질 비교.
그림에 대한 서술은 fig_attention_typical.png, fig_attention_errors.png 를 직접 보고 적은 육안 판독이다 (전문가 확인 없음, AGG 영역 라벨이 없어 대응 검증 불가).
출력: outputs_full/m13_attention_analysis/report_full_tiles.md, outputs_full/m13_attention_analysis/tile_set_effect.csv
"""
import csv
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
import sys  # noqa: E402
sys.path.insert(0, str(HERE.parent / "exp_group_cv"))
from gcommon import load_images, write_csv  # noqa: E402
from gcommon import load_config as _lc  # noqa: E402

cfg = _lc(HERE / "config_full.yaml")
ALL, FULL = HERE / "outputs", HERE / "outputs_full"
img = load_images(cfg)
ids = img["ids"]; pos = {i: k for k, i in enumerate(ids)}; N = len(ids)
y = np.isin(img["y"], cfg["positive_labels"]).astype(int)
conds = img["condition"]; cond_list = sorted(set(conds))


def rd(p):
    return list(csv.DictReader(open(p, encoding="utf-8-sig")))


def load_pred(p):
    P = {}
    for r in rd(p):
        R = 10 if r["scheme"] == "IMG" else 1
        P.setdefault(r["scheme"], np.full((N, R), -1, int))[pos[r["image_id"]], int(r["repeat"])] = int(r["pred"])
    return P


PM_all, PM_full = load_pred(ALL / "m10_train_eval_mil" / "mil_predictions.csv"), load_pred(FULL / "m10_train_eval_mil" / "mil_predictions.csv")
PB_all, PB_full = load_pred(ALL / "m15_baseline_pooled_logreg" / "baseline_predictions.csv"), load_pred(FULL / "m15_baseline_pooled_logreg" / "baseline_predictions.csv")
SM_all = {r["scheme"]: r for r in rd(ALL / "m10_train_eval_mil" / "mil_summary.csv")}
SM_full = {r["scheme"]: r for r in rd(FULL / "m10_train_eval_mil" / "mil_summary.csv")}
SB_all = {r["scheme"]: r for r in rd(ALL / "m15_baseline_pooled_logreg" / "baseline_summary.csv")}
SB_full = {r["scheme"]: r for r in rd(FULL / "m15_baseline_pooled_logreg" / "baseline_summary.csv")}
nb = cfg["bootstrap"]["n"]; rng = np.random.RandomState(cfg["bootstrap"]["seed"])
members = {c: np.where(np.array(conds) == c)[0] for c in cond_list}
idx_img = rng.randint(0, N, (nb, N))
idx_clu = np.array([np.concatenate([members[cond_list[k]] for k in rng.randint(0, len(cond_list), len(cond_list))]) for _ in range(nb)])


def accvec(Pm, idx):
    C = (Pm.reshape(N, -1) == y[:, None]).astype(float)
    return C[idx].mean(axis=1).mean(axis=1)


def ci(v):
    return np.percentile(v, [2.5, 97.5])


f = lambda r, k: float(r[k])
rows = []
for meth, Pa, Pf, Sa, Sf in (("baseline (패치 평균 + logreg)", PB_all, PB_full, SB_all, SB_full), ("MIL (gated attention)", PM_all, PM_full, SM_all, SM_full)):
    for sc in ("IMG", "LOCO", "LOMO"):
        d = accvec(Pf[sc], idx_img) - accvec(Pa[sc], idx_img); dc = accvec(Pf[sc], idx_clu) - accvec(Pa[sc], idx_clu)
        lo, hi = ci(d); lc, hc = ci(dc)
        rows.append(dict(method=meth, scheme=sc, acc_all=round(f(Sa[sc], "accuracy"), 4), acc_full=round(f(Sf[sc], "accuracy"), 4), diff=round(f(Sf[sc], "accuracy") - f(Sa[sc], "accuracy"), 4),
                         image_lo=round(lo, 4), image_hi=round(hi, 4), cluster_lo=round(lc, 4), cluster_hi=round(hc, 4)))
write_csv(FULL / "m13_attention_analysis" / "tile_set_effect.csv", rows)
ST_all = {(r["scheme"], r["group"]): r for r in rd(ALL / "m13_attention_analysis" / "attention_stats.csv")}
ST_full = {(r["scheme"], r["group"]): r for r in rd(FULL / "m13_attention_analysis" / "attention_stats.csv")}
STAB_all = {r["scheme"]: r for r in rd(ALL / "m13_attention_analysis" / "attention_stability.csv")}
STAB_full = {r["scheme"]: r for r in rd(FULL / "m13_attention_analysis" / "attention_stability.csv")}
perm = rd(FULL / "m11_permutation_mil" / "mil_permutation_runs.csv")
REP = rd(FULL / "m13_attention_analysis" / "representative_images.csv")
tiles_all = list(csv.DictReader(open(ALL / "m10_train_eval_mil" / "attention_tiles.csv", encoding="utf-8-sig")))
cnt = {}
for r in tiles_all:
    cnt.setdefault(int(r["image_idx"]), [0, 0])[1] += 1
    if float(r["mask_frac"]) >= 0.999999:
        cnt[int(r["image_idx"])][0] += 1
by_cls = {c: [cnt[i][0] for i in range(N) if img["y"][i] == c] for c in (0, 1, 2)}

SCH = {"IMG": "이미지 단위 분할", "LOCO": "LOCO", "LOMO": "LOMO"}
L = []
A = L.append
A("# 마스크 100% 타일만으로 MIL 을 다시 돌린 결과 (가장자리/배경이 섞인 타일 제거)\n")
A("> 가장자리와 배경이 섞인 타일(마스크 비율 100% 미만)을 모두 빼고, 시편만 있는 타일(이미지당 16~33개, 전체 1,309개)로만 같은 MIL 을 같은 하이퍼파라미터, 같은 분할, 같은 학습/평가 방식으로 다시 학습했다. "
  "이 타일의 특징은 모든 배경 증강 버전에서 원본과 같음을 코드로 확인했다 (시편만 있어 증강이 픽셀을 바꾸지 않는다). baseline(타일 평균 + logreg)도 같은 타일 집합으로 다시 학습했다.\n"
  "> 모든 타일 결과에서 attention 이 가장자리/배경이 섞인 타일에 쏠렸던 것이 점수나 해석의 교란인지 가르기 위한 실험이다.\n")
A("## 1. 관찰 (측정값)\n")
A("### 1.1 점수: 타일 집합을 줄이면 어떻게 달라지나 (accuracy, 이진)\n")
A("| 평가 | 방법 | 모든 타일 | 마스크 100% 타일 | 차이 | 이미지 단위 95% CI | 처리 조건 단위 95% CI |\n|---|---|---|---|---|---|---|")
for r in rows:
    A(f"| {SCH[r['scheme']]} | {r['method']} | {r['acc_all']:.3f} | {r['acc_full']:.3f} | {r['diff']:+.3f} | [{r['image_lo']:+.3f}, {r['image_hi']:+.3f}] | [{r['cluster_lo']:+.3f}, {r['cluster_hi']:+.3f}] |")
A("\n(차이 = 마스크 100% 타일 − 모든 타일, 같은 재표본 2000회의 쌍 차이.)\n")
A("### 1.2 점수: 마스크 100% 타일에서 MIL 대 baseline\n")
A("| 평가 | baseline | MIL | MIL − baseline | 이미지 단위 95% CI | 판정 |\n|---|---|---|---|---|---|")
for sc in ("IMG", "LOCO", "LOMO"):
    d = accvec(PM_full[sc], idx_img) - accvec(PB_full[sc], idx_img); lo, hi = ci(d); d0 = f(SM_full[sc], "accuracy") - f(SB_full[sc], "accuracy")
    v = "높다" if lo > 0 else ("낮다" if hi < 0 else "구분되지 않는다 (CI 가 0 포함)")
    A(f"| {SCH[sc]} | {f(SB_full[sc], 'accuracy'):.3f} | {f(SM_full[sc], 'accuracy'):.3f} | {d0:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {v} |")
pa = {sc: np.array([float(r["accuracy"]) for r in perm if r["scheme"] == sc]) for sc in ("IMG", "LOCO", "LOMO")}
A("\n- 라벨 순열 검사 (20회): " + ", ".join(f"{SCH[sc]} 실제 {f(SM_full[sc], 'accuracy'):.3f} vs 순열 평균 {pa[sc].mean():.3f} (최대 {pa[sc].max():.3f})" for sc in pa) + ". 세 평가 모두 실제 점수가 순열 최댓값을 넘는다.")
A("- 마스크 100% 타일 수는 클래스마다 다르다: 이미지당 중앙값 AGG0 " + f"{np.median(by_cls[0]):.0f}, AGG1 {np.median(by_cls[1]):.0f}, AGG2 {np.median(by_cls[2]):.0f}" + " (AGG2 이미지가 시편이 프레임에서 더 크게 찍힌 경향). 타일 평균과 softmax attention 은 타일 수에 직접 의존하지 않지만, bag 크기와 라벨의 연결은 교란이 될 수 있다.\n")
A("### 1.3 attention 의 성질 (LOCO fold 모델, 시드 5개 평균, 이미지 평균)\n")
A("| 이미지 그룹 | 타일 집합 | effective number / 타일 수 | 최대 가중치 | attention 과 시편 경계까지 거리의 Spearman (− = 경계 쪽 선호) | attention 가중 평균 경계 거리 / 균일 평균 경계 거리 (1 미만 = 경계 쪽 선호) |\n|---|---|---|---|---|---|")
for g in ("AGG0", "AGG1", "AGG2"):
    for nm, S in (("모든 타일", ST_all), ("마스크 100% 타일", ST_full)):
        r = S[("LOCO", g)]
        A(f"| {g} | {nm} | {float(r['effective_n_over_n']):.3f} | {float(r['max_attention']):.3f} | {float(r['spearman_attn_vs_boundary_dist']):+.2f} | {float(r['attn_weighted_dist_over_uniform']):.2f} |")
sf, sa = STAB_full["LOCO"], STAB_all["LOCO"]
A(f"\n- 시드끼리 attention 순위 상관 (Spearman) 평균: 모든 타일 {float(sa['seed_to_seed_spearman_mean']):.2f} (최소 {float(sa['seed_to_seed_min']):.2f}), 마스크 100% 타일 {float(sf['seed_to_seed_spearman_mean']):.2f} (최소 {float(sf['seed_to_seed_min']):.2f}). "
  "원본 대 배경 증강 비교는 마스크 100% 타일에서는 특징이 같아 상관 1.0 이고 정보가 없다.\n")
A("### 1.4 대표 이미지 오버레이 (`fig_attention_typical.png`, `fig_attention_errors.png`)\n")
A("대표 이미지는 이전과 같은 규칙(클래스마다 맞힌 이미지 중 양성 확률이 클래스 중앙값에 가장 가까운 1장, 틀린 이미지 중 오차가 가장 큰 1장)으로 새 예측에서 골랐다. 색은 사용한 타일(마스크 100%)에만 칠해지고, 가장자리 타일은 이번 bag 에 없다. 아래는 그림을 보고 적은 육안 판독이다.\n")
for r in REP:
    A(f"- {'맞힌 대표' if r['kind'] == 'typical' else '크게 틀린'} AGG{r['label3']}: `{r['image_id']}`, 모델 양성 확률 {float(r['prob_pos']):.2f}")
A("")
A("- AGG0 맞힌 대표 (BASE-LSi/PH1230-1035): attention 이 높은 타일이 사용된 타일 영역의 바깥쪽 모서리 쪽(왼쪽 위 두 타일, 오른쪽 위 두 타일)에 있고 안쪽 타일은 대부분 낮다.")
A("- AGG1 맞힌 대표 (BASE/PH1205-1050): 높은 attention 이 안쪽 한 타일(왼쪽 가운데)과 오른쪽 두 타일에 흩어져 있다.")
A("- AGG2 맞힌 대표 (BASE-LC/NPH1035): attention 이 모든 타일에 걸쳐 중간 값으로 퍼져 있고 한 안쪽 타일이 가장 높다.")
A("- AGG0 → 양성으로 크게 틀린 이미지 (REF-LN/PH1255-1035, 양성 확률 1.00): 높은 attention 이 왼쪽 가운데 두 타일과 안쪽 한 타일에 있다.")
A("- AGG1 → 음성으로 크게 틀린 이미지 (REF/PH1230-1050, 양성 확률 0.00): 높은 attention 이 오른쪽 가장자리 쪽 타일 두 개에 있다.")
A("- 이 그림에서 attention 이 높은 타일이 결정립이 편평한 영역인지는 판단하지 못했다 (AGG 영역 라벨 없음, 타일 한 장이 224px).\n")

dd = {(r["method"][:2], r["scheme"]): r for r in rows}
A("## 2. 해석 (가능한 설명이며 확정이 아님)\n")
A("1. **가장자리/배경이 섞인 타일에는 점수에 기여하는 정보가 있는 것으로 보인다.** 타일 집합을 시편만 있는 타일로 줄이면 baseline 의 이미지 단위 분할 accuracy 가 "
  f"{dd[('ba', 'IMG')]['acc_all']:.3f} → {dd[('ba', 'IMG')]['acc_full']:.3f} ({dd[('ba', 'IMG')]['diff']:+.3f}, 이미지 단위 CI [{dd[('ba', 'IMG')]['image_lo']:+.3f}, {dd[('ba', 'IMG')]['image_hi']:+.3f}]) 로 내려간다. "
  "LOCO, LOMO 에서도 baseline 은 내려가지만 MIL 은 LOMO 에서 오히려 올랐다 (표 1.1). 그 정보가 시편 가장자리의 질감인지, 윤곽 모양, 시편의 크기와 위치(bag 크기와 클래스의 연결)인지는 이 실험으로 가르지 못한다.")
A("2. **시편만 있는 타일만으로도 점수가 유지된다.** MIL 은 세 평가에서 " + ", ".join(f"{SCH[sc]} {f(SM_full[sc], 'accuracy'):.3f}" for sc in ("IMG", "LOCO", "LOMO")) + " 이고 순열 최댓값(0.60~0.63)을 모두 넘는다. "
  "따라서 MIL 이 \"가장자리/배경이 섞인 타일의 비율\" 같은 가장자리 정보만 세고 있었다는 설명으로는 이 점수 전부를 설명할 수 없다. 다만 이것이 AGG 형태를 읽은 결과라는 근거는 아니다 (처리 조건 혼동, 시편 크기 차이는 남아 있다).")
A("3. **MIL 과 baseline 은 마스크 100% 타일에서 구분되지 않는다** (차이 " + ", ".join(f"{SCH[sc]} {f(SM_full[sc], 'accuracy') - f(SB_full[sc], 'accuracy'):+.3f}" for sc in ("IMG", "LOCO", "LOMO")) + ", CI 모두 0 포함). "
  "모든 타일에서는 MIL 이 baseline 보다 점추정이 낮았으므로, 가장자리 타일을 뺀 입력에서는 두 방법의 격차가 점추정 수준에서도 사라진다. 이 차이는 표본 크기(이미지 60장)에 비해 작아서 단정하지 않는다.")
A("4. **attention 의 경계 쪽 선호는 배경 픽셀만의 효과가 아니다.** 시편만 있는 타일 안에서도 AGG0, AGG1 bag 의 attention 은 시편 경계에 가까운 타일을 선호한다 (Spearman " + f"{float(ST_full[('LOCO', 'AGG0')]['spearman_attn_vs_boundary_dist']):+.2f}" + ", " + f"{float(ST_full[('LOCO', 'AGG1')]['spearman_attn_vs_boundary_dist']):+.2f}" + "; 가중 평균 경계 거리 / 균일 평균 " + f"{float(ST_full[('LOCO', 'AGG0')]['attn_weighted_dist_over_uniform']):.2f}" + ", " + f"{float(ST_full[('LOCO', 'AGG1')]['attn_weighted_dist_over_uniform']):.2f}" + "). AGG2 bag 은 그런 선호가 없다 (" + f"{float(ST_full[('LOCO', 'AGG2')]['spearman_attn_vs_boundary_dist']):+.2f}" + ", " + f"{float(ST_full[('LOCO', 'AGG2')]['attn_weighted_dist_over_uniform']):.2f}" + "). "
  "원본 이미지에는 시편 가장자리 안쪽에 밝은 띠가 보이는 경우가 있어서(대표 이미지 참고, 육안), 경계 근처의 조명이나 질감을 반영했을 수 있다는 가설이 가능하지만 이 분석으로 확인하지는 못했다.")
A("5. attention 이 AGG 영역을 자동으로 찾는다고 말할 수 없다는 결론은 그대로다. 마스크 100% 타일로 제한해도 attention 은 클래스와 타일의 위치(경계 거리)에 연결돼 있고, AGG 영역 라벨이 없어 대응을 검증하지 못했다.")
A("6. 시드끼리 attention 순위 상관이 평균 " + f"{float(sf['seed_to_seed_spearman_mean']):.2f}" + " (최소 " + f"{float(sf['seed_to_seed_min']):.2f}" + ") 이므로 개별 타일 하이라이트를 근거로 삼지 않는다.\n")
A("## 3. 이번에 하지 않은 것\n")
A("- **가장자리 비율 자체만 feature 로 쓴 logreg 통제**는 하지 않았다 (요청이 중단되어 마스크 100% 타일 MIL 을 먼저 진행했다). 이 통제가 있으면 \"MIL 이 한 일이 가장자리 비율 세기\"라는 가설을 직접 검정할 수 있다. 필요하면 추가한다.")
A("- 가장자리/배경이 섞인 타일만으로 MIL 을 돌리는 반대 방향의 실험, 전문가가 표시한 AGG 영역 라벨과 attention 의 대응 평가.")
A("- 이미지가 60장이라 CI 가 넓고 attention 통계는 이미지 평균이며 CI 를 계산하지 않았다. 육안 판독은 전문가 확인을 받지 않았다.")
(FULL / "m13_attention_analysis" / "report_full_tiles.md").write_text("\n".join(L) + "\n", encoding="utf-8")
assert "시편 종류" not in "\n".join(L)
print("saved report_full_tiles.md", len(L), "lines")
for r in rows:
    print(f"  {r['method'][:10]:10s} {r['scheme']:5s} 모든 {r['acc_all']:.3f} -> 100% {r['acc_full']:.3f} ({r['diff']:+.3f}) 이미지 CI [{r['image_lo']:+.3f}, {r['image_hi']:+.3f}]")
