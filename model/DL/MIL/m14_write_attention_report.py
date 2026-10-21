"""m14: attention 해석 보고서(report_attention.md)를 쓴다. 관찰(측정값, 그림에서 본 것)과 해석(가능한 설명, 확정 아님)을 분리해서 적는다.

숫자는 m10~m13 의 CSV 에서 읽어 채우고, 그림에 대한 서술은 대표 이미지 그림(fig_attention_typical.png, fig_attention_errors.png)을 직접 보고 적은 육안 판독이다.
육안 판독은 도메인 전문가의 확인을 받지 않았다. 이미지에 AGG 영역 라벨이 없어서 attention 이 AGG 영역에 대응하는지는 검증하지 못했다.
출력: report_attention.md
"""
import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"


def rd(p):
    return list(csv.DictReader(open(OUT / p, encoding="utf-8-sig")))


S = {(r["scheme"], r["group"]): r for r in rd("m13_attention_analysis/attention_stats.csv")}
STAB = {r["scheme"]: r for r in rd("m13_attention_analysis/attention_stability.csv")}
REP = rd("m13_attention_analysis/representative_images.csv")
PD = {r["scheme"]: r for r in rd("m12_report/paired_diff.csv")}
SUM = {r["scheme"]: r for r in rd("m12_report/results_table.csv") if r["method"].startswith("MIL")}
BAS = {r["scheme"]: r for r in rd("m12_report/results_table.csv") if r["method"].startswith("baseline")}
f = lambda r, k: float(r[k])


def pct(x):
    return f"{float(x) * 100:.1f}%"


L = []
A = L.append
A("# MIL attention 해석 보고서 (관찰과 해석을 분리)\n")
A("> attention 은 모델이 bag 표현을 만들 때 타일에 준 가중치이지, 그 타일이 AGG 라는 라벨이 아니다. 이 문서는 attention 이 높은 타일이 실제 AGG 영역과 대응하는지를 **검증하지 못했다** (이미지에 AGG 영역 라벨이 없다). "
  "그림에 대한 서술은 연구자가 아닌 분석자의 육안 판독이며 도메인 전문가의 확인을 받지 않았다.\n")
A("## 0. 무엇을 분석했나\n")
A("- 모델: Ilse et al. (2018) gated attention MIL. bag = 이미지 1장 (분석 타일 36~48개, DINOv2 cls 768차원), 이진 라벨(AGG0 vs AGG1+2).")
A("- attention 은 **시험 단위를 학습에서 뺀 LOCO fold 모델**(처리 조건 하나를 통째로 뺀 모델이 그 조건의 이미지에 낸 attention)을 시드 5개 평균해서 썼다. 이미지 단위 분할이나 LOMO 의 값도 같은 지표로 계산했지만(`attention_stats.csv`) 아래는 LOCO 위주다.")
A("- 지표와 대표 이미지 선택 규칙은 결과를 보기 전에 `config.yaml` 과 `m13_attention_analysis.py` 에 고정했다.\n")

A("## 1. 관찰 (측정값)\n")
A("### 1.1 점수 (참고)\n")
for sc, nm in (("IMG", "이미지 단위 분할"), ("LOCO", "LOCO"), ("LOMO", "LOMO")):
    d = PD[sc]
    A(f"- {nm}: MIL accuracy {f(SUM[sc], 'accuracy'):.3f} vs baseline(패치 평균 + logreg) {f(BAS[sc], 'accuracy'):.3f}, 차이 {float(d['diff']):+.3f}, 이미지 단위 95% CI [{float(d['image_lo']):+.3f}, {float(d['image_hi']):+.3f}] → {d['verdict']}")
A("")
A("### 1.2 attention 의 집중도와 타일 구성 (LOCO fold 모델, 이미지 평균)\n")
A("| 이미지 그룹 | 이미지 수 | effective number / 타일 수 (1 = 균일) | 최대 attention 가중치 | 배경이 섞인 타일이 받은 attention 질량 | 그 타일의 비율 | 배율 | attention 상위 10% 타일 중 배경이 섞인 타일 | 기대 비율 | attention 과 mask_frac 의 Spearman |")
A("|---|---|---|---|---|---|---|---|---|---|")
for g in ("전체", "AGG0", "AGG1", "AGG2"):
    r = S[("LOCO", g)]
    A(f"| {g} | {r['n_images']} | {float(r['effective_n_over_n']):.3f} | {float(r['max_attention']):.3f} | {pct(r['attn_mass_on_partial_tiles'])} | {pct(r['tile_fraction_partial'])} | {float(r['ratio_partial']):.2f} | {pct(r['top10_partial_share'])} | {pct(r['top10_expected'])} | {float(r['spearman_attn_vs_maskfrac']):+.2f} |")
A("\n(배경이 섞인 타일 = 마스크 비율이 100% 가 아닌 타일, 즉 시편 가장자리와 배경이 들어 있는 타일. 배율 = attention 질량 비율 / 타일 비율; 1 이면 균일한 attention 과 같다.)\n")
A("- AGG0 bag 에서는 attention 이 배경이 섞인 타일에 쏠린다 (질량 " + pct(S[("LOCO", "AGG0")]["attn_mass_on_partial_tiles"]) + ", 타일 비율 " + pct(S[("LOCO", "AGG0")]["tile_fraction_partial"]) + ", 상위 10% 타일 중 " + pct(S[("LOCO", "AGG0")]["top10_partial_share"]) + ").")
A("- AGG2 bag 에서는 반대로 배경이 섞인 타일을 피한다 (질량 " + pct(S[("LOCO", "AGG2")]["attn_mass_on_partial_tiles"]) + ", 타일 비율 " + pct(S[("LOCO", "AGG2")]["tile_fraction_partial"]) + ", 상위 10% 타일 중 " + pct(S[("LOCO", "AGG2")]["top10_partial_share"]) + "). AGG1 은 그 중간이다.")
A("- AGG0 bag 의 attention 이 더 소수 타일에 집중하고 (effective number / 타일 수 " + f"{float(S[('LOCO', 'AGG0')]['effective_n_over_n']):.2f}" + ", 최대 가중치 " + f"{float(S[('LOCO', 'AGG0')]['max_attention']):.2f}" + "), AGG2 bag 은 더 퍼져 있다 (" + f"{float(S[('LOCO', 'AGG2')]['effective_n_over_n']):.2f}" + ", " + f"{float(S[('LOCO', 'AGG2')]['max_attention']):.2f}" + ").")
A("- 같은 패턴이 이미지 단위 분할 모델과 LOMO 모델의 attention 에서도 나타난다 (`attention_stats.csv`: AGG2 의 배경 섞인 타일 배율 IMG " + f"{float(S[('IMG', 'AGG2')]['ratio_partial']):.2f}" + ", LOMO " + f"{float(S[('LOMO', 'AGG2')]['ratio_partial']):.2f}" + "; AGG0 는 " + f"{float(S[('IMG', 'AGG0')]['ratio_partial']):.2f}" + ", " + f"{float(S[('LOMO', 'AGG0')]['ratio_partial']):.2f}" + ").\n")
A("### 1.3 안정성\n")
st = STAB["LOCO"]
A(f"- 같은 이미지, 같은 입력에서 **시드끼리** attention 의 타일 순위 상관 (Spearman) 평균 {float(st['seed_to_seed_spearman_mean']):.2f}, 최소 {float(st['seed_to_seed_min']):.2f}. 대부분의 이미지에서 재현되지만 일부 이미지는 시드에 따라 크게 달라진다.")
A(f"- **원본 bag 과 배경 증강 bag** (같은 이미지, 시편 픽셀과 마스크는 같고 배경 픽셀만 무작위로 바뀜) 의 attention 순위 상관 평균 {float(st['orig_vs_augmented_spearman_mean']):.2f}, 최소 {float(st['orig_vs_augmented_min']):.2f}. 배경 픽셀을 바꿔도 attention 순위가 대체로 유지된다.\n")
A("### 1.4 대표 이미지 오버레이 (`fig_attention_typical.png`, `fig_attention_errors.png`)\n")
A("대표 이미지는 규칙 (클래스마다 맞힌 이미지 중 양성 확률이 클래스 중앙값에 가장 가까운 1장, 틀린 이미지 중 오차가 가장 큰 1장) 으로 골랐다. 아래는 그림을 보고 적은 육안 판독이다 (타일 격자는 이미지 왼쪽 위 모서리에서 시작하는 224px 정사각형).\n")
for r in REP:
    A(f"- {'맞힌 대표' if r['kind'] == 'typical' else '크게 틀린'} AGG{r['label3']}: `{r['image_id']}`, 모델 양성 확률 {float(r['prob_pos']):.2f}")
A("")
A("- AGG0 맞힌 대표 (REF-LN/PH1255-1050): 높은 attention 이 왼쪽·오른쪽 가장자리 열의 타일 (마스크 윤곽이 지나가는 타일)과 안쪽 몇 개 타일에 있고, 대부분의 안쪽 타일은 낮다.")
A("- AGG1 맞힌 대표 (BASE/PH1180-1035): 높은 attention 이 오른쪽 가장자리 열 타일과 안쪽 두 타일에 있다.")
A("- AGG2 맞힌 대표 (BASE-LC/NPH1050): attention 이 안쪽 타일 전반에 퍼져 있고 (최대 3배 안팎) 가장자리 타일에 특별히 높지 않다.")
A("- AGG0 → 양성으로 크게 틀린 이미지 (BASE-LC/PH1205-1035, 양성 확률 0.99): 높은 attention 이 윗줄과 왼쪽 가장자리 타일에 있다.")
A("- AGG1 → 음성으로 크게 틀린 이미지 (REF/PH1230-1050, 양성 확률 0.00): 높은 attention 이 안쪽에 흩어진 타일 몇 개와 오른쪽 가장자리 타일에 있다.")
A("- 이 그림에서 attention 이 높은 타일을 눈으로 봐서 결정립이 편평해 보이는 영역이라고 판단하지는 못했다 (타일 한 장이 224px 이고 AGG 영역의 위치 라벨이 없어 대응 여부를 확인할 기준이 없다).\n")

A("## 2. 해석 (가능한 설명이며 확정이 아님)\n")
A("1. **attention 이 AGG 영역을 자동으로 찾아낸다고 말할 수 없다.** 위 관찰에서 attention 의 분포는 클래스와 타일 구성(가장자리/배경이 섞인 타일 대 시편만 있는 타일)에 강하게 연결돼 있다. 이는 모델이 타일의 구성(가장자리 여부)과 클래스를 연결해서 판단 근거로 삼았을 가능성과 양립한다. "
  "AGG0 의 음성 판단이 가장자리 타일에 기대고 있는지는 이 분석으로는 가릴 수 없다.")
A("2. 배경 픽셀을 바꿔도 attention 순위가 대체로 유지된다는 관찰 (상관 " + f"{float(st['orig_vs_augmented_spearman_mean']):.2f}" + ") 은 attention 이 배경의 색, 질감에 직접 의존한다는 가설과는 맞지 않는다. "
  "그러나 증강은 시편 픽셀과 마스크 윤곽(가장자리 모양)을 바꾸지 않으므로, **가장자리의 기하(윤곽 모양)나 가장자리 근처 시편 픽셀**에 의존하는 경우는 이 관찰로 배제되지 않는다.")
A("3. 시드끼리 순위 상관이 평균 " + f"{float(st['seed_to_seed_spearman_mean']):.2f}" + ", 최소 " + f"{float(st['seed_to_seed_min']):.2f}" + " 라서, 개별 이미지의 상위 타일 한두 개를 근거로 해석하면 안 된다. 클래스 단위 경향 (1.2절) 이 개별 타일 하이라이트보다 신뢰할 만하다.")
A("4. **점수 대 해석 가능성의 트레이드오프**: MIL 의 점추정은 세 평가 모두 baseline 보다 낮다 (차이 −0.017 ~ −0.037, 1.1절). 이미지 단위 CI 는 모두 0 을 포함해 통계적으로 구분되지는 않지만, 낮은 쪽을 숨기지 않고 그대로 보고한다. attention 으로 얻은 것은 \"모델이 어디를 보는가\"에 대한 단서이며, 위 1번이 말하듯 그 단서가 가리키는 곳이 AGG 영역이라는 근거는 아직 없다.")
A("5. 위 패턴이 교란 (가장자리 타일과 클래스의 연결) 인지 아닌지를 가르려면 다음이 필요하다: 시편만 있는 타일(마스크 100%)만으로 MIL 을 다시 돌려 점수와 attention 이 유지되는지 보기, 가장자리 타일을 가린 입력에서의 attention 확인, 전문가가 일부 이미지에 AGG 영역을 표시한 라벨과 attention 의 대응 평가. 이번에는 하지 않았다.\n")

A("## 3. 한계\n")
A("- 이미지가 60장이라 클래스별 이미지 수가 적다 (AGG0 28, AGG1 14, AGG2 18). 위 통계는 이미지 평균이고 CI 를 계산하지 않았다.")
A("- attention 은 모델의 내부 가중치이며 예측에 대한 인과적 설명이 아니다 (attention 이 높다고 그 타일이 결과를 바꾼다는 뜻이 아님).")
A("- 타일 격자는 이미지 모서리에서 시작하는 224px 정사각형이고 겹치지 않는다. 한 타일 안의 위치 정보는 attention 으로 알 수 없다.")
A("- 육안 판독은 전문가 확인을 받지 않았다. 대표 이미지 6장 미만을 본 것이고 전체 60장의 모든 경향을 대표하지 않는다 (전체 이미지별 값은 `attention_by_image.csv`).")
(HERE / "outputs" / "m13_attention_analysis" / "report_attention.md").write_text("\n".join(L) + "\n", encoding="utf-8")
assert "시편 종류" not in "\n".join(L)
print("saved report_attention.md", len(L), "lines")
