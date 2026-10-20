"""e04: 결과 보고서.

- 조합별 결과 표(accuracy, macro-F1, 클래스별 recall, 반복 간 평균±SD)와 e02 기준선(다수 클래스, 처리 조건 암기)을 한 표에 나란히 둔다.
  표는 config 의 조합 순서 그대로이며 성능순으로 정렬하지 않는다 (좋은 것만 고르지 않기 위해).
- R4: e02 의 불일치 목록 이미지(라벨이 자기 처리 조건의 다수 라벨과 다른 이미지)와 나머지 이미지의 정확도를 따로 보고. 통계 검정은 하지 않는다.
- R3: 이미지별 오분류 빈도(반복 10회 중 틀린 횟수)를 금속 종류(6), 처리 조건(10), 처리 방법(5), 온도(2) 네 가지로 집계.
- primary 의 혼동행렬, 부트스트랩 95% CI (이미지 단위 + 처리 조건 단위 클러스터; primary 와 e02 기준선).
- 금지어 검사 (config forbidden_terms).
출력: report.md, results_table.csv, discordant_vs_rest.csv, misclassification_by_group.csv, image_error_counts.csv, primary_confusion.csv, bootstrap_ci.csv, log.txt
"""
import argparse
import csv
import sys

import numpy as np

from common import Logger, bootstrap_ci, confusion, load_config, metrics, term_check, write_csv

SCRIPT = "e04_report"
BASE_IDS = {"baseline|majority_class": "다수 클래스 예측", "baseline|condition_memorization": "처리 조건 암기"}


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def ms(vals, R):
    return f"{np.mean(vals):.3f}±{np.std(vals):.3f}" if R > 1 else f"{vals[0]:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--output", help="outputs 폴더 (기본: config paths.outputs_dir)")
    a = ap.parse_args()
    cfg = load_config(a.config)
    out = a.output or cfg["paths"]["outputs_dir"]
    log = Logger(out, SCRIPT)
    log("config:", cfg["_config_path"])

    conds = list(csv.DictReader(open(f"{out}/e00_make_splits/conditions.csv", encoding="utf-8-sig")))
    ids = [r["image_id"] for r in conds]; pos = {i: k for k, i in enumerate(ids)}; N = len(ids)
    y = np.array([int(r["label"]) for r in conds])
    groupings = {"금속 종류": [r["metal"] for r in conds], "처리 조건": [r["condition"] for r in conds],
                 "처리 방법": [r["method"] for r in conds], "온도": [r["temp"] for r in conds]}
    cond_ids = np.array([sorted(set(groupings["처리 조건"])).index(c) for c in groupings["처리 조건"]])

    settings = list(csv.DictReader(open(f"{out}/e03_run_cv/settings.csv", encoding="utf-8-sig")))
    preds = list(csv.DictReader(open(f"{out}/e03_run_cv/cv_predictions.csv", encoding="utf-8-sig")))
    reps = sorted({int(r["repeat"]) for r in preds}); R = len(reps)
    series = {s["id"]: np.full((N, R), -1, int) for s in settings}
    for r in preds:
        series[r["setting"]][pos[r["image_id"]], int(r["repeat"])] = int(r["pred"])
    assert all((v >= 0).all() for v in series.values()), "예측이 비어 있는 칸이 있음"
    primary = next(s["id"] for s in settings if s["primary"] == "1")
    base = list(csv.DictReader(open(f"{out}/e02_reference_baselines/baseline_predictions.csv", encoding="utf-8-sig")))
    assert [b["image_id"] for b in base] == ids, "e02 와 e00 의 이미지 순서가 다름"
    assert [int(b["label"]) for b in base] == y.tolist()
    series["baseline|majority_class"] = np.array([[int(b["pred_majority"])] for b in base])
    series["baseline|condition_memorization"] = np.array([[int(b["pred_condition_memorization"])] for b in base])
    order = [s["id"] for s in settings] + list(BASE_IDS)
    disc = list(csv.DictReader(open(f"{out}/e02_reference_baselines/condition_discordant_images.csv", encoding="utf-8-sig")))
    D = np.array(sorted(pos[d["image_id"]] for d in disc)); rest = np.array([i for i in range(N) if i not in set(D.tolist())])
    log(f"설정 {len(settings)}개 + 기준선 2개, 반복 {R}회, 이미지 {N}장 | primary = {primary} | 불일치 이미지 {len(D)}장 / 나머지 {len(rest)}장")

    # ---------- 결과 표 ----------
    rows, md_rows = [], []
    for sid in order:
        S = series[sid]; Rn = S.shape[1]
        ms_ = [metrics(y, S[:, r]) for r in range(Rn)]
        acc = [m["accuracy"] for m in ms_]; f1 = [m["macro_f1"] for m in ms_]; rec = np.array([m["recall"] for m in ms_])
        tag = "primary" if sid == primary else ("기준선" if sid in BASE_IDS else "ablation")
        name = BASE_IDS.get(sid, sid)
        rows.append(dict(setting=sid, group=tag, accuracy_mean=np.mean(acc), accuracy_sd=np.std(acc), macro_f1_mean=np.mean(f1), macro_f1_sd=np.std(f1),
                         **{f"recall_AGG{c}_mean": float(rec[:, c].mean()) for c in range(3)}, **{f"recall_AGG{c}_sd": float(rec[:, c].std()) for c in range(3)}))
        md_rows.append([("**" + name + "**") if tag == "primary" else name, tag, ms(acc, Rn), ms(f1, Rn)] + [ms(rec[:, c], Rn) for c in range(3)])
    write_csv(f"{log.dir}/results_table.csv", rows)
    t_main = md_table(["설정", "구분", "accuracy", "macro-F1", "recall AGG0", "recall AGG1", "recall AGG2"], md_rows)

    # ---------- R4: 불일치 이미지 vs 나머지 ----------
    r4, md4 = [], []
    for sid in order:
        S = series[sid]; Rn = S.shape[1]
        kD = np.array([(S[D, r] == y[D]).sum() for r in range(Rn)])
        accR = np.array([(S[rest, r] == y[rest]).mean() for r in range(Rn)])
        r4.append(dict(setting=sid, discordant_k_mean=float(kD.mean()), discordant_k_min=int(kD.min()), discordant_k_max=int(kD.max()), discordant_n=len(D),
                       rest_accuracy_mean=float(accR.mean()), rest_accuracy_sd=float(accR.std()), rest_n=len(rest)))
        kd_txt = f"{kD.mean():.1f}/{len(D)} ({kD.min()}~{kD.max()})" if Rn > 1 else f"{int(kD[0])}/{len(D)}"
        md4.append([BASE_IDS.get(sid, sid), kd_txt, ms(accR, Rn)])
    write_csv(f"{log.dir}/discordant_vs_rest.csv", r4)
    t4 = md_table(["설정", f"불일치 {len(D)}장: 맞힌 수 k/{len(D)} (반복 평균, 최소~최대)", f"나머지 {len(rest)}장 accuracy"], md4)

    # ---------- R3: 이미지별 오분류 빈도 ----------
    img_rows, grp_rows = [], []
    for sid in order:
        S = series[sid]; Rn = S.shape[1]
        wrong = (S != y[:, None])
        for i in range(N):
            img_rows.append(dict(setting=sid, image_id=ids[i], label=int(y[i]), wrong_count=int(wrong[i].sum()), trials=Rn))
        for gname, labels_ in groupings.items():
            arr = np.array(labels_)
            for g in sorted(set(labels_)):
                idx = np.where(arr == g)[0]
                grp_rows.append(dict(setting=sid, grouping=gname, group=g, n_images=len(idx), wrong_rate=float(wrong[idx].mean()), total_wrong=int(wrong[idx].sum()), trials=len(idx) * Rn))
    write_csv(f"{log.dir}/image_error_counts.csv", img_rows)
    write_csv(f"{log.dir}/misclassification_by_group.csv", grp_rows)
    t3 = {}
    for gname in groupings:
        rr = [r for r in grp_rows if r["setting"] == primary and r["grouping"] == gname]
        t3[gname] = md_table([gname, "이미지 수", "오분류 빈도 (틀린 횟수 / 시도)"], [[r["group"], r["n_images"], f"{r['wrong_rate']:.3f} ({r['total_wrong']}/{r['trials']})"] for r in rr])

    # ---------- primary 혼동행렬 ----------
    Sp = series[primary]
    cm_sum = sum(confusion(y, Sp[:, r]) for r in range(R))
    cm_avg = cm_sum / R
    cm_norm = cm_sum / cm_sum.sum(1, keepdims=True)
    write_csv(f"{log.dir}/primary_confusion.csv", [dict(true=f"AGG{i}", **{f"pred_AGG{j}_mean_count": float(cm_avg[i, j]) for j in range(3)}, **{f"pred_AGG{j}_rate": float(cm_norm[i, j]) for j in range(3)}) for i in range(3)])
    t_cm = md_table(["정답 \\ 예측", "AGG0", "AGG1", "AGG2"], [[f"AGG{i}"] + [f"{cm_avg[i, j]:.1f} ({100 * cm_norm[i, j]:.0f}%)" for j in range(3)] for i in range(3)])

    # ---------- 부트스트랩 ----------
    bs_rows, md_bs = [], []
    n_b, seed_b = cfg["bootstrap"]["n"], cfg["bootstrap"]["seed"]
    for sid in (primary, "baseline|condition_memorization", "baseline|majority_class"):
        S = series[sid]
        ci = bootstrap_ci(y, S, cond_ids, n_b, seed_b)
        m0 = [metrics(y, S[:, r]) for r in range(S.shape[1])]
        for mode, mname in (("image", "이미지 단위"), ("cluster", "처리 조건 단위(클러스터)")):
            c = ci[mode]
            bs_rows.append(dict(setting=sid, mode=mode, accuracy=np.mean([m["accuracy"] for m in m0]), acc_lo=c["acc"][0], acc_hi=c["acc"][1],
                                macro_f1=np.mean([m["macro_f1"] for m in m0]), f1_lo=c["f1"][0], f1_hi=c["f1"][1], n_boot=c["n"], missing_class_resamples=c["missing_class_resamples"]))
            md_bs.append([BASE_IDS.get(sid, sid) if sid in BASE_IDS else "primary", mname + ("*" if mode == "image" else ""),
                          f"{np.mean([m['accuracy'] for m in m0]):.3f} [{c['acc'][0]:.3f}, {c['acc'][1]:.3f}]", f"{np.mean([m['macro_f1'] for m in m0]):.3f} [{c['f1'][0]:.3f}, {c['f1'][1]:.3f}]", c["missing_class_resamples"]])
        log(f"부트스트랩 {sid}: 이미지 단위 acc [{ci['image']['acc'][0]:.3f}, {ci['image']['acc'][1]:.3f}] | 클러스터 acc [{ci['cluster']['acc'][0]:.3f}, {ci['cluster']['acc'][1]:.3f}] | 클러스터 표본 중 클래스가 빠진 표본 {ci['cluster']['missing_class_resamples']}/{n_b}")
    write_csv(f"{log.dir}/bootstrap_ci.csv", bs_rows)
    t_bs = md_table(["대상", "방식", "accuracy [95% CI]", "macro-F1 [95% CI]", "클래스 빠진 표본 수"], md_bs)

    # ---------- report.md ----------
    pr = next(r for r in rows if r["setting"] == primary); cm_ = next(r for r in rows if r["setting"] == "baseline|condition_memorization")
    md = [
        "# 모델·실험 보고서 (고정 DINOv2 특징 → 이미지 단위 합치기 → 분류기)", "",
        "용어: 1단계 폴더 = 금속 종류(6종), 2단계 폴더 = 처리 조건(10종) = 처리 방법(5종) x 온도(2종).", "",
        "## 1. 설정과 평가 방식", "",
        f"- 입력: 10x 이미지 {N}장, 선택 타일 {cfg['data']['expected_tiles']:,}개 (`g00_tile_list`: 겹치지 않는 224px 부분 격자, 시편 픽셀 0인 타일 제외). 특징: DINOv2 ViT-B/14 (고정), cls 또는 cls+patch 평균.",
        f"- 평가: 이미지 단위 층화 {cfg['cv']['n_splits']}-fold x 반복 {R}회 (seed {cfg['cv']['seed']}). 한 이미지의 타일은 항상 같은 쪽. 반복마다 test 예측을 모아(pooled) 지표를 구하고 반복 간 평균±표준편차를 보고. **반복들은 독립이 아니라서 표준편차는 불확실성을 과소평가한다.**",
        f"- primary (결과를 보기 전에 고정): {primary}. 나머지는 모두 ablation이며 config 순서대로 전부 싣는다 (성능순 정렬 없음).",
        "- kNN은 k=5, cosine, **단순 거리 가중** (SEM 논문은 Gaussian kernel 가중이었음, 표준화 없음). logreg는 L2, C=1.0, balanced, 표준화는 train에서만 fit, mean_std의 표준편차는 ddof=0.", "",
        "## 2. 결과 표 (accuracy, macro-F1, 클래스별 recall)", "", t_main, "",
        f"- **처리 조건 암기 기준선**은 같은 처리 조건의 다른 이미지들의 다수 라벨로 예측해 accuracy {cm_['accuracy_mean']:.3f}, macro-F1 {cm_['macro_f1_mean']:.3f}다. 이 평가에서는 test 이미지와 같은 처리 조건의 다른 금속 이미지 5장이 항상 train에 있다.",
        f"- primary는 accuracy {pr['accuracy_mean']:.3f}±{pr['accuracy_sd']:.3f}, macro-F1 {pr['macro_f1_mean']:.3f}±{pr['macro_f1_sd']:.3f}. 기준선과의 차이의 의미는 아래 CI와 4절을 함께 볼 것 (통계 검정은 하지 않았다).", "",
        "## 3. primary 부트스트랩 95% CI", "", t_bs, "",
        f"- \\* 이미지 단위 CI는 이미지가 독립이라는 가정이라, 같은 처리 조건 안의 상관 때문에 실제보다 좁다. 처리 조건 단위(10개 조건 복원추출) 클러스터 CI가 더 보수적이다. 방식: {n_b}회 복원추출, 각 표본에서 이미지의 {R}회 반복 예측으로 지표를 구해 평균, 95% 백분위 구간. 클러스터 표본에서 클래스가 빠지면 macro-F1은 정답이 있는 클래스만 평균한다(빠진 표본 수는 표에 표기). 기준선은 예측이 하나뿐(반복 1)이다.", "",
        "## 4. 처리 조건과 라벨이 다른 이미지 (e02 목록) vs 나머지", "",
        f"불일치 목록 {len(D)}장: " + ", ".join(f"{d['metal']} {d['condition']} (AGG{d['label']}, 조건 다수 AGG{d['condition_majority_label']})" for d in disc) + ". 조건 암기는 구조상 이 이미지들을 모두 틀린다.", "",
        t4, "",
        "- 개수가 적어(불일치 6장) 통계 검정은 하지 않았다. k는 반복 10회에서 6장 중 맞힌 수의 평균(괄호는 최소~최대).", "",
        "## 5. 이미지별 오분류 빈도 (primary, 반복 10회 중 틀린 비율)", "",
        t3["금속 종류"], "", t3["처리 조건"], "", t3["처리 방법"], "", t3["온도"], "",
        "- 모든 설정과 기준선의 이미지별 값은 `image_error_counts.csv`, 네 가지 집계는 `misclassification_by_group.csv`.", "",
        "## 6. primary 혼동행렬 (반복 평균 개수와 행 비율)", "", t_cm, "",
        "## 7. 주의", "",
        "- 위 수치는 이 60장에서의 값이며 확인된 사실이다. 원인(예: 어떤 처리 조건에서 주로 틀리는 이유)에 대한 해석은 이 보고서에 넣지 않았다.",
        "- 기준선 평가는 이미지 단위 분할이므로 처리 조건을 알아보는 것만으로도 높은 점수가 나올 수 있다. 불일치 6장의 결과가 그 구분에 쓸 수 있는 유일한 이미지들이다.",
    ]
    rp = f"{log.dir}/report.md"
    open(rp, "w", encoding="utf-8").write("\n".join(md) + "\n")
    log(f"\nreport.md 저장: {rp}")

    # ---------- 금지어 검사 ----------
    bad = term_check(cfg)
    if bad:
        log("[실패] 금지어가 발견됨 (내용을 로그에 다시 쓰면 다음 검사에 걸리므로 파일명:줄 번호만 기록):")
        for b in bad:
            log(f"    {b[0]}:{b[1]}")
        sys.exit(5)
    log(f"금지어 검사 통과: 금지어 {len(cfg.get('forbidden_terms', []))}개 (config forbidden_terms)가 스크립트, config, 로그, 보고서에 없음")   # 금지어 자체를 로그에 찍으면 스스로 걸리므로 개수만 적는다


if __name__ == "__main__":
    main()
