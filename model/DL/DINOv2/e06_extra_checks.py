"""e06: 추가 확인 (새로 학습하지 않고 e00~e03 결과를 읽어서 정리한다. primary 와 설정은 바꾸지 않는다).

1. primary 의 3x3 혼동행렬(반복 10회를 합친 풀링)과 16개 설정 전체 표
   (accuracy, macro-F1, 클래스별 recall, 불일치 6장 k/6, 나머지 54장 정확도)
3. 비교 기준 추가: (a) 불일치 6장에 모두 AGG0 예측 -> 맞힌 수, (b) 금속 종류 단독 기준선
   (같은 금속 종류의 다른 이미지들의 다수 라벨, leave-one-image-out): accuracy, macro-F1, 6장 k/6.
   동률 처리: 기본 규칙은 e02 와 같이 가장 작은 클래스 번호. 동률 건수를 적고, 다른 규칙(동률은 오답 / 동률 클래스 중 균등 무작위의 기댓값)으로 민감도를 같이 보인다.
4. PH1255-1050 이미지 6장에 대한 primary 의 예측 분포 (반복 10회 중 클래스별 횟수)
5. e01 로그의 결정성(같은 입력 두 번 실행한 최대 절대 차이)과 NaN/Inf 검사 결과 + features.npz 에서 다시 검사
출력: extra_report.md, all_settings_table.csv, primary_confusion_pooled.csv, metal_baseline.csv, ph1255_1050_primary_predictions.csv, log.txt
"""
import argparse
import csv
import json
import sys
from collections import Counter

import numpy as np

from common import Logger, confusion, load_config, metrics, term_check, write_csv

SCRIPT = "e06_extra_checks"


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def macro_f1_with_none(y, pred):
    """pred 에 None(예측 없음)이 있으면 오답으로 세되 정밀도 분모에는 넣지 않는다."""
    f1s = []
    for c in (0, 1, 2):
        tp = sum(1 for t, p in zip(y, pred) if t == c and p == c)
        true = sum(1 for t in y if t == c)
        predc = sum(1 for p in pred if p == c)
        rec = tp / true if true else 0.0
        prec = tp / predc if predc else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--output", help="outputs 폴더 (기본: config paths.outputs_dir)")
    a = ap.parse_args()
    cfg = load_config(a.config)
    out = a.output or cfg["paths"]["outputs_dir"]
    log = Logger(out, SCRIPT)
    log("config:", cfg["_config_path"], "| primary/설정은 바꾸지 않음 (읽기만)")

    conds = list(csv.DictReader(open(f"{out}/e00_make_splits/conditions.csv", encoding="utf-8-sig")))
    ids = [r["image_id"] for r in conds]; pos = {i: k for k, i in enumerate(ids)}; N = len(ids)
    y = np.array([int(r["label"]) for r in conds]); metal = [r["metal"] for r in conds]; cond = [r["condition"] for r in conds]
    settings = list(csv.DictReader(open(f"{out}/e03_run_cv/settings.csv", encoding="utf-8-sig")))
    preds = list(csv.DictReader(open(f"{out}/e03_run_cv/cv_predictions.csv", encoding="utf-8-sig")))
    R = len({int(r["repeat"]) for r in preds})
    S = {s["id"]: np.full((N, R), -1, int) for s in settings}
    for r in preds:
        S[r["setting"]][pos[r["image_id"]], int(r["repeat"])] = int(r["pred"])
    assert all((v >= 0).all() for v in S.values())
    primary = next(s["id"] for s in settings if s["primary"] == "1")
    disc = list(csv.DictReader(open(f"{out}/e02_reference_baselines/condition_discordant_images.csv", encoding="utf-8-sig")))
    D = np.array(sorted(pos[d["image_id"]] for d in disc)); rest = np.array([i for i in range(N) if i not in set(D.tolist())])
    log(f"설정 {len(settings)}개, 반복 {R}회, 이미지 {N}장, primary = {primary}, 불일치 {len(D)}장 / 나머지 {len(rest)}장")

    # ---------- 1-a. primary 혼동행렬 (풀링) ----------
    Sp = S[primary]
    cm = sum(confusion(y, Sp[:, r]) for r in range(R))
    rate = cm / cm.sum(1, keepdims=True)
    assert cm.sum() == N * R
    write_csv(f"{log.dir}/primary_confusion_pooled.csv", [dict(true=f"AGG{i}", **{f"pred_AGG{j}_count": int(cm[i, j]) for j in range(3)}, **{f"pred_AGG{j}_rate": float(rate[i, j]) for j in range(3)}, row_total=int(cm[i].sum())) for i in range(3)])
    log(f"\n[1-a] primary 3x3 혼동행렬 (반복 {R}회를 합친 풀링, 총 {int(cm.sum())}개 예측; 행=정답, 열=예측)")
    for i in range(3):
        log(f"    정답 AGG{i} (행 합 {int(cm[i].sum())}): " + " | ".join(f"AGG{j} {int(cm[i, j])} ({100 * rate[i, j]:.1f}%)" for j in range(3)))
    t_cm = md_table(["정답 \\ 예측", "AGG0", "AGG1", "AGG2", "행 합"], [[f"AGG{i}"] + [f"{int(cm[i, j])} ({100 * rate[i, j]:.1f}%)" for j in range(3)] + [int(cm[i].sum())] for i in range(3)])

    # ---------- 1-b. 16개 설정 전체 표 ----------
    rows, md_rows = [], []
    for s in settings:
        P = S[s["id"]]
        ms = [metrics(y, P[:, r]) for r in range(R)]
        acc = np.array([m["accuracy"] for m in ms]); f1 = np.array([m["macro_f1"] for m in ms]); rec = np.array([m["recall"] for m in ms])
        kD = np.array([(P[D, r] == y[D]).sum() for r in range(R)]); accR = np.array([(P[rest, r] == y[rest]).mean() for r in range(R)])
        rows.append(dict(setting=s["id"], primary=int(s["id"] == primary), accuracy_mean=acc.mean(), accuracy_sd=acc.std(), macro_f1_mean=f1.mean(), macro_f1_sd=f1.std(),
                         recall_AGG0=rec[:, 0].mean(), recall_AGG1=rec[:, 1].mean(), recall_AGG2=rec[:, 2].mean(), k_of_6_mean=kD.mean(), k_of_6_min=int(kD.min()), k_of_6_max=int(kD.max()),
                         rest54_accuracy_mean=accR.mean(), rest54_accuracy_sd=accR.std()))
        md_rows.append([("**" + s["id"] + "** (primary)") if s["id"] == primary else s["id"], f"{acc.mean():.3f}±{acc.std():.3f}", f"{f1.mean():.3f}±{f1.std():.3f}",
                        f"{rec[:, 0].mean():.3f} / {rec[:, 1].mean():.3f} / {rec[:, 2].mean():.3f}", f"{kD.mean():.1f}/6 ({kD.min()}~{kD.max()})", f"{accR.mean():.3f}±{accR.std():.3f}"])
    write_csv(f"{log.dir}/all_settings_table.csv", rows)
    t_all = md_table(["설정", "accuracy", "macro-F1", "recall AGG0 / AGG1 / AGG2", "불일치 6장 k/6 (최소~최대)", "나머지 54장 accuracy"], md_rows)
    log(f"\n[1-b] 16개 설정 전체 표 -> all_settings_table.csv, extra_report.md (primary 행: acc {rows[0]['accuracy_mean']:.3f}, macro-F1 {rows[0]['macro_f1_mean']:.3f})")

    # ---------- 3-a. 6장 모두 AGG0 예측 ----------
    k_all0 = int((y[D] == 0).sum())
    log(f"\n[3-a] 불일치 6장에 모두 AGG0 을 예측하면 맞힌 수: {k_all0}/{len(D)} ({'4/6 확인됨' if k_all0 == 4 else '4/6 이 아님!'})  (6장의 라벨: " + ", ".join(f"{ids[i].split('/')[0]} {ids[i].split('/')[1]}=AGG{y[i]}" for i in D) + ")")

    # ---------- 3-b. 금속 종류 단독 기준선 ----------
    ties, p_i, p_iii = [], np.zeros(N, int), []
    soft = np.zeros((N, 3))
    for i in range(N):
        others = [j for j in range(N) if metal[j] == metal[i] and j != i]
        cnt = Counter(y[others].tolist()); top = max(cnt.values()); T = sorted(c for c, v in cnt.items() if v == top)
        p_i[i] = T[0]                                  # 규칙 i: 가장 작은 클래스 번호 (e02 와 같은 규칙)
        p_iii.append(T[0] if len(T) == 1 else None)    # 규칙 iii: 동률은 오답(예측 없음)
        for c in T:
            soft[i, c] = 1 / len(T)                    # 규칙 ii: 동률 클래스 중 균등 무작위의 기댓값
        if len(T) > 1:
            ties.append((ids[i], len(T), T))
    m_i = metrics(y, p_i)
    kD_i = int((p_i[D] == y[D]).sum())
    acc_ii = float(soft[np.arange(N), y].mean()); kD_ii = float(soft[D, y[D]].sum())
    acc_iii = float(np.mean([p is not None and p == t for p, t in zip(p_iii, y)])); kD_iii = int(sum(1 for i in D if p_iii[i] is not None and p_iii[i] == y[i]))
    f1_iii = macro_f1_with_none(y.tolist(), p_iii)
    by_metal_ties = Counter(t[0].split("/")[0] for t in ties)
    log(f"\n[3-b] 금속 종류 단독 기준선 (같은 금속 종류의 다른 9장의 다수 라벨로 예측, leave-one-image-out)")
    log(f"    동률 처리 (기본 규칙): 동률이면 가장 작은 클래스 번호로 예측 (e02 의 처리 조건 암기와 같은 규칙). 동률로 예측이 정해진 이미지 {len(ties)}장, 금속별 {dict(by_metal_ties)}")
    log(f"    기본 규칙: accuracy {m_i['accuracy']:.4f} ({int((p_i == y).sum())}/{N}), macro-F1 {m_i['macro_f1']:.4f}, recall " + ", ".join(f"AGG{c} {v:.3f}" for c, v in enumerate(m_i['recall'])) + f", 불일치 6장 k/6 = {kD_i}/6, 나머지 54장 accuracy {(p_i[rest] == y[rest]).mean():.3f}")
    log(f"    민감도: 동률을 오답으로 세면 accuracy {acc_iii:.4f}, macro-F1 {f1_iii:.4f}, 6장 {kD_iii}/6 | 동률 클래스 중 균등 무작위의 기댓값이면 accuracy {acc_ii:.4f}, 6장 {kD_ii:.2f}/6")
    cmm = confusion(y, p_i)
    log("    혼동행렬(기본 규칙, 행=정답 AGG0/1/2, 열=예측): " + " | ".join(str(r) for r in cmm.tolist()))
    for im, n, T in ties:
        pass
    write_csv(f"{log.dir}/metal_baseline.csv", [dict(rule="기본(동률은 가장 작은 클래스 번호)", accuracy=m_i["accuracy"], macro_f1=m_i["macro_f1"], k_of_6=kD_i, n_ties=len(ties)),
                                                 dict(rule="동률은 오답", accuracy=acc_iii, macro_f1=f1_iii, k_of_6=kD_iii, n_ties=len(ties)),
                                                 dict(rule="동률 클래스 중 균등 무작위의 기댓값", accuracy=acc_ii, macro_f1="", k_of_6=kD_ii, n_ties=len(ties))])

    # ---------- 4. PH1255-1050 ----------
    idx = [i for i in range(N) if cond[i] == "PH1255-1050"]
    ph = []
    for i in idx:
        c = Counter(Sp[i].tolist())
        ph.append(dict(image_id=ids[i], metal=metal[i], true_label=int(y[i]), pred_AGG0=c.get(0, 0), pred_AGG1=c.get(1, 0), pred_AGG2=c.get(2, 0), correct=c.get(int(y[i]), 0), repeats=R))
    write_csv(f"{log.dir}/ph1255_1050_primary_predictions.csv", ph)
    log(f"\n[4] PH1255-1050 이미지 {len(idx)}장에 대한 primary 의 예측 (반복 {R}회 중 클래스별 횟수)")
    for r in ph:
        log(f"    {r['metal']:<10} 정답 AGG{r['true_label']} | AGG0 {r['pred_AGG0']:>2} | AGG1 {r['pred_AGG1']:>2} | AGG2 {r['pred_AGG2']:>2} | 맞힌 횟수 {r['correct']}/{R}")
    tot = np.array([[r["pred_AGG0"], r["pred_AGG1"], r["pred_AGG2"]] for r in ph]).sum(0)
    log(f"    합계({len(idx)}장 x {R}회 = {len(idx) * R}): AGG0 {tot[0]}, AGG1 {tot[1]}, AGG2 {tot[2]} (정확도 {tot[0] / (len(idx) * R):.3f})")
    t_ph = md_table(["금속 종류", "정답", "AGG0 예측", "AGG1 예측", "AGG2 예측", f"맞힌 횟수 (/{R})"], [[r["metal"], f"AGG{r['true_label']}", r["pred_AGG0"], r["pred_AGG1"], r["pred_AGG2"], r["correct"]] for r in ph])

    # ---------- 5. e01 결정성 / NaN, Inf ----------
    e01_dir = f"{out}/e01_extract_features"
    lines = open(f"{e01_dir}/log.txt", encoding="utf-8").read().splitlines()
    pick = [l for l in lines if "결정성" in l or "두 번 실행 최대 절대 차이" in l or "NaN/Inf" in l]
    meta = json.load(open(f"{e01_dir}/meta.json", encoding="utf-8"))
    z = np.load(f"{e01_dir}/features.npz")
    fin = {k: bool(np.isfinite(z[k]).all()) for k in z.files}
    log("\n[5] e01 로그 (원문):")
    for l in pick:
        log("    " + l.strip())
    log(f"    meta.json 의 determinism_max_abs_diff: {meta['determinism_max_abs_diff']}")
    log(f"    features.npz 에서 다시 검사: " + ", ".join(f"{k} {z[k].shape} 모두 유한값={fin[k]}" for k in z.files) + f" | NaN 개수 {sum(int(np.isnan(z[k]).sum()) for k in z.files)}, Inf 개수 {sum(int(np.isinf(z[k]).sum()) for k in z.files)}")
    assert all(fin.values())

    # ---------- extra_report.md ----------
    base_rows = [["다수 클래스 예측 (항상 AGG0)", "0.467", "0.212", f"{k_all0}/6", "0.444"],
                 ["처리 조건 암기", "0.900", "0.893", "0/6", "1.000"],
                 [f"금속 종류 단독 (동률: 가장 작은 클래스 번호)", f"{m_i['accuracy']:.3f}", f"{m_i['macro_f1']:.3f}", f"{kD_i}/6", f"{(p_i[rest] == y[rest]).mean():.3f}"]]
    md = ["# 추가 확인 (e06)", "", "primary 와 설정은 바꾸지 않았다. 새로 학습하지 않고 e00~e03 의 결과를 읽어서 정리했다.", "",
          f"## 1. primary ({primary}) 3x3 혼동행렬 (반복 {R}회 풀링, 총 {int(cm.sum())}개 예측)", "", t_cm, "",
          "## 2. 16개 설정 전체 표", "", t_all, "",
          "k/6 는 불일치 6장(라벨이 자기 처리 조건의 다수 라벨과 다른 이미지) 중 반복 10회에서 맞힌 수의 평균(괄호는 최소~최대). 통계 검정은 하지 않았다.", "",
          "## 3. 비교 기준", "", md_table(["기준", "accuracy", "macro-F1", "불일치 6장 k/6", "나머지 54장 accuracy"], base_rows), "",
          f"- (a) 불일치 6장에 모두 AGG0 을 예측하면 {k_all0}/6 ({'4/6 맞음' if k_all0 == 4 else '4/6 아님'}).",
          f"- (b) 금속 종류 단독 기준선의 동률 처리: 동률이면 가장 작은 클래스 번호로 예측(e02 와 같은 규칙). 동률로 정해진 이미지 {len(ties)}장(금속별 {dict(by_metal_ties)}). 민감도: 동률을 오답으로 세면 accuracy {acc_iii:.3f}, macro-F1 {f1_iii:.3f}, 6장 {kD_iii}/6; 동률 클래스 중 균등 무작위의 기댓값이면 accuracy {acc_ii:.3f}, 6장 {kD_ii:.2f}/6.", "",
          f"## 4. PH1255-1050 이미지 {len(idx)}장에 대한 primary 의 예측 (반복 {R}회 중 클래스별 횟수)", "", t_ph, "",
          f"합계 {len(idx) * R}번 중 AGG0 {tot[0]}, AGG1 {tot[1]}, AGG2 {tot[2]}.", "",
          "## 5. e01 결정성과 NaN/Inf 검사", ""] + ["- `" + l.strip() + "`" for l in pick] + [
          f"- meta.json: {meta['determinism_max_abs_diff']}", f"- features.npz 를 다시 검사: " + ", ".join(f"{k} {z[k].shape} 유한값 {fin[k]}" for k in z.files)]
    open(f"{log.dir}/extra_report.md", "w", encoding="utf-8").write("\n".join(md) + "\n")
    log(f"\nextra_report.md 저장: {log.dir}/extra_report.md")
    bad = term_check(cfg)
    if bad:
        log("[실패] 금지어가 발견됨 (파일명:줄 번호만 기록):")
        for b in bad:
            log(f"    {b[0]}:{b[1]}")
        sys.exit(5)
    log(f"금지어 검사 통과: 금지어 {len(cfg.get('forbidden_terms', []))}개")


if __name__ == "__main__":
    main()
