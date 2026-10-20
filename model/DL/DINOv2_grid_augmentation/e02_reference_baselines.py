"""e02: 기준선 (특징 불필요).

- 다수 클래스 예측: 모든 이미지를 가장 많은 클래스(AGG0)로 예측.
- 처리 조건 암기: 각 이미지를 "같은 처리 조건(2단계 폴더)의 다른 이미지들의 다수 라벨"로 예측 (leave-one-image-out).
  처리 조건은 이미지 경로의 2단계 폴더에서 읽는다(R1 파싱 규칙).
- R2: 라벨이 자기 처리 조건의 다수 라벨(그 조건 6장 전체, 자기 포함)과 다른 이미지 목록을 저장한다.
  동률인 처리 조건이 있으면 알린다. leave-one-image-out 예측에서 동률이 생기면 가장 작은 클래스 번호로 정하고 건수를 기록한다.
- R5: 기대값(손 계산)과 비교해 로그에 기대 vs 실제를 적는다 (다르면 멈추지 않고 알린다).
출력: baseline_predictions.csv, baseline_metrics.csv, condition_discordant_images.csv, log.txt
"""
import argparse
from collections import Counter

import numpy as np

from common import Logger, assert_conditions, load_config, load_image_table, metrics, write_csv

SCRIPT = "e02_reference_baselines"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--output", help="outputs 폴더 (기본: config paths.outputs_dir)")
    a = ap.parse_args()
    cfg = load_config(a.config)
    out = a.output or cfg["paths"]["outputs_dir"]
    log = Logger(out, SCRIPT)
    log("config:", cfg["_config_path"])
    imgs = load_image_table(cfg)
    assert_conditions(imgs, cfg, log)
    y = np.array([r["label"] for r in imgs]); N = len(imgs)
    labels = cfg["data"]["labels"]

    # ---------- 다수 클래스 ----------
    cnt = Counter(y.tolist()); maj = min(c for c, n in cnt.items() if n == max(cnt.values()))
    p_maj = np.full(N, maj)
    mm = metrics(y, p_maj)
    log(f"\n[1] 다수 클래스 예측: 항상 AGG{maj} -> accuracy {mm['accuracy']:.4f} ({int((p_maj == y).sum())}/{N}), macro-F1 {mm['macro_f1']:.4f}, recall {np.round(mm['recall'], 3).tolist()}")

    # ---------- 처리 조건 암기 (leave-one-image-out) ----------
    by_cond = {}
    for i, r in enumerate(imgs):
        by_cond.setdefault(r["condition"], []).append(i)
    ties_full, ties_loo = [], 0
    cond_major = {}
    for c, idx in sorted(by_cond.items()):
        cc = Counter(y[idx].tolist()); top = max(cc.values())
        winners = sorted(k for k, v in cc.items() if v == top)
        if len(winners) > 1:
            ties_full.append((c, dict(cc)))
        cond_major[c] = winners[0]
    p_mem = np.zeros(N, int)
    for i, r in enumerate(imgs):
        others = [j for j in by_cond[r["condition"]] if j != i]
        cc = Counter(y[others].tolist()); top = max(cc.values())
        winners = sorted(k for k, v in cc.items() if v == top)
        ties_loo += len(winners) > 1
        p_mem[i] = winners[0]
    mem = metrics(y, p_mem)
    log(f"\n[2] 처리 조건 암기 (같은 처리 조건의 다른 5장의 다수 라벨로 예측, leave-one-image-out)")
    log(f"    accuracy {mem['accuracy']:.4f} ({int((p_mem == y).sum())}/{N}), macro-F1 {mem['macro_f1']:.4f}")
    log(f"    클래스별 recall: " + ", ".join(f"AGG{l} {v:.3f}" for l, v in zip(labels, mem['recall'])) + " | 클래스별 F1: " + ", ".join(f"AGG{l} {v:.3f}" for l, v in zip(labels, mem['f1'])))
    cm = np.zeros((3, 3), int)
    for t, p in zip(y, p_mem):
        cm[t, p] += 1
    log("    혼동행렬 (행=정답 AGG0/1/2, 열=예측): " + " | ".join(str(r) for r in cm.tolist()))
    log(f"    동률: 처리 조건 전체(6장)의 다수 라벨 동률 {len(ties_full)}건" + (f" {ties_full}" if ties_full else "") + f", leave-one-image-out 예측 동률 {ties_loo}건")

    # ---------- R2: 불일치 목록 ----------
    disc = [dict(image_id=r["image_id"], metal=r["metal"], condition=r["condition"], label=r["label"], condition_majority_label=cond_major[r["condition"]])
            for r in imgs if r["label"] != cond_major[r["condition"]]]
    write_csv(f"{log.dir}/condition_discordant_images.csv", disc, ["image_id", "metal", "condition", "label", "condition_majority_label"])
    by_metal = Counter(d["metal"] for d in disc)
    log(f"\n[3] 라벨이 자기 처리 조건의 다수 라벨과 다른 이미지: {len(disc)}장, 금속 종류별 {dict(by_metal)}")
    for d in disc:
        log(f"    {d['image_id']}: 라벨 AGG{d['label']} (그 처리 조건의 다수 라벨 AGG{d['condition_majority_label']})")

    # ---------- R5: 기대값 비교 ----------
    ex = cfg["expected_reference"]
    checks = [
        ("처리 조건 암기 맞힌 수", ex["condition_memorization_correct"], int((p_mem == y).sum()), True),
        ("처리 조건 암기 이미지 수", ex["condition_memorization_n"], N, True),
        ("처리 조건 암기 macro-F1", ex["condition_memorization_macro_f1"], round(mem["macro_f1"], 3), abs(mem["macro_f1"] - ex["condition_memorization_macro_f1"]) <= ex["macro_f1_tolerance"]),
        ("불일치 이미지 수", ex["discordant_n"], len(disc), True),
        ("불일치 이미지의 금속 종류별", ex["discordant_by_metal"], dict(by_metal), None),
        ("동률 처리 조건 수", ex["ties"], len(ties_full), True),
    ]
    log("\n[4] 기대값(손 계산) vs 실제")
    allok = True
    for name, e, v, ok in checks:
        if ok is None:
            ok = dict(e) == dict(v)
        elif ok is True:
            ok = e == v
        allok &= bool(ok)
        log(f"    {name:<26} 기대 {e} | 실제 {v} | {'일치' if ok else '불일치 -> 알림'}")
    log("    결론: " + ("기대값과 모두 일치" if allok else "기대값과 다른 항목이 있음 (위 표 확인; 멈추지 않고 알림)"))

    # ---------- 저장 ----------
    write_csv(f"{log.dir}/baseline_predictions.csv", [dict(image_id=r["image_id"], label=int(y[i]), pred_majority=int(p_maj[i]), pred_condition_memorization=int(p_mem[i])) for i, r in enumerate(imgs)])
    write_csv(f"{log.dir}/baseline_metrics.csv", [dict(baseline="majority_class", accuracy=mm["accuracy"], macro_f1=mm["macro_f1"], **{f"recall_AGG{l}": float(v) for l, v in zip(labels, mm["recall"])}),
                                                  dict(baseline="condition_memorization", accuracy=mem["accuracy"], macro_f1=mem["macro_f1"], **{f"recall_AGG{l}": float(v) for l, v in zip(labels, mem["recall"])})])
    log(f"\n저장: {log.dir}")


if __name__ == "__main__":
    main()
