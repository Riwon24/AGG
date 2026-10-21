"""m10: DINOv2_grid_augmentation 의 타일 특징(cls 768)으로 Attention-based MIL 을 이진(AGG0 = 0 vs AGG1+AGG2 = 1) 학습/평가한다 (WSL, torch + GPU).

- bag = 이미지 1장, instance = 그 이미지의 분석 타일(2,591개 분할, 이미지당 36~48개). 특징 추출 파이프라인과 분할(e00 이미지 단위 5-fold x 10 반복, LOCO, LOMO)은 기존 것을 그대로 쓴다.
- 학습 bag = 학습 이미지의 모든 버전(원본 + 배경 증강 5개 = 6개), 평가 bag = 원본(k=0) 만. 표준화는 학습 bag 의 타일로만 구해 적용한다.
- 모델: 타일 FC(768 -> 64) -> ReLU -> Dropout -> gated attention(tanh x sigmoid, 64) -> softmax 가중합 -> Dropout -> FC -> sigmoid. 고정 epoch(40), 조기 종료 없음, 시드 5개 앙상블.
- 누수 방지 assert: train/test 이미지 서로소, 학습 bag 이 모두 학습 이미지에서 온 것, 표준화 통계에 test 타일이 없음, test 점수는 원본 bag 만 사용(다른 버전 bag 의 attention 은 해석용으로만 저장하고 예측에 쓰지 않음).
- 지표는 fold 를 합친(pooled) 60장 예측으로 계산. IMG 는 반복 10회 각각 구해 평균 +- SD, LOCO/LOMO 는 한 번. AUROC 는 양성 확률로 계산 (LOCO/LOMO 는 fold 마다 모델이 달라 참고용).
출력: mil_predictions.csv, mil_summary.csv, mil_fold_accuracy.csv, mil_train_stats.csv, attention_tiles.csv, attention_arrays.npz (scheme 별 [이미지, 버전, 시드, 타일]), log.txt
"""
import argparse
import csv
import time
from collections import defaultdict

import numpy as np
import torch

from mcommon import (Logger, check_folds, load_config, load_features, load_images, make_folds, metrics_bin, predict_mil, standardize_fit, tile_keep, to_binary, train_mil, write_csv, pad_bags)

SCRIPT = "m10_train_eval_mil"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    out = cfg["paths"]["outputs_dir"]
    log = Logger(out, SCRIPT)
    log("config:", cfg["_config_path"])
    torch.use_deterministic_algorithms(True, warn_only=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"torch {torch.__version__} | device {dev}" + (f" ({torch.cuda.get_device_name(0)})" if dev == "cuda" else ""))
    img = load_images(cfg)
    ids = img["ids"]; y3 = img["y"]; y = to_binary(y3, cfg); N = len(ids); pos = {i: k for k, i in enumerate(ids)}
    L = load_features(cfg["paths"]["aug_out"], True, ids)
    F = L["F"][cfg["features"]]; K1 = L["K1"]
    keep = tile_keep(cfg, L["tiles_of"]); tiles_of = [L["tiles_of"][i][keep[i]] for i in range(N)]       # tile_filter: all | full (마스크 100% 타일만)
    nmax = max(len(t) for t in tiles_of)
    if cfg.get("tile_filter", "all") == "full":
        assert all(np.array_equal(F[k][tiles_of[i]], F[0][tiles_of[i]]) for i in range(N) for k in range(K1)), "마스크 100% 타일의 특징이 증강 버전 사이에 다름"
        log(f"tile_filter = full: 마스크 100% 타일만 사용 (전체 분석 타일 {sum(len(t) for t in L['tiles_of'])}개 중 {sum(len(t) for t in tiles_of)}개, 이미지당 {min(len(t) for t in tiles_of)}~{nmax}개). 이 타일들의 특징은 모든 증강 버전에서 원본과 같음을 assert 로 확인 (시편만 있어 배경 증강이 바꾸지 않음). 학습 절차와 하이퍼파라미터는 이전과 완전히 같다.")
    # bags[i][k] = (n_i, 768) float32 (버전 k 의 이미지 i 의 타일 특징)
    bags = [[F[k][tiles_of[i]].astype(np.float32) for k in range(K1)] for i in range(N)]
    mc = cfg["mil"]; S = mc["seeds"]
    log(f"bag {N}개 (이미지당 타일 {min(len(t) for t in tiles_of)}~{nmax}개), 버전 {K1}개, 특징 {cfg['features']} {bags[0][0].shape[1]}차원, 이진 라벨 음성 {int((y == 0).sum())} / 양성 {int((y == 1).sum())}")
    log(f"MIL: embed {mc['embed_dim']}, attn {mc['attn_dim']}, dropout {mc['dropout']}, L2 {mc['weight_decay']}, lr {mc['lr']}, batch {mc['batch_bags']} bag, epochs {mc['epochs']} (고정), 시드 {S}개 앙상블, 클래스 균형 손실 {mc['class_balanced_loss']}")

    # ---------- 분할 ----------
    sp = list(csv.DictReader(open(cfg["paths"]["splits_csv"], encoding="utf-8-sig")))
    byk = defaultdict(lambda: {"train": [], "test": []})
    for r in sp:
        byk[(int(r["repeat"]), int(r["fold"]))][r["role"]].append(pos[r["image_id"]])
    reps = sorted({k[0] for k in byk}); nf = sorted({k[1] for k in byk})
    plan = {"IMG": [[(np.array(byk[(r, f)]["train"]), np.array(byk[(r, f)]["test"]), f"{r}-{f}") for f in nf] for r in reps]}
    for sch, unit in cfg["schemes"].items():
        folds = make_folds(img, img[unit]); check_folds(img, folds, sch)
        plan[sch] = [[(tr, te, u) for u, tr, te in folds]]
    for sch, reps_ in plan.items():
        for folds in reps_:
            seen = np.zeros(N, int)
            for tr, te, _ in folds:
                assert set(tr.tolist()).isdisjoint(te.tolist()) and set(np.unique(y[tr]).tolist()) == {0, 1}, sch
                seen[te] += 1
            assert (seen == 1).all(), sch
    log("분할 assert 통과: 기존 e00/그룹 분할 그대로, 모든 repeat 에서 모든 이미지가 정확히 한 번 test, train/test 서로소, train 에 두 클래스")

    pred_rows, sum_rows, fold_rows, stat_rows = [], [], [], []
    ATT = {}
    t0 = time.time()
    for sch, reps_ in plan.items():
        R = len(reps_)
        P = np.zeros((N, R)); att = np.full((N, K1, S, nmax), np.nan, np.float32)
        for r, folds in enumerate(reps_):
            for fi, (tr, te, u) in enumerate(folds):
                # ---- 학습 bag: 학습 이미지의 모든 버전, 표준화는 학습 bag 타일로만 ----
                tr_bags = [bags[i][k] for i in tr for k in range(K1)]
                tr_img = [i for i in tr for k in range(K1)]
                assert set(tr_img) == set(tr.tolist()) and set(tr_img).isdisjoint(te.tolist()), "학습 bag 에 test 이미지가 있음"
                ytr = np.array([y[i] for i in tr_img])
                mu, sd = standardize_fit(tr_bags)
                assert sum(len(b) for b in tr_bags) == sum(len(bags[i][k]) for i in tr for k in range(K1)), "표준화에 학습 이외의 타일이 쓰임"
                tr_n = [(b - mu) / sd for b in tr_bags]
                te_all = [[(bags[i][k] - mu) / sd for k in range(K1)] for i in te]
                pv = np.zeros((len(te), S)); tl, ta = [], []
                for s in range(S):
                    seed = mc["base_seed"] + 100000 * r + 1000 * fi + s
                    model, loss, tacc = train_mil(tr_n, ytr, cfg, seed, dev, nmax)
                    tl.append(loss); ta.append(tacc)
                    flat = [b for ib in te_all for b in ib]                                   # test 이미지의 모든 버전 (예측은 버전 0 만 사용)
                    p, atts = predict_mil(model, flat, nmax, dev)
                    for j, i in enumerate(te):
                        pv[j, s] = p[j * K1]                                                  # 버전 0 = 원본
                        if r == 0:                                                           # attention 은 repeat 0 (LOCO/LOMO 는 유일한 repeat) 만 저장
                            for k in range(K1):
                                att[i, k, s, :len(atts[j * K1 + k])] = atts[j * K1 + k]
                P[te, r] = pv.mean(1)                                                        # 시드 앙상블 확률 (원본 bag)
                stat_rows.append(dict(scheme=sch, repeat=r, fold=fi, unit=u, train_bags=len(tr_n), train_loss=round(float(np.mean(tl)), 4), train_acc=round(float(np.mean(ta)), 4)))
                if sch != "IMG":
                    fold_rows.append(dict(scheme=sch, fold=fi, unit=u, n_test=len(te), accuracy=round(float(np.mean((P[te, r] > 0.5) == y[te])), 4)))
            print(f"  [{sch}] repeat {r + 1}/{R} 완료 ({time.time() - t0:.0f}s)", flush=True)
        ATT[sch] = att
        ms = [metrics_bin(y, (P[:, r] > 0.5).astype(int), P[:, r]) for r in range(R)]
        row = dict(scheme=sch, n_repeats=R)
        for k in ("accuracy", "balanced_acc", "macro_f1", "recall_neg", "recall_pos", "f1_pos", "auroc"):
            row[k] = round(float(np.mean([m[k] for m in ms])), 4); row[k + "_sd"] = round(float(np.std([m[k] for m in ms])), 4)
        sum_rows.append(row)
        for r in range(R):
            for i in range(N):
                pred_rows.append(dict(scheme=sch, repeat=r, image_id=ids[i], label=int(y[i]), label3=int(y3[i]), pred=int(P[i, r] > 0.5), prob=round(float(P[i, r]), 6)))
        log(f"[{sch}] MIL accuracy {row['accuracy']:.3f}" + (f" ± {row['accuracy_sd']:.3f}" if R > 1 else "") + f" | balanced {row['balanced_acc']:.3f} | macro-F1 {row['macro_f1']:.3f} | recall AGG0 {row['recall_neg']:.2f} / AGG1+2 {row['recall_pos']:.2f} | AUROC {row['auroc']:.3f}")
    log(f"\n학습 요약: 마지막 epoch 학습 손실과 학습 bag accuracy 의 평균 (참고, 평가 지표 아님): " + ", ".join(
        f"{sch} 손실 {np.mean([r['train_loss'] for r in stat_rows if r['scheme'] == sch]):.3f} / acc {np.mean([r['train_acc'] for r in stat_rows if r['scheme'] == sch]):.3f}" for sch in plan))
    write_csv(f"{log.dir}/mil_predictions.csv", pred_rows)
    write_csv(f"{log.dir}/mil_summary.csv", sum_rows)
    write_csv(f"{log.dir}/mil_fold_accuracy.csv", fold_rows)
    write_csv(f"{log.dir}/mil_train_stats.csv", stat_rows)
    # attention 저장: 타일 순서 정의 (이미지 i 의 j 번째 타일 = tiles_of[i][j])
    tindex = list(csv.DictReader(open(f"{cfg['paths']['aug_out']}/e01_extract_features/tile_index.csv", encoding="utf-8-sig")))
    mf = {(r["image_id"], r["tile_id"]): r for r in csv.DictReader(open(cfg["paths"]["tiles_csv"], encoding="utf-8-sig"))}
    rows = []
    for i in range(N):
        for j, ti in enumerate(tiles_of[i]):
            t = tindex[ti]; m = mf[(t["image_id"], t["tile_id"])]
            rows.append(dict(image_idx=i, image_id=ids[i], j=j, tile_id=t["tile_id"], x0=t["x0"], y0=t["y0"], mask_frac=m["mask_frac"], mask_px=m["mask_px"]))
    write_csv(f"{log.dir}/attention_tiles.csv", rows)
    np.savez(f"{log.dir}/attention_arrays.npz", **{f"att_{k}": v for k, v in ATT.items()}, tile_counts=np.array([len(t) for t in tiles_of]))
    log(f"\n누수 방지 확인: 모든 fold 에서 학습 bag 이 학습 이미지에서만 왔고(assert), 표준화 통계는 학습 bag 타일만 사용(assert), 조기 종료/하이퍼파라미터 탐색 없음(고정 epoch), test 점수는 원본 bag 만 사용")
    log(f"저장: {log.dir} (총 {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
