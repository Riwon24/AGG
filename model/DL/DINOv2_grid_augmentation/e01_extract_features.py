"""e01 (배경 증강 판): 분석 타일 2,591개와 이미지 전체(마스크 bbox 크롭)에서, 원본(k=0)과 배경 증강본(k=1..K)의 고정 DINOv2 ViT-B/14 특징을 뽑는다 (GPU).

- 증강은 시편이 아니라 배경 픽셀(마스크 밖, 경계에서 buffer 8px 바깥부터 feather 16px 에 걸쳐)에만 적용한다. 함수와 파라미터는 배경 증강 실험의
  `bg_common.augment_image` 와 `exp_bgaug/config.yaml` 의 bgaug 항목, b01 측정 풀(aug_params.json)을 읽기 전용으로 그대로 쓴다 (같은 이미지 번호, 같은 시드 -> 같은 증강본).
- 이미지 60장을 하나씩 읽어 k=0 과 k=1..K 버전을 만들고, 분석 타일을 224x224 로 잘라(resize 없음) 모델에 넣는다. 증강본에서 자른 타일이 원본과 픽셀이 같으면
  다시 계산하지 않고 원본 행을 가리킨다. 이미지 전체 특징은 버전마다 마스크 bbox 를 잘라 224x224 로 resize(BICUBIC) 한 한 장의 cls 이다.
- 고정, eval, no_grad, fp32, ImageNet 정규화 (DINOv2_grid 의 e01 과 같음). 모델 출처는 exp 의 e01 이 기록한 저장소 파일 sha256 과 체크포인트 sha256 으로 대조한다 (달라졌으면 멈춤).
- 검증: 결정성(같은 입력 두 번 실행의 최대 차이), 원본(k=0) 특징이 DINOv2_grid 의 e01 특징과 같은지 (같은 이미지, 같은 타일), 증강본에서 마스크 안쪽 픽셀이 원본과 같은지(assert).
출력: features.npz (tile_cls, tile_patchmean: 계산한 고유 타일 행, tile_row_of: 타일 x 버전 -> 행, whole_cls: 60 x 버전 x 768), tile_index.csv, whole_index.csv, meta.json, log.txt
"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

from common import Logger, load_config, load_image_table, load_selected_tiles, sha256_file, write_csv   # (이 폴더의 common 을 먼저 불러온다)

SCRIPT = "e01_extract_features"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--output", help="outputs 폴더 (기본: config paths.outputs_dir)")
    a = ap.parse_args()
    cfg = load_config(a.config)
    out = a.output or cfg["paths"]["outputs_dir"]
    log = Logger(out, SCRIPT)
    o = log.dir
    m = cfg["model"]
    log("config:", cfg["_config_path"])
    repo = Path(cfg["paths"]["dinov2_repo_dir"]); ckpt = Path(cfg["paths"]["dinov2_checkpoint"])
    assert repo.exists() and ckpt.exists(), "캐시된 DINOv2 저장소/체크포인트가 없음"
    # ---- 배경 증강 함수와 파라미터 (읽기 전용, 배경 증강 실험의 것) ----
    bg_cfg_path = Path(cfg["paths"]["bgaug_config"])
    sys.path.insert(0, str(bg_cfg_path.parent))
    import bg_common                                        # noqa: E402  (exp_bgaug/bg_common.py)
    bg = dict(yaml.safe_load(bg_cfg_path.read_text(encoding="utf-8"))["bgaug"])
    pj = json.loads(Path(cfg["paths"]["bgaug_aug_params"]).read_text(encoding="utf-8"))
    for k_, v in pj["params"].items():
        assert bg[k_] == v, f"bgaug 설정 {k_}={bg[k_]} 가 b01 측정 때 값 {v} 와 다름"
    pool = pj["pool"]
    K, feather = int(cfg["aug"]["n_aug"]), int(bg["feather_px"])
    assert K == int(bg["n_aug"]), "aug.n_aug 가 배경 증강 실험의 n_aug 와 다름"

    # ---------- 환경 / 출처 ----------
    torch.manual_seed(cfg["cv"]["seed"]); np.random.seed(cfg["cv"]["seed"])
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    import cv2
    import torchvision
    log(f"torch {torch.__version__} | torchvision {torchvision.__version__} | cuda build {torch.version.cuda} | opencv {cv2.__version__} | device {dev}" + (f" ({torch.cuda.get_device_name(0)})" if dev == "cuda" else ""))
    exp_e01 = Path(cfg["paths"]["exp_dir"]) / "outputs" / "e01_extract_features"
    old = {r["path"]: r["sha256"] for r in csv.DictReader(open(exp_e01 / "repo_files_sha256.csv", encoding="utf-8-sig"))}
    new = {f.relative_to(repo).as_posix(): sha256_file(f) for f in sorted(repo.rglob("*")) if f.is_file() and "__pycache__" not in f.parts}
    assert old == new, f"저장소 파일이 exp 의 e01 기록과 다름: {len(set(old.items()) ^ set(new.items()))}건"
    meta_old = json.loads((exp_e01 / "meta.json").read_text(encoding="utf-8"))
    ck_sha = sha256_file(ckpt)
    assert ck_sha == meta_old["checkpoint_sha256"], "체크포인트가 exp 의 e01 기록과 다름"
    assert meta_old["normalization"]["verified_against_repo"] and list(meta_old["normalization"]["mean"]) == m["mean"] and list(meta_old["normalization"]["std"]) == m["std"]
    log(f"모델 출처 검증: 저장소 파일 {len(new)}개 sha256 과 체크포인트 sha256 이 exp 의 e01 기록과 같음, 정규화 값이 e01 에서 공식 코드와 대조한 값과 같음")
    model = torch.hub.load(str(repo), m["name"], source="local", pretrained=False, trust_repo=True)
    res = model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=True)
    assert not res.missing_keys and not res.unexpected_keys
    model.eval().to(dev).float()
    for p in model.parameters():
        p.requires_grad_(False)
    n_param = sum(p.numel() for p in model.parameters())
    assert abs(n_param / 1e6 - m["expected_params_million"]) < 0.5
    log(f"가중치 로드(strict=True), 파라미터 {n_param / 1e6:.2f}M, 고정(requires_grad=False)")
    mean = torch.tensor(m["mean"], device=dev).view(1, 3, 1, 1); std = torch.tensor(m["std"], device=dev).view(1, 3, 1, 1)

    @torch.no_grad()
    def embed(arrs, want_patch):
        x = torch.from_numpy(np.stack(arrs)).to(dev).permute(0, 3, 1, 2).float() / 255.0
        o_ = model.forward_features((x - mean) / std)
        cls, pt = o_["x_norm_clstoken"], o_["x_norm_patchtokens"]
        assert cls.shape == (x.shape[0], 768) and pt.shape == (x.shape[0], (m["tile_input"] // 14) ** 2, 768)
        return cls.float().cpu().numpy(), (pt.mean(dim=1).float().cpu().numpy() if want_patch else None)

    def embed_many(arrs, want_patch=True):
        cl, pm = [], []
        for i in range(0, len(arrs), m["batch_size"]):
            c, p = embed(arrs[i:i + m["batch_size"]], want_patch); cl.append(c); pm.append(p)
        return np.concatenate(cl), (np.concatenate(pm) if want_patch else None)

    # ---------- 입력 ----------
    tiles = load_selected_tiles(cfg); imgs = load_image_table(cfg)
    assert len(tiles) == cfg["data"]["expected_tiles"] and len(imgs) == cfg["data"]["expected_images"]
    by_img = {}
    for ti, t in enumerate(tiles):
        by_img.setdefault(t["image_id"], []).append(ti)
    T, K1 = m["tile_input"], K + 1
    log(f"\n분석 타일 {len(tiles)}개, 이미지 {len(imgs)}장, 버전 {K1}개 (원본 + 배경 증강 {K}개), buffer {bg['buffer_px']}px, feather {feather}px")
    ex = cfg["data"]["scalebar_exclude_box"]
    row_of = np.full((len(tiles), K1), -1, np.int32)
    cls_all, pm_all, whole = [], [], np.zeros((len(imgs), K1, 768), np.float32)
    widx, nrows, n_same, n_new, t0, check = [], 0, 0, 0, time.time(), None
    for i, r in enumerate(imgs):
        img, mk = bg_common.read_rgb(r["path"]), bg_common.read_mask(r["mask"])
        d = bg_common.dist_to_mask(mk)
        ys, xs = np.where(mk)
        bx0, bx1, by0, by1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
        idxs = by_img[r["image_id"]]
        crops0 = [img[int(tiles[ti]["y0"]):int(tiles[ti]["y0"]) + T, int(tiles[ti]["x0"]):int(tiles[ti]["x0"]) + T].copy() for ti in idxs]
        c0, p0 = embed_many(crops0)
        base = nrows
        cls_all.append(c0); pm_all.append(p0); nrows += len(idxs)
        for j, ti in enumerate(idxs):
            row_of[ti, 0] = base + j
        if check is None:
            check = (crops0[:128], c0[:128].copy(), p0[:128].copy())
        wholes = [img]
        for k in range(1, K1):
            aug, alpha, _ = bg_common.augment_image(img, d, pool, bg, i, k, feather)
            assert np.array_equal(aug[mk], img[mk]), f"증강본에서 시편(마스크 안쪽) 픽셀이 바뀜: {r['image_id']} k={k}"
            wholes.append(aug)
            todo, todo_ti = [], []
            for j, ti in enumerate(idxs):
                x0, y0 = int(tiles[ti]["x0"]), int(tiles[ti]["y0"])
                cr = aug[y0:y0 + T, x0:x0 + T]
                if np.array_equal(cr, crops0[j]):
                    row_of[ti, k] = base + j; n_same += 1
                else:
                    todo.append(cr.copy()); todo_ti.append(ti)
            if todo:
                ck, pk = embed_many(todo)
                cls_all.append(ck); pm_all.append(pk)
                for q, ti in enumerate(todo_ti):
                    row_of[ti, k] = nrows + q
                nrows += len(todo); n_new += len(todo)
        # 이미지 전체: 마스크 bbox 를 잘라 224x224 로 resize (BICUBIC), 버전마다
        wa = [np.asarray(Image.fromarray(w_[by0:by1, bx0:bx1]).resize((T, T), Image.BICUBIC)) for w_ in wholes]
        wc, _ = embed_many(wa, want_patch=False)
        whole[i] = wc
        wcrop_w, wcrop_h = bx1 - bx0, by1 - by0
        widx.append(dict(row=i, image_id=r["image_id"], label=r["label"], bbox_x0=bx0, bbox_y0=by0, bbox_x1=bx1, bbox_y1=by1, w=wcrop_w, h=wcrop_h,
                         aspect_distortion=round(max(wcrop_w / wcrop_h, wcrop_h / wcrop_w), 4),
                         overlaps_scalebar_zone=int(bx0 < ex[2] and bx1 > ex[0] and by0 < ex[3] and by1 > ex[1])))
        print(f"  {i + 1}/{len(imgs)} {r['image_id']}  누적 행 {nrows}  {time.time() - t0:.0f}s", flush=True)
    cls_all, pm_all = np.concatenate(cls_all), np.concatenate(pm_all)
    assert (row_of >= 0).all() and cls_all.shape[0] == nrows == pm_all.shape[0]
    for name, arr in (("tile_cls", cls_all), ("tile_patchmean", pm_all), ("whole_cls", whole)):
        assert np.isfinite(arr).all(), f"{name} 에 NaN/Inf"
    log(f"특징 추출 {time.time() - t0:.0f}s: 타일 계산 {nrows}행 (원본 {len(tiles)} + 증강본에서 달라진 {n_new}), 증강본에서 원본과 픽셀이 같아 재사용 {n_same}개, 이미지 전체 {whole.shape[0] * whole.shape[1]}장")
    tot = len(tiles) * K
    log(f"  증강본 타일 {tot}개 중 달라진 것 {n_new}개 ({n_new / tot * 100:.1f}%), 같은 것 {n_same}개 ({n_same / tot * 100:.1f}%)")
    log("검증 통과: 증강본에서 모든 이미지, 모든 k 의 시편(마스크 안쪽) 픽셀이 원본과 완전히 같음")

    # ---------- 검증 ----------
    c2, p2 = embed_many(check[0])
    d1, d2 = float(np.abs(c2 - check[1]).max()), float(np.abs(p2 - check[2]).max())
    log(f"결정성: 같은 입력(타일 {len(check[0])}개)을 두 번 실행한 최대 절대 차이 cls {d1:.3e}, patchmean {d2:.3e}")
    assert d1 < 1e-3 and d2 < 1e-3
    grid_e01 = Path(cfg["paths"]["exp_grid_dir"]) / "outputs" / "e01_extract_features"
    dw = float("nan")
    if (grid_e01 / "features.npz").exists():
        og = np.load(grid_e01 / "features.npz")
        oidx = list(csv.DictReader(open(grid_e01 / "tile_index.csv", encoding="utf-8-sig")))
        assert [(t["image_id"], t["tile_id"]) for t in oidx] == [(t["image_id"], t["tile_id"]) for t in tiles], "DINOv2_grid 의 타일 순서가 다름"
        dc = float(np.abs(cls_all[row_of[:, 0]] - og["tile_cls"]).max()); dpm = float(np.abs(pm_all[row_of[:, 0]] - og["tile_patchmean"]).max())
        dw = float(np.abs(whole[:, 0] - og["whole_cls"]).max())
        log(f"DINOv2_grid 의 e01 특징과 비교(원본 k=0, 같은 타일 {len(tiles)}개, 이미지 {len(imgs)}장): cls 최대 절대 차이 {dc:.3e}, patchmean {dpm:.3e}, 이미지 전체 cls {dw:.3e}")
    else:
        log("DINOv2_grid 의 e01 출력이 없어 비교를 생략함")
    dist = np.array([r["aspect_distortion"] for r in widx])
    ov = sum(r["overlaps_scalebar_zone"] for r in widx)
    log(f"이미지 전체 특징 {whole.shape} | 종횡비 왜곡 평균 {dist.mean():.3f} 최소 {dist.min():.3f} 최대 {dist.max():.3f} | 마스크 bbox 가 스케일바 제외 구역과 겹치는 이미지 {ov}/{len(widx)}장")
    log(f"\n검증 통과: 타일 수 {len(tiles):,}, NaN/Inf 없음")
    for name, arr in (("tile_cls", cls_all), ("tile_patchmean", pm_all), ("whole_cls", whole)):
        log(f"  {name:<15} 전체 평균 {arr.mean():+.4f} | 차원별 표준편차 평균 {arr.std(0).mean() if arr.ndim == 2 else arr.reshape(-1, arr.shape[-1]).std(0).mean():.4f}")

    np.savez(o / "features.npz", tile_cls=cls_all.astype(np.float32), tile_patchmean=pm_all.astype(np.float32), tile_row_of=row_of, whole_cls=whole)
    write_csv(o / "tile_index.csv", [dict(row=i, image_id=t["image_id"], tile_id=t["tile_id"], x0=t["x0"], y0=t["y0"], label=t["label"]) for i, t in enumerate(tiles)])
    write_csv(o / "whole_index.csv", widx)
    meta = dict(torch=torch.__version__, torchvision=torchvision.__version__, opencv=cv2.__version__, device=dev, gpu=torch.cuda.get_device_name(0) if dev == "cuda" else None,
                model=m["name"], params=n_param, checkpoint_sha256=ck_sha, n_tiles=len(tiles), n_images=len(imgs), n_aug=K, buffer_px=bg["buffer_px"], feather_px=feather,
                n_rows=int(nrows), n_new_aug_rows=int(n_new), n_same_reused=int(n_same), determinism_max_abs_diff=dict(cls=d1, patchmean=d2),
                orig_vs_dinov2_grid_whole_max_abs_diff=dw, aug_params_sha256=sha256_file(cfg["paths"]["bgaug_aug_params"]))
    (o / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"\n저장: {o} (features.npz: tile_cls {cls_all.shape}, tile_patchmean {pm_all.shape}, tile_row_of {row_of.shape}, whole_cls {whole.shape})")


if __name__ == "__main__":
    main()
