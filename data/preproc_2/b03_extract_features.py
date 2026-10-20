"""b03: 원본(k=0)과 배경 증강본(k=1..K)에서 선택 타일의 고정 DINOv2 ViT-B/14 특징을 뽑는다 (WSL, GPU).

- 이미지 60장을 하나씩 읽어 k=0 (원본) 과 k=1..K (증강; b02 와 같은 함수 augment_image, feather_px 는 config) 버전을 만들고,
  타일(config tiles.select, 격자 전체 9,900개)를 224x224 로 잘라 (resize 없음) 모델에 넣는다. 고정, eval, no_grad, fp32, ImageNet 정규화 (exp/e01 과 같음).
- 증강본에서 자른 타일이 원본에서 자른 타일과 픽셀 단위로 같으면(배경이 없거나 배경이 모두 buffer 안) 다시 계산하지 않고 원본 행을 가리킨다.
- 타일마다 class token(768)과 patch token 평균(768) 저장. 모델 출처는 exp/e01 이 기록한 저장소 파일 sha256, 체크포인트 sha256 과 대조한다 (달라졌으면 멈춤).
- 검증: 결정성(같은 입력 두 번 실행의 최대 차이), 원본(k=0) 특징을 기존 e01 특징과 비교(같은 이미지, 같은 좌표의 타일; 이미지 디코더가 달라 미세 차이 가능).
출력: features.npz (cls, patchmean, row_of), tile_index.csv, meta.json, log.txt
"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from bg_common import (Logger, augment_image, dist_to_mask, inputs, load_config, read_mask, read_rgb, sha256_file, write_csv)

SCRIPT = "b03_extract_features"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    bg, m = dict(cfg["bgaug"]), cfg["model"]
    out = cfg["paths"]["outputs_dir"]
    log = Logger(out, SCRIPT)
    log("config:", cfg["_config_path"])
    exp_out = Path(cfg["paths"]["exp_dir"]) / "outputs" / "e01_extract_features"
    repo, ckpt = Path(cfg["paths"]["dinov2_repo_dir"]), Path(cfg["paths"]["dinov2_checkpoint"])
    assert repo.exists() and ckpt.exists(), "캐시된 DINOv2 저장소/체크포인트가 없음"

    # ---------- 환경 / 출처 검증 ----------
    torch.manual_seed(bg["seed"]); np.random.seed(bg["seed"])
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    import cv2
    import torchvision
    log(f"torch {torch.__version__} | torchvision {torchvision.__version__} | cuda build {torch.version.cuda} | opencv {cv2.__version__} | device {dev}" + (f" ({torch.cuda.get_device_name(0)})" if dev == "cuda" else ""))
    old = {r["path"]: r["sha256"] for r in csv.DictReader(open(exp_out / "repo_files_sha256.csv", encoding="utf-8-sig"))}
    new = {f.relative_to(repo).as_posix(): sha256_file(f) for f in sorted(repo.rglob("*")) if f.is_file() and "__pycache__" not in f.parts}
    assert old == new, f"저장소 파일이 e01 기록과 다름: {len(set(old.items()) ^ set(new.items()))}건"
    meta_old = json.loads((exp_out / "meta.json").read_text(encoding="utf-8"))
    ck_sha = sha256_file(ckpt)
    assert ck_sha == meta_old["checkpoint_sha256"], "체크포인트가 e01 기록과 다름"
    assert meta_old["normalization"]["verified_against_repo"] and list(meta_old["normalization"]["mean"]) == m["mean"] and list(meta_old["normalization"]["std"]) == m["std"]
    log(f"모델 출처 검증: 저장소 파일 {len(new)}개 sha256 이 e01 기록과 모두 같음, 체크포인트 sha256 같음 ({ck_sha[:16]}…), 정규화 값이 e01 에서 공식 코드와 대조한 값과 같음")

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
    def embed(arrs):
        x = torch.from_numpy(np.stack(arrs)).to(dev).permute(0, 3, 1, 2).float() / 255.0
        o_ = model.forward_features((x - mean) / std)
        cls, pt = o_["x_norm_clstoken"], o_["x_norm_patchtokens"]
        assert cls.shape == (x.shape[0], 768) and pt.shape == (x.shape[0], (m["tile_input"] // 14) ** 2, 768)
        return cls.float().cpu().numpy(), pt.mean(dim=1).float().cpu().numpy()

    def embed_many(arrs):
        cl, pm = [], []
        for i in range(0, len(arrs), m["batch_size"]):
            c, p = embed(arrs[i:i + m["batch_size"]]); cl.append(c); pm.append(p)
        return np.concatenate(cl), np.concatenate(pm)

    # ---------- 입력 ----------
    pool = json.loads((Path(out) / "b01_measure_background" / "aug_params.json").read_text(encoding="utf-8"))["pool"]
    imgs, tiles = inputs(cfg, extract=True)      # 특징은 격자 전체에 대해 만든다 (분석용 부분집합은 b04 가 고른다)
    K, f = int(bg["n_aug"]), int(bg["feather_px"])
    log(f"이미지 {len(imgs)}장, 타일 {len(tiles)}개 (특징 추출 대상: 격자 전체), 증강본 K={K}, buffer {bg['buffer_px']}px, feather {f}px")
    by_img = {}
    for ti, t in enumerate(tiles):
        by_img.setdefault(t["image_id"], []).append(ti)
    row_of = np.full((len(tiles), K + 1), -1, np.int32)
    cls_all, pm_all, check = [], [], None
    n_unique_aug, n_same, t0, nrows = 0, 0, time.time(), 0
    T = m["tile_input"]
    for i, r in enumerate(imgs):
        img, mk = read_rgb(r["path"]), read_mask(r["mask"])
        d = dist_to_mask(mk)
        idxs = by_img[r["image_id"]]
        crops0 = []
        for ti in idxs:
            t = tiles[ti]; x0, y0 = int(t["x0"]), int(t["y0"])
            crops0.append(img[y0:y0 + T, x0:x0 + T].copy())
        c0, p0 = embed_many(crops0)
        base = nrows
        cls_all.append(c0); pm_all.append(p0); nrows += len(idxs)
        for j, ti in enumerate(idxs):
            row_of[ti, 0] = base + j
        if check is None:
            check = (crops0[:128], c0[:128].copy(), p0[:128].copy())
        for k in range(1, K + 1):
            aug, alpha, _ = augment_image(img, d, pool, bg, i, k, f)
            todo, todo_ti = [], []
            for j, ti in enumerate(idxs):
                t = tiles[ti]; x0, y0 = int(t["x0"]), int(t["y0"])
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
                nrows += len(todo); n_unique_aug += len(todo)
        print(f"  {i + 1}/{len(imgs)} {r['image_id']}  누적 행 {nrows}  {time.time() - t0:.0f}s", flush=True)
    cls_all, pm_all = np.concatenate(cls_all), np.concatenate(pm_all)
    assert (row_of >= 0).all() and cls_all.shape[0] == nrows == pm_all.shape[0]
    assert np.isfinite(cls_all).all() and np.isfinite(pm_all).all()
    log(f"\n특징 추출 {time.time() - t0:.0f}s: 계산한 타일 {nrows}개 (원본 {len(tiles)} + 증강본에서 달라진 {n_unique_aug}), 증강본에서 원본과 픽셀이 같아 재사용 {n_same}개")
    tot_aug = len(tiles) * K
    log(f"  증강본 타일 {tot_aug}개 중 달라진 것 {n_unique_aug}개 ({n_unique_aug / tot_aug * 100:.1f}%), 같은 것 {n_same}개 ({n_same / tot_aug * 100:.1f}%)")
    changed_tiles = int((row_of[:, 1:] != row_of[:, [0]]).any(1).sum())
    log(f"  배경 증강으로 한 번이라도 달라진 타일 {changed_tiles}개 (마스크 100%가 아니면서 buffer 바깥에 배경이 있는 타일)")

    # ---------- 검증 ----------
    c2, p2 = embed_many(check[0])
    d1, d2 = float(np.abs(c2 - check[1]).max()), float(np.abs(p2 - check[2]).max())
    log(f"결정성: 같은 입력(타일 {len(check[0])}개)을 두 번 실행한 최대 절대 차이 cls {d1:.3e}, patchmean {d2:.3e}")
    assert d1 < 1e-3 and d2 < 1e-3
    oz = np.load(exp_out / "features.npz")
    oidx = list(csv.DictReader(open(exp_out / "tile_index.csv", encoding="utf-8-sig")))
    key_new = {(t["image_id"], int(t["x0"]), int(t["y0"])): ti for ti, t in enumerate(tiles)}
    pairs = [(key_new[(o["image_id"], int(o["x0"]), int(o["y0"]))], ri) for ri, o in enumerate(oidx) if (o["image_id"], int(o["x0"]), int(o["y0"])) in key_new]
    if pairs:
        ni, oi = np.array([row_of[p[0], 0] for p in pairs]), np.array([p[1] for p in pairs])
        dc = np.abs(cls_all[ni] - oz["tile_cls"][oi]); a_, b_ = cls_all[ni], oz["tile_cls"][oi]
        cosv = (a_ * b_).sum(1) / (np.linalg.norm(a_, axis=1) * np.linalg.norm(b_, axis=1))
        log(f"기존 e01 특징과 비교(같은 이미지, 같은 좌표 타일 {len(pairs)}/{len(oidx)}개): cls 최대 절대 차이 {dc.max():.3e}, 평균 {dc.mean():.3e}, 코사인 유사도 최소 {cosv.min():.6f} "
            f"(차이는 이미지 디코더 cv2/PIL 때문일 수 있음)")
    else:
        log("기존 e01 타일과 좌표가 같은 타일이 없음 (비교 생략)")

    # ---------- 저장 ----------
    np.savez(Path(log.dir) / "features.npz", cls=cls_all.astype(np.float32), patchmean=pm_all.astype(np.float32), row_of=row_of)
    write_csv(f"{log.dir}/tile_index.csv", [dict(row=ti, image_id=t["image_id"], tile_id=t["tile_id"], x0=t["x0"], y0=t["y0"], label=t["label"], mask_frac=t["mask_frac"]) for ti, t in enumerate(tiles)])
    meta = dict(torch=torch.__version__, torchvision=torchvision.__version__, opencv=cv2.__version__, device=dev, gpu=torch.cuda.get_device_name(0) if dev == "cuda" else None,
                model=m["name"], params=n_param, checkpoint_sha256=ck_sha, n_tiles=len(tiles), n_images=len(imgs), n_aug=K, buffer_px=bg["buffer_px"], feather_px=f,
                n_rows=int(nrows), n_unique_aug=int(n_unique_aug), n_same_reused=int(n_same), determinism_max_abs_diff=dict(cls=d1, patchmean=d2),
                aug_params_sha256=sha256_file(Path(out) / "b01_measure_background" / "aug_params.json"))
    (Path(log.dir) / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"\n저장: {log.dir} (features.npz: cls {cls_all.shape}, patchmean {pm_all.shape}, row_of {row_of.shape}) | sha256 {sha256_file(Path(log.dir) / 'features.npz')}")


if __name__ == "__main__":
    main()
