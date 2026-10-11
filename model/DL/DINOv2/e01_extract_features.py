"""e01: 선택 타일에서 고정 DINOv2 ViT-B/14 특징을 뽑는다 (이 실험에서 GPU 를 쓰는 유일한 단계).

- 선택 타일 1,514개를 원본에서 224x224 로 잘라(resize 없음) 모델에 넣는다. 고정, eval, no_grad, fp32, ImageNet 정규화.
- 타일마다 class token(768)과 patch token 평균(768)을 저장한다. 키는 공식 코드를 읽어 확인한다(x_norm_clstoken, x_norm_patchtokens).
- baseline용: 이미지마다 시편 마스크(s02)의 bbox 로 잘라 224x224 로 resize 한 한 장의 class token 도 저장 (종횡비 왜곡, 스케일바 겹침 로그만; 이미지는 수정하지 않음)
- 모델 출처: 캐시된 공식 저장소 (커밋 불명, 2026-10-09 다운로드). pretrained=False 로 만든 뒤 캐시된 체크포인트를 strict=True 로 불러온다(네트워크 호출 없음).
  저장소 파일들과 가중치의 sha256 목록(repo_files_sha256.csv)으로 검증 가능하게 한다. 이미 목록이 있으면 현재 파일과 대조해 달라졌으면 멈춘다.
- 출력: features.npz, tile_index.csv, whole_index.csv, meta.json, repo_files_sha256.csv, log.txt
"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # 결정적 cuBLAS (torch import 전에 설정)

import argparse
import ast
import csv
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from common import Logger, load_config, load_image_table, load_selected_tiles, sha256_file, write_csv

SCRIPT = "e01_extract_features"


def file_manifest(repo_dir):
    rows = []
    for f in sorted(Path(repo_dir).rglob("*")):
        if f.is_file() and "__pycache__" not in f.parts:
            rows.append(dict(path=f.relative_to(repo_dir).as_posix(), sha256=sha256_file(f), size=f.stat().st_size))
    return rows


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

    # ---------- 환경 / 출처 ----------
    torch.manual_seed(cfg["cv"]["seed"]); np.random.seed(cfg["cv"]["seed"])
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    import torchvision
    log(f"torch {torch.__version__} | torchvision {torchvision.__version__} | cuda build {torch.version.cuda} | device {dev}" + (f" ({torch.cuda.get_device_name(0)})" if dev == "cuda" else ""))
    log(f"모델 출처: {m['repo_provenance']} (캐시 폴더 수정 시각 {time.strftime('%Y-%m-%d %H:%M', time.localtime(repo.stat().st_mtime))}, 체크포인트 {time.strftime('%Y-%m-%d %H:%M', time.localtime(ckpt.stat().st_mtime))})")
    man = file_manifest(repo)
    mp = o / "repo_files_sha256.csv"
    if mp.exists():
        old = {r["path"]: r["sha256"] for r in csv.DictReader(open(mp, encoding="utf-8-sig"))}
        new = {r["path"]: r["sha256"] for r in man}
        assert old == new, f"저장소 파일이 이전 기록과 다름: 변경/추가/삭제 {len(set(old.items()) ^ set(new.items()))}건"
        log(f"저장소 파일 {len(man)}개의 sha256 이 이전 기록과 모두 같음")
    else:
        write_csv(mp, man); log(f"저장소 파일 {len(man)}개의 sha256 목록 기록 -> {mp.name}")
    ck_sha = sha256_file(ckpt)
    log(f"체크포인트 {ckpt.name}: {ckpt.stat().st_size / 1e6:.0f}MB, sha256 {ck_sha}")

    # ---------- 공식 코드 확인 (정규화, 키 이름) ----------
    tf_src = (repo / "dinov2" / "data" / "transforms.py").read_text(encoding="utf-8")
    off_mean = ast.literal_eval(re.search(r"IMAGENET_DEFAULT_MEAN\s*=\s*(\([^)]*\))", tf_src).group(1))
    off_std = ast.literal_eval(re.search(r"IMAGENET_DEFAULT_STD\s*=\s*(\([^)]*\))", tf_src).group(1))
    assert tuple(m["mean"]) == tuple(off_mean) and tuple(m["std"]) == tuple(off_std), "config 정규화 값이 공식 repo 와 다름"
    log(f"정규화 확인: config mean/std == 공식 repo IMAGENET_DEFAULT_MEAN/STD {off_mean} / {off_std}")
    ev = re.search(r"def make_classification_eval_transform\((.*?)\n\) ->", tf_src, re.S)
    log("참고: 공식 평가 변환은 Resize 256(bicubic) -> CenterCrop 224. 이 실험은 타일 224 를 그대로 쓴다 (resize 없음, 의도된 차이)")
    vt_src = (repo / "dinov2" / "models" / "vision_transformer.py").read_text(encoding="utf-8")
    ff = vt_src[vt_src.index("def forward_features(self, x, masks=None)"):]
    ff = ff[:ff.index("\n    def ", 10)]
    for key in ("x_norm_clstoken", "x_norm_patchtokens"):
        assert f'"{key}"' in ff, f"공식 코드 forward_features 에 키 {key} 가 없음"
    log("공식 코드 확인: forward_features 반환 딕셔너리에 'x_norm_clstoken', 'x_norm_patchtokens' 키가 있음 (patch token 은 register token 을 제외한 부분)")

    # ---------- 모델 ----------
    model = torch.hub.load(str(repo), m["name"], source="local", pretrained=False, trust_repo=True)
    sd = torch.load(ckpt, map_location="cpu")
    res = model.load_state_dict(sd, strict=True)
    log(f"가중치 로드(strict=True): missing {len(res.missing_keys)}, unexpected {len(res.unexpected_keys)}")
    assert not res.missing_keys and not res.unexpected_keys
    model.eval().to(dev).float()
    for p in model.parameters():
        p.requires_grad_(False)
    n_param = sum(p.numel() for p in model.parameters())
    log(f"파라미터 수 {n_param:,} ({n_param / 1e6:.2f}M), 기대 {m['expected_params_million']}M ±0.5, patch_size={model.patch_size}, register tokens={getattr(model, 'num_register_tokens', 0)}")
    assert abs(n_param / 1e6 - m["expected_params_million"]) < 0.5
    mean = torch.tensor(m["mean"], device=dev).view(1, 3, 1, 1); std = torch.tensor(m["std"], device=dev).view(1, 3, 1, 1)

    @torch.no_grad()
    def embed(arrs, want_patch):
        """arrs: list of (224,224,3) uint8 -> cls (B,768), patchmean (B,768)"""
        x = torch.from_numpy(np.stack(arrs)).to(dev).permute(0, 3, 1, 2).float() / 255.0
        x = (x - mean) / std
        o_ = model.forward_features(x)
        cls = o_["x_norm_clstoken"]; pt = o_["x_norm_patchtokens"]
        assert cls.shape == (x.shape[0], 768) and pt.shape == (x.shape[0], (m["tile_input"] // 14) ** 2, 768), (cls.shape, pt.shape)
        return cls.float().cpu().numpy(), (pt.mean(dim=1).float().cpu().numpy() if want_patch else None)

    # ---------- 타일 ----------
    tiles = load_selected_tiles(cfg); imgs = {r["image_id"]: r for r in load_image_table(cfg)}
    assert len(tiles) == cfg["data"]["expected_tiles"]
    log(f"\n선택 타일 {len(tiles)}개, 이미지 {len(imgs)}장")
    cache = {}

    def crop_tiles(batch):
        out_ = []
        for t in batch:
            if t["image_id"] not in cache:
                cache.clear(); cache[t["image_id"]] = Image.open(imgs[t["image_id"]]["path"]).convert("RGB")
            x0, y0, w, h = int(t["x0"]), int(t["y0"]), int(t["w"]), int(t["h"])
            assert (w, h) == (m["tile_input"],) * 2
            arr = np.asarray(cache[t["image_id"]].crop((x0, y0, x0 + w, y0 + h)))
            assert arr.shape == (h, w, 3) and arr.dtype == np.uint8
            out_.append(arr)
        return out_

    def extract_tiles():
        cl, pm = [], []
        for i in range(0, len(tiles), m["batch_size"]):
            c, p = embed(crop_tiles(tiles[i:i + m["batch_size"]]), True)
            cl.append(c); pm.append(p)
        return np.concatenate(cl), np.concatenate(pm)

    t0 = time.time(); tile_cls, tile_pm = extract_tiles(); t1 = time.time() - t0
    log(f"타일 특징 추출 {t1:.1f}s: cls {tile_cls.shape}, patchmean {tile_pm.shape}")
    # 같은 입력 두 번째 실행 -> 결정성
    tile_cls2, tile_pm2 = extract_tiles()
    d1, d2 = float(np.abs(tile_cls - tile_cls2).max()), float(np.abs(tile_pm - tile_pm2).max())
    log(f"결정성: 같은 입력을 두 번 실행한 최대 절대 차이 cls {d1:.3e}, patchmean {d2:.3e}")

    # ---------- 이미지 전체 (마스크 bbox -> 224 resize) ----------
    ex = cfg["data"]["scalebar_exclude_box"]
    whole, widx, overlap = [], [], 0
    for k, (iid, r) in enumerate(sorted(imgs.items())):
        mk = np.asarray(Image.open(r["mask"]).convert("L")) > 127
        ys, xs = np.where(mk)
        x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
        im = Image.open(r["path"]).convert("RGB")
        crop = im.crop((x0, y0, x1, y1)); w, h = crop.size
        arr = np.asarray(crop.resize((m["tile_input"], m["tile_input"]), Image.BICUBIC))
        ratio = max(w / h, h / w)
        ov = int(x0 < ex[2] and x1 > ex[0] and y0 < ex[3] and y1 > ex[1])
        overlap += ov
        whole.append(arr); widx.append(dict(row=k, image_id=iid, label=r["label"], bbox_x0=x0, bbox_y0=y0, bbox_x1=x1, bbox_y1=y1, w=w, h=h, aspect_distortion=round(ratio, 4), overlaps_scalebar_zone=ov))
    whole_cls = np.concatenate([embed(whole[i:i + m["batch_size"]], False)[0] for i in range(0, len(whole), m["batch_size"])])
    whole_cls2 = np.concatenate([embed(whole[i:i + m["batch_size"]], False)[0] for i in range(0, len(whole), m["batch_size"])])
    dw = float(np.abs(whole_cls - whole_cls2).max())
    dist = np.array([r["aspect_distortion"] for r in widx])
    log(f"\n이미지 전체 특징: {whole_cls.shape}, 두 번 실행 최대 절대 차이 {dw:.3e}")
    log(f"종횡비 왜곡(max(w/h, h/w)): 평균 {dist.mean():.3f}, 최소 {dist.min():.3f}, 최대 {dist.max():.3f} (1.0 이면 정사각형, resize 로 가로세로를 224x224 에 맞춤)")
    log(f"마스크 bbox 가 스케일바 제외 구역 {ex} 과 겹치는 이미지: {overlap}/{len(widx)}장 (수정하지 않고 기록만)")

    # ---------- 검증 ----------
    for name, arr in (("tile_cls", tile_cls), ("tile_patchmean", tile_pm), ("whole_cls", whole_cls)):
        assert np.isfinite(arr).all(), f"{name} 에 NaN/Inf"
    assert tile_cls.shape == (cfg["data"]["expected_tiles"], 768) and tile_pm.shape == (cfg["data"]["expected_tiles"], 768)
    log("\n검증 통과: 타일 수 1,514, NaN/Inf 없음")
    for name, arr in (("tile_cls", tile_cls), ("tile_patchmean", tile_pm), ("whole_cls", whole_cls)):
        log(f"  {name:<15} 전체 평균 {arr.mean():+.4f} | 차원별 평균의 범위 [{arr.mean(0).min():+.3f}, {arr.mean(0).max():+.3f}] | 차원별 표준편차 평균 {arr.std(0).mean():.4f} (범위 [{arr.std(0).min():.3f}, {arr.std(0).max():.3f}]) | L2 노름 평균 {np.linalg.norm(arr, axis=1).mean():.2f}")

    # ---------- 저장 ----------
    np.savez(o / "features.npz", tile_cls=tile_cls.astype(np.float32), tile_patchmean=tile_pm.astype(np.float32), whole_cls=whole_cls.astype(np.float32))
    write_csv(o / "tile_index.csv", [dict(row=i, image_id=t["image_id"], tile_id=t["tile_id"], x0=t["x0"], y0=t["y0"], label=t["label"]) for i, t in enumerate(tiles)])
    write_csv(o / "whole_index.csv", widx)
    meta = dict(torch=torch.__version__, torchvision=torchvision.__version__, cuda_build=torch.version.cuda, device=dev,
                gpu=torch.cuda.get_device_name(0) if dev == "cuda" else None, model=m["name"], params=n_param,
                repo_provenance=m["repo_provenance"], checkpoint_sha256=ck_sha, repo_files=len(man),
                normalization=dict(mean=m["mean"], std=m["std"], verified_against_repo=True),
                keys_verified=["x_norm_clstoken", "x_norm_patchtokens"], determinism_max_abs_diff=dict(cls=d1, patchmean=d2, whole_cls=dw),
                n_tiles=len(tiles), n_images=len(imgs), aspect_distortion_mean=float(dist.mean()), whole_overlap_scalebar=overlap)
    (o / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"\n저장: {o} (features.npz {np.load(o / 'features.npz')['tile_cls'].shape}, tile_index.csv, whole_index.csv, meta.json, repo_files_sha256.csv)")


if __name__ == "__main__":
    main()
